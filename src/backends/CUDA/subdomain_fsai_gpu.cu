#include "backends/CUDA/subdomain_preconditioner_backend.hpp"
#include "backends/CUDA/subdomain_sparse_solve_common.cuh"
#include "factor/symbolic/symbolic.hpp"

#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    using namespace ichol::precond::detail::subdomain_common;

    void cusparse_check_named(cusparseStatus_t status, const char *what)
    {
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed with cuSPARSE status " + std::to_string(static_cast<int>(status)));
    }

    void cusolver_check_named(cusolverStatus_t status, const char *what)
    {
        if (status != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed with cuSOLVER status " + std::to_string(static_cast<int>(status)));
    }

    struct SubdomainFsaiVariant
    {
        double *d_local_in = nullptr;
        double *d_local_mid = nullptr;
        double *d_local_out = nullptr;
        double *d_val_g = nullptr;
        double *d_val_gt = nullptr;

        cusparseSpMatDescr_t mat_g = nullptr;
        cusparseSpMatDescr_t mat_gt = nullptr;
        cusparseDnVecDescr_t vec_in = nullptr;
        cusparseDnVecDescr_t vec_mid = nullptr;
        cusparseDnVecDescr_t vec_out = nullptr;
        void *buf_g = nullptr;
        void *buf_gt = nullptr;
    };

    struct SubdomainFsaiContext
    {
        int nsub = 0;
        int nnz = 0;

        int *d_gidx = nullptr;
        int *d_row_ptr_g = nullptr;
        int *d_col_ind_g = nullptr;
        int *d_row_ptr_gt = nullptr;
        int *d_col_ind_gt = nullptr;

        SubdomainFsaiVariant fp64;
    };

    __global__ void k_extract_and_pad_batched(
        const int *A_row_ptr, const int *A_col_ind, const double *A_val,
        const int *G_row_ptr, const int *G_col_ind,
        double *d_system, int max_pattern, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsub)
            return;

        const int row_begin = G_row_ptr[i];
        const int len = G_row_ptr[i + 1] - row_begin;
        double *matrix_slice = d_system + i * max_pattern * max_pattern;

        for (int r = 0; r < max_pattern; ++r)
        {
            for (int c = 0; c < max_pattern; ++c)
            {
                if (r < len && c < len)
                {
                    const int prow = G_col_ind[row_begin + r];
                    const int pcol = G_col_ind[row_begin + c];

                    double val = 0.0;
                    const int ap_start = A_row_ptr[prow];
                    const int ap_end = A_row_ptr[prow + 1];
                    for (int k = ap_start; k < ap_end; ++k)
                    {
                        if (A_col_ind[k] == pcol)
                        {
                            val = A_val[k];
                            break;
                        }
                    }
                    matrix_slice[r + c * max_pattern] = val;
                }
                else
                {
                    matrix_slice[r + c * max_pattern] = (r == c) ? 1.0 : 0.0;
                }
            }
        }
    }

    __global__ void k_build_rhs_batched(
        const int *G_row_ptr, double *d_rhs, int max_pattern, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsub)
            return;

        const int len = G_row_ptr[i + 1] - G_row_ptr[i];
        double *rhs_slice = d_rhs + i * max_pattern;

        for (int r = 0; r < max_pattern; ++r)
        {
            rhs_slice[r] = (r == len - 1) ? 1.0 : 0.0;
        }
    }

    __global__ void k_setup_pointer_arrays(
        double **A_arr, double **B_arr, double *d_system, double *d_rhs,
        int max_pattern, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nsub)
        {
            A_arr[i] = d_system + i * max_pattern * max_pattern;
            B_arr[i] = d_rhs + i * max_pattern;
        }
    }

    __global__ void k_scale_store_batched(
        const double *d_rhs, double *d_val_g, const int *G_row_ptr,
        int max_pattern, const int *d_info, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsub)
            return;
        if (d_info[i] != 0)
            return;

        const int row_begin = G_row_ptr[i];
        const int len = G_row_ptr[i + 1] - row_begin;

        const double alpha = d_rhs[i * max_pattern + (len - 1)];
        const double scale = 1.0 / sqrt(alpha);

        for (int r = 0; r < len; ++r)
        {
            d_val_g[row_begin + r] = d_rhs[i * max_pattern + r] * scale;
        }
    }

    ichol::matrix::CsrMatrix<double> extract_full_subdomain_csr(
        const ichol::matrix::CsrMatrix<double> &A,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg)
    {
        const auto A_lower = extract_lower_subdomain_csr(A, global, reg);
        const int n = A_lower.num_rows;

        std::vector<int> counts(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i)
        {
            for (int p = A_lower.row_ptr[static_cast<std::size_t>(i)];
                 p < A_lower.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            {
                counts[static_cast<std::size_t>(i)]++;
                const int j = A_lower.col_ind[static_cast<std::size_t>(p)];
                if (i != j)
                {
                    counts[static_cast<std::size_t>(j)]++;
                }
            }
        }

        ichol::matrix::CsrMatrix<double> out;
        out.num_rows = n;
        out.num_cols = n;
        out.row_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
        for (int i = 0; i < n; ++i)
        {
            out.row_ptr[static_cast<std::size_t>(i) + 1] = out.row_ptr[static_cast<std::size_t>(i)] + counts[static_cast<std::size_t>(i)];
        }

        const int nnz_total = out.row_ptr.back();
        out.col_ind.resize(static_cast<std::size_t>(nnz_total));
        out.values.resize(static_cast<std::size_t>(nnz_total));
        out.nnz = nnz_total;

        std::vector<int> current = out.row_ptr;
        for (int i = 0; i < n; ++i)
        {
            for (int p = A_lower.row_ptr[static_cast<std::size_t>(i)];
                 p < A_lower.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            {
                const int j = A_lower.col_ind[static_cast<std::size_t>(p)];
                const double v = A_lower.values[static_cast<std::size_t>(p)];

                int pos_i = current[static_cast<std::size_t>(i)]++;
                out.col_ind[static_cast<std::size_t>(pos_i)] = j;
                out.values[static_cast<std::size_t>(pos_i)] = v;

                if (i != j)
                {
                    int pos_j = current[static_cast<std::size_t>(j)]++;
                    out.col_ind[static_cast<std::size_t>(pos_j)] = i;
                    out.values[static_cast<std::size_t>(pos_j)] = v;
                }
            }
        }

        for (int i = 0; i < n; ++i)
        {
            const int start = out.row_ptr[static_cast<std::size_t>(i)];
            const int end = out.row_ptr[static_cast<std::size_t>(i) + 1];
            const int len = end - start;
            std::vector<std::pair<int, double>> temp(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k)
            {
                temp[static_cast<std::size_t>(k)] = {out.col_ind[static_cast<std::size_t>(start + k)], out.values[static_cast<std::size_t>(start + k)]};
            }
            std::sort(temp.begin(), temp.end(), [](const auto &lhs, const auto &rhs)
                      { return lhs.first < rhs.first; });
            for (int k = 0; k < len; ++k)
            {
                out.col_ind[static_cast<std::size_t>(start + k)] = temp[static_cast<std::size_t>(k)].first;
                out.values[static_cast<std::size_t>(start + k)] = temp[static_cast<std::size_t>(k)].second;
            }
        }

        return out;
    }

    ichol::matrix::CsrMatrix<double> build_fsai_pattern_csr(
        const ichol::matrix::CsrMatrix<double> &A_lower,
        int level_k)
    {
        const auto pattern = ichol::symbolic::compute_ic_factor_pattern(A_lower, level_k);
        ichol::matrix::CsrMatrix<double> out;
        out.num_rows = A_lower.num_rows;
        out.num_cols = A_lower.num_cols;
        out.row_ptr = pattern.row_ptr_L;
        out.col_ind = pattern.col_ind_L;
        out.values.assign(pattern.col_ind_L.size(), 0.0);
        out.nnz = static_cast<int>(out.col_ind.size());
        return out;
    }

    void destroy_variant(SubdomainFsaiVariant &variant)
    {
        if (variant.vec_out)
            cusparseDestroyDnVec(variant.vec_out);
        if (variant.vec_mid)
            cusparseDestroyDnVec(variant.vec_mid);
        if (variant.vec_in)
            cusparseDestroyDnVec(variant.vec_in);
        if (variant.mat_gt)
            cusparseDestroySpMat(variant.mat_gt);
        if (variant.mat_g)
            cusparseDestroySpMat(variant.mat_g);

        cudaFree(variant.buf_gt);
        cudaFree(variant.buf_g);
        cudaFree(variant.d_local_out);
        cudaFree(variant.d_local_mid);
        cudaFree(variant.d_local_in);
        cudaFree(variant.d_val_gt);
        cudaFree(variant.d_val_g);
    }

    void destroy_ctx_impl(SubdomainFsaiContext *ctx)
    {
        if (!ctx)
            return;
        destroy_variant(ctx->fp64);
        cudaFree(ctx->d_col_ind_gt);
        cudaFree(ctx->d_row_ptr_gt);
        cudaFree(ctx->d_col_ind_g);
        cudaFree(ctx->d_row_ptr_g);
        cudaFree(ctx->d_gidx);
        delete ctx;
    }

    void init_variant(SubdomainFsaiContext *ctx, cusparseHandle_t cusparse)
    {
        cuda_check(cudaMalloc(&ctx->fp64.d_local_in, static_cast<std::size_t>(ctx->nsub) * sizeof(double)));
        cuda_check(cudaMalloc(&ctx->fp64.d_local_mid, static_cast<std::size_t>(ctx->nsub) * sizeof(double)));
        cuda_check(cudaMalloc(&ctx->fp64.d_local_out, static_cast<std::size_t>(ctx->nsub) * sizeof(double)));

        cusparse_check_named(cusparseCreateCsr(
                                 &ctx->fp64.mat_g,
                                 ctx->nsub, ctx->nsub, ctx->nnz,
                                 ctx->d_row_ptr_g, ctx->d_col_ind_g, ctx->fp64.d_val_g,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                             "cusparseCreateCsr(mat_g)");
        cusparse_check_named(cusparseCreateCsr(
                                 &ctx->fp64.mat_gt,
                                 ctx->nsub, ctx->nsub, ctx->nnz,
                                 ctx->d_row_ptr_gt, ctx->d_col_ind_gt, ctx->fp64.d_val_gt,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                             "cusparseCreateCsr(mat_gt)");

        cusparse_check_named(cusparseCreateDnVec(&ctx->fp64.vec_in, ctx->nsub, ctx->fp64.d_local_in, CUDA_R_64F), "cusparseCreateDnVec(vec_in)");
        cusparse_check_named(cusparseCreateDnVec(&ctx->fp64.vec_mid, ctx->nsub, ctx->fp64.d_local_mid, CUDA_R_64F), "cusparseCreateDnVec(vec_mid)");
        cusparse_check_named(cusparseCreateDnVec(&ctx->fp64.vec_out, ctx->nsub, ctx->fp64.d_local_out, CUDA_R_64F), "cusparseCreateDnVec(vec_out)");

        const double alpha = 1.0;
        const double beta = 0.0;
        size_t buf_size_g = 0;
        size_t buf_size_gt = 0;

        cusparse_check_named(cusparseSpMV_bufferSize(
                                 cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, ctx->fp64.mat_g, ctx->fp64.vec_in, &beta, ctx->fp64.vec_mid,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_g),
                             "cusparseSpMV_bufferSize(mat_g)");
        if (buf_size_g > 0)
            cuda_check(cudaMalloc(&ctx->fp64.buf_g, buf_size_g));

        cusparse_check_named(cusparseSpMV_bufferSize(
                                 cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, ctx->fp64.mat_gt, ctx->fp64.vec_mid, &beta, ctx->fp64.vec_out,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_gt),
                             "cusparseSpMV_bufferSize(mat_gt)");
        if (buf_size_gt > 0)
            cuda_check(cudaMalloc(&ctx->fp64.buf_gt, buf_size_gt));
    }

    void apply_fsai_impl(
        SubdomainFsaiContext *ctx,
        const double *d_r,
        double *d_z,
        cudaStream_t stream)
    {
        thread_local cusparseHandle_t thread_cusparse = nullptr;
        if (!thread_cusparse)
        {
            cusparse_check_named(cusparseCreate(&thread_cusparse), "cusparseCreate (thread_local)");
        }

        const int threads = 256;
        const int blocks = (ctx->nsub + threads - 1) / threads;
        const double alpha = 1.0;
        const double beta = 0.0;

        k_gather_subvec<<<blocks, threads, 0, stream>>>(d_r, ctx->d_gidx, ctx->fp64.d_local_in, ctx->nsub);
        cusparse_check_named(cusparseSetStream(thread_cusparse, stream), "cusparseSetStream");
        cusparse_check_named(cusparseSpMV(
                                 thread_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, ctx->fp64.mat_g, ctx->fp64.vec_in, &beta, ctx->fp64.vec_mid,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->fp64.buf_g),
                             "cusparseSpMV(mat_g)");
        cusparse_check_named(cusparseSpMV(
                                 thread_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, ctx->fp64.mat_gt, ctx->fp64.vec_mid, &beta, ctx->fp64.vec_out,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->fp64.buf_gt),
                             "cusparseSpMV(mat_gt)");
        k_scatter_subvec<<<blocks, threads, 0, stream>>>(ctx->fp64.d_local_out, ctx->d_gidx, d_z, ctx->nsub);
        cuda_check(cudaGetLastError());
    }
} // namespace

namespace ichol::precond::detail
{
    void *create_subdomain_fsai_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &reg,
        const SubdomainPreconditionerOptions &options)
    {
        auto *ctx = new SubdomainFsaiContext();

        int *d_row_ptr_full = nullptr;
        int *d_col_ind_full = nullptr;
        double *d_val_full = nullptr;
        double *d_system = nullptr;
        double *d_rhs = nullptr;
        double **d_A_array = nullptr;
        double **d_B_array = nullptr;
        int *d_info_array = nullptr;
        void *d_csr2csc_buf = nullptr;
        cusparseHandle_t cusparse_init = nullptr;
        cusolverDnHandle_t cusolver = nullptr;

        try
        {
            const int lw = reg.x1 - reg.x0;
            const int lh = reg.y1 - reg.y0;
            const int ld = reg.z1 - reg.z0;
            ctx->nsub = lw * lh * ld;
            if (ctx->nsub <= 0)
                throw std::runtime_error("create_subdomain_fsai_context: empty subdomain");

            const auto A_lower = extract_lower_subdomain_csr(A, global, reg);
            const auto A_full = extract_full_subdomain_csr(A, global, reg);
            const auto G_pattern = build_fsai_pattern_csr(A_lower, options.fsai_level_k);
            ctx->nnz = G_pattern.nnz;

            cuda_check(cudaMalloc(&ctx->d_gidx, static_cast<std::size_t>(ctx->nsub) * sizeof(int)));
            build_subdomain_gidx(ctx->d_gidx, lw, lh, ld, global.w, global.h, reg.x0, reg.y0, reg.z0);

            cuda_check(cudaMalloc(&ctx->d_row_ptr_g, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_g, static_cast<std::size_t>(ctx->nnz) * sizeof(int)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr_g, G_pattern.row_ptr.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind_g, G_pattern.col_ind.data(), static_cast<std::size_t>(ctx->nnz) * sizeof(int), cudaMemcpyHostToDevice));

            cuda_check(cudaMalloc(&ctx->fp64.d_val_g, static_cast<std::size_t>(ctx->nnz) * sizeof(double)));
            cuda_check(cudaMalloc(&ctx->fp64.d_val_gt, static_cast<std::size_t>(ctx->nnz) * sizeof(double)));

            const int nnz_full = A_full.nnz;
            cuda_check(cudaMalloc(&d_row_ptr_full, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&d_col_ind_full, static_cast<std::size_t>(nnz_full) * sizeof(int)));
            cuda_check(cudaMalloc(&d_val_full, static_cast<std::size_t>(nnz_full) * sizeof(double)));
            cuda_check(cudaMemcpy(d_row_ptr_full, A_full.row_ptr.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_col_ind_full, A_full.col_ind.data(), static_cast<std::size_t>(nnz_full) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_val_full, A_full.values.data(), static_cast<std::size_t>(nnz_full) * sizeof(double), cudaMemcpyHostToDevice));

            cusparse_check_named(cusparseCreate(&cusparse_init), "cusparseCreate");
            cusolver_check_named(cusolverDnCreate(&cusolver), "cusolverDnCreate");

            int max_pattern = 0;
            for (int i = 0; i < ctx->nsub; ++i)
                max_pattern = std::max(max_pattern, G_pattern.row_ptr[static_cast<std::size_t>(i) + 1] - G_pattern.row_ptr[static_cast<std::size_t>(i)]);
            if (max_pattern <= 0)
                throw std::runtime_error("create_subdomain_fsai_context: invalid row pattern size");

            const std::size_t batched_mat_size = static_cast<std::size_t>(ctx->nsub) * max_pattern * max_pattern * sizeof(double);
            const std::size_t batched_vec_size = static_cast<std::size_t>(ctx->nsub) * max_pattern * sizeof(double);
            cuda_check(cudaMalloc(&d_system, batched_mat_size));
            cuda_check(cudaMalloc(&d_rhs, batched_vec_size));
            cuda_check(cudaMalloc(&d_A_array, static_cast<std::size_t>(ctx->nsub) * sizeof(double *)));
            cuda_check(cudaMalloc(&d_B_array, static_cast<std::size_t>(ctx->nsub) * sizeof(double *)));
            cuda_check(cudaMalloc(&d_info_array, static_cast<std::size_t>(ctx->nsub) * sizeof(int)));

            const int threads = 256;
            const int blocks = (ctx->nsub + threads - 1) / threads;

            k_extract_and_pad_batched<<<blocks, threads>>>(
                d_row_ptr_full, d_col_ind_full, d_val_full,
                ctx->d_row_ptr_g, ctx->d_col_ind_g,
                d_system, max_pattern, ctx->nsub);

            k_build_rhs_batched<<<blocks, threads>>>(
                ctx->d_row_ptr_g, d_rhs, max_pattern, ctx->nsub);

            k_setup_pointer_arrays<<<blocks, threads>>>(
                d_A_array, d_B_array, d_system, d_rhs, max_pattern, ctx->nsub);

            cuda_check(cudaGetLastError());
            cuda_check(cudaDeviceSynchronize());

            cusolver_check_named(cusolverDnDpotrfBatched(
                                     cusolver, CUBLAS_FILL_MODE_LOWER, max_pattern, d_A_array, max_pattern, d_info_array, ctx->nsub),
                                 "cusolverDnDpotrfBatched");

            cusolver_check_named(cusolverDnDpotrsBatched(
                                     cusolver, CUBLAS_FILL_MODE_LOWER, max_pattern, 1, d_A_array, max_pattern, d_B_array, max_pattern, d_info_array, ctx->nsub),
                                 "cusolverDnDpotrsBatched");

            std::vector<int> h_info_array(static_cast<std::size_t>(ctx->nsub));
            cuda_check(cudaMemcpy(h_info_array.data(), d_info_array, static_cast<std::size_t>(ctx->nsub) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < ctx->nsub; ++i)
            {
                if (h_info_array[static_cast<std::size_t>(i)] != 0)
                {
                    throw std::runtime_error("create_subdomain_fsai_context: Batched solve failed at row " + std::to_string(i));
                }
            }

            k_scale_store_batched<<<blocks, threads>>>(
                d_rhs, ctx->fp64.d_val_g, ctx->d_row_ptr_g, max_pattern, d_info_array, ctx->nsub);
            cuda_check(cudaGetLastError());
            cuda_check(cudaDeviceSynchronize());

            cuda_check(cudaMalloc(&ctx->d_row_ptr_gt, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_gt, static_cast<std::size_t>(ctx->nnz) * sizeof(int)));

            size_t csr2csc_buf_size = 0;
            cusparse_check_named(cusparseCsr2cscEx2_bufferSize(
                                     cusparse_init,
                                     ctx->nsub, ctx->nsub, ctx->nnz,
                                     ctx->fp64.d_val_g, ctx->d_row_ptr_g, ctx->d_col_ind_g,
                                     ctx->fp64.d_val_gt, ctx->d_row_ptr_gt, ctx->d_col_ind_gt,
                                     CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG1, &csr2csc_buf_size),
                                 "cusparseCsr2cscEx2_bufferSize");

            if (csr2csc_buf_size > 0)
                cuda_check(cudaMalloc(&d_csr2csc_buf, csr2csc_buf_size));

            cusparse_check_named(cusparseCsr2cscEx2(
                                     cusparse_init,
                                     ctx->nsub, ctx->nsub, ctx->nnz,
                                     ctx->fp64.d_val_g, ctx->d_row_ptr_g, ctx->d_col_ind_g,
                                     ctx->fp64.d_val_gt, ctx->d_row_ptr_gt, ctx->d_col_ind_gt,
                                     CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG1, d_csr2csc_buf),
                                 "cusparseCsr2cscEx2");

            init_variant(ctx, cusparse_init);

            cudaFree(d_csr2csc_buf);
            cudaFree(d_info_array);
            cudaFree(d_B_array);
            cudaFree(d_A_array);
            cudaFree(d_rhs);
            cudaFree(d_system);
            cudaFree(d_val_full);
            cudaFree(d_col_ind_full);
            cudaFree(d_row_ptr_full);

            if (cusparse_init)
                cusparseDestroy(cusparse_init);
            if (cusolver)
                cusolverDnDestroy(cusolver);

            return ctx;
        }
        catch (...)
        {
            cudaFree(d_csr2csc_buf);
            cudaFree(d_info_array);
            cudaFree(d_B_array);
            cudaFree(d_A_array);
            cudaFree(d_rhs);
            cudaFree(d_system);
            cudaFree(d_val_full);
            cudaFree(d_col_ind_full);
            cudaFree(d_row_ptr_full);

            if (cusparse_init)
                cusparseDestroy(cusparse_init);
            if (cusolver)
                cusolverDnDestroy(cusolver);

            destroy_ctx_impl(ctx);
            throw;
        }
    }

    void apply_subdomain_fsai(
        void *vctx,
        const void *d_r,
        void *d_z,
        int /*N*/,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainFsaiContext *>(vctx);
        apply_fsai_impl(ctx, static_cast<const double *>(d_r), static_cast<double *>(d_z), stream);
    }

    void destroy_subdomain_fsai_context(void *vctx)
    {
        destroy_ctx_impl(reinterpret_cast<SubdomainFsaiContext *>(vctx));
    }
} // namespace ichol::precond::detail