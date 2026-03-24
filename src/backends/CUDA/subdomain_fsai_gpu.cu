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

    template <typename T>
    struct SubdomainFsaiVariant
    {
        T *d_local_in = nullptr;
        T *d_local_mid = nullptr;
        T *d_local_out = nullptr;
        T *d_val_g = nullptr;
        T *d_val_gt = nullptr;

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

        cusparseHandle_t cusparse = nullptr;
        SubdomainFsaiVariant<double> fp64;
        SubdomainFsaiVariant<float> fp32;
    };

    __global__ void k_extract_principal_submatrix(
        const double *a_dense,
        int lda_dense,
        const int *pattern,
        int len,
        double *a_sys,
        int lda_sys)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = len * len;
        if (idx >= total)
            return;

        const int row = idx % len;
        const int col = idx / len;
        const int prow = pattern[row];
        const int pcol = pattern[col];
        a_sys[row + col * lda_sys] = a_dense[prow * lda_dense + pcol];
    }

    __global__ void k_build_unit_rhs(double *rhs, int len)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < len)
            rhs[i] = (i == len - 1) ? 1.0 : 0.0;
    }

    __global__ void k_scale_store_row(const double *src, double *dst, int len, double scale)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < len)
            dst[i] = src[i] * scale;
    }

    ichol::matrix::CsrMatrix<double> extract_full_subdomain_csr(
        const ichol::matrix::CsrMatrix<double> &A,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg)
    {
        const auto A_lower = extract_lower_subdomain_csr(A, global, reg);
        const int n = A_lower.num_rows;

        std::vector<std::vector<std::pair<int, double>>> rows(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            for (int p = A_lower.row_ptr[static_cast<std::size_t>(i)];
                 p < A_lower.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            {
                const int j = A_lower.col_ind[static_cast<std::size_t>(p)];
                const double v = A_lower.values[static_cast<std::size_t>(p)];
                rows[static_cast<std::size_t>(i)].push_back({j, v});
                if (j != i)
                    rows[static_cast<std::size_t>(j)].push_back({i, v});
            }
        }

        ichol::matrix::CsrMatrix<double> out;
        out.num_rows = n;
        out.num_cols = n;
        out.row_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
        for (int i = 0; i < n; ++i)
        {
            auto &row = rows[static_cast<std::size_t>(i)];
            std::sort(row.begin(), row.end(), [](const auto &lhs, const auto &rhs)
                      { return lhs.first < rhs.first; });
            for (const auto &[j, v] : row)
            {
                out.col_ind.push_back(j);
                out.values.push_back(v);
            }
            out.row_ptr[static_cast<std::size_t>(i) + 1] = static_cast<int>(out.col_ind.size());
        }
        out.nnz = static_cast<int>(out.values.size());
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

    template <typename T>
    void destroy_variant(SubdomainFsaiVariant<T> &variant)
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

        destroy_variant(ctx->fp32);
        destroy_variant(ctx->fp64);
        if (ctx->cusparse)
            cusparseDestroy(ctx->cusparse);

        cudaFree(ctx->d_col_ind_gt);
        cudaFree(ctx->d_row_ptr_gt);
        cudaFree(ctx->d_col_ind_g);
        cudaFree(ctx->d_row_ptr_g);
        cudaFree(ctx->d_gidx);
        delete ctx;
    }

    template <typename T>
    void init_variant(SubdomainFsaiContext *ctx, SubdomainFsaiVariant<T> &variant)
    {
        cuda_check(cudaMalloc(&variant.d_local_in, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMalloc(&variant.d_local_mid, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMalloc(&variant.d_local_out, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));

        cusparse_check_named(cusparseCreateCsr(
            &variant.mat_g,
            ctx->nsub, ctx->nsub, ctx->nnz,
            ctx->d_row_ptr_g, ctx->d_col_ind_g, variant.d_val_g,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<T>()),
            "cusparseCreateCsr(mat_g)");
        cusparse_check_named(cusparseCreateCsr(
            &variant.mat_gt,
            ctx->nsub, ctx->nsub, ctx->nnz,
            ctx->d_row_ptr_gt, ctx->d_col_ind_gt, variant.d_val_gt,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<T>()),
            "cusparseCreateCsr(mat_gt)");

        cusparse_check_named(cusparseCreateDnVec(&variant.vec_in, ctx->nsub, variant.d_local_in, cuda_data_type<T>()), "cusparseCreateDnVec(vec_in)");
        cusparse_check_named(cusparseCreateDnVec(&variant.vec_mid, ctx->nsub, variant.d_local_mid, cuda_data_type<T>()), "cusparseCreateDnVec(vec_mid)");
        cusparse_check_named(cusparseCreateDnVec(&variant.vec_out, ctx->nsub, variant.d_local_out, cuda_data_type<T>()), "cusparseCreateDnVec(vec_out)");

        const T alpha = static_cast<T>(1);
        const T beta = static_cast<T>(0);
        size_t buf_size_g = 0;
        size_t buf_size_gt = 0;

        cusparse_check_named(cusparseSpMV_bufferSize(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_g, variant.vec_in, &beta, variant.vec_mid,
            cuda_data_type<T>(), CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_g),
            "cusparseSpMV_bufferSize(mat_g)");
        if (buf_size_g > 0)
            cuda_check(cudaMalloc(&variant.buf_g, buf_size_g));

        cusparse_check_named(cusparseSpMV_bufferSize(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_gt, variant.vec_mid, &beta, variant.vec_out,
            cuda_data_type<T>(), CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_gt),
            "cusparseSpMV_bufferSize(mat_gt)");
        if (buf_size_gt > 0)
            cuda_check(cudaMalloc(&variant.buf_gt, buf_size_gt));
    }

    template <typename T>
    void apply_fsai_impl(
        SubdomainFsaiContext *ctx,
        SubdomainFsaiVariant<T> &variant,
        const T *d_r,
        T *d_z,
        cudaStream_t stream)
    {
        const int threads = 256;
        const int blocks = (ctx->nsub + threads - 1) / threads;
        const T alpha = static_cast<T>(1);
        const T beta = static_cast<T>(0);

        k_gather_subvec<<<blocks, threads, 0, stream>>>(d_r, ctx->d_gidx, variant.d_local_in, ctx->nsub);
        cusparse_check_named(cusparseSetStream(ctx->cusparse, stream), "cusparseSetStream");
        cusparse_check_named(cusparseSpMV(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_g, variant.vec_in, &beta, variant.vec_mid,
            cuda_data_type<T>(), CUSPARSE_SPMV_ALG_DEFAULT, variant.buf_g), "cusparseSpMV(mat_g)");
        cusparse_check_named(cusparseSpMV(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_gt, variant.vec_mid, &beta, variant.vec_out,
            cuda_data_type<T>(), CUSPARSE_SPMV_ALG_DEFAULT, variant.buf_gt), "cusparseSpMV(mat_gt)");
        k_scatter_subvec<<<blocks, threads, 0, stream>>>(variant.d_local_out, ctx->d_gidx, d_z, ctx->nsub);
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
        double *d_a_dense = nullptr;
        void *d_sparse_to_dense_buf = nullptr;
        double *d_system = nullptr;
        double *d_rhs = nullptr;
        double *d_potrf_work = nullptr;
        int *d_info = nullptr;
        cusparseSpMatDescr_t mat_full = nullptr;
        cusparseDnMatDescr_t dn_a_dense = nullptr;
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
            cuda_check(cudaMalloc(&ctx->fp32.d_val_g, static_cast<std::size_t>(ctx->nnz) * sizeof(float)));
            cuda_check(cudaMalloc(&ctx->fp32.d_val_gt, static_cast<std::size_t>(ctx->nnz) * sizeof(float)));

            const int nnz_full = A_full.nnz;
            cuda_check(cudaMalloc(&d_row_ptr_full, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&d_col_ind_full, static_cast<std::size_t>(nnz_full) * sizeof(int)));
            cuda_check(cudaMalloc(&d_val_full, static_cast<std::size_t>(nnz_full) * sizeof(double)));
            cuda_check(cudaMemcpy(d_row_ptr_full, A_full.row_ptr.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_col_ind_full, A_full.col_ind.data(), static_cast<std::size_t>(nnz_full) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_val_full, A_full.values.data(), static_cast<std::size_t>(nnz_full) * sizeof(double), cudaMemcpyHostToDevice));

            cusparse_check_named(cusparseCreate(&ctx->cusparse), "cusparseCreate");
            cusolver_check_named(cusolverDnCreate(&cusolver), "cusolverDnCreate");

            cusparse_check_named(cusparseCreateCsr(
                &mat_full,
                ctx->nsub, ctx->nsub, nnz_full,
                d_row_ptr_full, d_col_ind_full, d_val_full,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "cusparseCreateCsr(mat_full)");

            cuda_check(cudaMalloc(&d_a_dense, static_cast<std::size_t>(ctx->nsub) * static_cast<std::size_t>(ctx->nsub) * sizeof(double)));
            cuda_check(cudaMemset(d_a_dense, 0, static_cast<std::size_t>(ctx->nsub) * static_cast<std::size_t>(ctx->nsub) * sizeof(double)));
            cusparse_check_named(cusparseCreateDnMat(
                &dn_a_dense,
                ctx->nsub, ctx->nsub, ctx->nsub, d_a_dense,
                CUDA_R_64F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(dn_a_dense)");

            size_t sparse_to_dense_buf_size = 0;
            cusparse_check_named(cusparseSparseToDense_bufferSize(
                ctx->cusparse, mat_full, dn_a_dense,
                CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &sparse_to_dense_buf_size),
                "cusparseSparseToDense_bufferSize");
            if (sparse_to_dense_buf_size > 0)
                cuda_check(cudaMalloc(&d_sparse_to_dense_buf, sparse_to_dense_buf_size));
            cusparse_check_named(cusparseSparseToDense(
                ctx->cusparse, mat_full, dn_a_dense,
                CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_sparse_to_dense_buf),
                "cusparseSparseToDense");

            int max_pattern = 0;
            for (int i = 0; i < ctx->nsub; ++i)
                max_pattern = std::max(max_pattern, G_pattern.row_ptr[static_cast<std::size_t>(i) + 1] - G_pattern.row_ptr[static_cast<std::size_t>(i)]);
            if (max_pattern <= 0)
                throw std::runtime_error("create_subdomain_fsai_context: invalid row pattern size");

            cuda_check(cudaMalloc(&d_system, static_cast<std::size_t>(max_pattern) * static_cast<std::size_t>(max_pattern) * sizeof(double)));
            cuda_check(cudaMalloc(&d_rhs, static_cast<std::size_t>(max_pattern) * sizeof(double)));
            cuda_check(cudaMalloc(&d_info, sizeof(int)));

            int potrf_lwork = 0;
            cusolver_check_named(cusolverDnDpotrf_bufferSize(
                cusolver, CUBLAS_FILL_MODE_LOWER, max_pattern, d_system, max_pattern, &potrf_lwork),
                "cusolverDnDpotrf_bufferSize");
            if (potrf_lwork > 0)
                cuda_check(cudaMalloc(&d_potrf_work, static_cast<std::size_t>(potrf_lwork) * sizeof(double)));

            for (int row = 0; row < ctx->nsub; ++row)
            {
                const int row_begin = G_pattern.row_ptr[static_cast<std::size_t>(row)];
                const int row_end = G_pattern.row_ptr[static_cast<std::size_t>(row) + 1];
                const int len = row_end - row_begin;
                const int threads = 256;
                const int mat_blocks = (len * len + threads - 1) / threads;
                const int vec_blocks = (len + threads - 1) / threads;

                k_extract_principal_submatrix<<<mat_blocks, threads>>>(
                    d_a_dense, ctx->nsub, ctx->d_col_ind_g + row_begin, len, d_system, max_pattern);
                k_build_unit_rhs<<<vec_blocks, threads>>>(d_rhs, len);
                cuda_check(cudaGetLastError());

                cusolver_check_named(cusolverDnDpotrf(
                    cusolver, CUBLAS_FILL_MODE_LOWER, len, d_system, max_pattern, d_potrf_work, potrf_lwork, d_info),
                    "cusolverDnDpotrf");
                int info = 0;
                cuda_check(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
                if (info != 0)
                    throw std::runtime_error("create_subdomain_fsai_context: FSAI principal block Cholesky failed at row " +
                                             std::to_string(row) + " with info=" + std::to_string(info));

                cusolver_check_named(cusolverDnDpotrs(
                    cusolver, CUBLAS_FILL_MODE_LOWER, len, 1, d_system, max_pattern, d_rhs, max_pattern, d_info),
                    "cusolverDnDpotrs");
                cuda_check(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
                if (info != 0)
                    throw std::runtime_error("create_subdomain_fsai_context: FSAI principal solve failed at row " +
                                             std::to_string(row) + " with info=" + std::to_string(info));

                double alpha = 0.0;
                cuda_check(cudaMemcpy(&alpha, d_rhs + (len - 1), sizeof(double), cudaMemcpyDeviceToHost));
                if (!(alpha > 0.0) || !std::isfinite(alpha))
                    throw std::runtime_error("create_subdomain_fsai_context: non-positive normalization pivot at row " + std::to_string(row));

                const double scale = 1.0 / std::sqrt(alpha);
                k_scale_store_row<<<vec_blocks, threads>>>(d_rhs, ctx->fp64.d_val_g + row_begin, len, scale);
                cuda_check(cudaGetLastError());
            }
            cuda_check(cudaDeviceSynchronize());

            cuda_check(cudaMalloc(&ctx->d_row_ptr_gt, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_gt, static_cast<std::size_t>(ctx->nnz) * sizeof(int)));

            size_t csr2csc_buf_size = 0;
            cusparse_check_named(cusparseCsr2cscEx2_bufferSize(
                ctx->cusparse,
                ctx->nsub, ctx->nsub, ctx->nnz,
                ctx->fp64.d_val_g, ctx->d_row_ptr_g, ctx->d_col_ind_g,
                ctx->fp64.d_val_gt, ctx->d_row_ptr_gt, ctx->d_col_ind_gt,
                CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                CUSPARSE_CSR2CSC_ALG1, &csr2csc_buf_size),
                "cusparseCsr2cscEx2_bufferSize");
            void *d_csr2csc_buf = nullptr;
            if (csr2csc_buf_size > 0)
                cuda_check(cudaMalloc(&d_csr2csc_buf, csr2csc_buf_size));
            cusparse_check_named(cusparseCsr2cscEx2(
                ctx->cusparse,
                ctx->nsub, ctx->nsub, ctx->nnz,
                ctx->fp64.d_val_g, ctx->d_row_ptr_g, ctx->d_col_ind_g,
                ctx->fp64.d_val_gt, ctx->d_row_ptr_gt, ctx->d_col_ind_gt,
                CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                CUSPARSE_CSR2CSC_ALG1, d_csr2csc_buf),
                "cusparseCsr2cscEx2");
            cudaFree(d_csr2csc_buf);

            const int cast_threads = 256;
            const int cast_blocks = (ctx->nnz + cast_threads - 1) / cast_threads;
            k_cast_vec<<<cast_blocks, cast_threads>>>(ctx->nnz, ctx->fp64.d_val_g, ctx->fp32.d_val_g);
            k_cast_vec<<<cast_blocks, cast_threads>>>(ctx->nnz, ctx->fp64.d_val_gt, ctx->fp32.d_val_gt);
            cuda_check(cudaGetLastError());

            init_variant(ctx, ctx->fp64);
            init_variant(ctx, ctx->fp32);

            if (dn_a_dense)
            {
                cusparseDestroyDnMat(dn_a_dense);
                dn_a_dense = nullptr;
            }
            if (mat_full)
            {
                cusparseDestroySpMat(mat_full);
                mat_full = nullptr;
            }
            cudaFree(d_potrf_work);
            cudaFree(d_info);
            cudaFree(d_rhs);
            cudaFree(d_system);
            cudaFree(d_sparse_to_dense_buf);
            cudaFree(d_a_dense);
            cudaFree(d_val_full);
            cudaFree(d_col_ind_full);
            cudaFree(d_row_ptr_full);
            if (cusolver)
            {
                cusolverDnDestroy(cusolver);
                cusolver = nullptr;
            }

            return ctx;
        }
        catch (...)
        {
            if (dn_a_dense)
                cusparseDestroyDnMat(dn_a_dense);
            if (mat_full)
                cusparseDestroySpMat(mat_full);
            if (cusolver)
                cusolverDnDestroy(cusolver);

            cudaFree(d_info);
            cudaFree(d_potrf_work);
            cudaFree(d_rhs);
            cudaFree(d_system);
            cudaFree(d_sparse_to_dense_buf);
            cudaFree(d_a_dense);
            cudaFree(d_val_full);
            cudaFree(d_col_ind_full);
            cudaFree(d_row_ptr_full);

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
        switch (normalize_sparse_solve_precision(prec))
        {
        case ichol::solver::ComputePrecision::FP64:
            apply_fsai_impl(ctx, ctx->fp64, static_cast<const double *>(d_r), static_cast<double *>(d_z), stream);
            break;
        case ichol::solver::ComputePrecision::FP32:
            apply_fsai_impl(ctx, ctx->fp32, static_cast<const float *>(d_r), static_cast<float *>(d_z), stream);
            break;
        default:
            break;
        }
    }

    void destroy_subdomain_fsai_context(void *vctx)
    {
        destroy_ctx_impl(reinterpret_cast<SubdomainFsaiContext *>(vctx));
    }
} // namespace ichol::precond::detail
