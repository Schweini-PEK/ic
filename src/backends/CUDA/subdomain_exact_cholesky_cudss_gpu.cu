#include "backends/CUDA/subdomain_preconditioner_backend.hpp"
#include "backends/CUDA/subdomain_sparse_solve_common.cuh"

#include <cudss.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace ichol::precond::detail::subdomain_common;

    struct SubdomainExactCholeskyContext
    {
        int nsub = 0;
        int *d_gidx = nullptr;
        int *d_row_ptr = nullptr;
        int *d_col_ind = nullptr;
        void *d_val = nullptr;
        void *d_rhs = nullptr;
        void *d_x = nullptr;

        ichol::solver::ComputePrecision storage_prec = ichol::solver::ComputePrecision::FP64;
        cudaDataType_t value_type = CUDA_R_64F;

        cudssHandle_t handle = nullptr;
        cudssConfig_t config = nullptr;
        cudssData_t data = nullptr;
        cudssMatrix_t matrix_a = nullptr;
        cudssMatrix_t matrix_b = nullptr;
        cudssMatrix_t matrix_x = nullptr;
    };

    void cudss_check(cudssStatus_t status, const char *what)
    {
        if (status != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed with cuDSS status " + std::to_string(static_cast<int>(status)));
    }

    template <typename T>
    void init_exact_storage(
        SubdomainExactCholeskyContext *ctx,
        const ichol::matrix::CsrMatrix<double> &A_sub)
    {
        std::vector<T> values_t;
        upload_values(values_t, A_sub.values);

        const int nnz = A_sub.nnz;
        cuda_check(cudaMalloc(&ctx->d_val, static_cast<std::size_t>(nnz) * sizeof(T)));
        cuda_check(cudaMalloc(&ctx->d_rhs, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMalloc(&ctx->d_x, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMemcpy(ctx->d_val, values_t.data(), static_cast<std::size_t>(nnz) * sizeof(T), cudaMemcpyHostToDevice));

        ctx->value_type = cuda_data_type<T>();
        cudss_check(cudssMatrixCreateCsr(
                        &ctx->matrix_a,
                        ctx->nsub,
                        ctx->nsub,
                        nnz,
                        ctx->d_row_ptr,
                        ctx->d_row_ptr + 1,
                        ctx->d_col_ind,
                        ctx->d_val,
                        CUDA_R_32I,
                        ctx->value_type,
                        CUDSS_MTYPE_SPD,
                        CUDSS_MVIEW_LOWER,
                        CUDSS_BASE_ZERO),
                    "cudssMatrixCreateCsr");
        cudss_check(cudssMatrixCreateDn(
                        &ctx->matrix_b,
                        ctx->nsub,
                        1,
                        ctx->nsub,
                        ctx->d_rhs,
                        ctx->value_type,
                        CUDSS_LAYOUT_COL_MAJOR),
                    "cudssMatrixCreateDn(rhs)");
        cudss_check(cudssMatrixCreateDn(
                        &ctx->matrix_x,
                        ctx->nsub,
                        1,
                        ctx->nsub,
                        ctx->d_x,
                        ctx->value_type,
                        CUDSS_LAYOUT_COL_MAJOR),
                    "cudssMatrixCreateDn(x)");

        cudss_check(cudssExecute(
                        ctx->handle,
                        CUDSS_PHASE_ANALYSIS,
                        ctx->config,
                        ctx->data,
                        ctx->matrix_a,
                        ctx->matrix_x,
                        ctx->matrix_b),
                    "cudssExecute(analysis)");
        cudss_check(cudssExecute(
                        ctx->handle,
                        CUDSS_PHASE_FACTORIZATION,
                        ctx->config,
                        ctx->data,
                        ctx->matrix_a,
                        ctx->matrix_x,
                        ctx->matrix_b),
                    "cudssExecute(factorization)");
    }

    template <typename T>
    void apply_exact_impl(
        SubdomainExactCholeskyContext *ctx,
        const T *d_r,
        T *d_z,
        cudaStream_t stream)
    {
        const int threads = 256;
        const int blocks = (ctx->nsub + threads - 1) / threads;
        auto *d_rhs = static_cast<T *>(ctx->d_rhs);
        auto *d_x = static_cast<T *>(ctx->d_x);

        k_gather_subvec<<<blocks, threads, 0, stream>>>(d_r, ctx->d_gidx, d_rhs, ctx->nsub);
        cuda_check(cudaGetLastError());
        cudss_check(cudssSetStream(ctx->handle, stream), "cudssSetStream");
        cudss_check(cudssExecute(
                        ctx->handle,
                        CUDSS_PHASE_SOLVE,
                        ctx->config,
                        ctx->data,
                        ctx->matrix_a,
                        ctx->matrix_x,
                        ctx->matrix_b),
                    "cudssExecute(solve)");
        k_scatter_subvec<<<blocks, threads, 0, stream>>>(d_x, ctx->d_gidx, d_z, ctx->nsub);
        cuda_check(cudaGetLastError());
    }

    void destroy_ctx_impl(SubdomainExactCholeskyContext *ctx)
    {
        if (!ctx)
            return;

        if (ctx->matrix_x)
            cudssMatrixDestroy(ctx->matrix_x);
        if (ctx->matrix_b)
            cudssMatrixDestroy(ctx->matrix_b);
        if (ctx->matrix_a)
            cudssMatrixDestroy(ctx->matrix_a);
        if (ctx->data)
            cudssDataDestroy(ctx->handle, ctx->data);
        if (ctx->config)
            cudssConfigDestroy(ctx->config);
        if (ctx->handle)
            cudssDestroy(ctx->handle);

        cudaFree(ctx->d_x);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_val);
        cudaFree(ctx->d_col_ind);
        cudaFree(ctx->d_row_ptr);
        cudaFree(ctx->d_gidx);
        delete ctx;
    }
} // namespace

namespace ichol::precond::detail
{
    void *create_subdomain_exact_cholesky_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &reg,
        const SubdomainPreconditionerOptions &options)
    {
        auto *ctx = new SubdomainExactCholeskyContext();
        try
        {
            const int lw = reg.x1 - reg.x0;
            const int lh = reg.y1 - reg.y0;
            const int ld = reg.z1 - reg.z0;
            ctx->nsub = lw * lh * ld;
            if (ctx->nsub <= 0)
                throw std::runtime_error("create_subdomain_exact_cholesky_context: empty subdomain");

            ctx->storage_prec = normalize_sparse_solve_precision(options.precision);

            cuda_check(cudaMalloc(&ctx->d_gidx, static_cast<std::size_t>(ctx->nsub) * sizeof(int)));
            build_subdomain_gidx(ctx->d_gidx, lw, lh, ld, global.w, global.h, reg.x0, reg.y0, reg.z0);

            const auto A_sub = extract_lower_subdomain_csr(A, global, reg);
            const int nnz = A_sub.nnz;
            cuda_check(cudaMalloc(&ctx->d_row_ptr, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind, static_cast<std::size_t>(nnz) * sizeof(int)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr, A_sub.row_ptr.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind, A_sub.col_ind.data(), static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));

            cudss_check(cudssCreate(&ctx->handle), "cudssCreate");
            cudss_check(cudssConfigCreate(&ctx->config), "cudssConfigCreate");
            cudss_check(cudssDataCreate(ctx->handle, &ctx->data), "cudssDataCreate");

            switch (ctx->storage_prec)
            {
            case ichol::solver::ComputePrecision::FP64:
                init_exact_storage<double>(ctx, A_sub);
                break;
            case ichol::solver::ComputePrecision::FP32:
                init_exact_storage<float>(ctx, A_sub);
                break;
            default:
                throw std::runtime_error("create_subdomain_exact_cholesky_context: unsupported precision");
            }
            return ctx;
        }
        catch (...)
        {
            destroy_ctx_impl(ctx);
            throw;
        }
    }

    void apply_subdomain_exact_cholesky(
        void *vctx,
        const void *d_r,
        void *d_z,
        int /*N*/,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainExactCholeskyContext *>(vctx);
        const auto requested_prec = normalize_sparse_solve_precision(prec);
        if (requested_prec != ctx->storage_prec)
            throw std::runtime_error("apply_subdomain_exact_cholesky: requested precision does not match stored exact preconditioner precision");

        switch (ctx->storage_prec)
        {
        case ichol::solver::ComputePrecision::FP64:
            apply_exact_impl(ctx, static_cast<const double *>(d_r), static_cast<double *>(d_z), stream);
            break;
        case ichol::solver::ComputePrecision::FP32:
            apply_exact_impl(ctx, static_cast<const float *>(d_r), static_cast<float *>(d_z), stream);
            break;
        default:
            throw std::runtime_error("apply_subdomain_exact_cholesky: unsupported precision");
        }
    }

    void destroy_subdomain_exact_cholesky_context(void *vctx)
    {
        destroy_ctx_impl(reinterpret_cast<SubdomainExactCholeskyContext *>(vctx));
    }
} // namespace ichol::precond::detail
