#include "backends/CUDA/subdomain_preconditioner_backend.hpp"
#include "backends/CUDA/subdomain_sparse_solve_common.cuh"

#if ICHOL_HAVE_CUDSS
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <numeric>
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
        std::vector<int> h_gidx;
        int debug_subdomain_index = -1;

        ichol::solver::ComputePrecision storage_prec = ichol::solver::ComputePrecision::FP64;
        cudaDataType_t value_type = CUDA_R_64F;

        // Dedicated stream: analysis, factorization, and solve all run here,
        // guaranteeing their ordering.  Caller streams interact via events.
        cudaStream_t internal_stream = nullptr;
        cudaEvent_t gather_done_event = nullptr;
        cudaEvent_t solve_done_event = nullptr;

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

    bool cudss_exact_debug_enabled()
    {
        static const bool enabled = []()
        {
            const char *env = std::getenv("ICHOL_DEBUG_CUDSS_EXACT");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }();
        return enabled;
    }

    template <typename T>
    double host_l2_norm(const std::vector<T> &values)
    {
        double accum = 0.0;
        for (const T value : values)
        {
            const double x = static_cast<double>(value);
            accum += x * x;
        }
        return std::sqrt(accum);
    }

    template <typename T>
    std::vector<T> copy_device_vector(const T *d_ptr, int n)
    {
        std::vector<T> host(static_cast<std::size_t>(n));
        if (n > 0)
            cuda_check(cudaMemcpy(host.data(), d_ptr, static_cast<std::size_t>(n) * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

    template <typename T>
    void debug_log_apply_state(
        const SubdomainExactCholeskyContext *ctx,
        const T *d_rhs,
        const T *d_x,
        const T *d_z,
        int n,
        cudaStream_t stream)
    {
        if (!cudss_exact_debug_enabled())
            return;

        cuda_check(cudaStreamSynchronize(stream));

        const auto h_rhs = copy_device_vector(d_rhs, ctx->nsub);
        const auto h_x = copy_device_vector(d_x, ctx->nsub);
        const auto h_z = copy_device_vector(d_z, n);

        std::vector<unsigned char> in_support(static_cast<std::size_t>(n), 0);
        for (int gi : ctx->h_gidx)
            in_support[static_cast<std::size_t>(gi)] = 1;

        double z_norm = 0.0;
        double outside_norm = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double val = static_cast<double>(h_z[static_cast<std::size_t>(i)]);
            z_norm += val * val;
            if (in_support[static_cast<std::size_t>(i)] == 0)
                outside_norm += val * val;
        }

        int info = -777;
        size_t size_written = 0;
        const cudssStatus_t info_status = cudssDataGet(
            ctx->handle,
            ctx->data,
            CUDSS_DATA_INFO,
            &info,
            sizeof(info),
            &size_written);

        const int preview = std::min<int>(4, static_cast<int>(ctx->h_gidx.size()));
        std::fprintf(stderr,
                     "[cudss-exact] subdomain=%d ctx=%p d_z=%p nsub=%d rhs_norm=%.17e x_norm=%.17e z_norm=%.17e outside_norm=%.17e info_status=%d info=%d size=%zu gidx0=%d gidx1=%d gidx2=%d gidx3=%d\n",
                     ctx->debug_subdomain_index,
                     static_cast<const void *>(ctx),
                     static_cast<const void *>(d_z),
                     ctx->nsub,
                     host_l2_norm(h_rhs),
                     host_l2_norm(h_x),
                     std::sqrt(z_norm),
                     std::sqrt(outside_norm),
                     static_cast<int>(info_status),
                     info,
                     size_written,
                     preview > 0 ? ctx->h_gidx[0] : -1,
                     preview > 1 ? ctx->h_gidx[1] : -1,
                     preview > 2 ? ctx->h_gidx[2] : -1,
                     preview > 3 ? ctx->h_gidx[3] : -1);
        std::fflush(stderr);
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
        int n,
        cudaStream_t stream)
    {
        const int threads = 256;
        const int blocks = (ctx->nsub + threads - 1) / threads;
        auto *d_rhs = static_cast<T *>(ctx->d_rhs);
        auto *d_x = static_cast<T *>(ctx->d_x);

        if (cudss_exact_debug_enabled())
        {
            cuda_check(cudaGetLastError());
            auto h_before = copy_device_vector(d_z, n);
            std::fprintf(stderr,
                         "[cudss-exact] subdomain=%d ctx=%p d_z=%p before_scatter_norm=%.17e\n",
                         ctx->debug_subdomain_index,
                         static_cast<void *>(ctx),
                         static_cast<void *>(d_z),
                         host_l2_norm(h_before));
            std::fflush(stderr);
        }

        // Gather on caller's stream, then hand off to the internal cuDSS stream.
        // The internal stream already holds the factorization from context creation,
        // so ordering FACTORIZATION -> SOLVE is guaranteed by the stream.
        k_gather_subvec<<<blocks, threads, 0, stream>>>(d_r, ctx->d_gidx, d_rhs, ctx->nsub);
        cuda_check(cudaGetLastError());
        cuda_check(cudaEventRecord(ctx->gather_done_event, stream));
        cuda_check(cudaStreamWaitEvent(ctx->internal_stream, ctx->gather_done_event, 0));
        cudss_check(cudssExecute(
                        ctx->handle,
                        CUDSS_PHASE_SOLVE,
                        ctx->config,
                        ctx->data,
                        ctx->matrix_a,
                        ctx->matrix_x,
                        ctx->matrix_b),
                    "cudssExecute(solve)");
        cuda_check(cudaEventRecord(ctx->solve_done_event, ctx->internal_stream));
        cuda_check(cudaStreamWaitEvent(stream, ctx->solve_done_event, 0));
        if (cudss_exact_debug_enabled())
            cuda_check(cudaStreamSynchronize(stream));
        k_scatter_subvec<<<blocks, threads, 0, stream>>>(d_x, ctx->d_gidx, d_z, ctx->nsub);
        cuda_check(cudaGetLastError());

        if (cudss_exact_debug_enabled())
        {
            cuda_check(cudaGetLastError());
            debug_log_apply_state(ctx, d_rhs, d_x, d_z, n, stream);
        }
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

        if (ctx->solve_done_event)
            cudaEventDestroy(ctx->solve_done_event);
        if (ctx->gather_done_event)
            cudaEventDestroy(ctx->gather_done_event);
        if (ctx->internal_stream)
            cudaStreamDestroy(ctx->internal_stream);

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
            ctx->debug_subdomain_index = options.debug_subdomain_index;

            cuda_check(cudaMalloc(&ctx->d_gidx, static_cast<std::size_t>(ctx->nsub) * sizeof(int)));
            build_subdomain_gidx(ctx->d_gidx, lw, lh, ld, global.w, global.h, reg.x0, reg.y0, reg.z0);
            ctx->h_gidx.resize(static_cast<std::size_t>(ctx->nsub));
            cuda_check(cudaMemcpy(ctx->h_gidx.data(), ctx->d_gidx, static_cast<std::size_t>(ctx->nsub) * sizeof(int), cudaMemcpyDeviceToHost));

            const auto A_sub = extract_lower_subdomain_csr(A, global, reg);
            const int nnz = A_sub.nnz;
            cuda_check(cudaMalloc(&ctx->d_row_ptr, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind, static_cast<std::size_t>(nnz) * sizeof(int)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr, A_sub.row_ptr.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind, A_sub.col_ind.data(), static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));

            // Create the dedicated internal stream and bind it to the cuDSS handle
            // before analysis and factorization.  All three phases (analysis,
            // factorization, solve) run on this stream, so their ordering is
            // guaranteed by the stream itself regardless of which caller stream is
            // used at apply time.
            cuda_check(cudaStreamCreateWithFlags(&ctx->internal_stream, cudaStreamNonBlocking));
            cuda_check(cudaEventCreateWithFlags(&ctx->gather_done_event, cudaEventDisableTiming));
            cuda_check(cudaEventCreateWithFlags(&ctx->solve_done_event, cudaEventDisableTiming));

            cudss_check(cudssCreate(&ctx->handle), "cudssCreate");
            cudss_check(cudssSetStream(ctx->handle, ctx->internal_stream), "cudssSetStream(init)");
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
        int N,
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
            apply_exact_impl(ctx, static_cast<const double *>(d_r), static_cast<double *>(d_z), N, stream);
            break;
        case ichol::solver::ComputePrecision::FP32:
            apply_exact_impl(ctx, static_cast<const float *>(d_r), static_cast<float *>(d_z), N, stream);
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
#else

#include <stdexcept>

namespace ichol::precond::detail
{
    void *create_subdomain_exact_cholesky_context(
        const ichol::matrix::CsrMatrix<double> & /*A*/,
        const GridShape & /*global*/,
        const SubdomainRegion & /*reg*/,
        const SubdomainPreconditionerOptions & /*options*/)
    {
        throw std::runtime_error("create_subdomain_exact_cholesky_context: cuDSS support is not enabled");
    }

    void apply_subdomain_exact_cholesky(
        void * /*vctx*/,
        const void * /*d_r*/,
        void * /*d_z*/,
        int /*N*/,
        ichol::solver::ComputePrecision /*prec*/,
        cudaStream_t /*stream*/)
    {
        throw std::runtime_error("apply_subdomain_exact_cholesky: cuDSS support is not enabled");
    }

    void destroy_subdomain_exact_cholesky_context(void * /*vctx*/)
    {
    }
} // namespace ichol::precond::detail

#endif
