#include "backends/CUDA/subdomain_precond_impl.hpp"

#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "factor/numerical/detail/numeric_plan.hpp"
#include "factor/numerical/factorize.hpp"
#include "factor/symbolic/symbolic.hpp"

namespace
{
    inline void cuda_check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
    }

    inline void cusparse_check(cusparseStatus_t s)
    {
        if (s != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("cuSPARSE Error");
    }

    __host__ __device__ inline int flatten_local_3d(int x, int y, int z, int w, int h)
    {
        return x + y * w + z * (w * h);
    }

    __host__ __device__ inline void unflatten_global_3d(int gi, int gw, int gh, int &x, int &y, int &z)
    {
        const int plane = gw * gh;
        z = gi / plane;
        const int rem = gi - z * plane;
        y = rem / gw;
        x = rem - y * gw;
    }

    inline int local_from_global_host(
        int gj,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg,
        int lw,
        int lh)
    {
        int x = 0;
        int y = 0;
        int z = 0;
        unflatten_global_3d(gj, global.w, global.h, x, y, z);
        if (x < reg.x0 || x >= reg.x1 || y < reg.y0 || y >= reg.y1 || z < reg.z0 || z >= reg.z1)
            return -1;
        return flatten_local_3d(x - reg.x0, y - reg.y0, z - reg.z0, lw, lh);
    }

    ichol::matrix::CsrMatrix<double> extract_lower_subdomain_csr(
        const ichol::matrix::CsrMatrix<double> &A,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg)
    {
        const int lw = reg.x1 - reg.x0;
        const int lh = reg.y1 - reg.y0;
        const int ld = reg.z1 - reg.z0;
        const int nsub = lw * lh * ld;

        ichol::matrix::CsrMatrix<double> sub;
        sub.num_rows = nsub;
        sub.num_cols = nsub;
        sub.row_ptr.resize((size_t)nsub + 1, 0);

        std::vector<int> cols;
        std::vector<double> vals;

        for (int li = 0; li < nsub; ++li)
        {
            const int plane = lw * lh;
            const int lz = li / plane;
            const int rem = li - lz * plane;
            const int ly = rem / lw;
            const int lx = rem - ly * lw;
            const int gi = (reg.x0 + lx) + (reg.y0 + ly) * global.w + (reg.z0 + lz) * (global.w * global.h);

            std::vector<std::pair<int, double>> row_entries;
            row_entries.reserve((size_t)(A.row_ptr[gi + 1] - A.row_ptr[gi]));

            for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
            {
                const int lj = local_from_global_host(A.col_ind[kk], global, reg, lw, lh);
                if (lj < 0 || lj > li)
                    continue;
                row_entries.push_back({lj, A.values[kk]});
            }

            std::sort(row_entries.begin(), row_entries.end(), [](const auto &a, const auto &b)
                      { return a.first < b.first; });

            int diag_pos = -1;
            for (int i = 0; i < (int)row_entries.size(); ++i)
            {
                if (row_entries[(size_t)i].first == li)
                {
                    diag_pos = i;
                    break;
                }
            }
            if (diag_pos < 0)
                throw std::runtime_error("extract_lower_subdomain_csr: missing diagonal entry");

            for (int i = 0; i < (int)row_entries.size(); ++i)
            {
                if (i == diag_pos)
                    continue;
                cols.push_back(row_entries[(size_t)i].first);
                vals.push_back(row_entries[(size_t)i].second);
            }
            cols.push_back(li);
            vals.push_back(row_entries[(size_t)diag_pos].second);

            sub.row_ptr[(size_t)li + 1] = (int)cols.size();
        }

        sub.col_ind = std::move(cols);
        sub.values = std::move(vals);
        sub.nnz = (int)sub.values.size();
        return sub;
    }

    void build_csr_transpose(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<double> &val,
        std::vector<int> &row_ptr_t,
        std::vector<int> &col_ind_t,
        std::vector<double> &val_t)
    {
        const int nnz = (int)val.size();
        row_ptr_t.assign((size_t)n + 1, 0);
        col_ind_t.assign((size_t)nnz, 0);
        val_t.assign((size_t)nnz, 0.0);

        for (int i = 0; i < nnz; ++i)
            ++row_ptr_t[(size_t)col_ind[i] + 1];

        for (int i = 0; i < n; ++i)
            row_ptr_t[(size_t)i + 1] += row_ptr_t[(size_t)i];

        std::vector<int> next = row_ptr_t;
        for (int i = 0; i < n; ++i)
        {
            for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
            {
                const int j = col_ind[p];
                const int dst = next[(size_t)j]++;
                col_ind_t[(size_t)dst] = i;
                val_t[(size_t)dst] = val[(size_t)p];
            }
        }
    }

    __global__ void k_gather_subvec(const double *src, const int *gidx, double *dst, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nsub)
            dst[i] = src[gidx[i]];
    }

    __global__ void k_scatter_subvec(const double *src, const int *gidx, double *dst, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nsub)
            dst[gidx[i]] = src[i];
    }

    __global__ void k_build_gidx(
        int *gidx,
        int lw,
        int lh,
        int ld,
        int gw,
        int gh,
        int x0,
        int y0,
        int z0)
    {
        const int li = blockIdx.x * blockDim.x + threadIdx.x;
        const int nsub = lw * lh * ld;
        if (li >= nsub)
            return;

        const int plane = lw * lh;
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;

        const int gx = x0 + lx;
        const int gy = y0 + ly;
        const int gz = z0 + lz;

        gidx[li] = gx + gy * gw + gz * (gw * gh);
    }
} // namespace

namespace ichol::precond::detail
{
    struct SubdomainIcContext
    {
        int nsub = 0;
        int *d_gidx = nullptr;
        double *d_rhs = nullptr;
        double *d_y = nullptr;
        double *d_x = nullptr;

        int *d_row_ptr_l = nullptr;
        int *d_col_ind_l = nullptr;
        double *d_val_l = nullptr;

        int *d_row_ptr_lt = nullptr;
        int *d_col_ind_lt = nullptr;
        double *d_val_lt = nullptr;

        cusparseHandle_t cusparse = nullptr;
        cusparseSpMatDescr_t mat_l = nullptr;
        cusparseSpMatDescr_t mat_lt = nullptr;
        cusparseDnVecDescr_t vec_rhs = nullptr;
        cusparseDnVecDescr_t vec_y = nullptr;
        cusparseDnVecDescr_t vec_x = nullptr;
        cusparseSpSVDescr_t spsv_l = nullptr;
        cusparseSpSVDescr_t spsv_lt = nullptr;
        void *buf_l = nullptr;
        void *buf_lt = nullptr;
    };

    static void destroy_ctx_impl(SubdomainIcContext *ctx)
    {
        if (!ctx)
            return;

        if (ctx->spsv_l)
            cusparseSpSV_destroyDescr(ctx->spsv_l);
        if (ctx->spsv_lt)
            cusparseSpSV_destroyDescr(ctx->spsv_lt);
        if (ctx->vec_rhs)
            cusparseDestroyDnVec(ctx->vec_rhs);
        if (ctx->vec_y)
            cusparseDestroyDnVec(ctx->vec_y);
        if (ctx->vec_x)
            cusparseDestroyDnVec(ctx->vec_x);
        if (ctx->mat_l)
            cusparseDestroySpMat(ctx->mat_l);
        if (ctx->mat_lt)
            cusparseDestroySpMat(ctx->mat_lt);
        if (ctx->cusparse)
            cusparseDestroy(ctx->cusparse);

        cudaFree(ctx->buf_l);
        cudaFree(ctx->buf_lt);
        cudaFree(ctx->d_gidx);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_y);
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_row_ptr_l);
        cudaFree(ctx->d_col_ind_l);
        cudaFree(ctx->d_val_l);
        cudaFree(ctx->d_row_ptr_lt);
        cudaFree(ctx->d_col_ind_lt);
        cudaFree(ctx->d_val_lt);
        delete ctx;
    }

    void *create_subdomain_ic_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &reg,
        const SubdomainPreconditionerOptions &options)
    {
        auto *ctx = new SubdomainIcContext();
        try
        {
            const int lw = reg.x1 - reg.x0;
            const int lh = reg.y1 - reg.y0;
            const int ld = reg.z1 - reg.z0;
            ctx->nsub = lw * lh * ld;
            if (ctx->nsub <= 0)
                throw std::runtime_error("create_subdomain_ic_context: empty subdomain");

            auto A_sub = extract_lower_subdomain_csr(A, global, reg);

            ichol::SymbolicOptions sym_opts;
            sym_opts.ordering = ichol::Ordering::Identity;
            sym_opts.level_k = (options.kind == SubdomainPreconditionerKind::ExactCholesky) ? -1 : options.ic_level_k;
            if (sym_opts.level_k < -1)
                throw std::runtime_error("create_subdomain_ic_context: ic_level_k must be >= -1");

            auto sym_plan = ichol::symbolic::ic_analyze(A_sub, sym_opts);

            ichol::IncompleteCholeskyOptions ic_opts;
            ic_opts.scaling = ichol::Scaling::None;
            ic_opts.pivot_shift_strategy = ichol::PivotShiftStrategy::None;
            ic_opts.algorithm = ichol::FactorizationAlgorithm::ICKDT;
            ic_opts.max_restarts = 1;
            ic_opts.verbose = false;
            ic_opts.lfil = A_sub.num_rows;
            ic_opts.drop_tol = 0.0;

            ichol::numeric::NumericPlan num_plan;
            auto L = ichol::numeric::incomplete_cholesky_preconditioner<double>(A_sub, sym_plan, num_plan, ic_opts);

            std::vector<int> lt_row_ptr;
            std::vector<int> lt_col_ind;
            std::vector<double> lt_val;
            build_csr_transpose(L.num_rows, L.row_ptr, L.col_ind, L.values, lt_row_ptr, lt_col_ind, lt_val);

            cuda_check(cudaMalloc(&ctx->d_gidx, (size_t)ctx->nsub * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_rhs, (size_t)ctx->nsub * sizeof(double)));
            cuda_check(cudaMalloc(&ctx->d_y, (size_t)ctx->nsub * sizeof(double)));
            cuda_check(cudaMalloc(&ctx->d_x, (size_t)ctx->nsub * sizeof(double)));

            const int threads = 256;
            const int blocks = (ctx->nsub + threads - 1) / threads;
            k_build_gidx<<<blocks, threads>>>(ctx->d_gidx, lw, lh, ld, global.w, global.h, reg.x0, reg.y0, reg.z0);
            cuda_check(cudaGetLastError());

            const int nnz_l = (int)L.values.size();
            const int nnz_lt = (int)lt_val.size();

            cuda_check(cudaMalloc(&ctx->d_row_ptr_l, ((size_t)ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_l, (size_t)nnz_l * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_val_l, (size_t)nnz_l * sizeof(double)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr_l, L.row_ptr.data(), ((size_t)ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind_l, L.col_ind.data(), (size_t)nnz_l * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_val_l, L.values.data(), (size_t)nnz_l * sizeof(double), cudaMemcpyHostToDevice));

            cuda_check(cudaMalloc(&ctx->d_row_ptr_lt, ((size_t)ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_lt, (size_t)nnz_lt * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_val_lt, (size_t)nnz_lt * sizeof(double)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr_lt, lt_row_ptr.data(), ((size_t)ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind_lt, lt_col_ind.data(), (size_t)nnz_lt * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_val_lt, lt_val.data(), (size_t)nnz_lt * sizeof(double), cudaMemcpyHostToDevice));

            cusparse_check(cusparseCreate(&ctx->cusparse));

            cusparse_check(cusparseCreateCsr(
                &ctx->mat_l, ctx->nsub, ctx->nsub, nnz_l,
                ctx->d_row_ptr_l, ctx->d_col_ind_l, ctx->d_val_l,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
            cusparse_check(cusparseCreateCsr(
                &ctx->mat_lt, ctx->nsub, ctx->nsub, nnz_lt,
                ctx->d_row_ptr_lt, ctx->d_col_ind_lt, ctx->d_val_lt,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

            cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
            cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
            cusparseDiagType_t non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
            cusparse_check(cusparseSpMatSetAttribute(ctx->mat_l, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)));
            cusparse_check(cusparseSpMatSetAttribute(ctx->mat_l, CUSPARSE_SPMAT_DIAG_TYPE, &non_unit, sizeof(non_unit)));
            cusparse_check(cusparseSpMatSetAttribute(ctx->mat_lt, CUSPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper)));
            cusparse_check(cusparseSpMatSetAttribute(ctx->mat_lt, CUSPARSE_SPMAT_DIAG_TYPE, &non_unit, sizeof(non_unit)));

            cusparse_check(cusparseCreateDnVec(&ctx->vec_rhs, ctx->nsub, ctx->d_rhs, CUDA_R_64F));
            cusparse_check(cusparseCreateDnVec(&ctx->vec_y, ctx->nsub, ctx->d_y, CUDA_R_64F));
            cusparse_check(cusparseCreateDnVec(&ctx->vec_x, ctx->nsub, ctx->d_x, CUDA_R_64F));

            cusparse_check(cusparseSpSV_createDescr(&ctx->spsv_l));
            cusparse_check(cusparseSpSV_createDescr(&ctx->spsv_lt));

            const double alpha = 1.0;
            size_t buf_size_l = 0;
            size_t buf_size_lt = 0;

            cusparse_check(cusparseSpSV_bufferSize(
                ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, ctx->mat_l, ctx->vec_rhs, ctx->vec_y, CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsv_l, &buf_size_l));
            cuda_check(cudaMalloc(&ctx->buf_l, buf_size_l));
            cusparse_check(cusparseSpSV_analysis(
                ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, ctx->mat_l, ctx->vec_rhs, ctx->vec_y, CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsv_l, ctx->buf_l));

            cusparse_check(cusparseSpSV_bufferSize(
                ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, ctx->mat_lt, ctx->vec_y, ctx->vec_x, CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsv_lt, &buf_size_lt));
            cuda_check(cudaMalloc(&ctx->buf_lt, buf_size_lt));
            cusparse_check(cusparseSpSV_analysis(
                ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, ctx->mat_lt, ctx->vec_y, ctx->vec_x, CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsv_lt, ctx->buf_lt));

            return ctx;
        }
        catch (...)
        {
            destroy_ctx_impl(ctx);
            throw;
        }
    }

    void apply_subdomain_ic(void *vctx, const double *d_r, double *d_z, int /*N*/, cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainIcContext *>(vctx);
        const int nsub = ctx->nsub;
        const int threads = 256;
        const int blocks = (nsub + threads - 1) / threads;
        const double alpha = 1.0;

        k_gather_subvec<<<blocks, threads, 0, stream>>>(d_r, ctx->d_gidx, ctx->d_rhs, nsub);
        cusparse_check(cusparseSetStream(ctx->cusparse, stream));
        cusparse_check(cusparseSpSV_solve(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, ctx->mat_l, ctx->vec_rhs, ctx->vec_y, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsv_l));
        cusparse_check(cusparseSpSV_solve(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, ctx->mat_lt, ctx->vec_y, ctx->vec_x, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsv_lt));
        k_scatter_subvec<<<blocks, threads, 0, stream>>>(ctx->d_x, ctx->d_gidx, d_z, nsub);
    }

    void destroy_subdomain_ic_context(void *vctx)
    {
        destroy_ctx_impl(reinterpret_cast<SubdomainIcContext *>(vctx));
    }
} // namespace ichol::precond::detail
