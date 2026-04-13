#include "backends/CUDA/subdomain_preconditioner_backend.hpp"
#include "backends/CUDA/subdomain_sparse_solve_common.cuh"
#include "factor/numerical/factorize.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"
#include "factor/symbolic/symbolic.hpp"

#include <cusparse.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace
{
    using namespace ichol::precond::detail::subdomain_common;

    void cusparse_check_named(cusparseStatus_t status, const char *what)
    {
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed with cuSPARSE status " + std::to_string(static_cast<int>(status)));
    }

    template <typename T>
    struct SubdomainTriangularSolveVariant
    {
        T *d_rhs = nullptr;
        T *d_y = nullptr;
        T *d_x = nullptr;
        T *d_val_l = nullptr;
        T *d_val_lt = nullptr;

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

    struct SubdomainIncompleteCholeskyContext
    {
        int nsub = 0;
        int *d_gidx = nullptr;

        int *d_row_ptr_l = nullptr;
        int *d_col_ind_l = nullptr;
        int *d_row_ptr_lt = nullptr;
        int *d_col_ind_lt = nullptr;

        cusparseHandle_t cusparse = nullptr;
        ichol::solver::ComputePrecision storage_prec = ichol::solver::ComputePrecision::FP64;
        SubdomainTriangularSolveVariant<double> fp64;
        SubdomainTriangularSolveVariant<float> fp32;
    };

    struct LegacyIc0Factorization
    {
        std::vector<int> row_ptr_l;
        std::vector<int> col_ind_l;
        std::vector<double> val_l;
        std::vector<int> row_ptr_lt;
        std::vector<int> col_ind_lt;
        std::vector<double> val_lt;
    };

    template <typename T>
    void destroy_variant(SubdomainTriangularSolveVariant<T> &v)
    {
        if (v.spsv_l)
            cusparseSpSV_destroyDescr(v.spsv_l);
        if (v.spsv_lt)
            cusparseSpSV_destroyDescr(v.spsv_lt);
        if (v.vec_rhs)
            cusparseDestroyDnVec(v.vec_rhs);
        if (v.vec_y)
            cusparseDestroyDnVec(v.vec_y);
        if (v.vec_x)
            cusparseDestroyDnVec(v.vec_x);
        if (v.mat_l)
            cusparseDestroySpMat(v.mat_l);
        if (v.mat_lt)
            cusparseDestroySpMat(v.mat_lt);

        cudaFree(v.buf_l);
        cudaFree(v.buf_lt);
        cudaFree(v.d_rhs);
        cudaFree(v.d_y);
        cudaFree(v.d_x);
        cudaFree(v.d_val_l);
        cudaFree(v.d_val_lt);
    }

    template <typename T>
    void init_variant(
        SubdomainIncompleteCholeskyContext *ctx,
        SubdomainTriangularSolveVariant<T> &variant,
        const std::vector<double> &l_val,
        const std::vector<double> &lt_val)
    {
        std::vector<T> l_val_t;
        std::vector<T> lt_val_t;
        upload_values(l_val_t, l_val);
        upload_values(lt_val_t, lt_val);

        const int nnz_l = static_cast<int>(l_val.size());
        const int nnz_lt = static_cast<int>(lt_val.size());

        cuda_check(cudaMalloc(&variant.d_rhs, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMalloc(&variant.d_y, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMalloc(&variant.d_x, static_cast<std::size_t>(ctx->nsub) * sizeof(T)));
        cuda_check(cudaMalloc(&variant.d_val_l, static_cast<std::size_t>(nnz_l) * sizeof(T)));
        cuda_check(cudaMalloc(&variant.d_val_lt, static_cast<std::size_t>(nnz_lt) * sizeof(T)));
        cuda_check(cudaMemcpy(variant.d_val_l, l_val_t.data(), static_cast<std::size_t>(nnz_l) * sizeof(T), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(variant.d_val_lt, lt_val_t.data(), static_cast<std::size_t>(nnz_lt) * sizeof(T), cudaMemcpyHostToDevice));

        cusparse_check_named(cusparseCreateCsr(
            &variant.mat_l, ctx->nsub, ctx->nsub, nnz_l,
            ctx->d_row_ptr_l, ctx->d_col_ind_l, variant.d_val_l,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<T>()), "cusparseCreateCsr(mat_l)");
        cusparse_check_named(cusparseCreateCsr(
            &variant.mat_lt, ctx->nsub, ctx->nsub, nnz_lt,
            ctx->d_row_ptr_lt, ctx->d_col_ind_lt, variant.d_val_lt,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<T>()), "cusparseCreateCsr(mat_lt)");

        cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
        cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparse_check_named(cusparseSpMatSetAttribute(variant.mat_l, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)), "cusparseSpMatSetAttribute(mat_l fill)");
        cusparse_check_named(cusparseSpMatSetAttribute(variant.mat_l, CUSPARSE_SPMAT_DIAG_TYPE, &non_unit, sizeof(non_unit)), "cusparseSpMatSetAttribute(mat_l diag)");
        cusparse_check_named(cusparseSpMatSetAttribute(variant.mat_lt, CUSPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper)), "cusparseSpMatSetAttribute(mat_lt fill)");
        cusparse_check_named(cusparseSpMatSetAttribute(variant.mat_lt, CUSPARSE_SPMAT_DIAG_TYPE, &non_unit, sizeof(non_unit)), "cusparseSpMatSetAttribute(mat_lt diag)");

        cusparse_check_named(cusparseCreateDnVec(&variant.vec_rhs, ctx->nsub, variant.d_rhs, cuda_data_type<T>()), "cusparseCreateDnVec(rhs)");
        cusparse_check_named(cusparseCreateDnVec(&variant.vec_y, ctx->nsub, variant.d_y, cuda_data_type<T>()), "cusparseCreateDnVec(y)");
        cusparse_check_named(cusparseCreateDnVec(&variant.vec_x, ctx->nsub, variant.d_x, cuda_data_type<T>()), "cusparseCreateDnVec(x)");

        cusparse_check_named(cusparseSpSV_createDescr(&variant.spsv_l), "cusparseSpSV_createDescr(L)");
        cusparse_check_named(cusparseSpSV_createDescr(&variant.spsv_lt), "cusparseSpSV_createDescr(LT)");

        const T alpha = static_cast<T>(1);
        size_t buf_size_l = 0;
        size_t buf_size_lt = 0;

        cusparse_check_named(cusparseSpSV_bufferSize(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_l, variant.vec_rhs, variant.vec_y, cuda_data_type<T>(),
            CUSPARSE_SPSV_ALG_DEFAULT, variant.spsv_l, &buf_size_l), "cusparseSpSV_bufferSize(L)");
        if (buf_size_l > 0)
            cuda_check(cudaMalloc(&variant.buf_l, buf_size_l));
        cusparse_check_named(cusparseSpSV_analysis(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_l, variant.vec_rhs, variant.vec_y, cuda_data_type<T>(),
            CUSPARSE_SPSV_ALG_DEFAULT, variant.spsv_l, variant.buf_l), "cusparseSpSV_analysis(L)");

        cusparse_check_named(cusparseSpSV_bufferSize(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_lt, variant.vec_y, variant.vec_x, cuda_data_type<T>(),
            CUSPARSE_SPSV_ALG_DEFAULT, variant.spsv_lt, &buf_size_lt), "cusparseSpSV_bufferSize(LT)");
        if (buf_size_lt > 0)
            cuda_check(cudaMalloc(&variant.buf_lt, buf_size_lt));
        cusparse_check_named(cusparseSpSV_analysis(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_lt, variant.vec_y, variant.vec_x, cuda_data_type<T>(),
            CUSPARSE_SPSV_ALG_DEFAULT, variant.spsv_lt, variant.buf_lt), "cusparseSpSV_analysis(LT)");
    }

    void destroy_ctx_impl(SubdomainIncompleteCholeskyContext *ctx)
    {
        if (!ctx)
            return;

        destroy_variant(ctx->fp64);
        destroy_variant(ctx->fp32);
        if (ctx->cusparse)
            cusparseDestroy(ctx->cusparse);

        cudaFree(ctx->d_gidx);
        cudaFree(ctx->d_row_ptr_l);
        cudaFree(ctx->d_col_ind_l);
        cudaFree(ctx->d_row_ptr_lt);
        cudaFree(ctx->d_col_ind_lt);
        delete ctx;
    }

    LegacyIc0Factorization factorize_subdomain_ic0(const ichol::matrix::CsrMatrix<double> &A_sub)
    {
        LegacyIc0Factorization out;
        out.row_ptr_l = A_sub.row_ptr;
        out.col_ind_l = A_sub.col_ind;
        out.val_l = A_sub.values;

        cusparseHandle_t handle = nullptr;
        cusparseMatDescr_t descr = nullptr;
        csric02Info_t info = nullptr;
        int *d_row_ptr = nullptr;
        int *d_col_ind = nullptr;
        double *d_val = nullptr;
        void *d_buf = nullptr;

        try
        {
            const int n = A_sub.num_rows;
            const int nnz = A_sub.nnz;
            cusparse_check_named(cusparseCreate(&handle), "cusparseCreate");
            cusparse_check_named(cusparseCreateMatDescr(&descr), "cusparseCreateMatDescr");
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
            cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
            cusparse_check_named(cusparseCreateCsric02Info(&info), "cusparseCreateCsric02Info");

            cuda_check(cudaMalloc(&d_row_ptr, static_cast<std::size_t>(n + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&d_col_ind, static_cast<std::size_t>(nnz) * sizeof(int)));
            cuda_check(cudaMalloc(&d_val, static_cast<std::size_t>(nnz) * sizeof(double)));
            cuda_check(cudaMemcpy(d_row_ptr, out.row_ptr_l.data(), static_cast<std::size_t>(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_col_ind, out.col_ind_l.data(), static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_val, out.val_l.data(), static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyHostToDevice));

            int buffer_size = 0;
            cusparse_check_named(cusparseDcsric02_bufferSize(handle, n, nnz, descr, d_val, d_row_ptr, d_col_ind, info, &buffer_size), "cusparseDcsric02_bufferSize");
            if (buffer_size > 0)
                cuda_check(cudaMalloc(&d_buf, static_cast<std::size_t>(buffer_size)));

            cusparse_check_named(cusparseDcsric02_analysis(handle, n, nnz, descr, d_val, d_row_ptr, d_col_ind, info,
                                                           CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_buf), "cusparseDcsric02_analysis");
            int pivot = -1;
            const cusparseStatus_t structural_status = cusparseXcsric02_zeroPivot(handle, info, &pivot);
            if (structural_status == CUSPARSE_STATUS_ZERO_PIVOT)
                throw std::runtime_error("legacy IC(0) analysis found a structural zero pivot at row " + std::to_string(pivot));
            cusparse_check_named(structural_status, "cusparseXcsric02_zeroPivot(structural)");

            cusparse_check_named(cusparseDcsric02(handle, n, nnz, descr, d_val, d_row_ptr, d_col_ind, info,
                                                  CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_buf), "cusparseDcsric02");
            const cusparseStatus_t numeric_status = cusparseXcsric02_zeroPivot(handle, info, &pivot);
            if (numeric_status == CUSPARSE_STATUS_ZERO_PIVOT)
                throw std::runtime_error("legacy IC(0) factorization found a zero pivot at row " + std::to_string(pivot));
            cusparse_check_named(numeric_status, "cusparseXcsric02_zeroPivot(numeric)");

            cuda_check(cudaMemcpy(out.val_l.data(), d_val, static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyDeviceToHost));
            build_csr_transpose(n, out.row_ptr_l, out.col_ind_l, out.val_l, out.row_ptr_lt, out.col_ind_lt, out.val_lt);
        }
        catch (...)
        {
            cudaFree(d_buf);
            cudaFree(d_val);
            cudaFree(d_col_ind);
            cudaFree(d_row_ptr);
            if (info)
                cusparseDestroyCsric02Info(info);
            if (descr)
                cusparseDestroyMatDescr(descr);
            if (handle)
                cusparseDestroy(handle);
            throw;
        }

        cudaFree(d_buf);
        cudaFree(d_val);
        cudaFree(d_col_ind);
        cudaFree(d_row_ptr);
        cusparseDestroyCsric02Info(info);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);
        return out;
    }

    LegacyIc0Factorization factorize_subdomain_ic(const ichol::matrix::CsrMatrix<double> &A_sub,
                                                  int ic_level_k)
    {
        if (ic_level_k == 0)
            return factorize_subdomain_ic0(A_sub);

        LegacyIc0Factorization out;
        auto A_sub_copy = A_sub;

        ichol::SymbolicOptions sym_opts;
        sym_opts.ordering = ichol::Ordering::Identity;
        sym_opts.level_k = ic_level_k;
        auto sym_plan = ichol::symbolic::ic_analyze(A_sub_copy, sym_opts);

        ichol::IncompleteCholeskyOptions ic_opts;
        ic_opts.scaling = ichol::Scaling::None;
        ic_opts.pivot_shift_strategy = ichol::PivotShiftStrategy::None;
        ic_opts.algorithm = ichol::FactorizationAlgorithm::ICKDT;
        ic_opts.max_restarts = 1;
        ic_opts.verbose = false;
        ic_opts.lfil = A_sub.num_rows;
        ic_opts.drop_tol = 0.0;

        ichol::numeric::NumericPlan num_plan;
        const auto L_sub = ichol::numeric::incomplete_cholesky_preconditioner<double>(A_sub_copy, sym_plan, num_plan, ic_opts);

        out.row_ptr_l = L_sub.row_ptr;
        out.col_ind_l = L_sub.col_ind;
        out.val_l = L_sub.values;
        build_csr_transpose(L_sub.num_rows, out.row_ptr_l, out.col_ind_l, out.val_l,
                            out.row_ptr_lt, out.col_ind_lt, out.val_lt);
        return out;
    }

    template <typename T>
    void apply_ic_impl(
        SubdomainIncompleteCholeskyContext *ctx,
        SubdomainTriangularSolveVariant<T> &variant,
        const T *d_r,
        T *d_z,
        cudaStream_t stream)
    {
        const int nsub = ctx->nsub;
        const int threads = 256;
        const int blocks = (nsub + threads - 1) / threads;
        const T alpha = static_cast<T>(1);

        k_gather_subvec<<<blocks, threads, 0, stream>>>(d_r, ctx->d_gidx, variant.d_rhs, nsub);
        cusparse_check_named(cusparseSetStream(ctx->cusparse, stream), "cusparseSetStream");
        cusparse_check_named(cusparseSpSV_solve(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_l, variant.vec_rhs, variant.vec_y, cuda_data_type<T>(),
            CUSPARSE_SPSV_ALG_DEFAULT, variant.spsv_l), "cusparseSpSV_solve(L)");
        cusparse_check_named(cusparseSpSV_solve(
            ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, variant.mat_lt, variant.vec_y, variant.vec_x, cuda_data_type<T>(),
            CUSPARSE_SPSV_ALG_DEFAULT, variant.spsv_lt), "cusparseSpSV_solve(LT)");
        k_scatter_subvec<<<blocks, threads, 0, stream>>>(variant.d_x, ctx->d_gidx, d_z, nsub);
        cuda_check(cudaGetLastError());
    }
} // namespace

namespace ichol::precond::detail
{
    void *create_subdomain_incomplete_cholesky_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &reg,
        const SubdomainPreconditionerOptions &options)
    {
        auto *ctx = new SubdomainIncompleteCholeskyContext();
        try
        {
            const int lw = reg.x1 - reg.x0;
            const int lh = reg.y1 - reg.y0;
            const int ld = reg.z1 - reg.z0;
            ctx->nsub = lw * lh * ld;
            if (ctx->nsub <= 0)
                throw std::runtime_error("create_subdomain_incomplete_cholesky_context: empty subdomain");

            ctx->storage_prec = normalize_sparse_solve_precision(options.precision);
            const auto A_sub = extract_lower_subdomain_csr(A, global, reg);
            const auto factor = factorize_subdomain_ic(A_sub, options.ic_level_k);

            cuda_check(cudaMalloc(&ctx->d_gidx, static_cast<std::size_t>(ctx->nsub) * sizeof(int)));
            build_subdomain_gidx(ctx->d_gidx, lw, lh, ld, global.w, global.h, reg.x0, reg.y0, reg.z0);

            const int nnz_l = static_cast<int>(factor.val_l.size());
            const int nnz_lt = static_cast<int>(factor.val_lt.size());
            cuda_check(cudaMalloc(&ctx->d_row_ptr_l, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_l, static_cast<std::size_t>(nnz_l) * sizeof(int)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr_l, factor.row_ptr_l.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind_l, factor.col_ind_l.data(), static_cast<std::size_t>(nnz_l) * sizeof(int), cudaMemcpyHostToDevice));

            cuda_check(cudaMalloc(&ctx->d_row_ptr_lt, static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind_lt, static_cast<std::size_t>(nnz_lt) * sizeof(int)));
            cuda_check(cudaMemcpy(ctx->d_row_ptr_lt, factor.row_ptr_lt.data(), static_cast<std::size_t>(ctx->nsub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind_lt, factor.col_ind_lt.data(), static_cast<std::size_t>(nnz_lt) * sizeof(int), cudaMemcpyHostToDevice));

            cusparse_check_named(cusparseCreate(&ctx->cusparse), "cusparseCreate(ctx->cusparse)");
            switch (ctx->storage_prec)
            {
            case ichol::solver::ComputePrecision::FP64:
                init_variant(ctx, ctx->fp64, factor.val_l, factor.val_lt);
                break;
            case ichol::solver::ComputePrecision::FP32:
                init_variant(ctx, ctx->fp32, factor.val_l, factor.val_lt);
                break;
            default:
                throw std::runtime_error("create_subdomain_incomplete_cholesky_context: unsupported precision");
            }
            return ctx;
        }
        catch (...)
        {
            destroy_ctx_impl(ctx);
            throw;
        }
    }

    void apply_subdomain_incomplete_cholesky(
        void *vctx,
        const void *d_r,
        void *d_z,
        int /*N*/,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainIncompleteCholeskyContext *>(vctx);
        const auto requested_prec = normalize_sparse_solve_precision(prec);
        if (requested_prec != ctx->storage_prec)
            throw std::runtime_error("apply_subdomain_incomplete_cholesky: requested precision does not match stored IC preconditioner precision");

        switch (ctx->storage_prec)
        {
        case ichol::solver::ComputePrecision::FP64:
            apply_ic_impl(ctx, ctx->fp64,
                          static_cast<const double *>(d_r),
                          static_cast<double *>(d_z),
                          stream);
            break;
        case ichol::solver::ComputePrecision::FP32:
            apply_ic_impl(ctx, ctx->fp32,
                          static_cast<const float *>(d_r),
                          static_cast<float *>(d_z),
                          stream);
            break;
        default:
            break;
        }
    }

    void destroy_subdomain_incomplete_cholesky_context(void *vctx)
    {
        destroy_ctx_impl(reinterpret_cast<SubdomainIncompleteCholeskyContext *>(vctx));
    }
} // namespace ichol::precond::detail
