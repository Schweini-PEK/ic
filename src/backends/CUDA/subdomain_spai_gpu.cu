#include "backends/CUDA/subdomain_precond_impl.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    // ── helpers ─────────────────────────────────────────────────────────────

    inline void cuda_check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
    }

    inline void cusparse_check(cusparseStatus_t s, const char *what)
    {
        if (s != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cuSPARSE ") + what + ": " +
                                     cusparseGetErrorString(s));
    }

    inline void petsc_check(PetscErrorCode ierr, const char *what)
    {
        if (ierr)
            throw std::runtime_error(std::string("PETSc ") + what +
                                     " returned error code " + std::to_string(ierr));
    }

    // ── PETSc lifecycle ─────────────────────────────────────────────────────
    // PETSc is not thread-safe; all PETSc calls are serialised via petsc_mutex.
    // ensure_petsc_initialized() must be called before taking the mutex.

    static std::once_flag petsc_init_flag;
    static std::mutex     petsc_mutex;

    bool env_var_is_set(const char *name)
    {
        const char *value = std::getenv(name);
        return value != nullptr && value[0] != '\0';
    }

    bool launched_under_mpi_job()
    {
        static constexpr std::array<const char *, 7> kLauncherVars = {
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "PMIX_RANK",
            "PMI_RANK",
            "PMI_SIZE",
            "MPI_LOCALRANKID",
            "SLURM_PROCID",
        };

        for (const char *name : kLauncherVars)
        {
            if (env_var_is_set(name))
                return true;
        }
        return false;
    }

    void configure_openmpi_for_standalone_startup()
    {
        // Disable the ext3x PMIx module unconditionally: it tries to connect to a
        // SLURM PMIx server that doesn't exist when running outside srun, causing
        // MPI_Init to abort.  This is safe when genuinely inside an srun job
        // because other PMIx transports (pmix3, pmix4, …) remain available.
        if (!env_var_is_set("OMPI_MCA_pmix"))
            ::setenv("OMPI_MCA_pmix", "^ext3x", 0);

        if (launched_under_mpi_job() || env_var_is_set("OMPI_MCA_ess"))
            return;

        // Not inside an MPI job: tell Open MPI to take the singleton (in-process)
        // path instead of trying to discover an external launcher.
        ::setenv("OMPI_MCA_ess", "singleton", 0);
        ::setenv("OMPI_MCA_ess_singleton_isolated", "1", 0);
        ::setenv("OMPI_MCA_plm", "isolated", 0);
    }

    void ensure_petsc_initialized()
    {
        std::call_once(petsc_init_flag, []() {
            PetscBool initialized = PETSC_FALSE;
            petsc_check(PetscInitialized(&initialized), "PetscInitialized");
            if (initialized)
                return;

            configure_openmpi_for_standalone_startup();
            petsc_check(PetscInitializeNoArguments(), "PetscInitializeNoArguments");
            std::atexit([]() {
                PetscBool petsc_initialized = PETSC_FALSE;
                PetscBool petsc_finalized = PETSC_FALSE;
                (void)PetscInitialized(&petsc_initialized);
                (void)PetscFinalized(&petsc_finalized);
                if (petsc_initialized && !petsc_finalized)
                    (void)PetscFinalize();
            });
        });
    }

    // ── gather / scatter kernels ─────────────────────────────────────────────

    template <typename T>
    __global__ void k_gather(const T *global_vec, const int *gidx, T *local_vec, int n)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            local_vec[i] = global_vec[gidx[i]];
    }

    template <typename T>
    __global__ void k_scatter(const T *local_vec, const int *gidx, T *global_vec, int n)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            global_vec[gidx[i]] = local_vec[i];
    }

} // namespace

namespace ichol::precond::detail
{
    // ── context ──────────────────────────────────────────────────────────────

    struct SubdomainSpaiContext
    {
        int n_sub = 0; // number of DOFs in this subdomain
        int nnz   = 0; // nonzeros in the SPAI matrix M

        // host-side global indices for this subdomain's DOFs
        std::vector<int> h_gidx;

        // GPU CSR for M  (n_sub × n_sub, local column indices 0..n_sub-1)
        int    *d_row_ptr    = nullptr;
        int    *d_col_ind    = nullptr;
        double *d_val64      = nullptr;
        float  *d_val32      = nullptr;

        // GPU global-index map and local gather/scatter buffers
        int    *d_gidx       = nullptr;
        double *d_local_in64 = nullptr;
        double *d_local_out64= nullptr;
        float  *d_local_in32 = nullptr;
        float  *d_local_out32= nullptr;

        // cuSPARSE handles and descriptors
        cusparseHandle_t     cusparse   = nullptr;
        cusparseSpMatDescr_t sp_mat64   = nullptr;
        cusparseSpMatDescr_t sp_mat32   = nullptr;
        cusparseDnVecDescr_t dn_in64    = nullptr;
        cusparseDnVecDescr_t dn_out64   = nullptr;
        cusparseDnVecDescr_t dn_in32    = nullptr;
        cusparseDnVecDescr_t dn_out32   = nullptr;
        void *buf64 = nullptr;
        void *buf32 = nullptr;
    };

    // ── create ───────────────────────────────────────────────────────────────

    void *create_subdomain_spai_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &reg,
        const SubdomainPreconditionerOptions &options)
    {
        ensure_petsc_initialized();

        auto *ctx = new SubdomainSpaiContext();
        try
        {
            const int lw = reg.x1 - reg.x0;
            const int lh = reg.y1 - reg.y0;
            const int ld = reg.z1 - reg.z0;
            ctx->n_sub = lw * lh * ld;

            // ── 1. Collect global DOF indices ─────────────────────────────────
            ctx->h_gidx.reserve(ctx->n_sub);
            for (int z = reg.z0; z < reg.z1; ++z)
                for (int y = reg.y0; y < reg.y1; ++y)
                    for (int x = reg.x0; x < reg.x1; ++x)
                        ctx->h_gidx.push_back(x + y * global.w + z * global.w * global.h);

            // ── 2. Build global→local map and extract the full symmetric subdomain block ──
            // The global A is stored lower-triangular only.  PETSc SPAI needs the full
            // SPD matrix M_i ≈ A_i^{-1} for correct approximate-inverse construction.
            // Symmetrize explicitly: for every stored lower-tri entry (i,j) with j < i,
            // add the transposed entry (j,i) so that PETSc sees the full operator.
            std::unordered_map<int, int> g2l;
            g2l.reserve(ctx->n_sub);
            for (int li = 0; li < ctx->n_sub; ++li)
                g2l[ctx->h_gidx[li]] = li;

            // lower-triangular extraction (source: global A is lower-tri only)
            std::vector<int>    lower_row_ptr(ctx->n_sub + 1, 0);
            std::vector<int>    lower_col_ind;
            std::vector<double> lower_val;
            lower_col_ind.reserve(ctx->n_sub * 7);
            lower_val.reserve(ctx->n_sub * 7);

            for (int li = 0; li < ctx->n_sub; ++li)
            {
                const int gi = ctx->h_gidx[li];
                for (int k = A.row_ptr[gi]; k < A.row_ptr[gi + 1]; ++k)
                {
                    auto it = g2l.find(A.col_ind[k]);
                    if (it != g2l.end())
                    {
                        lower_col_ind.push_back(it->second);
                        lower_val.push_back(A.values[k]);
                    }
                }
                lower_row_ptr[li + 1] = static_cast<int>(lower_col_ind.size());
            }

            // Reconstruct full symmetric matrix: reflect each off-diagonal lower entry.
            std::vector<std::vector<std::pair<int, double>>> rows(ctx->n_sub);
            for (int li = 0; li < ctx->n_sub; ++li)
            {
                for (int k = lower_row_ptr[li]; k < lower_row_ptr[li + 1]; ++k)
                {
                    const int    lj = lower_col_ind[k];
                    const double v  = lower_val[k];
                    rows[li].push_back({lj, v});
                    if (lj != li)
                        rows[lj].push_back({li, v}); // symmetric entry
                }
            }

            std::vector<int>    full_row_ptr(ctx->n_sub + 1, 0);
            std::vector<int>    full_col_ind;
            std::vector<double> full_val;
            full_col_ind.reserve(lower_col_ind.size() * 2);
            full_val.reserve(lower_val.size() * 2);

            for (int li = 0; li < ctx->n_sub; ++li)
            {
                auto &row = rows[li];
                std::sort(row.begin(), row.end(),
                          [](const auto &a, const auto &b) { return a.first < b.first; });
                for (const auto &[lj, v] : row)
                {
                    full_col_ind.push_back(lj);
                    full_val.push_back(v);
                }
                full_row_ptr[li + 1] = static_cast<int>(full_col_ind.size());
            }

            // ── 3. PETSc: build full A_sub, run SPAI, extract M directly ─────
            // PETSc is not thread-safe; serialise via petsc_mutex.
            // M is extracted via MatGetRow on PCSPAI's internally stored MATSEQAIJ.
            std::vector<int>    m_row_ptr_h;
            std::vector<int>    m_col_ind_h;
            std::vector<double> m_val_h;
            {
                std::lock_guard<std::mutex> lock(petsc_mutex);

                // Build PETSc SEQAIJ matrix for the full symmetric A_sub.
                Mat A_petsc;
                {
                    std::vector<PetscInt> nnz_per_row(ctx->n_sub);
                    for (int li = 0; li < ctx->n_sub; ++li)
                        nnz_per_row[li] = full_row_ptr[li + 1] - full_row_ptr[li];

                    petsc_check(MatCreateSeqAIJ(PETSC_COMM_SELF, ctx->n_sub, ctx->n_sub,
                                               0, nnz_per_row.data(), &A_petsc),
                                "MatCreateSeqAIJ");

                    for (int li = 0; li < ctx->n_sub; ++li)
                    {
                        for (int k = full_row_ptr[li]; k < full_row_ptr[li + 1]; ++k)
                        {
                            const PetscInt    col = full_col_ind[k];
                            const PetscScalar val = full_val[k];
                            petsc_check(MatSetValue(A_petsc, li, col, val, INSERT_VALUES),
                                        "MatSetValue");
                        }
                    }
                    petsc_check(MatAssemblyBegin(A_petsc, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
                    petsc_check(MatAssemblyEnd(A_petsc, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
                    // Inform PETSc of the symmetric sparsity pattern.
                    petsc_check(MatSetOption(A_petsc, MAT_SYMMETRIC, PETSC_TRUE),
                                "MatSetOption(SYMMETRIC)");
                }

                // Set up SPAI preconditioner with tuning parameters from options.
                PC pc;
                petsc_check(PCCreate(PETSC_COMM_SELF, &pc), "PCCreate");
                petsc_check(PCSetType(pc, PCSPAI), "PCSetType(PCSPAI)");
                petsc_check(PCSetOperators(pc, A_petsc, A_petsc), "PCSetOperators");
                petsc_check(PCSPAISetEpsilon(pc, static_cast<PetscReal>(options.spai_epsilon)),
                            "PCSPAISetEpsilon");
                petsc_check(PCSPAISetNBSteps(pc, static_cast<PetscInt>(options.spai_nbsteps)),
                            "PCSPAISetNBSteps");
                petsc_check(PCSPAISetMaxNew(pc, static_cast<PetscInt>(options.spai_maxnew)),
                            "PCSPAISetMaxNew");
                if (options.spai_symmetric)
                    petsc_check(PCSPAISetSp(pc, 1), "PCSPAISetSp");
                petsc_check(PCSetUp(pc), "PCSetUp");

                // Extract M column-by-column: PCApply(e_j) = M * e_j = column j of M.
                // PCSPAI stores M explicitly as a MATSEQAIJ and PCApply does MatMult;
                // each call is therefore O(nnz of column j), not a full solve.
                // PETSc has no public API to retrieve the stored M matrix directly
                // (PC_SPAI::PM is an implementation detail), so we use PCApply here.
                const double thresh = 1e-14;
                struct COOEntry { int row, col; double val; };
                std::vector<COOEntry> coo;
                coo.reserve(static_cast<std::size_t>(ctx->n_sub) * 7);

                Vec e_j, m_j;
                petsc_check(VecCreateSeq(PETSC_COMM_SELF, ctx->n_sub, &e_j), "VecCreateSeq e_j");
                petsc_check(VecDuplicate(e_j, &m_j), "VecDuplicate m_j");

                for (int j = 0; j < ctx->n_sub; ++j)
                {
                    petsc_check(VecZeroEntries(e_j), "VecZeroEntries");
                    petsc_check(VecSetValue(e_j, j, 1.0, INSERT_VALUES), "VecSetValue");
                    petsc_check(VecAssemblyBegin(e_j), "VecAssemblyBegin e_j");
                    petsc_check(VecAssemblyEnd(e_j),   "VecAssemblyEnd e_j");
                    petsc_check(PCApply(pc, e_j, m_j), "PCApply");

                    const PetscScalar *arr;
                    petsc_check(VecGetArrayRead(m_j, &arr), "VecGetArrayRead");
                    for (int i = 0; i < ctx->n_sub; ++i)
                    {
                        const double v = static_cast<double>(arr[i]);
                        if (std::abs(v) > thresh)
                            coo.push_back({i, j, v});
                    }
                    petsc_check(VecRestoreArrayRead(m_j, &arr), "VecRestoreArrayRead");
                }

                // ── COO → CSR via scatter ──────────────────────────────────────────
                // The PCApply loop above produces COO in column-major order
                // (j = 0..n_sub-1).  Within each row the column indices are already
                // ascending (j increases monotonically), so no explicit sort is needed
                // after the scatter step.  A direct copy would reproduce column-major
                // order in the value/col_ind arrays while row_ptr reflects row-major
                // layout — the two would be inconsistent.  Scatter fixes this.
                m_row_ptr_h.assign(ctx->n_sub + 1, 0);
                for (const auto &e : coo)
                    m_row_ptr_h[e.row + 1]++;
                for (int i = 0; i < ctx->n_sub; ++i)
                    m_row_ptr_h[i + 1] += m_row_ptr_h[i];

                m_col_ind_h.resize(coo.size());
                m_val_h.resize(coo.size());
                {
                    std::vector<int> cursor(m_row_ptr_h.begin(),
                                            m_row_ptr_h.begin() + ctx->n_sub);
                    for (const auto &e : coo)
                    {
                        const int pos = cursor[e.row]++;
                        m_col_ind_h[pos] = e.col;
                        m_val_h[pos]     = e.val;
                    }
                }

                // ── Validate: CPU CSR×ones must match PCApply(ones) ───────────────
                // PETSc resources are still alive here so we can call PCApply again.
                {
                    std::vector<double> csr_ones(ctx->n_sub, 0.0);
                    for (int i = 0; i < ctx->n_sub; ++i)
                        for (int p = m_row_ptr_h[i]; p < m_row_ptr_h[i + 1]; ++p)
                            csr_ones[static_cast<std::size_t>(i)] +=
                                m_val_h[static_cast<std::size_t>(p)];

                    petsc_check(VecSet(e_j, 1.0), "VecSet ones");
                    petsc_check(PCApply(pc, e_j, m_j), "PCApply ones");
                    const PetscScalar *arr_ones;
                    petsc_check(VecGetArrayRead(m_j, &arr_ones), "VecGetArrayRead ones");
                    double max_abs_diff = 0.0;
                    double max_abs_ref  = 0.0;
                    for (int i = 0; i < ctx->n_sub; ++i)
                    {
                        const double ref = static_cast<double>(arr_ones[i]);
                        max_abs_diff = std::max(max_abs_diff,
                            std::abs(csr_ones[static_cast<std::size_t>(i)] - ref));
                        max_abs_ref  = std::max(max_abs_ref, std::abs(ref));
                    }
                    petsc_check(VecRestoreArrayRead(m_j, &arr_ones), "VecRestoreArrayRead ones");
                    const double rel_diff =
                        max_abs_diff / (max_abs_ref > 0.0 ? max_abs_ref : 1.0);
                    if (rel_diff > 1e-10)
                        throw std::runtime_error(
                            "create_subdomain_spai_context: extracted CSR does not match "
                            "PCApply (all-ones check): max_rel_dev=" +
                            std::to_string(rel_diff));
                }

                petsc_check(VecDestroy(&e_j), "VecDestroy e_j");
                petsc_check(VecDestroy(&m_j), "VecDestroy m_j");
                petsc_check(PCDestroy(&pc), "PCDestroy");
                petsc_check(MatDestroy(&A_petsc), "MatDestroy");
            } // release petsc_mutex

            ctx->nnz = static_cast<int>(m_col_ind_h.size());

            // ── 4. Upload M CSR to GPU ────────────────────────────────────────
            cuda_check(cudaMalloc(&ctx->d_row_ptr, (ctx->n_sub + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_col_ind, ctx->nnz * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_val64,   ctx->nnz * sizeof(double)));
            cuda_check(cudaMalloc(&ctx->d_val32,   ctx->nnz * sizeof(float)));
            cuda_check(cudaMalloc(&ctx->d_gidx,    ctx->n_sub * sizeof(int)));
            cuda_check(cudaMalloc(&ctx->d_local_in64,  ctx->n_sub * sizeof(double)));
            cuda_check(cudaMalloc(&ctx->d_local_out64, ctx->n_sub * sizeof(double)));
            cuda_check(cudaMalloc(&ctx->d_local_in32,  ctx->n_sub * sizeof(float)));
            cuda_check(cudaMalloc(&ctx->d_local_out32, ctx->n_sub * sizeof(float)));

            cuda_check(cudaMemcpy(ctx->d_row_ptr, m_row_ptr_h.data(),
                                  (ctx->n_sub + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_col_ind, m_col_ind_h.data(),
                                  ctx->nnz * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(ctx->d_val64, m_val_h.data(),
                                  ctx->nnz * sizeof(double), cudaMemcpyHostToDevice));

            // Cast to FP32 on host, then upload
            std::vector<float> m_val32_h(ctx->nnz);
            for (int k = 0; k < ctx->nnz; ++k)
                m_val32_h[k] = static_cast<float>(m_val_h[k]);
            cuda_check(cudaMemcpy(ctx->d_val32, m_val32_h.data(),
                                  ctx->nnz * sizeof(float), cudaMemcpyHostToDevice));

            cuda_check(cudaMemcpy(ctx->d_gidx, ctx->h_gidx.data(),
                                  ctx->n_sub * sizeof(int), cudaMemcpyHostToDevice));

            // ── 5. Create cuSPARSE descriptors ────────────────────────────────
            cusparse_check(cusparseCreate(&ctx->cusparse), "cusparseCreate");

            cusparse_check(
                cusparseCreateCsr(&ctx->sp_mat64,
                                  ctx->n_sub, ctx->n_sub, ctx->nnz,
                                  ctx->d_row_ptr, ctx->d_col_ind, ctx->d_val64,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "cusparseCreateCsr(64)");

            cusparse_check(
                cusparseCreateCsr(&ctx->sp_mat32,
                                  ctx->n_sub, ctx->n_sub, ctx->nnz,
                                  ctx->d_row_ptr, ctx->d_col_ind, ctx->d_val32,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                "cusparseCreateCsr(32)");

            cusparse_check(cusparseCreateDnVec(&ctx->dn_in64,  ctx->n_sub,
                                               ctx->d_local_in64,  CUDA_R_64F), "CreateDnVec in64");
            cusparse_check(cusparseCreateDnVec(&ctx->dn_out64, ctx->n_sub,
                                               ctx->d_local_out64, CUDA_R_64F), "CreateDnVec out64");
            cusparse_check(cusparseCreateDnVec(&ctx->dn_in32,  ctx->n_sub,
                                               ctx->d_local_in32,  CUDA_R_32F), "CreateDnVec in32");
            cusparse_check(cusparseCreateDnVec(&ctx->dn_out32, ctx->n_sub,
                                               ctx->d_local_out32, CUDA_R_32F), "CreateDnVec out32");

            // Pre-compute SpMV workspace sizes
            const double alpha64 = 1.0, beta64 = 0.0;
            const float  alpha32 = 1.0f, beta32 = 0.0f;
            size_t buf_sz64 = 0, buf_sz32 = 0;

            cusparse_check(
                cusparseSpMV_bufferSize(ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha64, ctx->sp_mat64, ctx->dn_in64,
                                        &beta64, ctx->dn_out64,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_sz64),
                "SpMV_bufferSize(64)");

            cusparse_check(
                cusparseSpMV_bufferSize(ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha32, ctx->sp_mat32, ctx->dn_in32,
                                        &beta32, ctx->dn_out32,
                                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_sz32),
                "SpMV_bufferSize(32)");

            if (buf_sz64 > 0)
                cuda_check(cudaMalloc(&ctx->buf64, buf_sz64));
            if (buf_sz32 > 0)
                cuda_check(cudaMalloc(&ctx->buf32, buf_sz32));

            return ctx;
        }
        catch (...)
        {
            destroy_subdomain_spai_context(ctx);
            throw;
        }
    }

    // ── apply ────────────────────────────────────────────────────────────────

    void apply_subdomain_spai(void *vctx, const void *d_r, void *d_z, int /*N*/,
                              ichol::solver::ComputePrecision prec, cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainSpaiContext *>(vctx);

        const int threads = 256;
        const int blocks  = (ctx->n_sub + threads - 1) / threads;

        cusparse_check(cusparseSetStream(ctx->cusparse, stream), "cusparseSetStream");

        switch (prec)
        {
        case ichol::solver::ComputePrecision::FP64:
        {
            const double alpha = 1.0, beta = 0.0;
            k_gather<<<blocks, threads, 0, stream>>>(
                static_cast<const double *>(d_r), ctx->d_gidx,
                ctx->d_local_in64, ctx->n_sub);
            cusparse_check(
                cusparseSpMV(ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, ctx->sp_mat64, ctx->dn_in64,
                             &beta,  ctx->dn_out64,
                             CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->buf64),
                "cusparseSpMV(64)");
            k_scatter<<<blocks, threads, 0, stream>>>(
                ctx->d_local_out64, ctx->d_gidx,
                static_cast<double *>(d_z), ctx->n_sub);
            break;
        }
        case ichol::solver::ComputePrecision::FP32:
        case ichol::solver::ComputePrecision::TF32:
        {
            const float alpha = 1.0f, beta = 0.0f;
            k_gather<<<blocks, threads, 0, stream>>>(
                static_cast<const float *>(d_r), ctx->d_gidx,
                ctx->d_local_in32, ctx->n_sub);
            cusparse_check(
                cusparseSpMV(ctx->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, ctx->sp_mat32, ctx->dn_in32,
                             &beta,  ctx->dn_out32,
                             CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->buf32),
                "cusparseSpMV(32)");
            k_scatter<<<blocks, threads, 0, stream>>>(
                ctx->d_local_out32, ctx->d_gidx,
                static_cast<float *>(d_z), ctx->n_sub);
            break;
        }
        default:
            throw std::runtime_error("apply_subdomain_spai: unsupported precision");
        }
    }

    // ── destroy ──────────────────────────────────────────────────────────────

    void destroy_subdomain_spai_context(void *vctx)
    {
        auto *ctx = reinterpret_cast<SubdomainSpaiContext *>(vctx);
        if (!ctx)
            return;

        if (ctx->dn_out32) cusparseDestroyDnVec(ctx->dn_out32);
        if (ctx->dn_in32)  cusparseDestroyDnVec(ctx->dn_in32);
        if (ctx->dn_out64) cusparseDestroyDnVec(ctx->dn_out64);
        if (ctx->dn_in64)  cusparseDestroyDnVec(ctx->dn_in64);
        if (ctx->sp_mat32) cusparseDestroySpMat(ctx->sp_mat32);
        if (ctx->sp_mat64) cusparseDestroySpMat(ctx->sp_mat64);
        if (ctx->cusparse) cusparseDestroy(ctx->cusparse);

        cudaFree(ctx->buf32);
        cudaFree(ctx->buf64);
        cudaFree(ctx->d_local_out32);
        cudaFree(ctx->d_local_in32);
        cudaFree(ctx->d_local_out64);
        cudaFree(ctx->d_local_in64);
        cudaFree(ctx->d_gidx);
        cudaFree(ctx->d_val32);
        cudaFree(ctx->d_val64);
        cudaFree(ctx->d_col_ind);
        cudaFree(ctx->d_row_ptr);

        delete ctx;
    }

} // namespace ichol::precond::detail
