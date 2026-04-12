#ifndef ICHOL_PCG_HPP
#define ICHOL_PCG_HPP

#include <vector>

#include "ichol/precision.hpp"
#include "ichol/preconditioner.hpp"

namespace ichol::solver
{
    struct PCGParams
    {
        int maxits = 500;
        double tol = 1e-10;
        int restart = 0;

        ComputePrecision prec_gemm = ComputePrecision::FP64;
        ComputePrecision prec_spmm = ComputePrecision::FP64;
        ComputePrecision prec_precond = ComputePrecision::FP64;
        ComputePrecision prec_acc = ComputePrecision::FP64;

        ComputePrecision store_Znew = ComputePrecision::FP64;
        ComputePrecision store_Pnew = ComputePrecision::FP64;
        ComputePrecision store_Wnew = ComputePrecision::FP64;
        ComputePrecision store_P_hist = ComputePrecision::FP64;
        ComputePrecision store_W_hist = ComputePrecision::FP64;

        double rcond_base = 1e-15;
        // If ||P_{i+1}||_A falls below this fraction of ||Z_{i+1}||_A,
        // run one extra projection pass against the history window.
        double projection_anorm_drop_tol = 0.25;

        ichol::precond::PrecondApply *custom_precond = nullptr;

        bool verbose = false;
        bool precond_full = false;
        // When false (default): solve G*alpha=rhs via Cholesky (potrf/potrs), falling
        // back to the SVD path if the k×k system is detected as non-SPD.
        // When true: always use the SVD (pinv) path, preserving prior behaviour.
        bool use_svd = false;

        // ── SPAI subdomain preconditioner parameters ──────────────────────────
        // These control PETSc PCSPAI when subdomains use the SPAI preconditioner.
        // Defaults match PETSc's built-in PCSPAI defaults so that adding these
        // fields does not change numerical behaviour.
        double spai_epsilon = 0.4;   // approximation accuracy target (PETSc default)
        int spai_nbsteps = 5;        // minimisation steps per column  (PETSc default)
        int spai_max_cols = 5;       // max column growth per step     (PETSc default)
        int spai_maxnew = 5;         // max new columns per step       (PETSc default)
        bool spai_symmetric = false; // exploit SPD symmetry (false = PETSc default)
    };

    struct PCGResult
    {
        struct Timing
        {
            double total_ms = 0.0;
            double setup_ms = 0.0;
            double iter_ms = 0.0;
            double finalize_ms = 0.0;
            double preconditioner_apply_ms = 0.0;
            double orthogonalization_ms = 0.0;
            double spmm_ms = 0.0;
            double dense_ms = 0.0;
            double residual_reset_ms = 0.0;
            double other_iter_ms = 0.0;
        };

        int iterations = 0;
        double finalRes = 0.0;
        std::vector<double> relResiduals;
        Timing timing;
    };

    /**
     * @brief Solves a linear system with PCG on GPU.
     *
     * Given the original linear system Ax=b,
     * and the preconditioner L where LL^T \approx D^{-1} A D^{-1} + \alpha I,
     * this function solves By = \tilde{b} with PCG.
     * The stopping criteria is based on the residual norm
     * ||Ax-b||_2 / ||b||_2 \leq tol.
     *
     * Note that @param h_valA is the scaled A, i.e., B's values.
     *
     * @param h_csrRowPtrA CSR row pointer for matrix A (host).
     * @param h_csrColIndA CSR column indices for matrix A (host).
     * @param h_valA       Nonzero values of matrix A (host).
     * @param h_csrRowPtrL CSR row pointer for factor L (host).
     * @param h_csrColIndL CSR column indices for factor L (host).
     * @param h_valL       Nonzero values for factor L (host) in precision T_L.
     * @param h_b          Right-hand side vector b (host).
     * @param h_x          Solution vector x (host, output).
     * @param params       Solver parameters.
     */
    template <typename T_L>
    PCGResult pcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        const PCGParams &params = PCGParams{});

    template <typename T_L>
    PCGResult pcg_cusparse_spsv(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        const PCGParams &params = PCGParams{});

    /**
     * @brief A basic implementation of Multi-Preconditioner Conjugate Gradient (MPCG) on GPU.
     *
     * @param  restart 0 => treat as not truncated.
     *
     * @details The math during iteration i:
     * r_i = b - A x_i
     * Z_{i+1} = [M1^{-1} r_i | ... | Mk^{-1} r_i] (n×k)
     * P_{i+1} = Z_{i+1} - sum_{j in window} P_j * pinv(P_j^T A P_j) * (P_j^T A Z_{i+1})
     * alpha_{i+1} = pinv(P_{i+1}^T A P_{i+1}) * (P_{i+1}^T r_i) (k×1)
     * x_{i+1} = x_i + P_{i+1} * alpha_{i+1}
     * r_{i+1} = r_i - A P_{i+1} * alpha_{i+1}
     */
    template <typename T_L>
    PCGResult mpcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const PCGParams &params);

    template <typename T_L>
    PCGResult mpcg_cpu_baseline(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const PCGParams &params);

#ifndef ICHOL_USE_MKL_MPCG
    // Memory Saving MPCG: Recompute W_{i+1} in each iteration instead of storing it.
    template <typename T_L>
    PCGResult mpcg_ms(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const PCGParams &params);

    template <typename T_L>
    PCGResult mpcg_mixed(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        int m_64,
        int m_32,
        const PCGParams &params);
#endif
} // namespace ichol::solver

#endif // ICHOL_PCG_HPP
