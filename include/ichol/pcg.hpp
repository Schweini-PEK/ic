#ifndef ICHOL_PCG_HPP
#define ICHOL_PCG_HPP

#include <vector>

#include "ichol/preconditioner.hpp"

namespace ichol::solver
{
    enum class ComputePrecision
    {
        FP64,     // Standard Double
        FP32,     // Standard Float
        TF32,     // Tensor Float 32 (Internal to H100)
        FP16,     // Half Precision
        BF16,     // Brain Float 16
        FP8_E4M3, // Hopper FP8 (Max precision)
        FP8_E5M2  // Hopper FP8 (Max dynamic range)
    };

    struct PCGParams
    {
        int maxits = 500;
        double tol = 1e-10;
        int restart = 0;

        ComputePrecision prec_gemm = ComputePrecision::FP64;
        ComputePrecision prec_spmm = ComputePrecision::FP64;
        ComputePrecision prec_precond = ComputePrecision::FP64;
        ComputePrecision prec_acc = ComputePrecision::FP64;

        ComputePrecision store_P_hist = ComputePrecision::FP64;
        ComputePrecision store_W_hist = ComputePrecision::FP64;

        double rcond_base = 1e-15;
        // If ||P_{i+1}||_A falls below this fraction of ||Z_{i+1}||_A,
        // run one extra projection pass against the history window.
        double projection_anorm_drop_tol = 0.25;

        ichol::precond::PrecondApply *custom_precond = nullptr;

        bool verbose = false;
    };

    struct PCGResult
    {
        int iterations = 0;
        double finalRes = 0.0;
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
} // namespace ichol::solver

#endif // ICHOL_PCG_HPP
