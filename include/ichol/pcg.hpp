#ifndef ICHOL_PCG_HPP
#define ICHOL_PCG_HPP

#include <vector>

#include "ichol/preconditioner.hpp"

namespace ichol::solver
{
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
     * @param iterations   Number of iterations performed (output).
     * @param finalRes     Final residual norm (output).
     */
    template <typename T_L>
    void pcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        int &iterations,
        double &finalRes);

    template <typename T>
    struct PrecondApply
    {
        // Apply z = M^{-1} r
        void (*apply)(void *ctx, const double *d_r, double *d_z, int n, cudaStream_t stream);
        void *ctx;
    };

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
     * 
     * @note currently pinv is realized on CPU.
     */
    template <typename T_L>
    void mpcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        int maxits,
        double tol,
        int restart,
        int &iterations,
        double &finalRes);

} // namespace ichol::solver

#endif // ICHOL_PCG_HPP