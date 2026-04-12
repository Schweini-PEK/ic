// include/ichol/preconditioner.hpp

#pragma once
#ifndef ICHOL_PRECONDITIONER_HPP
#define ICHOL_PRECONDITIONER_HPP

#include <vector>

#include "ichol/cuda_compat.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/precision.hpp"

namespace ichol::precond
{
    enum class ADIDirection3D : int
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    struct ADIContext
    {
        int grid_n;        
        ADIDirection3D dir;
    };

    enum class ADIDirection2D
    {
        X,
        Y
    };

    struct ADI2DContext
    {
        int n;
        ADIDirection2D dir;
        double epsilon;
    };

    struct PrecondApply
    {
        using ApplyFn = void (*)(void *ctx,
                                 const void *d_r,
                                 void *d_z,
                                 int N, // total unknowns (should equal grid_n^3)
                                 ichol::solver::ComputePrecision prec,
                                 cudaStream_t stream);

        ApplyFn apply = nullptr;
        void *ctx = nullptr;

        PrecondApply(ApplyFn f = nullptr, void *c = nullptr) : apply(f), ctx(c) {}
    };

    // Apply ADI directional preconditioner:
    //   z = M_dir^{-1} r
    // where M_dir is the 1D Poisson operator (diag=2, off=-1) applied along lines in dir.
    void apply_adi3d_dir(void *ctx,
                         const void *d_r,
                         void *d_z,
                         int N,
                         ichol::solver::ComputePrecision prec,
                         cudaStream_t stream);

    template <typename T>
    std::vector<ichol::matrix::CsrMatrix<T>> gen_3dpoi_adi_preconds(int n);

    void apply_adi2d_dir(void *ctx,
                         const void *d_r,
                         void *d_z,
                         int N, // N = n*n
                         ichol::solver::ComputePrecision prec,
                         cudaStream_t stream);

    template <typename T>
    void generateADIPreconditioners(int n, T epsilon, ichol::matrix::CsrMatrix<T> &Mx, ichol::matrix::CsrMatrix<T> &My);

    ichol::matrix::CsrMatrix<double> build_block_diagonal_exact_preconditioner_3d(
        const ichol::matrix::CsrMatrix<double> &A,
        int n,
        int sub_w,
        int sub_h,
        int sub_d);

    ichol::matrix::CsrMatrix<double> build_block_diagonal_exact_preconditioner_2d(
        const ichol::matrix::CsrMatrix<double> &A,
        int n,
        int sub_w,
        int sub_h);
} // namespace ichol::precond

#endif // ICHOL_PRECONDITIONER_HPP
