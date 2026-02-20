// include/ichol/preconditioner.hpp

#pragma once
#ifndef ICHOL_PRECONDITIONER_HPP
#define ICHOL_PRECONDITIONER_HPP

#include <vector>
#include <cuda_runtime.h>

#include "ichol/matrix_formats.hpp"

namespace ichol::precond
{

    // Direction for 3D ADI line-solves.
    enum class ADIDirection3D : int
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    // Context carried by a PrecondApply entry.
    struct ADIContext
    {
        int grid_n;         // the "n" in n×n×n grid (so N = n^3 unknowns)
        ADIDirection3D dir; // which direction to solve along
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
                                 const double *d_r,
                                 double *d_z,
                                 int N, // total unknowns (should equal grid_n^3)
                                 cudaStream_t stream);

        ApplyFn apply = nullptr;
        void *ctx = nullptr;
        
        PrecondApply(ApplyFn f = nullptr, void* c = nullptr) : apply(f), ctx(c) {}
    };

    // Apply ADI directional preconditioner:
    //   z = M_dir^{-1} r
    // where M_dir is the 1D Poisson operator (diag=2, off=-1) applied along lines in dir.
    void apply_adi3d_dir(void *ctx,
                       const double *d_r,
                       double *d_z,
                       int N,
                       cudaStream_t stream);

    template <typename T>
    std::vector<ichol::matrix::CsrMatrix<T>> gen_3dpoi_adi_preconds(int n);

    void apply_adi2d_dir(void *ctx,
                         const double *d_r,
                         double *d_z,
                         int N, // N = n*n
                         cudaStream_t stream);

} // namespace ichol::precond

#endif // ICHOL_PRECONDITIONER_HPP