// src/backends/CUDA/adi_apply.cu

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

#include "ichol/preconditioner.hpp"

namespace ichol::precond
{

    static constexpr int ADI_MAX_N = 2048;

    // Solve n-length tridiagonal system with constant coefficients:
    //   a = -1 (subdiag), b = 2 (diag), c = -1 (superdiag)
    // for many independent lines of the 3D grid, using Thomas algorithm per line.
    //
    // This kernel realizes: z = M_dir^{-1} r
    // by solving n^2 independent 1D Poisson systems along the chosen direction.
    __global__ void k_adi_tridiag_solve_lines(const double *__restrict__ r,
                                              double *__restrict__ z,
                                              int grid_n,
                                              int dir_int)
    {
        const int line = blockIdx.x * blockDim.x + threadIdx.x;
        const int n = grid_n;
        const int nlines = n * n;
        if (line >= nlines)
            return;

        // Map "line id" -> (start, stride) in the global 1D unknown ordering:
        //   id(x,y,z) = x + y*n + z*n*n  (x fastest).
        int start = 0;
        int stride = 0;

        if (dir_int == (int)ADIDirection3D::X)
        {
            // Lines with x varying, fixed (y,z). These are contiguous segments.
            const int y = line % n;
            const int z0 = line / n;
            start = y * n + z0 * n * n; // id(0,y,z)
            stride = 1;
        }
        else if (dir_int == (int)ADIDirection3D::Y)
        {
            // Lines with y varying, fixed (x,z). Stride = n.
            const int x = line % n;
            const int z0 = line / n;
            start = x + z0 * n * n; // id(x,0,z)
            stride = n;
        }
        else
        {
            // dir == Z:
            // Lines with z varying, fixed (x,y). Stride = n*n.
            const int x = line % n;
            const int y = line / n;
            start = x + y * n; // id(x,y,0)
            stride = n * n;
        }

        // Thomas algorithm (forward sweep + back substitution).
        // Coefficients: a=-1, b=2, c=-1.
        // We store c' and d' in local arrays of fixed maximum size.
        double cprime[ADI_MAX_N];
        double dprime[ADI_MAX_N];

        // Forward sweep:
        //   c'[0] = c/b
        //   d'[0] = r0/b
        //   denom_i = b - a*c'[i-1] = 2 - (-1)*c' = 2 + c'
        //   c'[i] = c/denom_i
        //   d'[i] = (r_i - a*d'[i-1]) / denom_i = (r_i + d'[i-1]) / denom_i
        {
            const double b0 = 2.0;
            const double c0 = -1.0;

            const double r0 = r[start];
            double denom = b0;
            cprime[0] = c0 / denom; // = -0.5
            dprime[0] = r0 / denom;

            for (int i = 1; i < n; ++i)
            {
                const double ri = r[start + i * stride];
                denom = 2.0 + cprime[i - 1];                    // since a = -1
                cprime[i] = (i < n - 1) ? (-1.0 / denom) : 0.0; // last c' not used in back-sub
                dprime[i] = (ri + dprime[i - 1]) / denom;
            }
        }

        // Back substitution:
        //   x[n-1] = d'[n-1]
        //   x[i] = d'[i] - c'[i] * x[i+1]
        {
            double x_next = dprime[n - 1];
            z[start + (n - 1) * stride] = x_next;

            for (int i = n - 2; i >= 0; --i)
            {
                const double xi = dprime[i] - cprime[i] * x_next;
                z[start + i * stride] = xi;
                x_next = xi;
            }
        }
    }

    void apply_adi3d_dir(void *vctx,
                         const double *d_r,
                         double *d_z,
                         int N,
                         cudaStream_t stream)
    {
        if (!vctx)
            throw std::runtime_error("apply_adi3d_dir: null ctx");
        const ADIContext *ctx = reinterpret_cast<const ADIContext *>(vctx);
        const int n = ctx->grid_n;

        if (n <= 0)
            throw std::runtime_error("apply_adi3d_dir: ctx->grid_n <= 0");
        if (n > ADI_MAX_N)
            throw std::runtime_error("apply_adi3d_dir: grid_n too large for ADI_MAX_N");

        // Optional consistency check: N should equal n^3.
        const int64_t N_expected = 1LL * n * n * n;
        if ((int64_t)N != N_expected)
            throw std::runtime_error("apply_adi3d_dir: N != n^3 (ctx mismatch)");

        const int lines = n * n; // number of independent 1D systems
        const int block = 256;
        const int grid = (lines + block - 1) / block;

        // This launch realizes: z = M_dir^{-1} r (batched 1D Poisson line-solves).
        k_adi_tridiag_solve_lines<<<grid, block, 0, stream>>>(d_r, d_z, n, (int)ctx->dir);
    }

    __global__ void k_adi2d_tridiag_solve(const double *r, double *z, int n,
                                          int dir_int, double epsilon)
    {
        const int line = blockIdx.x * blockDim.x + threadIdx.x;
        if (line >= n)
            return;

        int start = (dir_int == (int)ADIDirection2D::X) ? (line * n) : line;
        int stride = (dir_int == (int)ADIDirection2D::X) ? 1 : n;

        // Coefficients: a = -val, b = 2*val, c = -val
        // For X: val = 1.0. For Y: val = epsilon.
        const double val = (dir_int == (int)ADIDirection2D::X) ? 1.0 : epsilon;
        const double b_coeff = 2.0 * val;
        const double ac_coeff = -1.0 * val;

        double cprime[ADI_MAX_N]; // Shared or local memory
        double dprime[ADI_MAX_N];

        // Forward sweep
        double denom = b_coeff;
        cprime[0] = ac_coeff / denom;
        dprime[0] = r[start] / denom;

        for (int i = 1; i < n; ++i)
        {
            denom = b_coeff - ac_coeff * cprime[i - 1];
            cprime[i] = ac_coeff / denom;
            dprime[i] = (r[start + i * stride] - ac_coeff * dprime[i - 1]) / denom;
        }

        // Back substitution
        double x_curr = dprime[n - 1];
        z[start + (n - 1) * stride] = x_curr;
        for (int i = n - 2; i >= 0; --i)
        {
            x_curr = dprime[i] - cprime[i] * x_curr;
            z[start + i * stride] = x_curr;
        }
    }

    void apply_adi2d_dir(void *vctx, const double *d_r, double *d_z, int N, cudaStream_t stream)
    {
        auto ctx = reinterpret_cast<ADI2DContext *>(vctx);
        int threads = 128;
        int blocks = (ctx->n + threads - 1) / threads;
        k_adi2d_tridiag_solve<<<blocks, threads, 0, stream>>>(d_r, d_z, ctx->n, (int)ctx->dir, ctx->epsilon);
    }

} // namespace ichol::precond
