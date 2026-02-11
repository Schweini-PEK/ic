#include <stdexcept>
#include <cstdint>
#include <limits>

#include "ichol/preconditioner.hpp"

// ADI preconditioners for 3D Poisson in the paper are the *directional* operators:
//   Mx ~ -u_xx,  My ~ -u_yy,  Mz ~ -u_zz.
// On the n×n×n interior grid with the same scaling as your gen_3dpoi():
// - Each directional operator contributes:
//     diag = 2, off-diagonal = -1 along that direction
// - And A = Mx + My + Mz gives diag=6 and off=-1 on each axis neighbor.
//
// Storage request: CSR of lower triangle + diagonal only, row sorted by col index.
// For each row i, only one lower neighbor exists in a given direction:
//   X: i-1    if x>0
//   Y: i-n    if y>0
//   Z: i-n^2  if z>0
// Then diagonal i. Emitting [lower, diag] keeps col_ind sorted.

template <typename T>
ichol::matrix::CsrMatrix<T> gen_3dpoi_adi_dir(int n, ichol::precond::ADIDirection3D dir)
{
    if (n <= 0)
        throw std::runtime_error("gen_3dpoi_adi_dir: n must be positive");

    // Total unknowns N = n^3 (interior grid points).
    const int64_t N64 = 1LL * n * n * n;
    if (N64 > std::numeric_limits<int>::max())
        throw std::runtime_error("gen_3dpoi_adi_dir: n^3 exceeds int index range");
    const int N = static_cast<int>(N64);

    // For one direction, stored nnz per row = 1 (diag) + indicator(lower neighbor exists).
    // Number of lower-direction edges:
    //   X: count(x>0) = (n-1)*n*n
    //   Y: count(y>0) = (n-1)*n*n
    //   Z: count(z>0) = (n-1)*n*n
    const int64_t edges64 = 1LL * (n - 1) * n * n;
    const int64_t nnz64 = 1LL * N + edges64;
    if (nnz64 > std::numeric_limits<int>::max())
        throw std::runtime_error("gen_3dpoi_adi_dir: nnz exceeds int range");

    ichol::matrix::CsrMatrix<T> M;
    M.num_rows = N;
    M.num_cols = N;
    M.nnz = static_cast<int>(nnz64);

    M.row_ptr.resize(static_cast<size_t>(N) + 1);
    M.col_ind.reserve(static_cast<size_t>(M.nnz));
    M.values.reserve(static_cast<size_t>(M.nnz));

    // Linearization consistent with gen_3dpoi():
    //   id(x,y,z) = x + y*n + z*n*n, with x fastest.
    auto id = [n](int x, int y, int z) -> int
    {
        return x + y * n + z * n * n;
    };

    // Direction step (in the linearized index).
    int step = 0;
    switch (dir)
    {
    case ichol::precond::ADIDirection3D::X:
        step = 1;
        break; // neighbor at x-1 is i-1
    case ichol::precond::ADIDirection3D::Y:
        step = n;
        break; // neighbor at y-1 is i-n
    case ichol::precond::ADIDirection3D::Z:
        step = n * n;
        break; // neighbor at z-1 is i-n^2
    default:
        throw std::runtime_error("gen_3dpoi_adi_dir: invalid dir");
    }

    int nnz_so_far = 0;
    M.row_ptr[0] = 0;

    for (int z = 0; z < n; ++z)
    {
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < n; ++x)
            {
                const int i = id(x, y, z);
                M.row_ptr[i] = nnz_so_far;

                // -----------------------------
                // Off-diagonal (lower only):
                // This realizes the directional coupling term:
                //   M_dir(i, i-step) = -1  if the "minus-direction" neighbor exists.
                // -----------------------------
                bool has_lower = false;
                if (dir == ichol::precond::ADIDirection3D::X)
                    has_lower = (x > 0);
                if (dir == ichol::precond::ADIDirection3D::Y)
                    has_lower = (y > 0);
                if (dir == ichol::precond::ADIDirection3D::Z)
                    has_lower = (z > 0);

                if (has_lower)
                {
                    const int j = i - step; // guaranteed j < i
                    M.col_ind.push_back(j);
                    M.values.push_back(static_cast<T>(-1));
                    ++nnz_so_far;
                }

                // -----------------------------
                // Diagonal:
                // This realizes the diagonal of the 1D Laplacian in that direction:
                //   M_dir(i,i) = 2
                // so that Mx+My+Mz matches your A (diag 6).
                // -----------------------------
                M.col_ind.push_back(i);
                M.values.push_back(static_cast<T>(2));
                ++nnz_so_far;

                // Row is sorted by construction: [i-step] (if present) then [i].
            }
        }
    }

    M.row_ptr[N] = nnz_so_far;
    M.nnz = nnz_so_far; // exact count

    return M;
}

// Convenience wrapper: return the 3 ADI preconditioners {Mx, My, Mz}.
namespace ichol::precond
{
    template <typename T>
    std::vector<ichol::matrix::CsrMatrix<T>> gen_3dpoi_adi_preconds(int n)
    {
        // Paper 4.1.2 uses three preconditioners in ADI fashion:
        // discrete operators for -u_xx, -u_yy, -u_zz => {Mx, My, Mz}.
        std::vector<ichol::matrix::CsrMatrix<T>> Ms;
        Ms.reserve(3);
        Ms.push_back(gen_3dpoi_adi_dir<T>(n, ichol::precond::ADIDirection3D::X)); // Mx
        Ms.push_back(gen_3dpoi_adi_dir<T>(n, ichol::precond::ADIDirection3D::Y)); // My
        Ms.push_back(gen_3dpoi_adi_dir<T>(n, ichol::precond::ADIDirection3D::Z)); // Mz
        return Ms;
    }

    template std::vector<ichol::matrix::CsrMatrix<double>> gen_3dpoi_adi_preconds(int n);

} // namespace ichol::precond
