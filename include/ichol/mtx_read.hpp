// include/ichol/mtx_read.hpp
#pragma once
#ifndef ICHOL_MTX_READ_HPP
#define ICHOL_MTX_READ_HPP

#include <string>
#include "ichol/matrix_formats.hpp"

namespace ichol::io
{
    /**
     * Read a mtx file and store it to a lower tri + diag in CSR format.
     */
    template <typename T>
    ichol::matrix::CsrMatrix<T> mtx_to_csr(const std::string &path, bool verify, double alpha = 1e-3);

    template <typename T>
    ichol::matrix::CscMatrix<T> mtx_to_csc(const std::string &path, bool verify, double alpha = 1e-3);

    template <typename T>
    ichol::matrix::CsrMatrix<T> coo_to_csr(const ichol::matrix::CooMatrix<T> &coo_in);

    template <typename T>
    ichol::matrix::CscMatrix<T> coo_to_csc(const ichol::matrix::CooMatrix<T> &coo_in);

    /**
     * Generate a 2D Laplacian matrix in CSR format.
     *
     * @param n The number of grid points per dimension. The matrix is n^2 x n^2.
     */
    template <typename T>
    ichol::matrix::CsrMatrix<T> gen_2dlap_csr(int n);

    /**
     * Generate a 3D Laplacian matrix in CSR format.
     *
     * @param n The number of grid points per dimension. The matrix is n^3 x n^3.
     */
    template <typename T>
    ichol::matrix::CsrMatrix<T> gen_3dlap_csr(int n);

    template <typename T>
    ichol::matrix::CsrMatrix<T> gen_3dpoi(int n);

    /**
     * Generate a 2D Poisson matrix in CSR format with a 5-point stencil, full version.
     */
    template <typename T>
    ichol::matrix::CsrMatrix<T> gen_2dpoi(int n, double epsilon);

    /**
     * Generate the right-hand side vector b for a 2D Poisson problem with manufactured solution.
     * 
     * Here we assume A corresponds to the standard lexicographic ordering on an n-by-n interior grid
     * over (0,1)^2 with h = 1/(n+1). The manufactured solution is u(x,y)=cos(pi x) cos(pi y), 
     * and this function computes b = A*u accordingly.
     * 
     * @return The right-hand side vector b corresponding to the manufactured solution.
     */
    std::vector<double> rhs_2d_poisson_manufactured(const ichol::matrix::CsrMatrix<double> &A, int n);
} // namespace ichol::io

#endif // ICHOL_MTX_READ_HPP
