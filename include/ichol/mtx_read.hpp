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
} // namespace ichol::io

#endif // ICHOL_MTX_READ_HPP
