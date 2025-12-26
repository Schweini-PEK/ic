#pragma once
#ifndef ICHOL_MTX_READ_HPP
#define ICHOL_MTX_READ_HPP

#include <string>
#include "matrix_formats.hpp"

namespace ichol::io
{
    /**
     * Read a mtx file and store it to a lower tri + diag in CSR format.
     */
    template <typename T>
    ichol::matrix::CsrMatrix<T> mtx_to_csr(const std::string &path, bool verify);

    template <typename T>
    ichol::matrix::CscMatrix<T> mtx_to_csc(const std::string &path, bool verify);

    // TODO: moved to another file
    template <typename T>
    std::vector<double> toDoubleVector(const std::vector<T> &input)
    {
        std::vector<double> output;
        output.reserve(input.size());
        for (const auto &v : input)
            output.push_back(static_cast<double>(v));
        return output;
    }
} // namespace ichol::io

#endif // ICHOL_MTX_READ_HPP