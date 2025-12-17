#ifndef INCHOL_MTX_READ_HPP
#define INCHOL_MTX_READ_HPP

#include <string>
#include "../include/ichol/matrix_formats.hpp"

namespace ichol
{
    /**
     * Read a mtx file and store it to a lower tri + diag in CSR format.
     */
    template <typename T>
    CSR<T> readMTXtoCSR(const std::string &path, bool verify);

    template <typename T>
    std::vector<double> toDoubleVector(const std::vector<T> &input)
    {
        std::vector<double> output;
        output.reserve(input.size());
        for (const auto &v : input) output.push_back(static_cast<double>(v));
        return output;
    }
} // namespace ichol

#endif // INCHOL_MTX_READ_HPP
