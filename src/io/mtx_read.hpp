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
} // namespace ichol

#endif // INCHOL_MTX_READ_HPP
