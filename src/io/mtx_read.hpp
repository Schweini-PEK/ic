#ifndef INCHOL_MTX_READ_HPP
#define INCHOL_MTX_READ_HPP

#include <string>
#include "../include/ichol/matrix_formats.hpp"

namespace ichol {
// Reads an MTX file and returns a CSR struct
template <typename T>
CSR<T> readMTXtoCSR(const std::string& path);

// Reads an MTX file and returns a CSC struct
template <typename T>
CSC<T> readMTXtoCSC(const std::string& path);
} // namespace ichol

#endif // INCHOL_MTX_READ_HPP
