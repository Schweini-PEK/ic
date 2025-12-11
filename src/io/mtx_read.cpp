#include "mtx_read.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <stdexcept>

namespace ichol {
template <typename T>
CSR<T> readMTXtoCSR(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::string line;
    // Skip header lines starting with %%
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read dimensions: rows, cols, nnz
    int num_rows, num_cols, nnz;
    std::istringstream iss(line);
    iss >> num_rows >> num_cols >> nnz;

    // Read triples: row, col, val (1-based)
    std::vector<std::tuple<int, int, T>> triples;
    triples.reserve(nnz);
    for (int i = 0; i < nnz; ++i) {
        int row, col;
        T val;
        std::getline(file, line);
        std::istringstream iss2(line);
        iss2 >> row >> col >> val;
        triples.emplace_back(row - 1, col - 1, val); // Convert to 0-based
    }

    // Sort by row, then by col
    std::sort(triples.begin(), triples.end());

    // Build CSR
    CSR<T> csr;
    csr.num_rows = num_rows;
    csr.num_cols = num_cols;
    csr.nnz = nnz;
    csr.row_ptr.resize(num_rows + 1, 0);
    csr.col_ind.resize(nnz);
    csr.values.resize(nnz);

    int current_row = 0;
    csr.row_ptr[0] = 0;
    for (int i = 0; i < nnz; ++i) {
        auto [row, col, val] = triples[i];
        while (current_row < row) {
            csr.row_ptr[++current_row] = i;
        }
        csr.col_ind[i] = col;
        csr.values[i] = val;
    }
    while (current_row < num_rows) {
        csr.row_ptr[++current_row] = nnz;
    }

    return csr;
}

template <typename T>
CSC<T> readMTXtoCSC(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::string line;
    // Skip header lines starting with %%
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read dimensions: rows, cols, nnz
    int num_rows, num_cols, nnz;
    std::istringstream iss(line);
    iss >> num_rows >> num_cols >> nnz;

    // Read triples: row, col, val (1-based)
    std::vector<std::tuple<int, int, T>> triples;
    triples.reserve(nnz);
    for (int i = 0; i < nnz; ++i) {
        int row, col;
        T val;
        std::getline(file, line);
        std::istringstream iss2(line);
        iss2 >> row >> col >> val;
        triples.emplace_back(row - 1, col - 1, val); // Convert to 0-based
    }

    // Sort by col, then by row
    std::sort(triples.begin(), triples.end(), [](const auto& a, const auto& b) {
        auto [r1, c1, v1] = a;
        auto [r2, c2, v2] = b;
        return (c1 < c2) || (c1 == c2 && r1 < r2);
    });

    // Build CSC
    CSC<T> csc;
    csc.num_rows = num_rows;
    csc.num_cols = num_cols;
    csc.nnz = nnz;
    csc.col_ptr.resize(num_cols + 1, 0);
    csc.row_ind.resize(nnz);
    csc.values.resize(nnz);

    int current_col = 0;
    csc.col_ptr[0] = 0;
    for (int i = 0; i < nnz; ++i) {
        auto [row, col, val] = triples[i];
        while (current_col < col) {
            csc.col_ptr[++current_col] = i;
        }
        csc.row_ind[i] = row;
        csc.values[i] = val;
    }
    while (current_col < num_cols) {
        csc.col_ptr[++current_col] = nnz;
    }

    return csc;
}

// Explicit instantiations (add more if needed, e.g., for float)
template CSR<double> readMTXtoCSR<double>(const std::string& path);
template CSC<double> readMTXtoCSC<double>(const std::string& path);
} // namespace ichol
