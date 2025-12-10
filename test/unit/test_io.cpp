#include "../../src/io/mtx_read.hpp"
#include <iostream>
#include <cassert>
#include <string>

int main() {
    std::string path = "test/data/bcsstk11.mtx";

    try {
        // Test CSR reading
        auto csr = ichol::readMTXtoCSR<double>(path);
        std::cout << "CSR loaded: rows=" << csr.num_rows << ", cols=" << csr.num_cols << ", nnz=" << csr.nnz << std::endl;

        // Test CSC reading
        auto csc = ichol::readMTXtoCSC<double>(path);
        std::cout << "CSC loaded: rows=" << csc.num_rows << ", cols=" << csc.num_cols << ", nnz=" << csc.nnz << std::endl;

        // Basic validation: dimensions should match
        assert(csr.num_rows == csc.num_rows);
        assert(csr.num_cols == csc.num_cols);
        assert(csr.nnz == csc.nnz);

        std::cout << "Test passed: CSR and CSC formats match in dimensions." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
