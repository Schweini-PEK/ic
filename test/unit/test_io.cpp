#include "ichol/mtx_read.hpp"
#include <iostream>
#include <cassert>
#include <string>

int main()
{
    std::string path = "test/data/HB/bcsstk06.mtx";

    try
    {
        ichol::CsrMatrix<double> csr = ichol::io::mtx_to_csr<double>(path, true);
        std::cout << "CSR loaded: rows=" << csr.num_rows << ", cols=" << csr.num_cols << ", nnz=" << csr.nnz << std::endl;
        std::cout << "nnz in csr matrix: " << csr.nnz << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
