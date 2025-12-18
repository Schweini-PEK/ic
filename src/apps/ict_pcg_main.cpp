// src/apps/ic_cg_main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/fact.hpp"
#include "ichol/pcg.hpp"
#include "ichol/half.hpp"

#include "../io/mtx_read.hpp"

static void usage(const char *argv0)
{
    std::cerr
        << "Usage:\n  " << argv0
        << " --mtx <path> [--prec double|float|half] [--lfil <int>] [--drop <double>]\n";
}

template <typename T>
int run(const std::string &path, int lfil_per_row, double drop_tol, double shift)
{
    ichol::CSR<double> Ahost = ichol::readMTXtoCSR<double>(path, /*keep_upper=*/false);
    const int n = Ahost.num_rows;

    ICTP_Params ictp_params;
    ictp_params.lfil_per_row = lfil_per_row;
    ictp_params.drop_tol = drop_tol;

    IC_Factorize_Params fparams;
    fparams.initial_shift = shift;
    fparams.shift_growth = 2.0;
    fparams.max_restarts = 8;

    IC_Factorize_Info out_info;

    ichol::core::IC_Symbolic Sym = ichol::core::build_ic_symbolic(Ahost, 4);

    ichol::CSR<T> L = ichol::IC_factorize<T>(Ahost, ictp_params, fparams, Sym, &out_info);

    std::vector<double> D(n, 1.0);
    if (!out_info.D.empty())
    {
        D = out_info.D;
    }

    ichol::CSR<double> B = apply_symm_prescaling(Ahost, D);

    std::vector<double> b_tilde(n, 1);
    for (int i = 0; i < n; ++i)
        b_tilde[i] = b_tilde[i] / D[i];

    std::vector<double> y;
    int iters = 0;
    double finalRes = 0;

    ichol::icPreconditionedCG_GPU<double>(
        B.row_ptr, B.col_ind, B.values,
        L.row_ptr, L.col_ind, ichol::toDoubleVector<T>(L.values),
        b_tilde, y, D,
        iters, finalRes);

    std::cout << "CG iters: " << iters << "\n";
    return 0;
}

int main(int argc, char **argv)
{
    std::string mtx_path;
    std::string prec = "double";
    int lfil = 20;
    double drop = 0.0;
    double shift = 1e-10;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need = [&](const char *opt)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << opt << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };

        if (a == "--mtx")
            mtx_path = need("--mtx");
        else if (a == "--prec")
            prec = need("--prec");
        else if (a == "--lfil")
            lfil = std::stoi(need("--lfil"));
        else if (a == "--drop")
            drop = std::stod(need("--drop"));
        else if (a == "--shift")
            shift = std::stod(need("--shift"));
        else
        {
            usage(argv[0]);
            return 2;
        }
    }

    if (mtx_path.empty())
    {
        usage(argv[0]);
        return 2;
    }

    if (prec == "double")
        return run<double>(mtx_path, lfil, drop, shift);
    if (prec == "float")
        return run<float>(mtx_path, lfil, drop, shift);
    if (prec == "half")
        return run<half_float::half>(mtx_path, lfil, drop, shift);

    std::cerr << "Unknown --prec: " << prec << "\n";
    return 2;
}
