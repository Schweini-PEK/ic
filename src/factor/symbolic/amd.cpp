#include "symbolic.hpp"

#include <numeric>
#include <string>

namespace ichol::symbolic
{
    ichol::symbolic::Permutation amd_from_csr(int n,
                                              const std::vector<int> &row_ptr,
                                              const std::vector<int> &col_ind)
    {
        if (n <= 0)
            return ichol::symbolic::Permutation{};

        std::vector<int> Ap, Ai;
        std::vector<int> col_csr_pos;
        ichol::matrix::csr_to_csc_pattern_only(n,
                                               row_ptr,
                                               col_ind,
                                               Ap,
                                               Ai,
                                               col_csr_pos);

        ichol::symbolic::Permutation P;
        P.perm.assign(n, 0);
        P.inv_perm.assign(n, 0);

        double Info[AMD_INFO];
        const int status = amd_order(n,
                                     Ap.data(),
                                     Ai.data(),
                                     P.perm.data(),
                                     nullptr,
                                     Info);

        if (status != AMD_OK)
        {
            // Common statuses: AMD_INVALID, AMD_OUT_OF_MEMORY
            throw std::runtime_error("amd_order failed (status=" + std::to_string(status) + ").");
        }

        for (int k = 0; k < n; ++k)
        {
            const int orig = P.perm[k];
            if (orig < 0 || orig >= n)
                throw std::runtime_error("amd_order returned invalid permutation.");
            P.inv_perm[orig] = k;
        }

        return P;
    }
} // namespace ichol::symbolic
