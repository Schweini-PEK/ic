#ifndef ICHOL_PCG_HPP
#define ICHOL_PCG_HPP

#include <vector>

namespace ichol
{

    template <typename T_L>
    void icPreconditionedCG_GPU(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        int &iterations,
        double &finalRes);

} // namespace ichol

#endif // ICHOL_PCG_HPP