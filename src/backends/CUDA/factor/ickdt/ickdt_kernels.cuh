#pragma once

namespace ichol::cuda
{
    template <class G>
    __global__ void ictp_level_kernel(
        int lev_begin, int lev_end,
        const int *level_rows,
        const int *rowPtrA, const int *colIndA, const G *valA,
        const int *rowPtrL, const int *colIndL, G *valL,
        const int *colPtrL, const int *colRowL, const int *colCsrPosL,
        int cap,
        G drop_tol,
        G pivot_tol,
        int max_off_level,
        int H_level,
        int N_level,
        int *d_status, int *d_fail_row);
} // namespace ichol::cuda