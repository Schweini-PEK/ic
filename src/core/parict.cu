// parict.cu
//
// ParICT-style implementation (ParILUT/ParICT loop specialized to SPD),
// assuming the caller already provides a symmetrically scaled SPD matrix.
//
// Fixes vs your current file:
// - Deterministically enforces LOWER+DIAG structure everywhere (j>i -> 0, excluded from dropping).
// - No uninitialized writes: every sweep writes every entry in L_new.
// - New entries are initialized using a residual-style estimate on candidate locations:
//     r_ij = a_ij - sum_{m<j} l_im l_jm   (using OLD values), then l_ij = r_ij / max(l_jj, diag_floor).
// - Initial guess uses A-based values on the initial pattern.
// - End-only validity check is reinstated, but checks what PCG needs:
//     all entries finite and all diagonals finite & strictly positive (=> LL^T SPD).
//   If it fails, returns empty and sets finfo to trigger outer shift-restart.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <limits>

#include "ichol/parict.hpp"
#include "ichol/half.hpp"
#include "ichol/fact.hpp"

namespace ichol
{

    static inline void cuda_check(cudaError_t e, const char *msg)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + msg + ": " + cudaGetErrorString(e));
    }

    // ------------------------
    // Host<->device scalar mapping
    // ------------------------
    template <class T>
    struct gpu_type
    {
        using type = __half;
    };
    template <>
    struct gpu_type<float>
    {
        using type = float;
    };
    template <>
    struct gpu_type<double>
    {
        using type = double;
    };
    template <>
    struct gpu_type<half_float::half>
    {
        using type = __half;
    };

    template <class G>
    struct accum_type
    {
        using type = G;
    };
    template <>
    struct accum_type<__half>
    {
        using type = float;
    };

    // ------------------------
    // Device scalar conversion helpers (specialized; no if constexpr)
    // ------------------------
    template <class G>
    __device__ __forceinline__ typename accum_type<G>::type g_to_acc(G x);

    template <>
    __device__ __forceinline__ float g_to_acc<__half>(__half x) { return __half2float(x); }

    template <>
    __device__ __forceinline__ float g_to_acc<float>(float x) { return x; }

    template <>
    __device__ __forceinline__ double g_to_acc<double>(double x) { return x; }

    template <class G>
    __device__ __forceinline__ G acc_to_g(typename accum_type<G>::type x);

    template <>
    __device__ __forceinline__ __half acc_to_g<__half>(float x) { return __float2half_rn(x); }

    template <>
    __device__ __forceinline__ float acc_to_g<float>(float x) { return x; }

    template <>
    __device__ __forceinline__ double acc_to_g<double>(double x) { return x; }

    template <class Acc>
    __device__ __forceinline__ Acc acc_sqrt(Acc x)
    {
        if (std::is_same<Acc, float>::value)
            return sqrtf((float)x);
        return (Acc)::sqrt((double)x);
    }

    template <class Acc>
    __device__ __forceinline__ Acc acc_abs(Acc x)
    {
        if (std::is_same<Acc, float>::value)
            return fabsf((float)x);
        return (Acc)::fabs((double)x);
    }

    template <class Acc>
    __device__ __forceinline__ Acc acc_max(Acc a, Acc b) { return (a > b) ? a : b; }

    __device__ __forceinline__ bool acc_isfinite(float x) { return isfinite(x); }
    __device__ __forceinline__ bool acc_isfinite(double x) { return isfinite(x); }

    template <class G>
    __device__ __forceinline__ typename accum_type<G>::type g_abs_acc(G x)
    {
        using Acc = typename accum_type<G>::type;
        Acc v = g_to_acc<G>(x);
        return acc_abs<Acc>(v);
    }

    // A small internal floor to prevent diag_floor=0 even if user passes 0.
    // (User still controls pivot_tol; this just avoids division by 0.)
    template <class Acc>
    struct pivot_eps;
    template <>
    struct pivot_eps<float>
    {
        static constexpr float value = 1e-30f;
    };
    template <>
    struct pivot_eps<double>
    {
        static constexpr double value = 1e-300;
    };

    // ------------------------
    // Host conversions (no if constexpr)
    // ------------------------
    template <class G, class T>
    struct host_to_gpu
    {
        static inline G cvt(T x) { return (G)x; }
    };
    template <class T>
    struct host_to_gpu<__half, T>
    {
        static inline __half cvt(T x) { return __float2half_rn((float)x); }
    };

    template <class T, class G>
    struct gpu_to_host
    {
        static inline T cvt(G x) { return (T)x; }
    };
    template <class T>
    struct gpu_to_host<T, __half>
    {
        static inline T cvt(__half x) { return (T)__half2float(x); }
    };

    // ------------------------
    // CSR row binary search (assumes row sorted by col)
    // ------------------------
    template <class G>
    __device__ __forceinline__ G csr_row_get(const int *row_ptr, const int *col_ind, const G *val,
                                             int row, int col)
    {
        int lo = row_ptr[row];
        int hi = row_ptr[row + 1];
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            int c = col_ind[mid];
            if (c < col)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < row_ptr[row + 1] && col_ind[lo] == col)
            return val[lo];
        return (G)0;
    }

    // ------------------------
    // Fill CSR row_ids[nnz] from row_ptr (row_ids[p]=row index)
    // ------------------------
    __global__ void kernel_fill_row_ids(int n, const int *row_ptr, int *row_ids)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        int b = row_ptr[i];
        int e = row_ptr[i + 1];
        for (int p = b; p < e; ++p)
            row_ids[p] = i;
    }

    // ------------------------
    // Build CSC from CSR(L)
    // ------------------------
    __global__ void kernel_count_csc_from_csr(int nnz, const int *col_ind, int n, int *col_counts)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnz)
            return;
        int c = col_ind[p];
        if (0 <= c && c < n)
            atomicAdd(&col_counts[c], 1);
    }

    template <class G>
    __global__ void kernel_fill_csc_from_csr(int nnz,
                                             const int *row_ids,
                                             const int *csr_col, const G *csr_val,
                                             const int *csc_col_ptr, int *csc_next,
                                             int *csc_row, G *csc_val,
                                             int n)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnz)
            return;

        int c = csr_col[p];
        if (!(0 <= c && c < n))
            return;

        int r = row_ids[p];
        int off = atomicAdd(&csc_next[c], 1);
        int dst = csc_col_ptr[c] + off;
        csc_row[dst] = r;
        csc_val[dst] = csr_val[p];
    }

    // ------------------------
    // Product-contrib: for each L(i,m) contribute nnz in CSC column m
    // ------------------------
    __global__ void kernel_product_contrib(int nnzL, const int *L_col, const int *csc_col_ptr, int n, int *contrib)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnzL)
            return;
        int m = L_col[p];
        if (!(0 <= m && m < n))
        {
            contrib[p] = 0;
            return;
        }
        contrib[p] = csc_col_ptr[m + 1] - csc_col_ptr[m];
    }

    // Fill product keys for pattern(L*L^T): for each L(i,m), for each r in col(m): key(i,r) if r<=i else UINT64_MAX
    __global__ void kernel_fill_product_keys(int nnzL,
                                             const int *L_row_ids, const int *L_col,
                                             const int *csc_col_ptr, const int *csc_row,
                                             const int *offset, const int *contrib,
                                             uint64_t *out_keys,
                                             int n)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnzL)
            return;
        int i = L_row_ids[p];
        int m = L_col[p];
        if (!(0 <= m && m < n) || !(0 <= i && i < n))
        {
            // still need to write something to all positions for this p-range
            int len = contrib[p];
            int out = offset[p];
            for (int t = 0; t < len; ++t)
                out_keys[out + t] = UINT64_MAX;
            return;
        }

        int b = csc_col_ptr[m];
        int len = contrib[p];
        int out = offset[p];

        for (int t = 0; t < len; ++t)
        {
            int r = csc_row[b + t];
            if (0 <= r && r <= i)
                out_keys[out + t] = (uint64_t(uint32_t(i)) << 32) | uint32_t(r);
            else
                out_keys[out + t] = UINT64_MAX;
        }
    }

    // Make CSR keys for lower triangle only: key(i,j) if j<=i else UINT64_MAX
    __global__ void kernel_make_csr_keys_lower(int nnz, const int *row_ids, const int *col_ind, int n, uint64_t *keys)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnz)
            return;
        int i = row_ids[p];
        int j = col_ind[p];
        if (0 <= i && i < n && 0 <= j && j <= i)
            keys[p] = (uint64_t(uint32_t(i)) << 32) | uint32_t(j);
        else
            keys[p] = UINT64_MAX;
    }

    // Count keys per row (keys are row-major sorted; atomic is fine for reference)
    __global__ void kernel_row_counts_from_keys(int nnz, const uint64_t *keys, int n, int *row_counts)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnz)
            return;
        uint64_t k = keys[p];
        if (k == UINT64_MAX)
            return;
        int i = int(k >> 32);
        if (0 <= i && i < n)
            atomicAdd(&row_counts[i], 1);
    }

    // Fill col_ind array from keys (keys assumed sorted row-major)
    __global__ void kernel_fill_cols_from_keys(int nnz, const uint64_t *keys, int *col_ind)
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnz)
            return;
        col_ind[p] = int(uint32_t(keys[p] & 0xFFFFFFFFu));
    }

    // ------------------------
    // Residual-style init on expanded pattern:
    // - if existed in old pattern -> copy old value
    // - else:
    //   diag: sqrt(max(Aii - sum Lim^2, pivot_tol_eff))
    //   off : r_ij = Aij - sum_{m<j} Lim*Ljm (OLD values), then lij = r_ij / max(Ljj, diag_floor)
    // Always enforces j>i -> 0.
    // ------------------------
    template <class G>
    __global__ void kernel_init_values_residual(int n,
                                                const int *A_rp, const int *A_ci, const G *A_v,
                                                const int *old_rp, const int *old_ci, const G *old_v,
                                                const int *new_rp, const int *new_ci, G *new_v,
                                                typename accum_type<G>::type pivot_tol_in)
    {
        using Acc = typename accum_type<G>::type;

        Acc pivot_tol = pivot_tol_in;
        pivot_tol = acc_max<Acc>(pivot_tol, (Acc)pivot_eps<Acc>::value);
        Acc diag_floor = acc_sqrt<Acc>(pivot_tol);

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        int ob = old_rp[i], oe = old_rp[i + 1];
        int nb = new_rp[i], ne = new_rp[i + 1];

        for (int p = nb; p < ne; ++p)
        {
            int j = new_ci[p];

            // enforce lower+diag only
            if (!(0 <= j && j <= i))
            {
                new_v[p] = (G)0;
                continue;
            }

            // binary search in old row for (i,j)
            int lo = ob, hi = oe;
            while (lo < hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                int c = old_ci[mid];
                if (c < j)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            if (lo < oe && old_ci[lo] == j)
            {
                // copy old
                new_v[p] = old_v[lo];
                continue;
            }

            // new entry
            if (j == i)
            {
                Acc aii = g_to_acc<G>(csr_row_get<G>(A_rp, A_ci, A_v, i, i));
                Acc sumsq = (Acc)0;
                for (int q = ob; q < oe; ++q)
                {
                    int m = old_ci[q];
                    if (m < 0 || m >= i)
                        continue;
                    Acc lim = g_to_acc<G>(old_v[q]);
                    sumsq += lim * lim;
                }
                Acc d = aii - sumsq;
                d = acc_max<Acc>(d, pivot_tol);
                Acc Lii = acc_sqrt<Acc>(d);
                if (!(Lii > (Acc)0))
                    Lii = diag_floor;
                if (Lii < diag_floor)
                    Lii = diag_floor;
                new_v[p] = acc_to_g<G>(Lii);
            }
            else
            {
                // find Ljj in old row j
                int jb = old_rp[j], je = old_rp[j + 1];
                int diagj_pos = -1;
                for (int q = jb; q < je; ++q)
                {
                    if (old_ci[q] == j)
                    {
                        diagj_pos = q;
                        break;
                    }
                }
                Acc Ljj = (diagj_pos >= 0) ? g_to_acc<G>(old_v[diagj_pos]) : (Acc)0;
                if (!(Ljj > (Acc)0))
                    Ljj = diag_floor;
                if (Ljj < diag_floor)
                    Ljj = diag_floor;

                Acc aij = g_to_acc<G>(csr_row_get<G>(A_rp, A_ci, A_v, i, j));

                // r_ij = aij - sum_{m<j} Lim*Ljm using OLD values
                Acc s = (Acc)0;
                for (int qi = ob; qi < oe; ++qi)
                {
                    int m = old_ci[qi];
                    if (m < 0 || m >= j)
                        continue;

                    Acc Lim = g_to_acc<G>(old_v[qi]);
                    if (Lim == (Acc)0)
                        continue;

                    // find Ljm in old row j (binary search)
                    int l2 = jb, h2 = je;
                    while (l2 < h2)
                    {
                        int mid = l2 + ((h2 - l2) >> 1);
                        int c = old_ci[mid];
                        if (c < m)
                            l2 = mid + 1;
                        else
                            h2 = mid;
                    }
                    if (l2 < je && old_ci[l2] == m)
                    {
                        Acc Ljm = g_to_acc<G>(old_v[l2]);
                        s += Lim * Ljm;
                    }
                }

                Acc rij = aij - s;
                Acc lij = rij / Ljj;

                if (!acc_isfinite(lij))
                    lij = (Acc)0;
                new_v[p] = acc_to_g<G>(lij);
            }
        }
    }

    // ------------------------
    // Guarded Jacobi fixed-point sweep on CSR pattern (SPD Cholesky form)
    // - Writes EVERY entry of L_new deterministically.
    // - Enforces j>i -> 0.
    // - Floors diagonals with pivot_tol_eff.
    // - Replaces non-finite offdiagonals with 0.
    // ------------------------
    template <class G>
    __global__ void kernel_parict_fp_sweep_csr_guarded(int n,
                                                       const int *A_rp, const int *A_ci, const G *A_v,
                                                       const int *L_rp, const int *L_ci,
                                                       const G *L_old,
                                                       typename accum_type<G>::type pivot_tol_in,
                                                       G *L_new)
    {
        using Acc = typename accum_type<G>::type;

        Acc pivot_tol = pivot_tol_in;
        pivot_tol = acc_max<Acc>(pivot_tol, (Acc)pivot_eps<Acc>::value);
        Acc diag_floor = acc_sqrt<Acc>(pivot_tol);

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        int ib = L_rp[i];
        int ie = L_rp[i + 1];

        // deterministic init: lower keeps old, upper forced to 0
        for (int p = ib; p < ie; ++p)
        {
            int j = L_ci[p];
            if (j > i)
                L_new[p] = (G)0;
            else
                L_new[p] = L_old[p];
        }

        // find diagonal in row i (assumed present in valid pattern)
        int diag_pos = -1;
        for (int p = ib; p < ie; ++p)
        {
            if (L_ci[p] == i)
            {
                diag_pos = p;
                break;
            }
        }
        if (diag_pos < 0)
            return;

        // diagonal: Lii = sqrt(max(Aii - sum_{m<i} Lim^2, pivot_tol))
        Acc aii = g_to_acc<G>(csr_row_get<G>(A_rp, A_ci, A_v, i, i));

        Acc sumsq = (Acc)0;
        for (int p = ib; p < ie; ++p)
        {
            int m = L_ci[p];
            if (m < 0 || m >= i)
                continue;
            Acc lim = g_to_acc<G>(L_old[p]);
            sumsq += lim * lim;
        }

        Acc d = aii - sumsq;
        d = acc_max<Acc>(d, pivot_tol);
        Acc Lii = acc_sqrt<Acc>(d);
        if (!(Lii > (Acc)0))
            Lii = diag_floor;
        if (Lii < diag_floor)
            Lii = diag_floor;
        L_new[diag_pos] = acc_to_g<G>(Lii);

        // off-diagonals: Lij = (Aij - sum_{m<j} Lim*Ljm) / max(Ljj, diag_floor)
        for (int p = ib; p < ie; ++p)
        {
            int j = L_ci[p];
            if (j < 0 || j >= i)
                continue;

            Acc aij = g_to_acc<G>(csr_row_get<G>(A_rp, A_ci, A_v, i, j));

            // find Ljj
            int jb = L_rp[j];
            int je = L_rp[j + 1];

            int diagj_pos = -1;
            for (int q = jb; q < je; ++q)
            {
                if (L_ci[q] == j)
                {
                    diagj_pos = q;
                    break;
                }
            }
            if (diagj_pos < 0)
            {
                L_new[p] = (G)0;
                continue;
            }

            Acc Ljj = g_to_acc<G>(L_old[diagj_pos]);
            if (!(Ljj > (Acc)0))
                Ljj = diag_floor;
            if (Ljj < diag_floor)
                Ljj = diag_floor;

            Acc s = (Acc)0;
            for (int pi = ib; pi < ie; ++pi)
            {
                int m = L_ci[pi];
                if (m < 0 || m >= j)
                    continue;

                Acc Lim = g_to_acc<G>(L_old[pi]);
                if (Lim == (Acc)0)
                    continue;

                // find Ljm in row j (binary search)
                int lo = jb, hi = je;
                while (lo < hi)
                {
                    int mid = lo + ((hi - lo) >> 1);
                    int c = L_ci[mid];
                    if (c < m)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                if (lo < je && L_ci[lo] == m)
                {
                    Acc Ljm = g_to_acc<G>(L_old[lo]);
                    s += Lim * Ljm;
                }
            }

            Acc lij = (aij - s) / Ljj;
            if (!acc_isfinite(lij))
                lij = (Acc)0;
            L_new[p] = acc_to_g<G>(lij);
        }
    }

    // ------------------------
    // End-of-run validity check for PCG:
    // - every row has a diagonal entry
    // - diagonal is finite and strictly positive
    // - all entries are finite
    // Writes first failing row into fail_row (initialized to -1).
    // ------------------------
    template <class G>
    __global__ void kernel_check_posdiag_finite(int n,
                                                const int *L_rp, const int *L_ci, const G *L_v,
                                                int *fail_row)
    {
        using Acc = typename accum_type<G>::type;

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        if (atomicAdd(fail_row, 0) >= 0)
            return;

        int ib = L_rp[i];
        int ie = L_rp[i + 1];

        int diag_pos = -1;
        for (int p = ib; p < ie; ++p)
        {
            int j = L_ci[p];
            if (j == i)
                diag_pos = p;

            Acc v = g_to_acc<G>(L_v[p]);
            if (!acc_isfinite(v))
            {
                atomicCAS(fail_row, -1, i);
                return;
            }
        }

        if (diag_pos < 0)
        {
            atomicCAS(fail_row, -1, i);
            return;
        }

        Acc d = g_to_acc<G>(L_v[diag_pos]);
        if (!(d > (Acc)0) || !acc_isfinite(d))
        {
            atomicCAS(fail_row, -1, i);
            return;
        }
    }

    // ------------------------
    // Drop helpers
    // ------------------------
    template <class G>
    __global__ void kernel_build_offdiag_list(int nnz,
                                              const int *row_ids, const int *col_ind,
                                              const G *vals,
                                              int *counter,
                                              int *off_pos,
                                              typename accum_type<G>::type *off_mag)
    {
        using Acc = typename accum_type<G>::type;

        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= nnz)
            return;

        int i = row_ids[p];
        int j = col_ind[p];

        // only strict lower entries are droppable
        if (!(j >= 0 && j < i))
            return;

        Acc v = g_to_acc<G>(vals[p]);
        if (!acc_isfinite(v))
            v = (Acc)0;

        int idx = atomicAdd(counter, 1);
        off_pos[idx] = p;
        off_mag[idx] = acc_abs<Acc>(v);
    }

    __global__ void kernel_set_all_int(int n, int *a, int v)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            a[i] = v;
    }

    __global__ void kernel_scatter_drop(int drop_count, const int *drop_positions, int *keep)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= drop_count)
            return;
        int p = drop_positions[t];
        keep[p] = 0;
    }

    __global__ void kernel_count_kept_per_row(int n, const int *row_ptr, const int *keep, int *row_counts)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        int b = row_ptr[i], e = row_ptr[i + 1];
        int c = 0;
        for (int p = b; p < e; ++p)
            c += (keep[p] != 0);
        row_counts[i] = c;
    }

    template <class G>
    __global__ void kernel_compact_csr(int n,
                                       const int *old_rp, const int *old_ci, const G *old_v,
                                       const int *keep,
                                       const int *new_rp, int *new_ci, G *new_v)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        int ob = old_rp[i], oe = old_rp[i + 1];
        int out = new_rp[i];

        for (int p = ob; p < oe; ++p)
        {
            if (keep[p])
            {
                new_ci[out] = old_ci[p];
                new_v[out] = old_v[p];
                out++;
            }
        }
    }

    // ------------------------
    // Iteration count selection (C++14-safe). Uses whichever field exists:
    // iterations -> sweeps -> max_iters -> default(5)
    // ------------------------
    template <typename P>
    static inline auto pick_steps_impl(const P &p, int) -> decltype(p.iterations, int())
    {
        return (int)p.iterations;
    }
    template <typename P>
    static inline auto pick_steps_impl(const P &p, long) -> decltype(p.sweeps, int())
    {
        return (int)p.sweeps;
    }
    template <typename P>
    static inline auto pick_steps_impl(const P &p, long long) -> decltype(p.max_iters, int())
    {
        return (int)p.max_iters;
    }
    static inline int pick_steps_impl(...) { return 5; }

    static inline int get_num_steps(const ICTP_Params &p)
    {
        int s = pick_steps_impl(p, 0);
        if (s < 1)
            s = 1;
        return s;
    }

    // ------------------------
    // Host CSR get (A rows assumed sorted by col)
    // ------------------------
    template <class T>
    static inline double host_csr_get(const CsrMatrix<T> &A, int i, int j)
    {
        int lo = A.row_ptr[i];
        int hi = A.row_ptr[i + 1];
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            int c = A.col_ind[mid];
            if (c < j)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < A.row_ptr[i + 1] && A.col_ind[lo] == j)
            return (double)A.values[lo];
        return 0.0;
    }

    // ------------------------
    // Initialize L from Sym pattern using A-based guess, but PRUNE to lower+diag only.
    // diag = sqrt(Aii), offdiag = Aij / max(sqrt(Ajj), tiny)
    // ------------------------
    template <class T>
    static CsrMatrix<T> init_L_from_Sym_lower(const CsrMatrix<T> &A, const core::IC_Symbolic &Sym)
    {
        const int n = A.num_rows;

        // Count lower+diag nnz per row from Sym
        std::vector<int> rp((size_t)n + 1, 0);
        for (int i = 0; i < n; ++i)
        {
            int cnt = 0;
            for (int p = Sym.row_ptr_L[i]; p < Sym.row_ptr_L[i + 1]; ++p)
            {
                int j = Sym.col_ind_L[p];
                if (0 <= j && j <= i)
                    cnt++;
            }
            rp[i + 1] = cnt;
        }
        for (int i = 0; i < n; ++i)
            rp[i + 1] += rp[i];
        int nnz = rp[n];

        CsrMatrix<T> L;
        L.num_rows = n;
        L.num_cols = n;
        L.nnz = nnz;
        L.row_ptr = rp;
        L.col_ind.resize((size_t)nnz);
        L.values.resize((size_t)nnz);

        // Precompute sqrt(diag)
        std::vector<double> dj((size_t)n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            double aii = host_csr_get(A, i, i);
            dj[i] = (aii > 0.0) ? std::sqrt(aii) : 0.0;
            if (!(dj[i] > 0.0))
                dj[i] = 1e-300; // avoid div0 in init
        }

        // Fill
        for (int i = 0; i < n; ++i)
        {
            int out = rp[i];
            for (int p = Sym.row_ptr_L[i]; p < Sym.row_ptr_L[i + 1]; ++p)
            {
                int j = Sym.col_ind_L[p];
                if (!(0 <= j && j <= i))
                    continue;

                L.col_ind[out] = j;
                if (j == i)
                {
                    L.values[out] = (T)dj[i];
                }
                else
                {
                    double aij = host_csr_get(A, i, j);
                    L.values[out] = (T)(aij / dj[j]);
                }
                out++;
            }

            // ensure sorted within row (Sym is typically sorted but do not assume)
            int b = rp[i], e = rp[i + 1];
            // simple insertion sort by col on small rows; acceptable as reference
            for (int p = b + 1; p < e; ++p)
            {
                int cj = L.col_ind[p];
                T vj = L.values[p];
                int q = p - 1;
                while (q >= b && L.col_ind[q] > cj)
                {
                    L.col_ind[q + 1] = L.col_ind[q];
                    L.values[q + 1] = L.values[q];
                    --q;
                }
                L.col_ind[q + 1] = cj;
                L.values[q + 1] = vj;
            }
        }

        return L;
    }

    // ------------------------
    // Download device CSR to host CSR<T>
    // ------------------------
    template <class T, class G>
    static CsrMatrix<T> download_L(int n, int nnz, const int *d_rp, const int *d_ci, const G *d_v)
    {
        CsrMatrix<T> L;
        L.num_rows = n;
        L.num_cols = n;
        L.nnz = nnz;

        L.row_ptr.resize((size_t)n + 1);
        L.col_ind.resize((size_t)nnz);
        L.values.resize((size_t)nnz);

        cuda_check(cudaMemcpy(L.row_ptr.data(), d_rp, sizeof(int) * (n + 1), cudaMemcpyDeviceToHost), "download L rp");
        cuda_check(cudaMemcpy(L.col_ind.data(), d_ci, sizeof(int) * nnz, cudaMemcpyDeviceToHost), "download L ci");

        std::vector<G> tmp((size_t)nnz);
        cuda_check(cudaMemcpy(tmp.data(), d_v, sizeof(G) * (size_t)nnz, cudaMemcpyDeviceToHost), "download L val");
        for (int i = 0; i < nnz; ++i)
            L.values[i] = gpu_to_host<T, G>::cvt(tmp[i]);

        return L;
    }

    // ------------------------
    // Top-level ParICT wrapper
    // ------------------------
    template <class T>
    CsrMatrix<T> parict(const CsrMatrix<T> &Ahost,
                  const ICTP_Params &row_params,
                  const IC_Attempt_Params &fparams,
                  const core::IC_Symbolic &Sym,
                  ICTP_Factor_Info *info)
    {
        using G = typename gpu_type<T>::type;
        using Acc = typename accum_type<G>::type;

        const int n = Ahost.num_rows;
        if (Ahost.num_cols != n)
            throw std::runtime_error("parict(ParICT): A must be square.");
        if (Sym.n != n)
            throw std::runtime_error("parict(ParICT): Sym.n mismatch.");

        const int steps = get_num_steps(row_params);

        // Initial L from Sym, pruned to lower+diag, A-based guess
        CsrMatrix<T> L0 = init_L_from_Sym_lower<T>(Ahost, Sym);
        const int nnz_target = L0.nnz;
        const int off_target = std::max(0, nnz_target - n);

        // ---------- Stage A to device ----------
        int *dA_rp = nullptr, *dA_ci = nullptr;
        G *dA_v = nullptr;

        cuda_check(cudaMalloc(&dA_rp, sizeof(int) * (n + 1)), "malloc A rp");
        cuda_check(cudaMalloc(&dA_ci, sizeof(int) * (size_t)Ahost.nnz), "malloc A ci");
        cuda_check(cudaMalloc(&dA_v, sizeof(G) * (size_t)Ahost.nnz), "malloc A v");

        cuda_check(cudaMemcpy(dA_rp, Ahost.row_ptr.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice), "cpy A rp");
        cuda_check(cudaMemcpy(dA_ci, Ahost.col_ind.data(), sizeof(int) * (size_t)Ahost.nnz, cudaMemcpyHostToDevice), "cpy A ci");
        {
            std::vector<G> tmp((size_t)Ahost.nnz);
            for (int i = 0; i < Ahost.nnz; ++i)
                tmp[i] = host_to_gpu<G, T>::cvt(Ahost.values[i]);
            cuda_check(cudaMemcpy(dA_v, tmp.data(), sizeof(G) * (size_t)Ahost.nnz, cudaMemcpyHostToDevice), "cpy A v");
        }

        // ---------- Stage initial L to device ----------
        int *dL_rp = nullptr, *dL_ci = nullptr;
        G *dL_v0 = nullptr, *dL_v1 = nullptr;

        int nnzL = L0.nnz;

        cuda_check(cudaMalloc(&dL_rp, sizeof(int) * (n + 1)), "malloc L rp");
        cuda_check(cudaMalloc(&dL_ci, sizeof(int) * (size_t)nnzL), "malloc L ci");
        cuda_check(cudaMalloc(&dL_v0, sizeof(G) * (size_t)nnzL), "malloc L v0");
        cuda_check(cudaMalloc(&dL_v1, sizeof(G) * (size_t)nnzL), "malloc L v1");

        cuda_check(cudaMemcpy(dL_rp, L0.row_ptr.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice), "cpy L rp");
        cuda_check(cudaMemcpy(dL_ci, L0.col_ind.data(), sizeof(int) * (size_t)nnzL, cudaMemcpyHostToDevice), "cpy L ci");
        {
            std::vector<G> tmp((size_t)nnzL);
            for (int i = 0; i < nnzL; ++i)
                tmp[i] = host_to_gpu<G, T>::cvt(L0.values[i]);
            cuda_check(cudaMemcpy(dL_v0, tmp.data(), sizeof(G) * (size_t)nnzL, cudaMemcpyHostToDevice), "cpy L v0");
        }

        int *dL_row_ids = nullptr;
        cuda_check(cudaMalloc(&dL_row_ids, sizeof(int) * (size_t)nnzL), "malloc L row_ids");

        int *d_fail = nullptr;
        cuda_check(cudaMalloc(&d_fail, sizeof(int)), "malloc fail");

        Acc pivot_tol = (Acc)fparams.pivot_tol;

        // ---------- Step loop ----------
        for (int s = 0; s < steps; ++s)
        {
            // row_ids for current L
            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_fill_row_ids<<<blocks, threads>>>(n, dL_rp, dL_row_ids);
                cuda_check(cudaGetLastError(), "kernel_fill_row_ids(L)");
            }

            // ---------- Build CSC of current L ----------
            int *d_col_counts = nullptr, *d_col_ptr = nullptr, *d_col_next = nullptr, *d_csc_row = nullptr;
            G *d_csc_val = nullptr;

            cuda_check(cudaMalloc(&d_col_counts, sizeof(int) * (n + 1)), "malloc col_counts");
            cuda_check(cudaMalloc(&d_col_ptr, sizeof(int) * (n + 1)), "malloc col_ptr");
            cuda_check(cudaMalloc(&d_col_next, sizeof(int) * (n + 1)), "malloc col_next");
            cuda_check(cudaMalloc(&d_csc_row, sizeof(int) * (size_t)nnzL), "malloc csc_row");
            cuda_check(cudaMalloc(&d_csc_val, sizeof(G) * (size_t)nnzL), "malloc csc_val");

            cuda_check(cudaMemset(d_col_counts, 0, sizeof(int) * (n + 1)), "memset col_counts");
            {
                int threads = 256;
                int blocks = (nnzL + threads - 1) / threads;
                kernel_count_csc_from_csr<<<blocks, threads>>>(nnzL, dL_ci, n, d_col_counts);
                cuda_check(cudaGetLastError(), "kernel_count_csc_from_csr");
            }
            {
                thrust::device_ptr<int> counts(d_col_counts);
                thrust::device_ptr<int> ptr(d_col_ptr);
                thrust::exclusive_scan(thrust::device, counts, counts + (n + 1), ptr);
            }
            cuda_check(cudaMemset(d_col_next, 0, sizeof(int) * (n + 1)), "memset col_next");
            {
                int threads = 256;
                int blocks = (nnzL + threads - 1) / threads;
                kernel_fill_csc_from_csr<G><<<blocks, threads>>>(nnzL, dL_row_ids, dL_ci, dL_v0,
                                                                 d_col_ptr, d_col_next, d_csc_row, d_csc_val, n);
                cuda_check(cudaGetLastError(), "kernel_fill_csc_from_csr");
            }

            // ---------- Keys for L, A, and product pattern ----------
            // L keys (lower only)
            uint64_t *d_keys_L = nullptr;
            cuda_check(cudaMalloc(&d_keys_L, sizeof(uint64_t) * (size_t)nnzL), "malloc keys_L");
            {
                int threads = 256;
                int blocks = (nnzL + threads - 1) / threads;
                kernel_make_csr_keys_lower<<<blocks, threads>>>(nnzL, dL_row_ids, dL_ci, n, d_keys_L);
                cuda_check(cudaGetLastError(), "kernel_make_csr_keys_lower(L)");
            }

            // A keys (lower only)
            int nnzA = Ahost.nnz;
            int *dA_row_ids = nullptr;
            cuda_check(cudaMalloc(&dA_row_ids, sizeof(int) * (size_t)nnzA), "malloc A row_ids");
            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_fill_row_ids<<<blocks, threads>>>(n, dA_rp, dA_row_ids);
                cuda_check(cudaGetLastError(), "kernel_fill_row_ids(A)");
            }
            uint64_t *d_keys_A = nullptr;
            cuda_check(cudaMalloc(&d_keys_A, sizeof(uint64_t) * (size_t)nnzA), "malloc keys_A");
            {
                int threads = 256;
                int blocks = (nnzA + threads - 1) / threads;
                kernel_make_csr_keys_lower<<<blocks, threads>>>(nnzA, dA_row_ids, dA_ci, n, d_keys_A);
                cuda_check(cudaGetLastError(), "kernel_make_csr_keys_lower(A)");
            }

            // product keys
            int *d_contrib = nullptr, *d_offset = nullptr;
            cuda_check(cudaMalloc(&d_contrib, sizeof(int) * (size_t)nnzL), "malloc contrib");
            cuda_check(cudaMalloc(&d_offset, sizeof(int) * (size_t)nnzL), "malloc offset");

            {
                int threads = 256;
                int blocks = (nnzL + threads - 1) / threads;
                kernel_product_contrib<<<blocks, threads>>>(nnzL, dL_ci, d_col_ptr, n, d_contrib);
                cuda_check(cudaGetLastError(), "kernel_product_contrib");
            }
            {
                thrust::device_ptr<int> c(d_contrib);
                thrust::device_ptr<int> o(d_offset);
                thrust::exclusive_scan(thrust::device, c, c + nnzL, o);
            }

            int last_off = 0, last_c = 0;
            cuda_check(cudaMemcpy(&last_off, d_offset + (nnzL - 1), sizeof(int), cudaMemcpyDeviceToHost), "cpy last offset");
            cuda_check(cudaMemcpy(&last_c, d_contrib + (nnzL - 1), sizeof(int), cudaMemcpyDeviceToHost), "cpy last contrib");
            int nnzP = last_off + last_c;

            uint64_t *d_keys_P = nullptr;
            cuda_check(cudaMalloc(&d_keys_P, sizeof(uint64_t) * (size_t)nnzP), "malloc keys_P");
            {
                int threads = 256;
                int blocks = (nnzL + threads - 1) / threads;
                kernel_fill_product_keys<<<blocks, threads>>>(nnzL, dL_row_ids, dL_ci,
                                                              d_col_ptr, d_csc_row,
                                                              d_offset, d_contrib, d_keys_P, n);
                cuda_check(cudaGetLastError(), "kernel_fill_product_keys");
            }

            // concatenate keys_all = [keys_L, keys_A, keys_P]
            int nnzAll = nnzL + nnzA + nnzP;
            uint64_t *d_keys_all = nullptr;
            cuda_check(cudaMalloc(&d_keys_all, sizeof(uint64_t) * (size_t)nnzAll), "malloc keys_all");

            cuda_check(cudaMemcpy(d_keys_all, d_keys_L, sizeof(uint64_t) * (size_t)nnzL, cudaMemcpyDeviceToDevice), "cpy keys_L");
            cuda_check(cudaMemcpy(d_keys_all + nnzL, d_keys_A, sizeof(uint64_t) * (size_t)nnzA, cudaMemcpyDeviceToDevice), "cpy keys_A");
            cuda_check(cudaMemcpy(d_keys_all + nnzL + nnzA, d_keys_P, sizeof(uint64_t) * (size_t)nnzP, cudaMemcpyDeviceToDevice), "cpy keys_P");

            // remove UINT64_MAX, sort+unique (expanded pattern keys)
            thrust::device_ptr<uint64_t> kbeg(d_keys_all);
            thrust::device_ptr<uint64_t> kend = kbeg + nnzAll;
            kend = thrust::remove(thrust::device, kbeg, kend, UINT64_MAX);
            nnzAll = (int)(kend - kbeg);

            thrust::sort(thrust::device, kbeg, kbeg + nnzAll);
            kend = thrust::unique(thrust::device, kbeg, kbeg + nnzAll);
            nnzAll = (int)(kend - kbeg);

            // ---------- Build expanded CSR pattern from keys ----------
            int nnzL2 = nnzAll;

            int *dL2_rp = nullptr, *dL2_ci = nullptr;
            G *dL2_v0 = nullptr, *dL2_v1 = nullptr;

            cuda_check(cudaMalloc(&dL2_rp, sizeof(int) * (n + 1)), "malloc L2 rp");

            int *d_row_counts = nullptr;
            cuda_check(cudaMalloc(&d_row_counts, sizeof(int) * (n + 1)), "malloc row_counts");
            cuda_check(cudaMemset(d_row_counts, 0, sizeof(int) * (n + 1)), "memset row_counts");

            {
                int threads = 256;
                int blocks = (nnzL2 + threads - 1) / threads;
                kernel_row_counts_from_keys<<<blocks, threads>>>(nnzL2, d_keys_all, n, d_row_counts);
                cuda_check(cudaGetLastError(), "kernel_row_counts_from_keys");
            }
            {
                thrust::device_ptr<int> rc(d_row_counts);
                thrust::device_ptr<int> rp(dL2_rp);
                thrust::exclusive_scan(thrust::device, rc, rc + (n + 1), rp);
            }

            cuda_check(cudaMalloc(&dL2_ci, sizeof(int) * (size_t)nnzL2), "malloc L2 ci");
            cuda_check(cudaMalloc(&dL2_v0, sizeof(G) * (size_t)nnzL2), "malloc L2 v0");
            cuda_check(cudaMalloc(&dL2_v1, sizeof(G) * (size_t)nnzL2), "malloc L2 v1");

            {
                int threads = 256;
                int blocks = (nnzL2 + threads - 1) / threads;
                kernel_fill_cols_from_keys<<<blocks, threads>>>(nnzL2, d_keys_all, dL2_ci);
                cuda_check(cudaGetLastError(), "kernel_fill_cols_from_keys");
            }

            // init values (old values kept; new ones residual-initialized)
            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_init_values_residual<G><<<blocks, threads>>>(n, dA_rp, dA_ci, dA_v,
                                                                    dL_rp, dL_ci, dL_v0,
                                                                    dL2_rp, dL2_ci, dL2_v0,
                                                                    pivot_tol);
                cuda_check(cudaGetLastError(), "kernel_init_values_residual");
            }

            // ---------- Sweep #1 (expanded) ----------
            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_parict_fp_sweep_csr_guarded<G><<<blocks, threads>>>(n, dA_rp, dA_ci, dA_v,
                                                                           dL2_rp, dL2_ci, dL2_v0,
                                                                           pivot_tol, dL2_v1);
                cuda_check(cudaGetLastError(), "kernel_parict_fp_sweep_csr_guarded #1");
            }

            // ---------- Global drop to restore nnz target ----------
            int *dL2_row_ids = nullptr;
            cuda_check(cudaMalloc(&dL2_row_ids, sizeof(int) * (size_t)nnzL2), "malloc L2 row_ids");
            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_fill_row_ids<<<blocks, threads>>>(n, dL2_rp, dL2_row_ids);
                cuda_check(cudaGetLastError(), "kernel_fill_row_ids(L2)");
            }

            int *d_off_counter = nullptr;
            int *d_off_pos = nullptr;
            Acc *d_off_mag = nullptr;

            cuda_check(cudaMalloc(&d_off_counter, sizeof(int)), "malloc off_counter");
            cuda_check(cudaMemset(d_off_counter, 0, sizeof(int)), "memset off_counter");

            cuda_check(cudaMalloc(&d_off_pos, sizeof(int) * (size_t)nnzL2), "malloc off_pos");
            cuda_check(cudaMalloc(&d_off_mag, sizeof(Acc) * (size_t)nnzL2), "malloc off_mag");

            {
                int threads = 256;
                int blocks = (nnzL2 + threads - 1) / threads;
                kernel_build_offdiag_list<G><<<blocks, threads>>>(nnzL2, dL2_row_ids, dL2_ci, dL2_v1,
                                                                  d_off_counter, d_off_pos, d_off_mag);
                cuda_check(cudaGetLastError(), "kernel_build_offdiag_list");
            }

            int off_nnz = 0;
            cuda_check(cudaMemcpy(&off_nnz, d_off_counter, sizeof(int), cudaMemcpyDeviceToHost), "read off_nnz");

            int drop_count = off_nnz - off_target;
            if (drop_count < 0)
                drop_count = 0;

            {
                thrust::device_ptr<Acc> mbeg(d_off_mag);
                thrust::device_ptr<int> pbeg(d_off_pos);
                thrust::sort_by_key(thrust::device, mbeg, mbeg + off_nnz, pbeg); // ascending
            }

            int *d_keep = nullptr;
            cuda_check(cudaMalloc(&d_keep, sizeof(int) * (size_t)nnzL2), "malloc keep");
            {
                int threads = 256;
                int blocks = (nnzL2 + threads - 1) / threads;
                kernel_set_all_int<<<blocks, threads>>>(nnzL2, d_keep, 1);
                cuda_check(cudaGetLastError(), "kernel_set_all_int");
            }
            if (drop_count > 0)
            {
                int threads = 256;
                int blocks = (drop_count + threads - 1) / threads;
                kernel_scatter_drop<<<blocks, threads>>>(drop_count, d_off_pos, d_keep);
                cuda_check(cudaGetLastError(), "kernel_scatter_drop");
            }

            // compact CSR to pruned pattern
            int *dL3_rp = nullptr, *dL3_ci = nullptr;
            G *dL3_v0 = nullptr, *dL3_v1 = nullptr;

            cuda_check(cudaMalloc(&dL3_rp, sizeof(int) * (n + 1)), "malloc L3 rp");
            int *d_row_counts3 = nullptr;
            cuda_check(cudaMalloc(&d_row_counts3, sizeof(int) * (n + 1)), "malloc row_counts3");
            cuda_check(cudaMemset(d_row_counts3, 0, sizeof(int) * (n + 1)), "memset row_counts3");

            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_count_kept_per_row<<<blocks, threads>>>(n, dL2_rp, d_keep, d_row_counts3);
                cuda_check(cudaGetLastError(), "kernel_count_kept_per_row");
            }
            {
                thrust::device_ptr<int> rc(d_row_counts3);
                thrust::device_ptr<int> rp(dL3_rp);
                thrust::exclusive_scan(thrust::device, rc, rc + (n + 1), rp);
            }

            int nnzL3 = 0;
            cuda_check(cudaMemcpy(&nnzL3, dL3_rp + n, sizeof(int), cudaMemcpyDeviceToHost), "read nnzL3");

            cuda_check(cudaMalloc(&dL3_ci, sizeof(int) * (size_t)nnzL3), "malloc L3 ci");
            cuda_check(cudaMalloc(&dL3_v0, sizeof(G) * (size_t)nnzL3), "malloc L3 v0");
            cuda_check(cudaMalloc(&dL3_v1, sizeof(G) * (size_t)nnzL3), "malloc L3 v1");

            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_compact_csr<G><<<blocks, threads>>>(n, dL2_rp, dL2_ci, dL2_v1,
                                                           d_keep, dL3_rp, dL3_ci, dL3_v0);
                cuda_check(cudaGetLastError(), "kernel_compact_csr");
            }

            // ---------- Sweep #2 (pruned) ----------
            {
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_parict_fp_sweep_csr_guarded<G><<<blocks, threads>>>(n, dA_rp, dA_ci, dA_v,
                                                                           dL3_rp, dL3_ci, dL3_v0,
                                                                           pivot_tol, dL3_v1);
                cuda_check(cudaGetLastError(), "kernel_parict_fp_sweep_csr_guarded #2");
            }

            // ---------- Replace current L with L3 ----------
            cudaFree(dL_rp);
            cudaFree(dL_ci);
            cudaFree(dL_v0);
            cudaFree(dL_v1);
            dL_rp = dL3_rp;
            dL_ci = dL3_ci;
            dL_v0 = dL3_v1; // newest values
            dL_v1 = dL3_v0; // reuse buffer next step
            nnzL = nnzL3;

            cudaFree(dL_row_ids);
            cuda_check(cudaMalloc(&dL_row_ids, sizeof(int) * (size_t)nnzL), "realloc L row_ids");

            // ---------- Free step temporaries ----------
            cudaFree(d_col_counts);
            cudaFree(d_col_ptr);
            cudaFree(d_col_next);
            cudaFree(d_csc_row);
            cudaFree(d_csc_val);
            cudaFree(d_keys_L);
            cudaFree(dA_row_ids);
            cudaFree(d_keys_A);
            cudaFree(d_contrib);
            cudaFree(d_offset);
            cudaFree(d_keys_P);
            cudaFree(d_keys_all);
            cudaFree(d_row_counts);

            cudaFree(dL2_rp);
            cudaFree(dL2_ci);
            cudaFree(dL2_v0);
            cudaFree(dL2_v1);
            cudaFree(dL2_row_ids);

            cudaFree(d_off_counter);
            cudaFree(d_off_pos);
            cudaFree(d_off_mag);
            cudaFree(d_keep);
            cudaFree(d_row_counts3);
        }

        // ---------- End-of-run validity check (must trigger restart if bad) ----------
        cuda_check(cudaMemset(d_fail, 0xFF, sizeof(int)), "init fail endcheck");
        {
            int threads = 128;
            int blocks = (n + threads - 1) / threads;
            kernel_check_posdiag_finite<G><<<blocks, threads>>>(n, dL_rp, dL_ci, dL_v0, d_fail);
            cuda_check(cudaGetLastError(), "kernel_check_posdiag_finite");
        }
        int hfail = -1;
        cuda_check(cudaMemcpy(&hfail, d_fail, sizeof(int), cudaMemcpyDeviceToHost), "read fail endcheck");
        if (hfail >= 0)
        {
            if (info)
            {
                info->code = IC_Breakdown::B1_SmallOrNegativePivot;
                info->step = hfail;
            }
            cudaFree(dA_rp);
            cudaFree(dA_ci);
            cudaFree(dA_v);
            cudaFree(dL_rp);
            cudaFree(dL_ci);
            cudaFree(dL_v0);
            cudaFree(dL_v1);
            cudaFree(dL_row_ids);
            cudaFree(d_fail);
            return CsrMatrix<T>{};
        }

        // ---------- Download final L ----------
        CsrMatrix<T> L = download_L<T, G>(n, nnzL, dL_rp, dL_ci, dL_v0);

        if (info)
        {
            info->code = IC_Breakdown::None;
            info->step = 0;
        }

        // ---------- Cleanup ----------
        cudaFree(dA_rp);
        cudaFree(dA_ci);
        cudaFree(dA_v);
        cudaFree(dL_rp);
        cudaFree(dL_ci);
        cudaFree(dL_v0);
        cudaFree(dL_v1);
        cudaFree(dL_row_ids);
        cudaFree(d_fail);

        return L;
    }

    // explicit instantiations
    template CsrMatrix<double> parict(const CsrMatrix<double> &Ahost,
                                const ICTP_Params &row_params,
                                const IC_Attempt_Params &fparams,
                                const core::IC_Symbolic &Sym,
                                ICTP_Factor_Info *info);

    template CsrMatrix<float> parict(const CsrMatrix<float> &Ahost,
                               const ICTP_Params &row_params,
                               const IC_Attempt_Params &fparams,
                               const core::IC_Symbolic &Sym,
                               ICTP_Factor_Info *info);

    template CsrMatrix<half_float::half> parict(const CsrMatrix<half_float::half> &Ahost,
                                          const ICTP_Params &row_params,
                                          const IC_Attempt_Params &fparams,
                                          const core::IC_Symbolic &Sym,
                                          ICTP_Factor_Info *info);

} // namespace ichol
