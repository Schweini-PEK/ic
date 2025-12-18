// parict.cu
//
// Reference ParICT (parallel threshold incomplete Cholesky) in CUDA.
//
// What this is:
// - A ParICT-style outer iteration that (1) adapts the sparsity pattern using
//   residual candidates from A - L*L^T, and (2) performs a fixed-point (Jacobi)
//   sweep to update L for the current pattern.
// - Wrapped with the exact same signature as your current parict<T>(...).
//
// What this is NOT:
// - Not built on your existing fixed-pattern ICTP kernels.
// - Not tuned for performance.
//
// Notes:
// - The symbolic argument (Sym) is accepted for API compatibility; this
//   implementation does not require it (it is ignored).
// - Uses a fixed per-row capacity k = row_params.lfil_per_row (including diag).
// - For half, all device arithmetic uses __half intrinsics (no explicit float temporaries).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <cstdint>
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "ichol/parict.hpp"
#include "ichol/half.hpp"
#include "ichol/fact.hpp"

namespace ichol
{

    // ------------------------
    // Minimal CUDA error check
    // ------------------------
    static inline void cuda_check(cudaError_t e, const char *msg)
    {
        if (e != cudaSuccess)
        {
            throw std::runtime_error(std::string("CUDA error: ") + msg + ": " + cudaGetErrorString(e));
        }
    }

    // ------------------------
    // Host<->device type mapping
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

    // half_float::half maps to __half
    template <>
    struct gpu_type<half_float::half>
    {
        using type = __half;
    };

    // ------------------------
    // Device math (double/float/half)
    // ------------------------
    template <class G>
    struct GMath;

    template <>
    struct GMath<float>
    {
        using M = float;
        __device__ __forceinline__ static float zero() { return 0.0f; }
        __device__ __forceinline__ static float one() { return 1.0f; }
        __device__ __forceinline__ static float add(float a, float b) { return a + b; }
        __device__ __forceinline__ static float sub(float a, float b) { return a - b; }
        __device__ __forceinline__ static float mul(float a, float b) { return a * b; }
        __device__ __forceinline__ static float div(float a, float b) { return a / b; }
        __device__ __forceinline__ static float fma(float a, float b, float c) { return fmaf(a, b, c); }
        __device__ __forceinline__ static float sqrt(float a) { return sqrtf(a); }
        __device__ __forceinline__ static float abs(float a) { return fabsf(a); }
        __device__ __forceinline__ static bool gt(float a, float b) { return a > b; }
        __device__ __forceinline__ static bool lt(float a, float b) { return a < b; }
        __device__ __forceinline__ static bool le(float a, float b) { return a <= b; }
        __device__ __forceinline__ static bool ge(float a, float b) { return a >= b; }
        __device__ __forceinline__ static bool eq0(float a) { return a == 0.0f; }
    };

    template <>
    struct GMath<double>
    {
        using M = double;
        __device__ __forceinline__ static double zero() { return 0.0; }
        __device__ __forceinline__ static double one() { return 1.0; }
        __device__ __forceinline__ static double add(double a, double b) { return a + b; }
        __device__ __forceinline__ static double sub(double a, double b) { return a - b; }
        __device__ __forceinline__ static double mul(double a, double b) { return a * b; }
        __device__ __forceinline__ static double div(double a, double b) { return a / b; }
        __device__ __forceinline__ static double fma(double a, double b, double c) { return fma(a, b, c); }
        __device__ __forceinline__ static double sqrt(double a) { return ::sqrt(a); }
        __device__ __forceinline__ static double abs(double a) { return ::fabs(a); }
        __device__ __forceinline__ static bool gt(double a, double b) { return a > b; }
        __device__ __forceinline__ static bool lt(double a, double b) { return a < b; }
        __device__ __forceinline__ static bool le(double a, double b) { return a <= b; }
        __device__ __forceinline__ static bool ge(double a, double b) { return a >= b; }
        __device__ __forceinline__ static bool eq0(double a) { return a == 0.0; }
    };

    template <>
    struct GMath<__half>
    {
        using M = __half;
        __device__ __forceinline__ static __half zero() { return __float2half_rn(0.0f); }
        __device__ __forceinline__ static __half one() { return __float2half_rn(1.0f); }

        __device__ __forceinline__ static __half add(__half a, __half b) { return __hadd(a, b); }
        __device__ __forceinline__ static __half sub(__half a, __half b) { return __hsub(a, b); }
        __device__ __forceinline__ static __half mul(__half a, __half b) { return __hmul(a, b); }
        __device__ __forceinline__ static __half div(__half a, __half b) { return __hdiv(a, b); }
        __device__ __forceinline__ static __half fma(__half a, __half b, __half c) { return __hadd(__hmul(a, b), c); }

        __device__ __forceinline__ static __half sqrt(__half a) { return hsqrt(a); }

        __device__ __forceinline__ static __half abs(__half a)
        {
            union
            {
                __half h;
                unsigned short u;
            } x;
            x.h = a;
            x.u &= 0x7FFFu;
            return x.h;
        }

        __device__ __forceinline__ static bool gt(__half a, __half b) { return __hgt(a, b); }
        __device__ __forceinline__ static bool lt(__half a, __half b) { return __hlt(a, b); }
        __device__ __forceinline__ static bool le(__half a, __half b) { return __hle(a, b); }
        __device__ __forceinline__ static bool ge(__half a, __half b) { return __hge(a, b); }

        __device__ __forceinline__ static bool eq0(__half a)
        {
            union
            {
                __half h;
                unsigned short u;
            } x;
            x.h = a;
            return (x.u & 0x7FFFu) == 0;
        }
    };

    // ------------------------
    // Param introspection: pick an iteration count without assuming field names
    // ------------------------
    template <typename P, typename = void>
    struct has_iterations : std::false_type
    {
    };
    template <typename P>
    struct has_iterations<P, std::void_t<decltype(std::declval<P>().iterations)>> : std::true_type
    {
    };

    template <typename P, typename = void>
    struct has_max_iters : std::false_type
    {
    };
    template <typename P>
    struct has_max_iters<P, std::void_t<decltype(std::declval<P>().max_iters)>> : std::true_type
    {
    };

    template <typename P, typename = void>
    struct has_sweeps : std::false_type
    {
    };
    template <typename P>
    struct has_sweeps<P, std::void_t<decltype(std::declval<P>().sweeps)>> : std::true_type
    {
    };

    static inline int get_num_sweeps(const ICTP_Params &p)
    {
        int it = 5;
        // if constexpr (has_iterations<ICTP_Params>::value)
        //     it = (int)p.iterations;
        // else if constexpr (has_sweeps<ICTP_Params>::value)
        //     it = (int)p.sweeps;
        // else if constexpr (has_max_iters<ICTP_Params>::value)
        //     it = (int)p.max_iters;
        // if (it < 1)
        //     it = 1;
        return it;
    }

    // ------------------------
    // Host conversions (only for staging / I/O)
    // ------------------------
    template <class G, class T>
    static inline G host_to_gpu_val(T x)
    {
        if constexpr (std::is_same<G, __half>::value)
        {
            // half_float::half -> float -> __half
            return __float2half_rn((float)x);
        }
        else
        {
            return (G)x;
        }
    }

    template <class T, class G>
    static inline T gpu_to_host_val(G x)
    {
        if constexpr (std::is_same<G, __half>::value)
        {
            return (T)__half2float(x);
        }
        else
        {
            return (T)x;
        }
    }

    // ------------------------
    // Device: CSR binary search for value in a row (assumes col_ind sorted)
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
        return GMath<G>::zero();
    }

    // ------------------------
    // Device: count & build CSC for ELL L (cols/vals with fixed width k)
    // ------------------------
    __global__ void kernel_count_csc(int n, int k, const int *L_cols, int *col_counts)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int N = n * k;
        if (tid >= N)
            return;
        int c = L_cols[tid];
        if (c >= 0)
            atomicAdd(&col_counts[c], 1);
    }

    template <class G>
    __global__ void kernel_fill_csc(int n, int k,
                                    const int *L_cols, const G *L_vals,
                                    const int *col_ptr, int *col_next,
                                    int *csc_row, G *csc_val)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int N = n * k;
        if (tid >= N)
            return;

        int c = L_cols[tid];
        if (c < 0)
            return;

        int r = tid / k;
        int off = atomicAdd(&col_next[c], 1);
        int dst = col_ptr[c] + off;
        csc_row[dst] = r;
        csc_val[dst] = L_vals[tid];
    }

    // ------------------------
    // Device: small shared-memory hash table for (key=row, value=accum)
    // ------------------------
    __device__ __forceinline__ unsigned hash_u32(unsigned x)
    {
        return x * 2654435761u;
    }

    __device__ __forceinline__ void shash_init(int *keys, int H)
    {
        for (int t = threadIdx.x; t < H; t += blockDim.x)
        {
            keys[t] = -1;
        }
    }

    template <class G>
    __device__ __forceinline__ void shash_init_vals(G *vals, int H)
    {
        for (int t = threadIdx.x; t < H; t += blockDim.x)
        {
            vals[t] = GMath<G>::zero();
        }
    }

    __device__ __forceinline__ int shash_find_or_insert(int *keys, int H, int key)
    {
        unsigned h = hash_u32((unsigned)key) & (unsigned)(H - 1);
        for (int it = 0; it < H; ++it)
        {
            int slot = (int)((h + (unsigned)it) & (unsigned)(H - 1));
            int prev = atomicCAS(&keys[slot], -1, key);
            if (prev == -1 || prev == key)
                return slot;
        }
        return -1;
    }

    // ------------------------
    // Kernel: pattern update using residual candidates from A - L*L^T
    // - Input: A in CSR, current L in ELL (cols/vals), and CSC view of L
    // - Output: next L pattern (cols_next) and values (vals_next), both ELL width k
    // ------------------------

    template <class G, int H>
    __global__ void kernel_parict_pattern_update(
        int n, int k,
        const int *A_row_ptr, const int *A_col, const G *A_val,
        const int *L_cols, const G *L_vals,
        const int *csc_col_ptr, const int *csc_row, const G *csc_val,
        G drop_tol,
        int *L_cols_next, G *L_vals_next)
    {
        using M = GMath<G>;

        int i = blockIdx.x;
        if (i >= n)
            return;

        // Shared hash for candidates j and accumulator C(i,j) = (L*L^T)(i,j)
        __shared__ int sh_keys[H];
        __shared__ G sh_acc[H];

        shash_init(sh_keys, H);
        shash_init_vals<G>(sh_acc, H);
        __syncthreads();

        // Accumulate C(i, j) over columns kcol appearing in row i
        // For each nonzero Lik at (i, kcol), traverse CSC column kcol to get rows j
        for (int t = threadIdx.x; t < k; t += blockDim.x)
        {
            int idx = i * k + t;
            int kcol = L_cols[idx];
            if (kcol < 0)
                continue;
            G Lik = L_vals[idx];
            if (M::eq0(Lik))
                continue;

            int p0 = csc_col_ptr[kcol];
            int p1 = csc_col_ptr[kcol + 1];
            for (int p = p0; p < p1; ++p)
            {
                int j = csc_row[p];
                if (j > i)
                    continue; // lower triangle only
                G Ljk = csc_val[p];
                if (M::eq0(Ljk))
                    continue;

                int slot = shash_find_or_insert(sh_keys, H, j);
                if (slot >= 0)
                {
                    // sh_acc[slot] += Lik * Ljk
                    // Note: uses atomic-like behavior because multiple threads may hit same slot.
                    // For float/double this isn't atomic; but contention is small for reference code.
                    // We serialize updates with atomicCAS on key; value update may race.
                    // To avoid races cheaply, only thread0 performs accumulation.
                }
            }
        }
        __syncthreads();

        // Race-free accumulation: single thread does the accumulation.
        if (threadIdx.x == 0)
        {
            for (int t = 0; t < k; ++t)
            {
                int idx = i * k + t;
                int kcol = L_cols[idx];
                if (kcol < 0)
                    continue;
                G Lik = L_vals[idx];
                if (M::eq0(Lik))
                    continue;

                int p0 = csc_col_ptr[kcol];
                int p1 = csc_col_ptr[kcol + 1];
                for (int p = p0; p < p1; ++p)
                {
                    int j = csc_row[p];
                    if (j > i)
                        continue;
                    G Ljk = csc_val[p];
                    if (M::eq0(Ljk))
                        continue;

                    unsigned h = hash_u32((unsigned)j) & (unsigned)(H - 1);
                    for (int it = 0; it < H; ++it)
                    {
                        int slot = (int)((h + (unsigned)it) & (unsigned)(H - 1));
                        int key = sh_keys[slot];
                        if (key == j)
                        {
                            sh_acc[slot] = M::add(sh_acc[slot], M::mul(Lik, Ljk));
                            break;
                        }
                        if (key == -1)
                            break;
                    }
                }
            }

            // Row norm of A (lower triangle) for thresholding
            G row_norm = M::zero();
            {
                int b = A_row_ptr[i];
                int e = A_row_ptr[i + 1];
                for (int p = b; p < e; ++p)
                {
                    int c = A_col[p];
                    if (c > i)
                        continue;
                    G mag = M::abs(A_val[p]);
                    if (M::gt(mag, row_norm))
                        row_norm = mag;
                }
            }
            G tau = M::mul(drop_tol, row_norm);

            // Select up to (k-1) off-diagonal entries by score = max(|r|, |lij_old|)
            // using a simple top-k insertion list.
            const int keep = (k > 1 ? (k - 1) : 0);

            // Local arrays (in registers / local memory)
            // store descending by score
            int top_col[128];
            G top_score[128];
            int top_n = 0;

            auto insert_top = [&](int col, G score)
            {
                // de-duplicate: if col already present, update score and bubble
                for (int u = 0; u < top_n; ++u)
                {
                    if (top_col[u] == col)
                    {
                        if (M::gt(score, top_score[u]))
                        {
                            top_score[u] = score;
                            int pos = u;
                            while (pos > 0 && M::gt(top_score[pos], top_score[pos - 1]))
                            {
                                int tc = top_col[pos - 1];
                                G ts = top_score[pos - 1];
                                top_col[pos - 1] = top_col[pos];
                                top_score[pos - 1] = top_score[pos];
                                top_col[pos] = tc;
                                top_score[pos] = ts;
                                --pos;
                            }
                        }
                        return;
                    }
                }

                if (keep == 0)
                    return;
                if (top_n < keep)
                {
                    int pos = top_n++;
                    top_col[pos] = col;
                    top_score[pos] = score;
                    // bubble up to keep descending
                    while (pos > 0 && M::gt(top_score[pos], top_score[pos - 1]))
                    {
                        int tc = top_col[pos - 1];
                        G ts = top_score[pos - 1];
                        top_col[pos - 1] = top_col[pos];
                        top_score[pos - 1] = top_score[pos];
                        top_col[pos] = tc;
                        top_score[pos] = ts;
                        --pos;
                    }
                }
                else
                {
                    // if score <= smallest, ignore
                    if (!M::gt(score, top_score[keep - 1]))
                        return;
                    // replace smallest
                    top_col[keep - 1] = col;
                    top_score[keep - 1] = score;
                    int pos = keep - 1;
                    while (pos > 0 && M::gt(top_score[pos], top_score[pos - 1]))
                    {
                        int tc = top_col[pos - 1];
                        G ts = top_score[pos - 1];
                        top_col[pos - 1] = top_col[pos];
                        top_score[pos - 1] = top_score[pos];
                        top_col[pos] = tc;
                        top_score[pos] = ts;
                        --pos;
                    }
                }
            };

            // Helper: get old L(i,j) (linear scan over k)
            auto get_old_lij = [&](int j) -> G
            {
                for (int t = 0; t < k; ++t)
                {
                    int idx = i * k + t;
                    int c = L_cols[idx];
                    if (c == j)
                        return L_vals[idx];
                }
                return M::zero();
            };

            // Iterate candidates from hash table
            for (int slot = 0; slot < H; ++slot)
            {
                int j = sh_keys[slot];
                if (j < 0)
                    continue;
                if (j == i)
                    continue;
                if (j > i)
                    continue;

                G cij = sh_acc[slot];
                G aij = csr_row_get<G>(A_row_ptr, A_col, A_val, i, j);
                G rij = M::sub(aij, cij);

                G score = M::abs(rij);
                G lij_old = M::abs(get_old_lij(j));
                if (M::gt(lij_old, score))
                    score = lij_old;

                if (M::gt(score, tau))
                {
                    // strictly lower
                    if (j < i)
                        insert_top(j, score);
                }
            }

            // Also consider all explicit A(i,j) (j<i) entries as candidates.
            // If a candidate wasn't discovered via CSC gather, we compute C(i,j) by a direct
            // row-intersection dot (O(k^2), reference simplicity).
            {
                int Ab = A_row_ptr[i];
                int Ae = A_row_ptr[i + 1];

                // hash lookup helper (returns -1 if absent)
                auto hash_find = [&](int key) -> int
                {
                    unsigned h = hash_u32((unsigned)key) & (unsigned)(H - 1);
                    for (int it2 = 0; it2 < H; ++it2)
                    {
                        int slot = (int)((h + (unsigned)it2) & (unsigned)(H - 1));
                        int k0 = sh_keys[slot];
                        if (k0 == key)
                            return slot;
                        if (k0 == -1)
                            return -1;
                    }
                    return -1;
                };

                // direct dot-product C(i,j) = (L*L^T)_{i,j}
                auto dot_rows = [&](int j) -> G
                {
                    G acc = M::zero();
                    for (int ti = 0; ti < k; ++ti)
                    {
                        int colk = L_cols[i * k + ti];
                        if (colk < 0)
                            continue;
                        G lik = L_vals[i * k + ti];
                        if (M::eq0(lik))
                            continue;
                        // find in row j
                        for (int tj = 0; tj < k; ++tj)
                        {
                            if (L_cols[j * k + tj] == colk)
                            {
                                G ljk = L_vals[j * k + tj];
                                if (!M::eq0(ljk))
                                    acc = M::add(acc, M::mul(lik, ljk));
                                break;
                            }
                        }
                    }
                    return acc;
                };

                for (int p = Ab; p < Ae; ++p)
                {
                    int j = A_col[p];
                    if (j < 0 || j >= i)
                        continue;

                    int slot = hash_find(j);
                    G cij = (slot >= 0) ? sh_acc[slot] : dot_rows(j);
                    G aij = A_val[p];
                    G rij = M::sub(aij, cij);

                    G score = M::abs(rij);
                    G lij_old = M::abs(get_old_lij(j));
                    if (M::gt(lij_old, score))
                        score = lij_old;

                    if (M::gt(score, tau))
                        insert_top(j, score);
                }
            }

            // Sort selected cols ascending
            for (int a = 0; a < top_n; ++a)
            {
                for (int b = a + 1; b < top_n; ++b)
                {
                    if (top_col[b] < top_col[a])
                    {
                        int tc = top_col[a];
                        top_col[a] = top_col[b];
                        top_col[b] = tc;
                        G ts = top_score[a];
                        top_score[a] = top_score[b];
                        top_score[b] = ts;
                    }
                }
            }

            // Write next row (fill unused slots with -1)
            // Keep old values for kept columns; new ones are initialized to 0.
            for (int t = 0; t < k; ++t)
            {
                int idx = i * k + t;
                L_cols_next[idx] = -1;
                L_vals_next[idx] = M::zero();
            }

            int out = 0;
            for (int p = 0; p < top_n && out < keep; ++p)
            {
                int j = top_col[p];
                int idx = i * k + out;
                L_cols_next[idx] = j;
                // keep old if existed
                G oldv = get_old_lij(j);
                L_vals_next[idx] = oldv; // may be zero
                out++;
            }

            // Diagonal last
            {
                int didx = i * k + (k - 1);
                L_cols_next[didx] = i;
                // keep old diag if present; if absent, init with sqrt(A_ii)
                G oldd = get_old_lij(i);
                if (M::eq0(oldd))
                {
                    G aii = csr_row_get<G>(A_row_ptr, A_col, A_val, i, i);
                    // If aii <= 0, diag will fail in sweep; set 0 here.
                    if (M::gt(aii, M::zero()))
                        oldd = M::sqrt(aii);
                }
                L_vals_next[didx] = oldd;
            }
        }
    }

    // ------------------------
    // Kernel: one Jacobi fixed-point sweep to update L values for a given pattern
    // Pattern is L_cols (ELL width k, diag at slot k-1).
    // ------------------------

    template <class G>
    __global__ void kernel_parict_fp_sweep(
        int n, int k,
        const int *A_row_ptr, const int *A_col, const G *A_val,
        const int *L_cols, const G *L_vals_old,
        G pivot_tol,
        G *L_vals_new,
        int *fail_row) // single int on device, init -1
    {
        using M = GMath<G>;

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        // If someone already failed, early out.
        if (atomicAdd(fail_row, 0) >= 0)
            return;

        // Diagonal index
        int diag_idx = i * k + (k - 1);

        // Compute diagonal: L_ii = sqrt(A_ii - sum_{m<i} L_im^2)
        G aii = csr_row_get<G>(A_row_ptr, A_col, A_val, i, i);
        G sum = M::zero();

        for (int t = 0; t < k - 1; ++t)
        {
            int idx = i * k + t;
            int col = L_cols[idx];
            if (col < 0 || col >= i)
                continue;
            G v = L_vals_old[idx];
            sum = M::add(sum, M::mul(v, v));
        }

        G d = M::sub(aii, sum);
        // breakdown if d <= pivot_tol or d <= 0
        if (!M::gt(d, pivot_tol) || !M::gt(d, M::zero()))
        {
            atomicCAS(fail_row, -1, i);
            return;
        }

        G Lii = M::sqrt(d);

        // Write diagonal
        L_vals_new[diag_idx] = Lii;

        // Update off-diagonal entries
        for (int t = 0; t < k - 1; ++t)
        {
            int idx = i * k + t;
            int j = L_cols[idx];
            if (j < 0 || j >= i)
            {
                L_vals_new[idx] = M::zero();
                continue;
            }

            // aij
            G aij = csr_row_get<G>(A_row_ptr, A_col, A_val, i, j);

            // sum_{m<j} L_im * L_jm
            G s = M::zero();
            for (int t2 = 0; t2 < k - 1; ++t2)
            {
                int idx2 = i * k + t2;
                int mcol = L_cols[idx2];
                if (mcol < 0 || mcol >= j)
                    continue;
                G Lim = L_vals_old[idx2];
                if (M::eq0(Lim))
                    continue;

                // find L_jm in row j
                G Ljm = M::zero();
                int basej = j * k;
                for (int u = 0; u < k - 1; ++u)
                {
                    int cc = L_cols[basej + u];
                    if (cc == mcol)
                    {
                        Ljm = L_vals_old[basej + u];
                        break;
                    }
                }
                if (M::eq0(Ljm))
                    continue;
                s = M::add(s, M::mul(Lim, Ljm));
            }

            // divide by L_jj
            G Ljj = L_vals_old[j * k + (k - 1)];
            if (M::eq0(Ljj))
            {
                atomicCAS(fail_row, -1, i);
                return;
            }

            G lij = M::div(M::sub(aij, s), Ljj);
            L_vals_new[idx] = lij;
        }
    }

    // ------------------------
    // Host: initialize L pattern from A (lower triangle) with capacity k
    // - Keep strongest |A(i,j)| entries (j<i) up to k-1, diag last
    // - Initial values: offdiag=0, diag=sqrt(A_ii)
    // ------------------------

    template <class T, class G>
    static void init_L_from_A_lower(const CSR<T> &A, int k,
                                    std::vector<int> &L_cols_h,
                                    std::vector<G> &L_vals_h)
    {
        const int n = A.num_rows;
        L_cols_h.assign((size_t)n * k, -1);
        L_vals_h.assign((size_t)n * k, host_to_gpu_val<G>(T(0)));

        for (int i = 0; i < n; ++i)
        {
            // gather lower entries
            struct Entry
            {
                int c;
                T score;
            };
            std::vector<Entry> tmp;
            tmp.reserve(64);

            // diag
            T aii_T = T(0);
            {
                int b = A.row_ptr[i];
                int e = A.row_ptr[i + 1];
                for (int p = b; p < e; ++p)
                {
                    int c = A.col_ind[p];
                    if (c == i)
                    {
                        aii_T = A.values[p];
                        break;
                    }
                }
            }

            // offdiag
            int b = A.row_ptr[i];
            int e = A.row_ptr[i + 1];
            for (int p = b; p < e; ++p)
            {
                int c = A.col_ind[p];
                if (c < 0 || c >= i)
                    continue;
                T v = A.values[p];
                T sc = (v < T(0)) ? -v : v; // keep scoring in T
                tmp.push_back({c, sc});
            }

            std::sort(tmp.begin(), tmp.end(), [](const Entry &a, const Entry &b)
                      {
            if (a.score != b.score) return a.score > b.score;
            return a.c < b.c; });

            int keep = std::max(0, k - 1);
            if ((int)tmp.size() > keep)
                tmp.resize(keep);

            std::sort(tmp.begin(), tmp.end(), [](const Entry &a, const Entry &b)
                      { return a.c < b.c; });

            int out = 0;
            for (auto &e2 : tmp)
            {
                L_cols_h[(size_t)i * k + out] = e2.c;
                L_vals_h[(size_t)i * k + out] = host_to_gpu_val<G>(T(0));
                out++;
            }

            // diag last
            L_cols_h[(size_t)i * k + (k - 1)] = i;

            // diag init sqrt(aii) (best-effort in T; device sweep validates pivot)
            T di = (aii_T > T(0)) ? sqrt(aii_T) : T(0);
            L_vals_h[(size_t)i * k + (k - 1)] = host_to_gpu_val<G>(di);
        }
    }

    // ------------------------
    // Host: compress ELL L into CSR<T> (drop col=-1 slots, keep diag last)
    // ------------------------

    template <class T, class G>
    static CSR<T> compress_L_ell_to_csr(int n, int k,
                                        const std::vector<int> &L_cols_h,
                                        const std::vector<G> &L_vals_h)
    {
        CSR<T> L;
        L.num_rows = n;
        L.num_cols = n;

        L.row_ptr.resize(n + 1);
        L.row_ptr[0] = 0;

        // count
        for (int i = 0; i < n; ++i)
        {
            int cnt = 0;
            // offdiag
            for (int t = 0; t < k - 1; ++t)
            {
                int c = L_cols_h[(size_t)i * k + t];
                if (c >= 0 && c < i)
                    cnt++;
            }
            // diag
            cnt++;
            L.row_ptr[i + 1] = L.row_ptr[i] + cnt;
        }

        L.nnz = L.row_ptr[n];
        L.col_ind.resize(L.nnz);
        L.values.resize(L.nnz);

        // fill
        int outp = 0;
        for (int i = 0; i < n; ++i)
        {
            // gather offdiag cols, sort (already mostly sorted, but be safe)
            struct P
            {
                int c;
                G v;
            };
            std::vector<P> tmp;
            tmp.reserve(k);
            for (int t = 0; t < k - 1; ++t)
            {
                int c = L_cols_h[(size_t)i * k + t];
                if (c >= 0 && c < i)
                {
                    G v = L_vals_h[(size_t)i * k + t];
                    tmp.push_back({c, v});
                }
            }
            std::sort(tmp.begin(), tmp.end(), [](const P &a, const P &b)
                      { return a.c < b.c; });

            for (auto &p : tmp)
            {
                L.col_ind[outp] = p.c;
                L.values[outp] = gpu_to_host_val<T, G>(p.v);
                outp++;
            }

            // diag last
            L.col_ind[outp] = i;
            L.values[outp] = gpu_to_host_val<T, G>(L_vals_h[(size_t)i * k + (k - 1)]);
            outp++;
        }

        return L;
    }

    // ------------------------
    // Top-level wrapper: SAME signature as your current parict<T>(...)
    // ------------------------

    template <class T>
    CSR<T> parict(const CSR<T> &Ahost,
                    const ICTP_Params &row_params,
                    const IC_Attempt_Params &fparams,
                    const core::IC_Symbolic &Sym,
                    ICTP_Factor_Info *info)
    {
        (void)Sym; // API compatibility; ParICT does not need the symbolic pattern

        using G = typename gpu_type<T>::type;
        using M = GMath<G>;

        const int n = Ahost.num_rows;
        if (Ahost.num_cols != n)
        {
            throw std::runtime_error("parict (ParICT): A must be square.");
        }

        const int k = std::max(1, row_params.lfil_per_row); // includes diag (slot k-1)
        const int sweeps = get_num_sweeps(row_params);

        if (k > 128)
        {
            throw std::runtime_error("parict (ParICT): lfil_per_row > 128 not supported by reference kernels.");
        }

        // stage A to device
        int *dA_rp = nullptr, *dA_ci = nullptr;
        G *dA_val = nullptr;

        cuda_check(cudaMalloc(&dA_rp, sizeof(int) * (n + 1)), "malloc A row_ptr");
        cuda_check(cudaMalloc(&dA_ci, sizeof(int) * Ahost.nnz), "malloc A col_ind");
        cuda_check(cudaMalloc(&dA_val, sizeof(G) * Ahost.nnz), "malloc A val");

        cuda_check(cudaMemcpy(dA_rp, Ahost.row_ptr.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice), "cpy A rp");
        cuda_check(cudaMemcpy(dA_ci, Ahost.col_ind.data(), sizeof(int) * Ahost.nnz, cudaMemcpyHostToDevice), "cpy A ci");

        // convert A values to device type
        {
            std::vector<G> tmp((size_t)Ahost.nnz);
            for (int i = 0; i < Ahost.nnz; ++i)
                tmp[i] = host_to_gpu_val<G, T>(Ahost.values[i]);
            cuda_check(cudaMemcpy(dA_val, tmp.data(), sizeof(G) * Ahost.nnz, cudaMemcpyHostToDevice), "cpy A val");
        }

        // Initialize L (ELL) on host then upload
        std::vector<int> L_cols_h;
        std::vector<G> L_vals_h;
        init_L_from_A_lower<T, G>(Ahost, k, L_cols_h, L_vals_h);

        int *dL_cols0 = nullptr, *dL_cols1 = nullptr;
        G *dL_vals0 = nullptr, *dL_vals1 = nullptr, *dL_vals2 = nullptr;

        const size_t Lsz_i = (size_t)n * k;
        cuda_check(cudaMalloc(&dL_cols0, sizeof(int) * Lsz_i), "malloc L cols0");
        cuda_check(cudaMalloc(&dL_cols1, sizeof(int) * Lsz_i), "malloc L cols1");
        cuda_check(cudaMalloc(&dL_vals0, sizeof(G) * Lsz_i), "malloc L vals0");
        cuda_check(cudaMalloc(&dL_vals1, sizeof(G) * Lsz_i), "malloc L vals1");
        cuda_check(cudaMalloc(&dL_vals2, sizeof(G) * Lsz_i), "malloc L vals2");

        cuda_check(cudaMemcpy(dL_cols0, L_cols_h.data(), sizeof(int) * Lsz_i, cudaMemcpyHostToDevice), "cpy L cols0");
        cuda_check(cudaMemcpy(dL_vals0, L_vals_h.data(), sizeof(G) * Lsz_i, cudaMemcpyHostToDevice), "cpy L vals0");

        // CSC buffers
        int *d_col_counts = nullptr;
        int *d_col_ptr = nullptr;
        int *d_col_next = nullptr;
        int *d_csc_row = nullptr;
        G *d_csc_val = nullptr;

        cuda_check(cudaMalloc(&d_col_counts, sizeof(int) * (n + 1)), "malloc col_counts");
        cuda_check(cudaMalloc(&d_col_ptr, sizeof(int) * (n + 1)), "malloc col_ptr");
        cuda_check(cudaMalloc(&d_col_next, sizeof(int) * (n + 1)), "malloc col_next");
        cuda_check(cudaMalloc(&d_csc_row, sizeof(int) * Lsz_i), "malloc csc_row");
        cuda_check(cudaMalloc(&d_csc_val, sizeof(G) * Lsz_i), "malloc csc_val");

        // failure row
        int *d_fail = nullptr;
        cuda_check(cudaMalloc(&d_fail, sizeof(int)), "malloc fail");

        // constants
        G drop_tol_g = host_to_gpu_val<G, T>(static_cast<T>(row_params.drop_tol));

        G pivot_tol_g = host_to_gpu_val<G, T>(static_cast<T>(fparams.pivot_tol));

        // iteration
        int *dL_cols = dL_cols0;
        int *dL_cols_next = dL_cols1;
        G *dL_vals = dL_vals0;
        G *dL_vals_pat = dL_vals1;
        G *dL_vals_new = dL_vals2;

        for (int it = 0; it < sweeps; ++it)
        {
            // Build CSC of current L
            cuda_check(cudaMemset(d_col_counts, 0, sizeof(int) * (n + 1)), "memset col_counts");
            {
                int threads = 256;
                int blocks = (int)((Lsz_i + threads - 1) / threads);
                kernel_count_csc<<<blocks, threads>>>(n, k, dL_cols, d_col_counts);
                cuda_check(cudaGetLastError(), "kernel_count_csc launch");
            }
            // exclusive scan counts -> col_ptr
            {
                thrust::device_ptr<int> counts(d_col_counts);
                thrust::device_ptr<int> ptr(d_col_ptr);
                thrust::exclusive_scan(counts, counts + (n + 1), ptr);
            }
            cuda_check(cudaMemset(d_col_next, 0, sizeof(int) * (n + 1)), "init col_next");
            {
                int threads = 256;
                int blocks = (int)((Lsz_i + threads - 1) / threads);
                kernel_fill_csc<G><<<blocks, threads>>>(n, k, dL_cols, dL_vals, d_col_ptr, d_col_next, d_csc_row, d_csc_val);
                cuda_check(cudaGetLastError(), "kernel_fill_csc launch");
            }

            // Pattern update
            {
                cuda_check(cudaMemset(d_fail, 0xFF, sizeof(int)), "init fail (-1)");
                // One block per row; simple and deterministic.
                // Hash size H: keep it proportional to k.
                // We compile a few common H; pick at runtime.
                int threads = 128;
                if (k <= 16)
                {
                    kernel_parict_pattern_update<G, 128><<<n, threads>>>(
                        n, k, dA_rp, dA_ci, dA_val,
                        dL_cols, dL_vals,
                        d_col_ptr, d_csc_row, d_csc_val,
                        drop_tol_g,
                        dL_cols_next, dL_vals_pat);
                }
                else
                {
                    kernel_parict_pattern_update<G, 256><<<n, threads>>>(
                        n, k, dA_rp, dA_ci, dA_val,
                        dL_cols, dL_vals,
                        d_col_ptr, d_csc_row, d_csc_val,
                        drop_tol_g,
                        dL_cols_next, dL_vals_pat);
                }
                cuda_check(cudaGetLastError(), "kernel_parict_pattern_update launch");
            }

            // Fixed-point sweep (Jacobi) on new pattern; input values are dL_vals_pat
            {
                cuda_check(cudaMemset(d_fail, 0xFF, sizeof(int)), "init fail (-1)");
                int threads = 128;
                int blocks = (n + threads - 1) / threads;
                kernel_parict_fp_sweep<G><<<blocks, threads>>>(
                    n, k, dA_rp, dA_ci, dA_val,
                    dL_cols_next, dL_vals_pat,
                    pivot_tol_g,
                    dL_vals_new,
                    d_fail);
                cuda_check(cudaGetLastError(), "kernel_parict_fp_sweep launch");

                int hfail = -1;
                cuda_check(cudaMemcpy(&hfail, d_fail, sizeof(int), cudaMemcpyDeviceToHost), "read fail");
                if (hfail >= 0)
                {
                    if (info)
                    {
                        info->code = IC_Breakdown::B1_SmallOrNegativePivot;
                        info->step = hfail;
                    }
                    // cleanup
                    cudaFree(dA_rp);
                    cudaFree(dA_ci);
                    cudaFree(dA_val);
                    cudaFree(dL_cols0);
                    cudaFree(dL_cols1);
                    cudaFree(dL_vals0);
                    cudaFree(dL_vals1);
                    cudaFree(dL_vals2);
                    cudaFree(d_col_counts);
                    cudaFree(d_col_ptr);
                    cudaFree(d_col_next);
                    cudaFree(d_csc_row);
                    cudaFree(d_csc_val);
                    cudaFree(d_fail);
                    return CSR<T>{};
                }

                // Swap: L <- (cols_next, vals1)
                std::swap(dL_cols, dL_cols_next);
                // dL_vals becomes vals1; reuse dL_vals_tmp as scratch next time
                std::swap(dL_vals, dL_vals_new);
                // keep dL_vals_tmp as temporary values buffer
            }
        }

        // Download final L (ELL)
        L_cols_h.resize(Lsz_i);
        L_vals_h.resize(Lsz_i);
        cuda_check(cudaMemcpy(L_cols_h.data(), dL_cols, sizeof(int) * Lsz_i, cudaMemcpyDeviceToHost), "download L cols");
        cuda_check(cudaMemcpy(L_vals_h.data(), dL_vals, sizeof(G) * Lsz_i, cudaMemcpyDeviceToHost), "download L vals");

        CSR<T> L = compress_L_ell_to_csr<T, G>(n, k, L_cols_h, L_vals_h);

        if (info)
        {
            info->code = IC_Breakdown::None;
            info->step = 0;
        }

        // cleanup
        cudaFree(dA_rp);
        cudaFree(dA_ci);
        cudaFree(dA_val);
        cudaFree(dL_cols0);
        cudaFree(dL_cols1);
        cudaFree(dL_vals0);
        cudaFree(dL_vals1);
        cudaFree(dL_vals2);
        cudaFree(d_col_counts);
        cudaFree(d_col_ptr);
        cudaFree(d_col_next);
        cudaFree(d_csc_row);
        cudaFree(d_csc_val);
        cudaFree(d_fail);

        return L;
    }

    // Explicit instantiations (match your existing ones)
    template CSR<double> parict(const CSR<double> &Ahost,
                                  const ICTP_Params &row_params,
                                  const IC_Attempt_Params &fparams,
                                  const core::IC_Symbolic &Sym,
                                  ICTP_Factor_Info *info);

    template CSR<float> parict(const CSR<float> &Ahost,
                                 const ICTP_Params &row_params,
                                 const IC_Attempt_Params &fparams,
                                 const core::IC_Symbolic &Sym,
                                 ICTP_Factor_Info *info);

    template CSR<half_float::half> parict(const CSR<half_float::half> &Ahost,
                                            const ICTP_Params &row_params,
                                            const IC_Attempt_Params &fparams,
                                            const core::IC_Symbolic &Sym,
                                            ICTP_Factor_Info *info);

} // namespace ichol
