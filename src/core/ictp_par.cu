// ictp_symbolic_parallel.cu
//
// Parallel (level-scheduled) ICTP/IC with Saad-style dual dropping on a FIXED symbolic pattern Sym.
// - Uses CSR pattern for rows (Sym.row_ptr_L / Sym.col_ind_L), diagonal must be LAST entry in each row.
// - Builds a CSC "view" of the same pattern to fetch Ljk in O(1) (no row scans).
// - Factors rows in parallel by symbolic level scheduling.
// - Applies dropping twice (during elimination + end-of-row) and keep-top-(lfil-1) on the L part.
//
// This revision fixes:
//   (1) misaligned shared-memory layout for double (absbuf/idxbuf alignment), by aligning offsets in-kernel
//       AND computing host shmem size with the same padding.
//   (2) launch "invalid argument" due to empty levels (blocks==0) and/or shmem exceeding device limit,
//       by skipping empty levels and checking/opt-in shared mem.
//   (3) non-uniform early-abort that could deadlock at __syncthreads, by making it block-uniform.
//
// Assumptions:
//   - A is stored as lower-triangular + diagonal only.
//   - Sym rows are strictly increasing in off-diagonals, all < i, and diag==i is last.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <type_traits>

#include "ichol/ictp_par.hpp"
#include "ichol/half.hpp"
#include "ichol/fact.hpp"
#include "ichol/cuda_utils.hpp"

namespace ichol
{
    // ------------------------
    // Minimal host<->gpu casts
    // ------------------------
    template <class G>
    static __host__ __forceinline__ G host_cast(double x) { return (G)x; }

    template <>
    __host__ __forceinline__ __half host_cast<__half>(double x) { return __float2half_rn((float)x); }

    template <class G>
    static __host__ __forceinline__ double host_to_double(G x) { return (double)x; }

    template <>
    __host__ __forceinline__ double host_to_double<__half>(__half x) { return (double)__half2float(x); }

    __host__ __device__ __forceinline__ size_t align_up(size_t x, size_t a)
    {
        return (x + a - 1) & ~(a - 1);
    }

    // ------------------------
    // Device math
    // ------------------------
    template <class G>
    struct GMath;

    template <>
    struct GMath<float>
    {
        __device__ __forceinline__ static float zero() { return 0.0f; }
        __device__ __forceinline__ static float add(float a, float b) { return a + b; }
        __device__ __forceinline__ static float sub(float a, float b) { return a - b; }
        __device__ __forceinline__ static float mul(float a, float b) { return a * b; }
        __device__ __forceinline__ static float div(float a, float b) { return a / b; }
        __device__ __forceinline__ static float fma(float a, float b, float c) { return fmaf(a, b, c); }
        __device__ __forceinline__ static float sqrt(float a) { return sqrtf(a); }
        __device__ __forceinline__ static float abs(float a) { return fabsf(a); }
        __device__ __forceinline__ static bool lt(float a, float b) { return a < b; }
        __device__ __forceinline__ static bool gt(float a, float b) { return a > b; }
        __device__ __forceinline__ static bool le(float a, float b) { return a <= b; }
        __device__ __forceinline__ static bool ge(float a, float b) { return a >= b; }
        __device__ __forceinline__ static bool eq0(float a) { return a == 0.0f; }
    };

    template <>
    struct GMath<double>
    {
        __device__ __forceinline__ static double zero() { return 0.0; }
        __device__ __forceinline__ static double add(double a, double b) { return a + b; }
        __device__ __forceinline__ static double sub(double a, double b) { return a - b; }
        __device__ __forceinline__ static double mul(double a, double b) { return a * b; }
        __device__ __forceinline__ static double div(double a, double b) { return a / b; }
        __device__ __forceinline__ static double fma(double a, double b, double c) { return ::fma(a, b, c); }
        __device__ __forceinline__ static double sqrt(double a) { return ::sqrt(a); }
        __device__ __forceinline__ static double abs(double a) { return ::fabs(a); }
        __device__ __forceinline__ static bool lt(double a, double b) { return a < b; }
        __device__ __forceinline__ static bool gt(double a, double b) { return a > b; }
        __device__ __forceinline__ static bool le(double a, double b) { return a <= b; }
        __device__ __forceinline__ static bool ge(double a, double b) { return a >= b; }
        __device__ __forceinline__ static bool eq0(double a) { return a == 0.0; }
    };

    template <>
    struct GMath<__half>
    {
        __device__ __forceinline__ static __half zero() { return __float2half_rn(0.0f); }
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

        __device__ __forceinline__ static bool lt(__half a, __half b) { return __hlt(a, b); }
        __device__ __forceinline__ static bool gt(__half a, __half b) { return __hgt(a, b); }
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
    // host helpers: CSC + levels + Sym validation
    // ------------------------
    static inline void build_csc_from_sym(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        std::vector<int> &col_ptr,
        std::vector<int> &col_row,
        std::vector<int> &col_csr_pos)
    {
        const int nnz = (int)col_ind.size();
        col_ptr.assign(n + 1, 0);

        for (int p = 0; p < nnz; ++p)
        {
            int c = col_ind[p];
            if (c < 0 || c >= n)
                throw std::runtime_error("Sym col index out of range");
            col_ptr[c + 1]++;
        }

        for (int c = 0; c < n; ++c)
            col_ptr[c + 1] += col_ptr[c];

        col_row.assign(nnz, 0);
        col_csr_pos.assign(nnz, 0);

        std::vector<int> next = col_ptr;
        for (int i = 0; i < n; ++i)
        {
            for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
            {
                int c = col_ind[p];
                int dst = next[c]++;
                col_row[dst] = i;
                col_csr_pos[dst] = p; // CSR position of (i,c)
            }
        }
    }

    static inline void build_level_schedule_from_sym(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        std::vector<int> &level_ptr,
        std::vector<int> &level_rows)
    {
        std::vector<int> lvl(n, 0);
        int maxlvl = 0;

        for (int i = 0; i < n; ++i)
        {
            int best = 0;
            for (int p = row_ptr[i]; p < row_ptr[i + 1] - 1; ++p)
            {
                int k = col_ind[p];
                if (k >= 0 && k < i)
                    best = std::max(best, lvl[k]);
            }
            lvl[i] = best + 1;
            maxlvl = std::max(maxlvl, lvl[i]);
        }

        level_ptr.assign(maxlvl + 2, 0); // levels are 1..maxlvl, rows for lev ℓ in [level_ptr[ℓ-1], level_ptr[ℓ])
        for (int i = 0; i < n; ++i)
            level_ptr[lvl[i]]++;
        for (int L = 1; L <= maxlvl + 1; ++L)
            level_ptr[L] += level_ptr[L - 1];

        level_rows.assign(n, 0);
        std::vector<int> next = level_ptr;
        for (int i = 0; i < n; ++i)
        {
            int L = lvl[i];
            level_rows[next[L - 1]++] = i;
        }
    }

    static inline void validate_sym_lower_diag_last(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind)
    {
        if ((int)row_ptr.size() != n + 1)
            throw std::runtime_error("Sym.row_ptr_L size mismatch");
        if (row_ptr.front() != 0)
            throw std::runtime_error("Sym.row_ptr_L[0] must be 0");
        if (row_ptr.back() != (int)col_ind.size())
            throw std::runtime_error("Sym.row_ptr_L end mismatch");

        for (int i = 0; i < n; ++i)
        {
            int r0 = row_ptr[i], r1 = row_ptr[i + 1];
            if (r1 <= r0)
                throw std::runtime_error("Sym row has no diagonal");
            if (col_ind[r1 - 1] != i)
                throw std::runtime_error("Sym diagonal must be last and equal to row index");

            int prev = -1;
            for (int p = r0; p < r1 - 1; ++p)
            {
                int c = col_ind[p];
                if (c < 0 || c >= i)
                    throw std::runtime_error("Sym off-diagonal must satisfy 0 <= col < row");
                if (c <= prev)
                    throw std::runtime_error("Sym off-diagonal must be strictly increasing (sorted+unique)");
                prev = c;
            }
        }
    }

    // ------------------------
    // Device: shared hash (col -> local slot)
    // ------------------------
    __device__ __forceinline__ unsigned hash_u32(unsigned x) { return x * 2654435761u; }

    __device__ __forceinline__ void hash_init(int *keys, int *vals, int H)
    {
        for (int t = threadIdx.x; t < H; t += blockDim.x)
        {
            keys[t] = -1;
            vals[t] = -1;
        }
    }

    __device__ __forceinline__ void hash_insert(int *keys, int *vals, int H, int key, int val)
    {
        unsigned h = hash_u32((unsigned)key) & (unsigned)(H - 1);
        for (int it = 0; it < H; ++it)
        {
            int slot = (int)((h + (unsigned)it) & (unsigned)(H - 1));
            int prev = atomicCAS(&keys[slot], -1, key);
            if (prev == -1 || prev == key)
            {
                vals[slot] = val;
                return;
            }
        }
    }

    __device__ __forceinline__ int hash_find(const int *keys, const int *vals, int H, int key)
    {
        unsigned h = hash_u32((unsigned)key) & (unsigned)(H - 1);
        for (int it = 0; it < H; ++it)
        {
            int slot = (int)((h + (unsigned)it) & (unsigned)(H - 1));
            int k = keys[slot];
            if (k == key)
                return vals[slot];
            if (k == -1)
                return -1;
        }
        return -1;
    }

    // ------------------------
    // Device: bitonic sort (descending by abs)
    // ------------------------
    template <class G>
    __device__ __forceinline__ void bitonic_sort_desc_abs(G *absv, int *idx, int N)
    {
        for (int k = 2; k <= N; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                for (int i = threadIdx.x; i < N; i += blockDim.x)
                {
                    int ixj = i ^ j;
                    if (ixj > i)
                    {
                        bool up = ((i & k) == 0);
                        bool swap_needed = up ? GMath<G>::lt(absv[i], absv[ixj])
                                              : GMath<G>::gt(absv[i], absv[ixj]);
                        if (swap_needed)
                        {
                            G ta = absv[i];
                            absv[i] = absv[ixj];
                            absv[ixj] = ta;
                            int ti = idx[i];
                            idx[i] = idx[ixj];
                            idx[ixj] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // ------------------------
    // Kernel: factor one level (block per row)
    // ------------------------
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
        int *d_status, int *d_fail_row)
    {
        // block-uniform early abort
        __shared__ int abort;
        if (threadIdx.x == 0)
            abort = *d_status;
        __syncthreads();
        if (abort)
            return;

        int b = blockIdx.x + lev_begin;
        if (b >= lev_end)
            return;
        int i = level_rows[b];

        int r0 = rowPtrL[i];
        int r1 = rowPtrL[i + 1];
        int m = r1 - r0;
        int m_off = m - 1;
        int diag_pos = r1 - 1;

        if (m_off > max_off_level)
        {
            if (threadIdx.x == 0)
            {
                if (atomicCAS(d_status, 0, 1) == 0)
                    *d_fail_row = i;
            }
            return;
        }

        __shared__ G sh_wii;
        __shared__ int sh_fail;

        extern __shared__ unsigned char smem[];
        int *hash_keys = (int *)smem;
        int *hash_vals = hash_keys + H_level;

        size_t off = 2ull * H_level * sizeof(int);

        off = align_up(off, alignof(G));
        G *w_val = (G *)(smem + off);
        G *lik_val = w_val + max_off_level;
        off += 2ull * max_off_level * sizeof(G);

        off = align_up(off, alignof(int));
        int *keep = (int *)(smem + off);
        off += 1ull * max_off_level * sizeof(int);

        off = align_up(off, alignof(G));
        G *absbuf = (G *)(smem + off);
        off += 1ull * N_level * sizeof(G);

        off = align_up(off, alignof(int));
        int *idxbuf = (int *)(smem + off);

        hash_init(hash_keys, hash_vals, H_level);
        for (int t = threadIdx.x; t < max_off_level; t += blockDim.x)
        {
            w_val[t] = GMath<G>::zero();
            lik_val[t] = GMath<G>::zero();
            keep[t] = 0;
        }
        if (threadIdx.x == 0)
        {
            sh_wii = GMath<G>::zero();
            sh_fail = 0;
        }
        __syncthreads();

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            int col = colIndL[r0 + t];
            hash_insert(hash_keys, hash_vals, H_level, col, t);
        }
        __syncthreads();

        G tau_i = drop_tol;

        if (threadIdx.x == 0)
        {
            G diag = GMath<G>::zero();
            int a0 = rowPtrA[i], a1 = rowPtrA[i + 1];
            for (int p = a0; p < a1; ++p)
            {
                if (colIndA[p] == i)
                {
                    diag = valA[p];
                    break;
                }
            }
            sh_wii = diag;
        }
        __syncthreads();

        {
            int a0 = rowPtrA[i], a1 = rowPtrA[i + 1];
            for (int p = a0 + threadIdx.x; p < a1; p += blockDim.x)
            {
                int c = colIndA[p];
                if (c < i)
                {
                    int slot = hash_find(hash_keys, hash_vals, H_level, c);
                    if (slot >= 0)
                        w_val[slot] = valA[p];
                }
            }
            __syncthreads();
        }

        for (int t = 0; t < m_off; ++t)
        {
            int k = colIndL[r0 + t];

            G wk = w_val[t];
            if (GMath<G>::eq0(wk))
            {
                lik_val[t] = GMath<G>::zero();
                __syncthreads();
                continue;
            }

            int k_diag_pos = rowPtrL[k + 1] - 1;
            G Lkk = valL[k_diag_pos];

            G lik = GMath<G>::div(wk, Lkk);

            if (GMath<G>::lt(GMath<G>::abs(lik), tau_i))
            {
                lik = GMath<G>::zero();
                lik_val[t] = lik;
                __syncthreads();
                continue;
            }

            lik_val[t] = lik;

            if (threadIdx.x == 0)
                sh_wii = GMath<G>::sub(sh_wii, GMath<G>::mul(lik, lik));
            __syncthreads();

            int c0 = colPtrL[k];
            int c1 = colPtrL[k + 1];
            for (int p = c0 + threadIdx.x; p < c1; p += blockDim.x)
            {
                int j = colRowL[p];
                if (j <= k)
                    continue;
                if (j >= i)
                    continue;

                int slot = hash_find(hash_keys, hash_vals, H_level, j);
                if (slot < 0)
                    continue;

                G Ljk = valL[colCsrPosL[p]];
                if (!GMath<G>::eq0(Ljk))
                    w_val[slot] = GMath<G>::sub(w_val[slot], GMath<G>::mul(lik, Ljk));
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            if (GMath<G>::le(sh_wii, pivot_tol))
            {
                sh_fail = 1;
                if (atomicCAS(d_status, 0, 1) == 0)
                    *d_fail_row = i;
            }
            else
            {
                valL[diag_pos] = GMath<G>::sqrt(sh_wii);
            }
        }
        __syncthreads();
        if (sh_fail)
            return;

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            G v = lik_val[t];
            if (GMath<G>::lt(GMath<G>::abs(v), tau_i))
                v = GMath<G>::zero();
            lik_val[t] = v;
        }
        __syncthreads();

        for (int t = threadIdx.x; t < N_level; t += blockDim.x)
        {
            if (t < m_off)
            {
                absbuf[t] = GMath<G>::abs(lik_val[t]);
                idxbuf[t] = t;
            }
            else
            {
                absbuf[t] = GMath<G>::zero();
                idxbuf[t] = -1;
            }
        }
        __syncthreads();

        bitonic_sort_desc_abs<G>(absbuf, idxbuf, N_level);
        __syncthreads();

        int pkeep = cap - 1;
        if (pkeep > m_off)
            pkeep = m_off;

        for (int t = threadIdx.x; t < pkeep; t += blockDim.x)
        {
            int s = idxbuf[t];
            if (s >= 0 && !GMath<G>::eq0(absbuf[t]))
                keep[s] = 1;
        }
        __syncthreads();

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            if (keep[t] == 0)
                lik_val[t] = GMath<G>::zero();
        }
        __syncthreads();

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
            valL[r0 + t] = lik_val[t];
        __syncthreads();
    }

    // ------------------------
    // Host: compute shmem size with SAME alignment/padding as kernel
    // ------------------------
    template <class G>
    static inline size_t shmem_bytes_for_level(int max_off_level, int H_level, int N_level)
    {
        size_t off = 0;

        // hash_keys + hash_vals
        off += 2ull * (size_t)H_level * sizeof(int);

        // w_val + lik_val
        off = align_up(off, alignof(G));
        off += 2ull * (size_t)max_off_level * sizeof(G);

        // keep
        off = align_up(off, alignof(int));
        off += 1ull * (size_t)max_off_level * sizeof(int);

        // absbuf
        off = align_up(off, alignof(G));
        off += 1ull * (size_t)N_level * sizeof(G);

        // idxbuf
        off = align_up(off, alignof(int));
        off += 1ull * (size_t)N_level * sizeof(int);

        return off;
    }

    template <class T, class G>
    static void ictp_symbolic_parallel_gpu_fixedpattern(
        const CSR<T> &Ahost,
        const ICTP_Params &row_params,
        const IC_Attempt_Params &fparams,
        const core::IC_Symbolic &Sym,
        std::vector<int> &L_row_ptr_out,
        std::vector<int> &L_col_ind_out,
        std::vector<G> &L_val_fixed_out,
        int &fail_row_out)
    {
        const int n = Ahost.num_rows;
        const int cap = row_params.lfil_per_row;
        if (Sym.n != n)
            throw std::runtime_error("Sym.n mismatch");

        validate_sym_lower_diag_last(n, Sym.row_ptr_L, Sym.col_ind_L);

        std::vector<int> col_ptr, col_row, col_csr_pos;
        build_csc_from_sym(n, Sym.row_ptr_L, Sym.col_ind_L, col_ptr, col_row, col_csr_pos);

        std::vector<int> level_ptr, level_rows;
        build_level_schedule_from_sym(n, Sym.row_ptr_L, Sym.col_ind_L, level_ptr, level_rows);

        int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
        G *d_valA = nullptr;

        int *d_rowPtrL = nullptr, *d_colIndL = nullptr;
        G *d_valL = nullptr;

        int *d_colPtrL = nullptr, *d_colRowL = nullptr, *d_colCsrPosL = nullptr;
        int *d_levelRows = nullptr;

        int *d_status = nullptr, *d_fail_row = nullptr;

        const int nnzA = (int)Ahost.col_ind.size();
        const int nnzL = (int)Sym.col_ind_L.size();

        CUDA_CHECK(cudaMalloc(&d_rowPtrA, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(G)));

        CUDA_CHECK(cudaMemcpy(d_rowPtrA, Ahost.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIndA, Ahost.col_ind.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));

        {
            std::vector<G> tmp(nnzA);
            for (int p = 0; p < nnzA; ++p)
                tmp[p] = host_cast<G>((double)Ahost.values[p]);
            CUDA_CHECK(cudaMemcpy(d_valA, tmp.data(), nnzA * sizeof(G), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMalloc(&d_rowPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIndL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valL, nnzL * sizeof(G)));

        CUDA_CHECK(cudaMemcpy(d_rowPtrL, Sym.row_ptr_L.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIndL, Sym.col_ind_L.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_valL, 0, nnzL * sizeof(G)));

        CUDA_CHECK(cudaMalloc(&d_colPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colRowL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colCsrPosL, nnzL * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_colPtrL, col_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colRowL, col_row.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colCsrPosL, col_csr_pos.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_levelRows, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_levelRows, level_rows.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_fail_row, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_fail_row, -1, sizeof(int)));

        int threads = 128;

        fail_row_out = -1;
        int h_status = 0;

        // shared memory capability check
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int maxShmOptin = 0;
        int maxShmDefault = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxShmOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxShmDefault, cudaDevAttrMaxSharedMemoryPerBlock, dev));

        int maxlvl = (int)level_ptr.size() - 2;
        for (int lev = 1; lev <= maxlvl; ++lev)
        {
            int lev_begin = level_ptr[lev - 1];
            int lev_end = level_ptr[lev];
            int nrows = lev_end - lev_begin;
            if (nrows <= 0)
                continue;

            int max_off_level = 0;
            for (int bi = lev_begin; bi < lev_end; ++bi)
            {
                int row = level_rows[bi];
                int r0 = Sym.row_ptr_L[row];
                int r1 = Sym.row_ptr_L[row + 1];
                int m_off = (r1 - r0) - 1;
                max_off_level = std::max(max_off_level, m_off);
            }
            if (max_off_level < 1)
                max_off_level = 1;

            int H_level = 1;
            while (H_level < 4 * max_off_level)
                H_level <<= 1;

            int N_level = 1;
            while (N_level < max_off_level)
                N_level <<= 1;

            size_t shmem = shmem_bytes_for_level<G>(max_off_level, H_level, N_level);

            // if dynamic shmem > default, opt in; if > opt-in limit, fail early (prevents launch invalid argument)
            if ((int)shmem > maxShmOptin)
                throw std::runtime_error("ictp_par: required dynamic shared memory exceeds device limit");

            if ((int)shmem > maxShmDefault)
            {
                CUDA_CHECK(cudaFuncSetAttribute(
                    ictp_level_kernel<G>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)shmem));
            }

            // grid.x = number of rows in this level (block per row)
            int blocks = nrows;

            ictp_level_kernel<G><<<blocks, threads, shmem>>>(
                lev_begin, lev_end,
                d_levelRows,
                d_rowPtrA, d_colIndA, d_valA,
                d_rowPtrL, d_colIndL, d_valL,
                d_colPtrL, d_colRowL, d_colCsrPosL,
                cap,
                host_cast<G>(row_params.drop_tol),
                host_cast<G>(fparams.pivot_tol),
                max_off_level, H_level, N_level,
                d_status, d_fail_row);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_status != 0)
            {
                CUDA_CHECK(cudaMemcpy(&fail_row_out, d_fail_row, sizeof(int), cudaMemcpyDeviceToHost));
                break;
            }
        }

        L_row_ptr_out = Sym.row_ptr_L;
        L_col_ind_out = Sym.col_ind_L;
        L_val_fixed_out.resize(nnzL);
        CUDA_CHECK(cudaMemcpy(L_val_fixed_out.data(), d_valL, nnzL * sizeof(G), cudaMemcpyDeviceToHost));

        cudaFree(d_rowPtrA);
        cudaFree(d_colIndA);
        cudaFree(d_valA);
        cudaFree(d_rowPtrL);
        cudaFree(d_colIndL);
        cudaFree(d_valL);
        cudaFree(d_colPtrL);
        cudaFree(d_colRowL);
        cudaFree(d_colCsrPosL);
        cudaFree(d_levelRows);
        cudaFree(d_status);
        cudaFree(d_fail_row);
    }

    // ------------------------
    // Host: compress fixed-pattern L (remove zeros) into CSR<T>, keep diag last
    // ------------------------
    template <class T, class G>
    static CSR<T> compress_fixed_pattern_L(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<G> &val_fixed)
    {
        CSR<T> L;
        L.num_rows = n;
        L.num_cols = n;
        L.row_ptr.assign(n + 1, 0);

        for (int i = 0; i < n; ++i)
        {
            int r0 = row_ptr[i], r1 = row_ptr[i + 1];
            int diagp = r1 - 1;

            int cnt = 0;
            for (int p = r0; p < diagp; ++p)
            {
                if (host_to_double(val_fixed[p]) != 0.0)
                    cnt++;
            }
            cnt += 1;
            L.row_ptr[i + 1] = L.row_ptr[i] + cnt;
        }

        L.nnz = L.row_ptr[n];
        L.col_ind.resize(L.nnz);
        L.values.resize(L.nnz);

        for (int i = 0; i < n; ++i)
        {
            int r0 = row_ptr[i], r1 = row_ptr[i + 1];
            int diagp = r1 - 1;
            int outp = L.row_ptr[i];

            for (int p = r0; p < diagp; ++p)
            {
                G v = val_fixed[p];
                if (host_to_double(v) != 0.0)
                {
                    L.col_ind[outp] = col_ind[p];
                    L.values[outp] = (T)host_to_double(v);
                    outp++;
                }
            }
            L.col_ind[outp] = i;
            L.values[outp] = (T)host_to_double(val_fixed[diagp]);
        }

        return L;
    }

    // ------------------------
    // Top-level
    // ------------------------
    template <class T>
    CSR<T> ictp_par(const CSR<T> &Ahost,
                    const ICTP_Params &row_params,
                    const IC_Attempt_Params &fparams,
                    const core::IC_Symbolic &Sym,
                    ICTP_Factor_Info *info)
    {
        using G =
            std::conditional_t<std::is_same<T, double>::value, double,
                               std::conditional_t<std::is_same<T, float>::value, float,
                                                  __half>>;

        std::vector<int> L_row_ptr_fixed, L_col_ind_fixed;
        std::vector<G> L_val_fixed;
        int fail_row = -1;

        try
        {
            ictp_symbolic_parallel_gpu_fixedpattern<T, G>(
                Ahost, row_params, fparams, Sym,
                L_row_ptr_fixed, L_col_ind_fixed, L_val_fixed,
                fail_row);

            if (fail_row >= 0)
            {
                if (info)
                {
                    info->code = IC_Breakdown::B1_SmallOrNegativePivot;
                    info->step = fail_row;
                }
                return CSR<T>{};
            }

            CSR<T> Lhost = compress_fixed_pattern_L<T, G>(
                Ahost.num_rows, L_row_ptr_fixed, L_col_ind_fixed, L_val_fixed);

            if (info)
            {
                info->code = IC_Breakdown::None;
                info->step = 0;
            }
            return Lhost;
        }
        catch (...)
        {
            if (info)
            {
                info->code = IC_Breakdown::OtherNumericalIssue;
                info->step = 0;
            }
            throw;
        }
    }

    template CSR<double> ictp_par(const CSR<double> &Ahost,
                                  const ICTP_Params &row_params,
                                  const IC_Attempt_Params &fparams,
                                  const core::IC_Symbolic &Sym,
                                  ICTP_Factor_Info *info);

    template CSR<float> ictp_par(const CSR<float> &Ahost,
                                 const ICTP_Params &row_params,
                                 const IC_Attempt_Params &fparams,
                                 const core::IC_Symbolic &Sym,
                                 ICTP_Factor_Info *info);

    template CSR<half_float::half> ictp_par(const CSR<half_float::half> &Ahost,
                                            const ICTP_Params &row_params,
                                            const IC_Attempt_Params &fparams,
                                            const core::IC_Symbolic &Sym,
                                            ICTP_Factor_Info *info);

} // namespace ichol
