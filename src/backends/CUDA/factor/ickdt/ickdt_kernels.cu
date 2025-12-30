#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "backends/CUDA/util/gmath.cuh"
#include "backends/CUDA/util/hash.cuh"
#include "backends/CUDA/util/sort.cuh"
#include "backends/CUDA/util/memory.cuh"

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
            w_val[t] = cuda::GMath<G>::zero();
            lik_val[t] = cuda::GMath<G>::zero();
            keep[t] = 0;
        }
        if (threadIdx.x == 0)
        {
            sh_wii = cuda::GMath<G>::zero();
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
            G diag = cuda::GMath<G>::zero();
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
            if (cuda::GMath<G>::eq0(wk))
            {
                lik_val[t] = cuda::GMath<G>::zero();
                __syncthreads();
                continue;
            }

            int k_diag_pos = rowPtrL[k + 1] - 1;
            G Lkk = valL[k_diag_pos];

            G lik = cuda::GMath<G>::div(wk, Lkk);

            if (cuda::GMath<G>::lt(cuda::GMath<G>::abs(lik), tau_i))
            {
                lik = cuda::GMath<G>::zero();
                lik_val[t] = lik;
                __syncthreads();
                continue;
            }

            lik_val[t] = lik;

            if (threadIdx.x == 0)
                sh_wii = cuda::GMath<G>::sub(sh_wii, cuda::GMath<G>::mul(lik, lik));
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
                if (!cuda::GMath<G>::eq0(Ljk))
                    w_val[slot] = cuda::GMath<G>::sub(w_val[slot], cuda::GMath<G>::mul(lik, Ljk));
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            if (cuda::GMath<G>::le(sh_wii, pivot_tol))
            {
                sh_fail = 1;
                if (atomicCAS(d_status, 0, 1) == 0)
                    *d_fail_row = i;
            }
            else
            {
                valL[diag_pos] = cuda::GMath<G>::sqrt(sh_wii);
            }
        }
        __syncthreads();
        if (sh_fail)
            return;

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            G v = lik_val[t];
            if (cuda::GMath<G>::lt(cuda::GMath<G>::abs(v), tau_i))
                v = cuda::GMath<G>::zero();
            lik_val[t] = v;
        }
        __syncthreads();

        for (int t = threadIdx.x; t < N_level; t += blockDim.x)
        {
            if (t < m_off)
            {
                absbuf[t] = cuda::GMath<G>::abs(lik_val[t]);
                idxbuf[t] = t;
            }
            else
            {
                absbuf[t] = cuda::GMath<G>::zero();
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
            if (s >= 0 && !cuda::GMath<G>::eq0(absbuf[t]))
                keep[s] = 1;
        }
        __syncthreads();

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            if (keep[t] == 0)
                lik_val[t] = cuda::GMath<G>::zero();
        }
        __syncthreads();

        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
            valL[r0 + t] = lik_val[t];
        __syncthreads();
    }

    template __global__ void ictp_level_kernel<double>(
        int lev_begin, int lev_end,
        const int *level_rows,
        const int *rowPtrA, const int *colIndA, const double *valA,
        const int *rowPtrL, const int *colIndL, double *valL,
        const int *colPtrL, const int *colRowL, const int *colCsrPosL,
        int cap,
        double drop_tol,
        double pivot_tol,
        int max_off_level,
        int H_level,
        int N_level,
        int *d_status, int *d_fail_row);
    template __global__ void ictp_level_kernel<float>(
        int lev_begin, int lev_end,
        const int *level_rows,
        const int *rowPtrA, const int *colIndA, const float *valA,
        const int *rowPtrL, const int *colIndL, float *valL,
        const int *colPtrL, const int *colRowL, const int *colCsrPosL,
        int cap,
        float drop_tol,
        float pivot_tol,
        int max_off_level,
        int H_level,
        int N_level,
        int *d_status, int *d_fail_row);
    template __global__ void ictp_level_kernel<__half>(
        int lev_begin, int lev_end,
        const int *level_rows,
        const int *rowPtrA, const int *colIndA, const __half *valA,
        const int *rowPtrL, const int *colIndL, __half *valL,
        const int *colPtrL, const int *colRowL, const int *colCsrPosL,
        int cap,
        __half drop_tol,
        __half pivot_tol,
        int max_off_level,
        int H_level,
        int N_level,
        int *d_status, int *d_fail_row);
} // namespace ichol::cuda