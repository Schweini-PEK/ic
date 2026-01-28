#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "backends/CUDA/util/gmath.cuh"
#include "backends/CUDA/util/hash.cuh"
#include "backends/CUDA/util/sort.cuh"
#include "backends/CUDA/util/memory.cuh"

namespace ichol::cuda
{
    /**
     * A GPU kernel that processes one level of rows for incomplete Cholesky with threshold. Each CUDA block processes one row in the level.
     *
     * Threads are grouped into "blocks". blockIdx.x selects the block id,
     * and threadIdx.x selects the thread id within that block. Threads in a block cooperate via shared memory and __syncthreads().
     * The input matrix should be lower triangular, sorted in each row by column index, and diagonal entries should be at the end of each row.
     *
     * No sanity checks are performed on the input data; it is assumed to be valid.
     */
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
        // Block-uniform early abort:
        // If any previous block reported failure, stop all work quickly.
        __shared__ int abort;
        if (threadIdx.x == 0)
            abort = *d_status;
        __syncthreads();
        if (abort)
            return;

        // Map this block to one row in the current level.
        int b = blockIdx.x + lev_begin;
        if (b >= lev_end)
            return;
        int i = level_rows[b];

        // Row i in L (CSR-like): [r0, r1) contains all entries of row i.
        // The last entry is the diagonal (by convention in this data structure).
        int r0 = rowPtrL[i];
        int r1 = rowPtrL[i + 1];
        int m = r1 - r0;
        int m_off = m - 1;
        int diag_pos = r1 - 1;

        // Safety check: if this row has too many off-diagonal entries for the
        // allocated shared-memory buffers, fail this row and abort the factorization.
        if (m_off > max_off_level)
        {
            if (threadIdx.x == 0)
            {
                if (atomicCAS(d_status, 0, 1) == 0)
                    *d_fail_row = i;
            }
            return;
        }

        // Shared scalars used by the block:
        // - sh_wii: diagonal work value for row i
        // - sh_fail: whether this row failed the pivot test
        __shared__ G sh_wii;
        __shared__ int sh_fail;

        // Dynamic shared memory layout. We pack several arrays into one buffer:
        // - hash_keys/hash_vals: hash table from column index -> slot
        // - w_val: working values for the row (size max_off_level)
        // - lik_val: L(i,k) values after scaling (size max_off_level)
        // - keep: flags indicating which entries survive dropping
        // - absbuf/idxbuf: temporary buffers for sorting by magnitude
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

        // Initialize the hash table (empty).
        hash_init(hash_keys, hash_vals, H_level);
        // Initialize per-entry buffers in parallel.
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

        // Build a hash table from column index -> position t within this row.
        // This lets us update row entries quickly by column index.
        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            int col = colIndL[r0 + t];
            hash_insert(hash_keys, hash_vals, H_level, col, t);
        }
        __syncthreads();

        // Drop tolerance for this row.
        G tau_i = drop_tol;

        int a0 = rowPtrA[i];
        int a1 = rowPtrA[i + 1];

        // diag at last position
        if (threadIdx.x == 0)
        {
            sh_wii = valA[a1 - 1];
        }
        __syncthreads();

        // all off-diagonals are in [a0, a1-1)
        for (int p = a0 + threadIdx.x; p < a1 - 1; p += blockDim.x)
        {
            int c = colIndA[p]; // guaranteed c < i (given your invariant)
            int slot = hash_find(hash_keys, hash_vals, H_level, c);
            if (slot >= 0)
                w_val[slot] = valA[p];
        }
        __syncthreads();

        // Main elimination loop over off-diagonal entries (k < i).
        // Each iteration updates the work row using previously computed L(:,k).
        __shared__ int sh_k;
        __shared__ G sh_lik;
        __shared__ int sh_active;

        for (int t = 0; t < m_off; ++t)
        {
            if (threadIdx.x == 0)
            {
                int k = colIndL[r0 + t];
                sh_k = k;

                G wk = w_val[t];
                if (cuda::GMath<G>::eq0(wk))
                {
                    sh_active = 0;
                    sh_lik = cuda::GMath<G>::zero();
                }
                else
                {
                    int k_diag_pos = rowPtrL[k + 1] - 1;
                    G Lkk = valL[k_diag_pos];
                    G lik = cuda::GMath<G>::div(wk, Lkk);

                    if (cuda::GMath<G>::lt(cuda::GMath<G>::abs(lik), tau_i))
                    {
                        sh_active = 0;
                        sh_lik = cuda::GMath<G>::zero();
                    }
                    else
                    {
                        sh_active = 1;
                        sh_lik = lik;
                        sh_wii = cuda::GMath<G>::sub(sh_wii, cuda::GMath<G>::mul(lik, lik));
                    }
                }
            }
            __syncthreads();

            if (sh_active)
            {
                int k = sh_k;
                G lik = sh_lik;

                int c0 = colPtrL[k];
                int c1 = colPtrL[k + 1];
                for (int p = c0 + threadIdx.x; p < c1; p += blockDim.x)
                {
                    int j = colRowL[p];
                    if (j <= k || j >= i)
                        continue;

                    int slot = hash_find(hash_keys, hash_vals, H_level, j);
                    if (slot < 0)
                        continue;

                    G Ljk = valL[colCsrPosL[p]];
                    if (!cuda::GMath<G>::eq0(Ljk))
                        w_val[slot] = cuda::GMath<G>::sub(w_val[slot], cuda::GMath<G>::mul(lik, Ljk));
                }
            }
            __syncthreads();

            if (threadIdx.x == 0)
                lik_val[t] = sh_active ? sh_lik : cuda::GMath<G>::zero();
            __syncthreads();
        }

        // Finalize the diagonal. If it's too small, fail the factorization.
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

        // Apply drop tolerance again to the computed L(i,k) values.
        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            G v = lik_val[t];
            if (cuda::GMath<G>::lt(cuda::GMath<G>::abs(v), tau_i))
                v = cuda::GMath<G>::zero();
            lik_val[t] = v;
        }
        __syncthreads();

        // Prepare buffers for selecting the largest |L(i,k)| values.
        // absbuf holds magnitudes, idxbuf holds original indices.
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

        // Sort by descending magnitude (bitonic sort in shared memory).
        bitonic_sort_desc_abs<G>(absbuf, idxbuf, N_level);
        __syncthreads();

        // Keep at most (cap - 1) off-diagonal entries (cap includes the diagonal).
        int pkeep = cap - 1;
        if (pkeep > m_off)
            pkeep = m_off;

        // Mark the strongest entries to keep.
        for (int t = threadIdx.x; t < pkeep; t += blockDim.x)
        {
            int s = idxbuf[t];
            if (s >= 0 && !cuda::GMath<G>::eq0(absbuf[t]))
                keep[s] = 1;
        }
        __syncthreads();

        // Zero out entries that were not selected.
        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
        {
            if (keep[t] == 0)
                lik_val[t] = cuda::GMath<G>::zero();
        }
        __syncthreads();

        // Write final L(i,k) values back to global memory.
        for (int t = threadIdx.x; t < m_off; t += blockDim.x)
            valL[r0 + t] = lik_val[t];
        __syncthreads();
    }

    // Explicit instantiations for supported numeric types.
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
