#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "ichol/ictp.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/fact.hpp"

#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA error in " << __FILE__ << ':' << __LINE__ \
                      << " " << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

template <typename T>
__host__ __device__ inline void swap_dev(T &a, T &b)
{
    T t = a;
    a = b;
    b = t;
}

template <typename T>
__device__ inline double absd(T x)
{
    return fabs((double)x);
}

template <typename T, int MAXK>
__device__ inline void topk_insert_by_abs(
    int *js, T *vs, int &sz, int limit, int j, T v)
{
    if (limit <= 0)
        return;

    if (sz < limit)
    {
        js[sz] = j;
        vs[sz] = v;
        ++sz;
        return;
    }

    int minpos = 0;
    double minabs = absd(vs[0]);
    for (int t = 1; t < sz; ++t)
    {
        double a = absd(vs[t]);
        if (a < minabs)
        {
            minabs = a;
            minpos = t;
        }
    }

    if (absd(v) > minabs)
    {
        js[minpos] = j;
        vs[minpos] = v;
    }
}

template <typename T>
__device__ inline void sort_by_j(int *js, T *vs, int sz)
{
    for (int a = 0; a < sz; ++a)
    {
        int best = a;
        for (int b = a + 1; b < sz; ++b)
            if (js[b] < js[best])
                best = b;

        if (best != a)
        {
            swap_dev(js[a], js[best]);
            swap_dev(vs[a], vs[best]);
        }
    }
}

// Host-visible symbolic structure
namespace ichol
{
    struct IC_Symbolic
    {
        int n;
        std::vector<int> row_ptr_L; // size n+1
        std::vector<int> col_ind_L; // strictly lower, sorted within each row
    };
} // namespace ichol

// Drop-in replacement idea: no precomputed symbolic candidate list.
// Dynamic fill discovery uses an incremental column-to-rows adjacency list of the *stored* L.
// Storage of L stays fixed-cap per row (cap), same as your current v4 code.
//
// You must allocate/initialize on device once (outside the row loop):
//   colHead[n] = -1
//   nodeCounter = 0
//   colRow[maxNodes], colNext[maxNodes]
// where maxNodes >= n*(cap-1) for fixed-cap L.
//
// Each time a row i is finalized, this kernel appends i into the column list of each kept (i,k).

template <typename T, int MAX_WORK>
__device__ __forceinline__ void w_add_bounded(
    int i, // current row (to ignore j>=i)
    int *w_col, T *w_val, int &wSz,
    int col, T delta,
    T dropTol)
{
    if (col < 0 || col >= i)
        return;
    if (delta == T(0))
        return;

    // accumulate if present
    for (int t = 0; t < wSz; ++t)
    {
        if (w_col[t] == col)
        {
            w_val[t] = w_val[t] + delta;
            return;
        }
    }

    // optional early threshold
    if (dropTol > T(0) && absd(delta) < (double)dropTol)
        return;

    // insert if space
    if (wSz < MAX_WORK)
    {
        w_col[wSz] = col;
        w_val[wSz] = delta;
        ++wSz;
        return;
    }

    // bounded growth: replace smallest-magnitude entry if new is larger
    int minpos = 0;
    double minabs = absd(w_val[0]);
    for (int t = 1; t < wSz; ++t)
    {
        double a = absd(w_val[t]);
        if (a < minabs)
        {
            minabs = a;
            minpos = t;
        }
    }

    if (absd(delta) <= minabs)
        return;

    w_col[minpos] = col;
    w_val[minpos] = delta;
}

template <typename T>
__device__ __forceinline__ T get_Ljk_rowwise(
    int j, int k, int cap,
    const int *__restrict__ rowCountL,
    const int *__restrict__ colIndL,
    const T *__restrict__ valL)
{
    const int jBase = j * cap;
    const int jCount = rowCountL[j];
    // off-diagonals are stored sorted by column (you already sort before storing)
    for (int q = 1; q < jCount; ++q)
    {
        int colq = colIndL[jBase + q];
        if (colq == k)
            return valL[jBase + q];
        if (colq > k)
            break;
    }
    return T(0);
}

template <typename T, int MAX_CAP, int MAX_WORK>
__global__ void ictp_row_kernel_dynamic(
    int n,
    const int *__restrict__ rowPtrA,
    const int *__restrict__ colIndA,
    const T *__restrict__ valA,
    int cap,
    T dropTol,
    T pivotTol,
    int *__restrict__ rowCountL,
    int *__restrict__ colIndL,
    T *__restrict__ valL,
    T *__restrict__ diagL,
    // column adjacency for fill discovery (incrementally built)
    int *__restrict__ colHead,     // size n, init -1
    int *__restrict__ colNext,     // size maxNodes
    int *__restrict__ colRow,      // size maxNodes
    int *__restrict__ nodeCounter, // single int, init 0
    int maxNodes,
    int i,
    int *__restrict__ status,
    int *__restrict__ fail_row,
    T *__restrict__ fail_pivot)
{
    if (threadIdx.x != 0)
        return;
    if (cap < 1 || cap > MAX_CAP)
        return;

    const int keep_max = cap - 1;

    const int rowStartA = rowPtrA[i];
    const int rowEndA = rowPtrA[i + 1];

    // robust diagonal extract (do not assume "last entry is diagonal")
    T a_ii = T(0);
    if (rowEndA > rowStartA)
    {
        int last = rowEndA - 1;
        if (colIndA[last] == i)
            a_ii = valA[last];
        else
        {
            for (int p = rowStartA; p < rowEndA; ++p)
                if (colIndA[p] == i)
                {
                    a_ii = valA[p];
                    break;
                }
        }
    }

    // workspace w over columns < i (dynamic, bounded)
    int w_col[MAX_WORK];
    T w_val[MAX_WORK];
    int wSz = 0;

    // seed w from A(i, 0:i-1)
    for (int p = rowStartA; p < rowEndA; ++p)
    {
        int j = colIndA[p];
        if (j < 0)
            continue;
        if (j >= i)
            continue; // lower+diag input expected
        w_add_bounded<T, MAX_WORK>(i, w_col, w_val, wSz, j, valA[p], dropTol);
    }

    // diagonal accumulator
    T w_ii = a_ii;

    // selected multipliers (final lfil)
    int sel_j[MAX_CAP];
    T sel_l[MAX_CAP];
    int selSz = 0;

    // elimination: repeatedly pick smallest column k present in w
    while (true)
    {
        int idxMin = -1;
        int kMin = n;

        for (int t = 0; t < wSz; ++t)
        {
            int k = w_col[t];
            T v = w_val[t];
            if (v == T(0))
                continue;
            if (k < kMin)
            {
                kMin = k;
                idxMin = t;
            }
        }

        if (idxMin < 0)
            break; // no more nonzeros in workspace

        const int k = kMin;
        const T wk = w_val[idxMin];

        // remove entry k from workspace (swap with last)
        --wSz;
        w_col[idxMin] = w_col[wSz];
        w_val[idxMin] = w_val[wSz];

        if (wk == T(0))
            continue;

        T Lkk = diagL[k];
        if (Lkk == T(0))
            continue; // earlier breakdown would have set status

        T lik = wk / Lkk;

        // drop multiplier
        if (dropTol > T(0) && absd(lik) < (double)dropTol)
            continue;

        // diagonal update
        w_ii = w_ii - lik * lik;

        // sparse update: for all j in column k of stored L (rows j>k), update w(j)
        for (int node = colHead[k]; node != -1; node = colNext[node])
        {
            int j = colRow[node];
            if (j <= k)
                continue;
            if (j >= i)
                continue; // only need j<i for this row i

            T Ljk = get_Ljk_rowwise<T>(j, k, cap, rowCountL, colIndL, valL);
            if (Ljk != T(0))
            {
                w_add_bounded<T, MAX_WORK>(i, w_col, w_val, wSz, j, -lik * Ljk, dropTol);
            }
        }

        // keep for final row (top keep_max by |lik|)
        topk_insert_by_abs<T, MAX_CAP>(sel_j, sel_l, selSz, keep_max, k, lik);
    }

    // pivot check
    T pivot = w_ii;
    if (pivot <= pivotTol)
    {
        *status = 1;
        *fail_row = i;
        *fail_pivot = pivot;
        return;
    }

    // store diagonal
    T lii = (T)sqrt((double)pivot);
    diagL[i] = lii;

    // write row i into fixed-cap row storage: [diag | selected off-diagonals]
    const int iBase = i * cap;

    colIndL[iBase + 0] = i;
    valL[iBase + 0] = lii;

    sort_by_j(sel_j, sel_l, selSz);
    for (int t = 0; t < selSz; ++t)
    {
        colIndL[iBase + 1 + t] = sel_j[t];
        valL[iBase + 1 + t] = sel_l[t];
    }
    rowCountL[i] = 1 + selSz;

    // append (i,k) into column adjacency lists for future rows (dynamic fill discovery)
    for (int t = 0; t < selSz; ++t)
    {
        int k = sel_j[t];
        int node = atomicAdd(nodeCounter, 1);
        if (node >= maxNodes)
        {
            *status = 3; // adjacency pool overflow
            *fail_row = i;
            *fail_pivot = T(0);
            return;
        }
        colRow[node] = i;
        colNext[node] = atomicExch(&colHead[k], node);
    }
}

/**
 * Row-wise kernel for ICT.
 *
 * Using symbolic candidate pattern for that row.
 */
template <typename T, int MAX_CAP, int MAX_CAND>
__global__ void ictp_row_kernel(
    int n,
    const int *__restrict__ rowPtrA,
    const int *__restrict__ colIndA,
    const T *__restrict__ valA,
    const int *__restrict__ rowPtrSymL,
    const int *__restrict__ colIndSymL,
    int cap,
    T dropTol,
    T pivotTol,
    int *__restrict__ rowCountL,
    int *__restrict__ colIndL,
    T *__restrict__ valL,
    T *__restrict__ diagL,
    int i,
    int *__restrict__ status,
    int *__restrict__ fail_row,
    T *__restrict__ fail_pivot)
{
    if (threadIdx.x != 0)
        return;

    if (cap < 1 || cap > MAX_CAP)
        return;

    const int keep_max = cap - 1;

    const int rowStartA = rowPtrA[i];
    const int rowEndA = rowPtrA[i + 1];

    // Extract a_ii
    T a_ii = valA[rowEndA - 1];

    // Symbolic candidate set for row i
    const int rowStartSym = rowPtrSymL[i];
    const int rowEndSym = rowPtrSymL[i + 1];
    const int candSz = rowEndSym - rowStartSym;

    if (candSz > MAX_CAND)
    {
        *status = 2; // symbolic overflow
        *fail_row = i;
        *fail_pivot = T(0);
        return;
    }
    // -------------------------------
    // Working row w for row i.
    // w is a scratch copy of the i-th row values (restricted to symbolic pattern),
    // and is updated in-place during elimination.
    //
    // After elimination:
    //   w(j) for j<i approximates L(i,j) before final dropping/truncation.
    //   w_ii approximates the Schur-complement diagonal used to form L(i,i)=sqrt(w_ii).
    // -------------------------------

    // Step 2 (Saad line 2): w := A(i,:)  (sparse copy on the candidate pattern)
    int w_j[MAX_CAND];
    T w_val[MAX_CAND];
    int wSz = 0;

    int pA = rowStartA;
    for (int idx = rowStartSym; idx < rowEndSym; ++idx)
    {
        int j = colIndSymL[idx];
        if (j > i)
            continue; // Do not consider upper triangular

        while (pA < rowEndA && colIndA[pA] < j)
            ++pA;
        T aij = (pA < rowEndA && colIndA[pA] == j) ? valA[pA] : T(0);

        w_j[wSz] = j;
        w_val[wSz] = aij;
        ++wSz;
    }

    // working diagonal entry w(i) = a_ii, updated during elimination
    T w_ii = a_ii;

    // -------------------------------
    // Step 3 (Saad line 3): for k<i where w(k) != 0 do
    // (restricted to k in the symbolic candidate set)
    // -------------------------------
    for (int pos = 0; pos < wSz; ++pos)
    {
        int k = w_j[pos];
        T wk = w_val[pos];
        if (wk == T(0))
            continue;

        // -------------------------------
        // Step 4 (Saad line 4): scale multiplier by diagonal pivot of row k
        // For IC: lik = w(k) / L(k,k)
        // -------------------------------
        T Lkk = diagL[k];
        if (Lkk == T(0))
            continue;

        T lik = wk / Lkk;

        // -------------------------------
        // Step 5 (Saad line 5): threshold drop the multiplier (ABSOLUTE threshold)
        // If dropped, it should not participate in later updates.
        // -------------------------------
        if (dropTol > T(0) && absd(lik) < (double)dropTol)
        {
            w_val[pos] = T(0);
            continue;
        }

        // keep multiplier in working row (w(k) becomes L(i,k))
        w_val[pos] = lik;

        // -------------------------------
        // Step 7 (Saad line 7): sparse update of remaining part of working row
        // w(j) := w(j) - lik * U(k,j),  j>k
        //
        // For Cholesky: U(k,j) = L(j,k).  Fetch L(j,k) from previously computed row j.
        // -------------------------------
        for (int pos2 = pos + 1; pos2 < wSz; ++pos2)
        {
            int j = w_j[pos2];
            if (j <= k)
                continue;

            // lookup L(j,k) in stored row j
            T Ljk = T(0);
            const int jBase = j * cap;
            const int jCount = rowCountL[j];
            for (int q = 1; q < jCount; ++q)
            { // skip diagonal at q=0
                int colq = colIndL[jBase + q];
                if (colq == k)
                {
                    Ljk = valL[jBase + q];
                    break;
                }
                if (colq > k)
                    break;
            }

            if (Ljk != T(0))
            {
                w_val[pos2] = w_val[pos2] - lik * Ljk;
            }
        }

        // diagonal update corresponds to j=i:
        // w(i) := w(i) - lik * U(k,i) with U(k,i)=L(i,k)=lik => w_ii -= lik^2
        w_ii = w_ii - lik * lik;
    }

    // -------------------------------
    // Pivot check (IC diagonal): pivot = w(i) after elimination
    // -------------------------------
    T pivot = w_ii;
    if (pivot <= pivotTol)
    {
        *status = 1;
        *fail_row = i;
        *fail_pivot = pivot;
        return;
    }

    // store diagonal
    T lii = (T)sqrt((double)pivot);
    diagL[i] = lii;

    // -------------------------------
    // Step 10 (Saad line 10): final dropping on the row (ABS threshold), then lfil
    // Keep only keep_max largest |w(j)| entries in the L-part, diagonal always kept.
    // -------------------------------
    int sel_j[MAX_CAP];
    T sel_l[MAX_CAP];
    int selSz = 0;

    for (int t = 0; t < wSz; ++t)
    {
        T lij = w_val[t];
        if (lij == T(0))
            continue;

        // threshold again (absolute threshold)
        if (dropTol > T(0) && absd(lij) < (double)dropTol)
            continue;

        // lfil: keep top keep_max by abs
        topk_insert_by_abs<T, MAX_CAP>(sel_j, sel_l, selSz, keep_max, w_j[t], lij);
    }

    // -------------------------------
    // Store row i into fixed-cap storage: [diag | selected off-diagonals]
    // -------------------------------
    const int iBase = i * cap;

    colIndL[iBase + 0] = i;
    valL[iBase + 0] = lii;

    sort_by_j(sel_j, sel_l, selSz);
    for (int t = 0; t < selSz; ++t)
    {
        colIndL[iBase + 1 + t] = sel_j[t];
        valL[iBase + 1 + t] = sel_l[t];
    }

    rowCountL[i] = 1 + selSz;
}

// Host utilities (reuse your validate_csr)
template <class T>
static void validate_csr(const ichol::CSR<T> &A)
{
    int n = A.num_rows;
    if (n < 0)
        throw std::runtime_error("n < 0");
    if ((int)A.row_ptr.size() != n + 1)
        throw std::runtime_error("rowPtr size != n+1");
    if (A.col_ind.size() != A.values.size())
        throw std::runtime_error("colInd/val size mismatch");
    if (A.row_ptr.empty() || A.row_ptr[0] != 0)
        throw std::runtime_error("rowPtr[0] != 0");
    for (int i = 0; i < n; ++i)
        if (A.row_ptr[i] > A.row_ptr[i + 1])
            throw std::runtime_error("rowPtr not nondecreasing");
    int nnz = A.row_ptr[n];
    if (nnz < 0 || nnz != (int)A.col_ind.size())
        throw std::runtime_error("rowPtr[n] != colInd.size()");
}

static void validate_symbolic(const ichol::core::IC_Symbolic &S, int n)
{
    if (S.n != n)
        throw std::runtime_error("IC_Symbolic: n mismatch");

    if ((int)S.row_ptr_L.size() != n + 1)
        throw std::runtime_error("IC_Symbolic: row_ptr_L size != n+1");

    if (S.row_ptr_L.empty() || S.row_ptr_L[0] != 0)
        throw std::runtime_error("IC_Symbolic: row_ptr_L[0] != 0");

    if (S.row_ptr_L.back() != (int)S.col_ind_L.size())
        throw std::runtime_error("IC_Symbolic: row_ptr_L[n] != col_ind_L.size()");

    for (int i = 0; i < n; ++i)
    {
        if (S.row_ptr_L[i] > S.row_ptr_L[i + 1])
            throw std::runtime_error("IC_Symbolic: row_ptr_L not nondecreasing");

        for (int p = S.row_ptr_L[i]; p < S.row_ptr_L[i + 1]; ++p)
        {
            int j = S.col_ind_L[p];
            if (j < 0 || j > i)
                throw std::runtime_error("IC_Symbolic: candidate j not in [0, i)");
            if (p > S.row_ptr_L[i] && S.col_ind_L[p - 1] > S.col_ind_L[p])
                throw std::runtime_error("IC_Symbolic: col_ind_L not sorted within row");
        }
    }
}

template <typename T>
static bool ictp_rowwise_gpu(
    const ichol::CSR<T> &Ahost,
    const ichol::core::IC_Symbolic &Sym,
    const ICTP_Params &row_params,
    const IC_Attempt_Params &attempt_params,
    ichol::CSR<T> &Lhost_out,
    ICTP_Factor_Info *info)
{
    const int n = Ahost.num_rows;
    const int cap = row_params.lfil_per_row;

    if (info)
        *info = ICTP_Factor_Info{};

    validate_csr(Ahost);
    validate_symbolic(Sym, n);

    if (cap < 1)
        return false;

    constexpr int MAX_CAP = 1000;
    constexpr int MAX_CAND = 1000; // upper bound on symbolic candidates per row

    if (cap > MAX_CAP)
    {
        if (info)
        {
            info->code = IC_Breakdown::OtherNumericalIssue;
            info->step = 0;
        }
        return false;
    }

    const size_t nnzA = Ahost.col_ind.size();
    const size_t nnzSymL = Sym.col_ind_L.size();

    // Device A
    int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
    T *d_valA = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrA, Ahost.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIndA, Ahost.col_ind.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valA, Ahost.values.data(), nnzA * sizeof(T), cudaMemcpyHostToDevice));

    // Device symbolic L pattern
    int *d_rowPtrSymL = nullptr;
    int *d_colIndSymL = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rowPtrSymL, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndSymL, nnzSymL * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrSymL, Sym.row_ptr_L.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIndSymL, Sym.col_ind_L.data(), nnzSymL * sizeof(int), cudaMemcpyHostToDevice));

    // Device L storage (fixed cap per row)
    const size_t maxNnzL = (size_t)n * (size_t)cap;

    int *d_rowCountL = nullptr;
    int *d_colIndL = nullptr;
    T *d_valL = nullptr;
    T *d_diagL = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rowCountL, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndL, maxNnzL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valL, maxNnzL * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_diagL, n * sizeof(T)));

    CUDA_CHECK(cudaMemset(d_rowCountL, 0, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_diagL, 0, n * sizeof(T)));

    // Status + failure details
    int *d_status = nullptr;
    int *d_fail_row = nullptr;
    T *d_fail_pivot = nullptr;

    CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_row, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_pivot, sizeof(T)));

    CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fail_row, -1, sizeof(int)));

    const T dropTol = (T)row_params.drop_tol;
    const T pivotTol = (T)attempt_params.pivot_tol;

    int host_status = 0;

    for (int i = 0; i < n; ++i)
    {
        ictp_row_kernel<T, MAX_CAP, MAX_CAND>
            <<<1, 1>>>(
                n,
                d_rowPtrA, d_colIndA, d_valA,
                d_rowPtrSymL, d_colIndSymL,
                cap,
                dropTol,
                pivotTol,
                d_rowCountL,
                d_colIndL,
                d_valL,
                d_diagL,
                i,
                d_status,
                d_fail_row,
                d_fail_pivot);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&host_status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_status != 0)
        {
            if (info)
            {
                int fr = -1;
                T fp = T(0);
                CUDA_CHECK(cudaMemcpy(&fr, d_fail_row, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&fp, d_fail_pivot, sizeof(T), cudaMemcpyDeviceToHost));

                std::fprintf(stderr, "ICTP breakdown or symbolic overflow at row %d, pivot=%g, pivotTol=%g\n",
                             fr, (double)fp, (double)pivotTol);

                info->code = (host_status == 1)
                                 ? IC_Breakdown::B1_SmallOrNegativePivot
                                 : IC_Breakdown::OtherNumericalIssue;
                info->step = fr;
                info->pivot_value = (double)fp;
            }
            break;
        }
    }

    if (host_status != 0)
    {
        cudaFree(d_rowPtrA);
        cudaFree(d_colIndA);
        cudaFree(d_valA);
        cudaFree(d_rowPtrSymL);
        cudaFree(d_colIndSymL);
        cudaFree(d_rowCountL);
        cudaFree(d_colIndL);
        cudaFree(d_valL);
        cudaFree(d_diagL);
        cudaFree(d_status);
        cudaFree(d_fail_row);
        cudaFree(d_fail_pivot);
        return false;
    }

    // Copy L back and compact to CSR
    std::vector<int> rowCountL(n);
    std::vector<int> colIndL(maxNnzL);
    std::vector<T> valL(maxNnzL);

    CUDA_CHECK(cudaMemcpy(rowCountL.data(), d_rowCountL, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colIndL.data(), d_colIndL, maxNnzL * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valL.data(), d_valL, maxNnzL * sizeof(T), cudaMemcpyDeviceToHost));

    ichol::CSR<T> L;
    L.num_rows = n;
    L.num_cols = n;
    L.row_ptr.assign(n + 1, 0);

    int total = 0;
    for (int i = 0; i < n; ++i)
    {
        L.row_ptr[i] = total;
        total += rowCountL[i];
    }
    L.row_ptr[n] = total;

    L.col_ind.resize(total);
    L.values.resize(total);

    for (int i = 0; i < n; ++i)
    {
        int base = i * cap;
        int cnt = rowCountL[i];
        int dst = L.row_ptr[i];

        for (int t = 0; t < cnt; ++t)
        {
            L.col_ind[dst + t] = colIndL[base + t];
            L.values[dst + t] = valL[base + t];
        }

        // Ensure CSR rows are sorted by column
        for (int a = 0; a < cnt; ++a)
        {
            int best = a;
            for (int b = a + 1; b < cnt; ++b)
                if (L.col_ind[dst + b] < L.col_ind[dst + best])
                    best = b;

            if (best != a)
            {
                swap_dev(L.col_ind[dst + a], L.col_ind[dst + best]);
                swap_dev(L.values[dst + a], L.values[dst + best]);
            }
        }
    }

    Lhost_out = std::move(L);

    cudaFree(d_rowPtrA);
    cudaFree(d_colIndA);
    cudaFree(d_valA);
    cudaFree(d_rowPtrSymL);
    cudaFree(d_colIndSymL);
    cudaFree(d_rowCountL);
    cudaFree(d_colIndL);
    cudaFree(d_valL);
    cudaFree(d_diagL);
    cudaFree(d_status);
    cudaFree(d_fail_row);
    cudaFree(d_fail_pivot);

    return true;
}

template <typename T>
static bool ictp_rowwise_gpu_dynamic(
    const ichol::CSR<T> &Ahost,
    const ICTP_Params &row_params,
    const IC_Attempt_Params &attempt_params,
    ichol::CSR<T> &Lhost_out,
    ICTP_Factor_Info *info)
{
    const int n = Ahost.num_rows;
    const int cap = row_params.lfil_per_row;

    if (info)
        *info = ICTP_Factor_Info{};

    validate_csr(Ahost);

    if (cap < 1)
        return false;

    constexpr int MAX_CAP = 1000;
    constexpr int MAX_WORK = 2000; // workspace bound inside ictp_row_kernel_dynamic

    if (cap > MAX_CAP)
    {
        if (info)
        {
            info->code = IC_Breakdown::OtherNumericalIssue;
            info->step = 0;
        }
        return false;
    }

    const size_t nnzA = Ahost.col_ind.size();

    // -------------------------
    // Device pointers (init null)
    // -------------------------
    int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
    T *d_valA = nullptr;

    int *d_rowCountL = nullptr;
    int *d_colIndL = nullptr;
    T *d_valL = nullptr;
    T *d_diagL = nullptr;

    // Dynamic fill discovery structures
    int *d_colHead = nullptr;     // size n, init -1
    int *d_colNext = nullptr;     // size maxNodes (optional if maxNodes==0)
    int *d_colRow = nullptr;      // size maxNodes (optional if maxNodes==0)
    int *d_nodeCounter = nullptr; // single int, init 0

    // Status + failure details
    int *d_status = nullptr;
    int *d_fail_row = nullptr;
    T *d_fail_pivot = nullptr;

    auto cleanup = [&]()
    {
        cudaFree(d_rowPtrA);
        cudaFree(d_colIndA);
        cudaFree(d_valA);

        cudaFree(d_rowCountL);
        cudaFree(d_colIndL);
        cudaFree(d_valL);
        cudaFree(d_diagL);

        cudaFree(d_colHead);
        cudaFree(d_colNext);
        cudaFree(d_colRow);
        cudaFree(d_nodeCounter);

        cudaFree(d_status);
        cudaFree(d_fail_row);
        cudaFree(d_fail_pivot);
    };

    // -------------------------
    // Upload A
    // -------------------------
    CUDA_CHECK(cudaMalloc(&d_rowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrA, Ahost.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIndA, Ahost.col_ind.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valA, Ahost.values.data(), nnzA * sizeof(T), cudaMemcpyHostToDevice));

    // -------------------------
    // Fixed-cap storage for L (row-major, cap entries per row)
    // -------------------------
    const size_t maxNnzL = (size_t)n * (size_t)cap;

    CUDA_CHECK(cudaMalloc(&d_rowCountL, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndL, maxNnzL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valL, maxNnzL * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_diagL, n * sizeof(T)));

    CUDA_CHECK(cudaMemset(d_rowCountL, 0, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_diagL, 0, n * sizeof(T)));

    // -------------------------
    // Dynamic fill discovery allocations
    // -------------------------
    CUDA_CHECK(cudaMalloc(&d_colHead, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_colHead, 0xFF, n * sizeof(int))); // -1

    CUDA_CHECK(cudaMalloc(&d_nodeCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nodeCounter, 0, sizeof(int)));

    const int maxNodes = n * (cap - 1); // worst-case adjacency nodes
    if (maxNodes > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_colNext, (size_t)maxNodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colRow, (size_t)maxNodes * sizeof(int)));
    }
    else
    {
        d_colNext = nullptr;
        d_colRow = nullptr;
    }

    // -------------------------
    // Status buffers
    // -------------------------
    CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_row, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_pivot, sizeof(T)));

    CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fail_row, -1, sizeof(int)));

    const T dropTol = (T)row_params.drop_tol;
    const T pivotTol = (T)attempt_params.pivot_tol;

    int host_status = 0;

    for (int i = 0; i < n; ++i)
    {
        ictp_row_kernel_dynamic<T, MAX_CAP, MAX_WORK>
            <<<1, 1>>>(
                n,
                d_rowPtrA, d_colIndA, d_valA,
                cap,
                dropTol,
                pivotTol,
                d_rowCountL,
                d_colIndL,
                d_valL,
                d_diagL,
                d_colHead,
                d_colNext,
                d_colRow,
                d_nodeCounter,
                maxNodes,
                i,
                d_status,
                d_fail_row,
                d_fail_pivot);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&host_status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_status != 0)
        {
            if (info)
            {
                int fr = -1;
                T fp = T(0);
                CUDA_CHECK(cudaMemcpy(&fr, d_fail_row, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&fp, d_fail_pivot, sizeof(T), cudaMemcpyDeviceToHost));

                std::fprintf(stderr,
                             "ICTP failure at row %d, pivot=%g, pivotTol=%g, status=%d\n",
                             fr, (double)fp, (double)pivotTol, host_status);

                if (host_status == 1)
                    info->code = IC_Breakdown::B1_SmallOrNegativePivot;
                else
                    info->code = IC_Breakdown::OtherNumericalIssue; // e.g., adjacency pool overflow

                info->step = fr;
                info->pivot_value = (double)fp;
            }

            cleanup();
            return false;
        }
    }

    // -------------------------
    // Copy fixed-cap L back and compact to CSR
    // -------------------------
    std::vector<int> rowCountL(n);
    std::vector<int> colIndL(maxNnzL);
    std::vector<T> valL(maxNnzL);

    CUDA_CHECK(cudaMemcpy(rowCountL.data(), d_rowCountL, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colIndL.data(), d_colIndL, maxNnzL * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valL.data(), d_valL, maxNnzL * sizeof(T), cudaMemcpyDeviceToHost));

    ichol::CSR<T> L;
    L.num_rows = n;
    L.num_cols = n;
    L.row_ptr.assign(n + 1, 0);

    int total = 0;
    for (int i = 0; i < n; ++i)
    {
        L.row_ptr[i] = total;
        total += rowCountL[i];
    }
    L.row_ptr[n] = total;

    L.col_ind.resize(total);
    L.values.resize(total);

    for (int i = 0; i < n; ++i)
    {
        const int base = i * cap;
        const int cnt = rowCountL[i];
        const int dst = L.row_ptr[i];

        for (int t = 0; t < cnt; ++t)
        {
            L.col_ind[dst + t] = colIndL[base + t];
            L.values[dst + t] = valL[base + t];
        }

        // cnt is small (<=cap), O(cnt^2) selection-sort is fine.
        for (int a = 0; a < cnt; ++a)
        {
            int best = a;
            for (int b = a + 1; b < cnt; ++b)
                if (L.col_ind[dst + b] < L.col_ind[dst + best])
                    best = b;

            if (best != a)
            {
                std::swap(L.col_ind[dst + a], L.col_ind[dst + best]);
                std::swap(L.values[dst + a], L.values[dst + best]);
            }
        }
    }

    Lhost_out = std::move(L);

    cleanup();
    return true;
}

// Public API wrapper
namespace ichol
{
    template <class T>
    CSR<T> ictp(const CSR<T> &Ahost,
                const ICTP_Params &row_params,
                const IC_Attempt_Params &fparams,
                const core::IC_Symbolic &Sym,
                ICTP_Factor_Info *info)
    {
        CSR<T> L;
        L.num_rows = Ahost.num_rows;
        L.num_cols = Ahost.num_cols;
        L.row_ptr.assign(Ahost.num_rows + 1, 0);

        // bool ok = ictp_rowwise_gpu<T>(Ahost, Sym, row_params, fparams, L, info);
        bool ok = ictp_rowwise_gpu_dynamic<T>(Ahost, row_params, fparams, L, info);
        if (!ok)
        {
            if (info && info->code == IC_Breakdown::None)
            {
                info->code = IC_Breakdown::OtherNumericalIssue;
                info->step = -1;
            }
        }
        return L;
    }

    template CSR<double> ictp<double>(const CSR<double> &,
                                      const ICTP_Params &,
                                      const IC_Attempt_Params &,
                                      const core::IC_Symbolic &,
                                      ICTP_Factor_Info *);

    template CSR<float> ictp<float>(const CSR<float> &,
                                    const ICTP_Params &,
                                    const IC_Attempt_Params &,
                                    const core::IC_Symbolic &,
                                    ICTP_Factor_Info *);
} // namespace ichol
