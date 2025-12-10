// parict.cu
// CUDA-oriented ParICT implementation skeleton faithful to Algorithm 2 (ParILUT outline)
// adapted to SPD case (ParICT computes only L such that A ≈ L * L^T).
// Based on ParILUT paper description of ParICT and Algorithm 2. :contentReference[oaicite:0]{index=0}

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <ichol/parict.hpp>

// ------------------------------
// Error handling
// ------------------------------
#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                     \
            std::abort();                                        \
        }                                                        \
    } while (0)

#define CUSPARSE_CHECK(call)                                      \
    do                                                            \
    {                                                             \
        cusparseStatus_t st = call;                               \
        if (st != CUSPARSE_STATUS_SUCCESS)                        \
        {                                                         \
            printf("cuSPARSE error %s:%d\n", __FILE__, __LINE__); \
            std::abort();                                         \
        }                                                         \
    } while (0)

// ------------------------------
// Basic CSR container
// ------------------------------
struct CsrDevice
{
    int num_rows = 0;
    int nnz = 0;
    int *row_ptr = nullptr;
    int *col_ind = nullptr;
    double *values = nullptr; // lower-triangular entries for L and full for A (assumed sorted per row)
};

// Candidate location for lower triangle (i >= j)
struct CandidateIJ
{
    int i, j;
};

// ------------------------------
// Device helpers
// ------------------------------
__device__ __forceinline__ double csr_get_value_sorted_row(const int *row_ptr, const int *col_ind, const double *values,
                                                           int row, int col)
{
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    // binary search in sorted colind
    int lo = start, hi = end - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        int c = col_ind[mid];
        if (c == col)
            return values[mid];
        if (c < col)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return 0.0;
}

// Dot product of two L rows restricted to columns < limit_col.
// Assumes L rows sorted by column index and only stores j <= i entries.
__device__ __forceinline__ double dot_L_rows_lt(const int *L_row_ptr, const int *L_col_ind, const double *L_values,
                                                int i, int j, int limit_col)
{
    int pi = L_row_ptr[i], ei = L_row_ptr[i + 1];
    int pj = L_row_ptr[j], ej = L_row_ptr[j + 1];

    double sum = 0.0;

    while (pi < ei && pj < ej)
    {
        int ci = L_col_ind[pi];
        int cj = L_col_ind[pj];

        if (ci >= limit_col)
        { // row i surpassed limit
            if (cj >= limit_col)
                break;
        }
        if (cj >= limit_col)
        { // row j surpassed limit
            if (ci >= limit_col)
                break;
        }

        if (ci == cj)
        {
            if (ci < limit_col)
            {
                sum += L_values[pi] * L_values[pj];
            }
            ++pi;
            ++pj;
        }
        else if (ci < cj)
        {
            ++pi;
        }
        else
        {
            ++pj;
        }
    }
    return sum;
}

// ------------------------------
// Kernels
// ------------------------------

// Compute residuals r_ij = a_ij - (L L^T)_ij for candidate list.
// Algorithm 2 line: "Compute ILU residual at candidate locations" (adapted to IC).
__global__ void compute_candidates_residual_kernel(const int *A_row_ptr, const int *A_col_ind, const double *A_values,
                                                   const int *L_row_ptr, const int *L_col_ind, const double *L_values,
                                                   const CandidateIJ *cand, double *rvals, int ncand)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ncand)
        return;

    int i = cand[tid].i;
    int j = cand[tid].j; // j <= i

    double aij = csr_get_value_sorted_row(A_row_ptr, A_col_ind, A_values, i, j);

    // (L L^T)_ij = sum_k l_ik * l_jk
    // For Cholesky-style update, only k < = min(i,j); here j <= i.
    double lij_dot = dot_L_rows_lt(L_row_ptr, L_col_ind, L_values, i, j, /*limit_col=*/j);

    // Include k=j term if present in both rows? The dot helper uses < j.
    // Cholesky off-diagonal formula uses sum_{k<j}. We'll follow that.
    double llT_ij = lij_dot;

    rvals[tid] = aij - llT_ij;
}

// One synchronous fixed-point sweep for ParICT.
// Algorithm 2 lines: "Do one sweep of the fixed-point ILU algorithm"
// adapted to incomplete Cholesky fixed-point update.
// We update all stored nonzeros in L.
// For i>j:
//   l_ij = (a_ij - sum_{k<j} l_ik * l_jk) / l_jj
// For i==j:
//   l_ii = sqrt(a_ii - sum_{k<i} l_ik^2)
// This matches the paper's note that square roots are used in ParICT. :contentReference[oaicite:1]{index=1}
__global__ void parict_fixed_point_sweep_kernel(const int *A_row_ptr, const int *A_col_ind, const double *A_values,
                                                const int *L_row_ptr, const int *L_col_ind,
                                                const double *L_old, double *L_new,
                                                int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    int start = L_row_ptr[row];
    int end = L_row_ptr[row + 1];

    // Iterate over nonzeros in L row. Each thread handles one row for simplicity.
    for (int p = start; p < end; ++p)
    {
        int col = L_col_ind[p]; // col <= row

        if (col == row)
        {
            // diagonal update
            double aii = csr_get_value_sorted_row(A_row_ptr, A_col_ind, A_values, row, row);

            // sum_{k < i} l_ik^2
            double ss = 0.0;
            int p2 = start;
            while (p2 < end)
            {
                int ck = L_col_ind[p2];
                if (ck >= row)
                    break;
                double vik = L_old[p2];
                ss += vik * vik;
                ++p2;
            }

            double val = aii - ss;
            // Guard against negative due to incompleteness; leave as-is if negative.
            L_new[p] = (val > 0.0) ? sqrt(val) : L_old[p];
        }
        else
        {
            // off-diagonal update
            double aij = csr_get_value_sorted_row(A_row_ptr, A_col_ind, A_values, row, col);

            double sum = dot_L_rows_lt(L_row_ptr, L_col_ind, L_old, row, col, /*limit_col=*/col);

            // divide by l_jj
            double ljj = csr_get_value_sorted_row(L_row_ptr, L_col_ind, L_old, col, col);

            double numer = aij - sum;
            L_new[p] = (ljj != 0.0) ? (numer / ljj) : L_old[p];
        }
    }
}

// Compute absolute values of L (excluding diagonal mask) into array for selection.
__global__ void abs_offdiag_kernel(const int *L_row_ptr, const int *L_col_ind, const double *L_values,
                                   double *out_abs, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    int start = L_row_ptr[row];
    int end = L_row_ptr[row + 1];

    for (int p = start; p < end; ++p)
    {
        int col = L_col_ind[p];
        double v = L_values[p];
        out_abs[p] = (col == row) ? 1e300 : fabs(v); // huge sentinel so diagonal won't be selected
    }
}

// Mark entries to remove based on threshold tau and keep diagonals.
__global__ void mark_remove_kernel(const int *L_row_ptr, const int *L_col_ind, const double *L_values,
                                   uint8_t *remove_flag, double tau, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    int start = L_row_ptr[row];
    int end = L_row_ptr[row + 1];

    for (int p = start; p < end; ++p)
    {
        int col = L_col_ind[p];
        if (col == row)
        {
            remove_flag[p] = 0;
        }
        else
        {
            remove_flag[p] = (fabs(L_values[p]) <= tau) ? 1 : 0;
        }
    }
}

// ------------------------------
// Host-side building blocks
// ------------------------------

// Utility: copy CSR host->device
static CsrDevice csr_to_device(const ichol::CSR<double> &h)
{
    CsrDevice d;
    d.num_rows = h.num_rows;
    d.nnz = h.nnz;

    CUDA_CHECK(cudaMalloc(&d.row_ptr, sizeof(int) * (d.num_rows + 1)));
    CUDA_CHECK(cudaMalloc(&d.col_ind, sizeof(int) * d.nnz));
    CUDA_CHECK(cudaMalloc(&d.values, sizeof(double) * d.nnz));

    CUDA_CHECK(cudaMemcpy(d.row_ptr, h.row_ptr.data(), sizeof(int) * (d.num_rows + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.col_ind, h.col_ind.data(), sizeof(int) * d.nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.values, h.values.data(), sizeof(double) * d.nnz, cudaMemcpyHostToDevice));

    return d;
}

static void csr_free(CsrDevice &d)
{
    if (d.row_ptr)
        CUDA_CHECK(cudaFree(d.row_ptr));
    if (d.col_ind)
        CUDA_CHECK(cudaFree(d.col_ind));
    if (d.values)
        CUDA_CHECK(cudaFree(d.values));
    d = {};
}

// Build initial L from lower triangle of A (level-0 pattern).
// Matches paper's initialization choice for ParILUT/ParICT. :contentReference[oaicite:2]{index=2}
static ichol::CSR<double> init_L_from_A_lower(const ichol::CSR<double> &A)
{
    ichol::CSR<double> L;
    L.num_rows = A.num_rows;
    L.num_cols = A.num_cols;
    L.row_ptr.resize(L.num_rows + 1, 0);

    // Count lower-triangular nnz per row
    for (int i = 0; i < A.num_rows; ++i)
    {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
        {
            int j = A.col_ind[p];
            if (j <= i)
                L.row_ptr[i + 1]++;
        }
        // Ensure diagonal exists
        // If A missing diagonal, this simplistic initializer won't add it.
    }
    // Prefix sum
    for (int i = 0; i < A.num_rows; ++i)
        L.row_ptr[i + 1] += L.row_ptr[i];

    int nnzL = L.row_ptr[L.num_rows];
    L.nnz = nnzL;
    L.col_ind.resize(nnzL);
    L.values.resize(nnzL);

    std::vector<int> offset = L.row_ptr;

    for (int i = 0; i < A.num_rows; ++i)
    {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
        {
            int j = A.col_ind[p];
            if (j <= i)
            {
                int q = offset[i]++;
                L.col_ind[q] = j;
                L.values[q] = A.values[p];
            }
        }
    }

    // Sort each row by col
    for (int i = 0; i < L.num_rows; ++i)
    {
        int s = L.row_ptr[i], e = L.row_ptr[i + 1];
        std::vector<int> idx(e - s);
        for (int k = 0; k < e - s; ++k)
            idx[k] = s + k;
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return L.col_ind[a] < L.col_ind[b]; });

        std::vector<int> cols(e - s);
        std::vector<double> vals(e - s);
        for (int k = 0; k < e - s; ++k)
        {
            cols[k] = L.col_ind[idx[k]];
            vals[k] = L.values[idx[k]];
        }
        for (int k = 0; k < e - s; ++k)
        {
            L.col_ind[s + k] = cols[k];
            L.values[s + k] = vals[k];
        }
    }

    return L;
}

// Naive host-side candidate identification:
// candidates = pattern(A_lower ∪ (L*L^T)_lower) \ pattern(L)
// This matches "Identify candidate locations" for ParICT adaptation. :contentReference[oaicite:3]{index=3}
static std::vector<CandidateIJ> identify_candidates_host(const ichol::CSR<double> &A, const ichol::CSR<double> &L)
{
    const int n = A.num_rows;

    // Build quick row-wise sets for current L pattern
    std::vector<std::vector<int>> Lpat(n);
    for (int i = 0; i < n; ++i)
    {
        for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
        {
            Lpat[i].push_back(L.col_ind[p]);
        }
        std::sort(Lpat[i].begin(), Lpat[i].end());
    }

    // Candidates collected per row with an ordered set-like vector
    std::vector<std::vector<int>> candCols(n);

    // 1) Add A lower triangle locations not in L
    for (int i = 0; i < n; ++i)
    {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
        {
            int j = A.col_ind[p];
            if (j <= i)
            {
                bool inL = std::binary_search(Lpat[i].begin(), Lpat[i].end(), j);
                if (!inL)
                    candCols[i].push_back(j);
            }
        }
    }

    // 2) Add locations from symbolic pattern of L*L^T (very naive)
    // For each row i, for each pair of columns k in row i and column j sharing k.
    // We approximate by: for each i, for each k in row i, for each j where L(j,k) exists,
    // add (i,j) to candidates if j <= i.
    // This is an O(nnz * avg_row) host heuristic.
    std::vector<std::vector<int>> rowsWithK(n);
    for (int i = 0; i < n; ++i)
    {
        for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
        {
            int k = L.col_ind[p];
            rowsWithK[k].push_back(i);
        }
    }
    for (int k = 0; k < n; ++k)
    {
        auto &rw = rowsWithK[k];
        std::sort(rw.begin(), rw.end());
    }

    for (int i = 0; i < n; ++i)
    {
        // gather k list from row i
        for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
        {
            int k = L.col_ind[p];
            // all rows j that also have column k
            for (int j : rowsWithK[k])
            {
                if (j <= i)
                {
                    bool inL = std::binary_search(Lpat[i].begin(), Lpat[i].end(), j);
                    if (!inL)
                        candCols[i].push_back(j);
                }
            }
        }
    }

    // Unique per row
    std::vector<CandidateIJ> candidates;
    for (int i = 0; i < n; ++i)
    {
        auto &v = candCols[i];
        if (v.empty())
            continue;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (int j : v)
        {
            if (j <= i)
                candidates.push_back({i, j});
        }
    }

    return candidates;
}

// Merge candidates into L pattern with initial values l_ij = r_ij (formula (5) choice).
// This implements Algorithm 2 line: "Add mL nonzeros to L" adapted to adding all candidates. :contentReference[oaicite:4]{index=4}
static ichol::CSR<double> add_candidates_to_L_host(const ichol::CSR<double> &L,
                                                   const std::vector<CandidateIJ> &cand,
                                                   const std::vector<double> &rvals_host,
                                                   int &mL_added_out)
{
    const int n = L.num_rows;
    mL_added_out = (int)cand.size();

    // Group candidates by row
    std::vector<std::vector<std::pair<int, double>>> addByRow(n);
    for (size_t t = 0; t < cand.size(); ++t)
    {
        addByRow[cand[t].i].push_back({cand[t].j, rvals_host[t]}); // init value = residual
    }

    ichol::CSR<double> Lnew;
    Lnew.num_rows = n;
    Lnew.row_ptr.resize(n + 1, 0);

    // Count new nnz per row
    for (int i = 0; i < n; ++i)
    {
        int oldCount = L.row_ptr[i + 1] - L.row_ptr[i];
        int addCount = (int)addByRow[i].size();
        Lnew.row_ptr[i + 1] = oldCount + addCount;
    }
    for (int i = 0; i < n; ++i)
        Lnew.row_ptr[i + 1] += Lnew.row_ptr[i];

    int nnzNew = Lnew.row_ptr[n];
    Lnew.col_ind.resize(nnzNew);
    Lnew.values.resize(nnzNew);

    // Fill rows by merging then sorting
    for (int i = 0; i < n; ++i)
    {
        int sNew = Lnew.row_ptr[i];
        int sOld = L.row_ptr[i], eOld = L.row_ptr[i + 1];

        int pos = sNew;
        for (int p = sOld; p < eOld; ++p)
        {
            Lnew.col_ind[pos] = L.col_ind[p];
            Lnew.values[pos] = L.values[p];
            ++pos;
        }
        for (auto &pr : addByRow[i])
        {
            Lnew.col_ind[pos] = pr.first;
            Lnew.values[pos] = pr.second;
            ++pos;
        }

        // sort by col and combine duplicates (keep newest value preference)
        int eNew = Lnew.row_ptr[i + 1];
        std::vector<int> idx(eNew - sNew);
        for (int k = 0; k < eNew - sNew; ++k)
            idx[k] = sNew + k;
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return Lnew.col_ind[a] < Lnew.col_ind[b]; });

        std::vector<int> cols;
        std::vector<double> vals;
        cols.reserve(eNew - sNew);
        vals.reserve(eNew - sNew);

        for (int id : idx)
        {
            if (!cols.empty() && cols.back() == Lnew.col_ind[id])
            {
                vals.back() = Lnew.values[id];
            }
            else
            {
                cols.push_back(Lnew.col_ind[id]);
                vals.push_back(Lnew.values[id]);
            }
        }

        // Write back (may leave leftover; compact later)
        for (size_t k = 0; k < cols.size(); ++k)
        {
            Lnew.col_ind[sNew + (int)k] = cols[k];
            Lnew.values[sNew + (int)k] = vals[k];
        }
        // If duplicates collapsed, we will compact globally next.
    }

    // Global compaction to remove duplicate slack
    ichol::CSR<double> Lcompact;
    Lcompact.num_rows = n;
    Lcompact.row_ptr.resize(n + 1, 0);

    // First count unique per row by scanning sorted row segments
    for (int i = 0; i < n; ++i)
    {
        int s = Lnew.row_ptr[i], e = Lnew.row_ptr[i + 1];
        int count = 0;
        int last = -1;
        for (int p = s; p < e; ++p)
        {
            int c = Lnew.col_ind[p];
            if (c == last)
                continue;
            last = c;
            ++count;
        }
        Lcompact.row_ptr[i + 1] = count;
    }
    for (int i = 0; i < n; ++i)
        Lcompact.row_ptr[i + 1] += Lcompact.row_ptr[i];

    int nnzC = Lcompact.row_ptr[n];
    Lcompact.col_ind.resize(nnzC);
    Lcompact.values.resize(nnzC);

    for (int i = 0; i < n; ++i)
    {
        int sOld = Lnew.row_ptr[i], eOld = Lnew.row_ptr[i + 1];
        int sC = Lcompact.row_ptr[i];
        int posC = sC;
        int last = -1;
        for (int p = sOld; p < eOld; ++p)
        {
            int c = Lnew.col_ind[p];
            if (c == last)
                continue;
            last = c;
            Lcompact.col_ind[posC] = c;
            Lcompact.values[posC] = Lnew.values[p];
            ++posC;
        }
    }

    Lcompact.num_cols = Lcompact.num_rows;
    Lcompact.nnz = Lcompact.row_ptr.back();

    return Lcompact;
}

// Remove approximately mL smallest-magnitude off-diagonal entries from L.
// Implements Algorithm 2 line: "Remove the mL ... smallest magnitude elements from L" adapted to ParICT. :contentReference[oaicite:5]{index=5}
static ichol::CSR<double> remove_smallest_magnitude_L_gpu(const ichol::CSR<double> &Lhost, int mL_to_remove)
{
    if (mL_to_remove <= 0)
        return Lhost;

    // Move to device
    CsrDevice Ld = csr_to_device(Lhost);

    // Build abs array over nnz
    thrust::device_vector<double> absvals(Ld.nnz);
    {
        int threads = 128;
        int blocks = (Ld.num_rows + threads - 1) / threads;
        abs_offdiag_kernel<<<blocks, threads>>>(Ld.row_ptr, Ld.col_ind, Ld.values,
                                                thrust::raw_pointer_cast(absvals.data()),
                                                Ld.num_rows);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy absvals to host to select threshold (simple faithful-but-not-fast approach).
    thrust::host_vector<double> h_abs = absvals;

    // We want threshold tau such that about mL_to_remove off-diagonals are <= tau.
    // Ignore sentinel huge values.
    std::vector<double> filt;
    filt.reserve(h_abs.size());
    for (double v : h_abs)
    {
        if (v < 1e200)
            filt.push_back(v);
    }
    if ((int)filt.size() <= mL_to_remove)
    {
        csr_free(Ld);
        // If asked to remove too many, return diagonal-only-ish fallback is not handled here.
        return Lhost;
    }

    std::nth_element(filt.begin(), filt.begin() + mL_to_remove - 1, filt.end());
    double tau = filt[mL_to_remove - 1];

    // Mark removal flags on device
    thrust::device_vector<uint8_t> remove_flag(Ld.nnz);
    {
        int threads = 128;
        int blocks = (Ld.num_rows + threads - 1) / threads;
        mark_remove_kernel<<<blocks, threads>>>(Ld.row_ptr, Ld.col_ind, Ld.values,
                                                thrust::raw_pointer_cast(remove_flag.data()),
                                                tau, Ld.num_rows);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Bring flags back to host and rebuild CSR without flagged entries.
    thrust::host_vector<uint8_t> h_flag = remove_flag;

    ichol::CSR<double> Lnew;
    Lnew.num_rows = Lhost.num_rows;
    Lnew.row_ptr.resize(Lnew.num_rows + 1, 0);

    // Count kept per row
    for (int i = 0; i < Lnew.num_rows; ++i)
    {
        int s = Lhost.row_ptr[i], e = Lhost.row_ptr[i + 1];
        int count = 0;
        for (int p = s; p < e; ++p)
        {
            if (h_flag[p] == 0)
                ++count;
        }
        Lnew.row_ptr[i + 1] = count;
    }
    for (int i = 0; i < Lnew.num_rows; ++i)
        Lnew.row_ptr[i + 1] += Lnew.row_ptr[i];

    int nnzN = Lnew.row_ptr[Lnew.num_rows];
    Lnew.col_ind.resize(nnzN);
    Lnew.values.resize(nnzN);

    // Fill
    for (int i = 0; i < Lnew.num_rows; ++i)
    {
        int s = Lhost.row_ptr[i], e = Lhost.row_ptr[i + 1];
        int pos = Lnew.row_ptr[i];
        for (int p = s; p < e; ++p)
        {
            if (h_flag[p] == 0)
            {
                Lnew.col_ind[pos] = Lhost.col_ind[p];
                Lnew.values[pos] = Lhost.values[p];
                ++pos;
            }
        }
    }

    csr_free(Ld);

    Lnew.num_cols = Lnew.num_rows;
    Lnew.nnz = Lnew.row_ptr.back();

    return Lnew;
}

// Perform one ParICT fixed-point sweep on GPU, producing updated Lhost.
static ichol::CSR<double> parict_sweep_gpu(const ichol::CSR<double> &Ahost, const ichol::CSR<double> &Lhost)
{
    CsrDevice Ad = csr_to_device(Ahost);
    CsrDevice Ld = csr_to_device(Lhost);

    // Allocate new L values
    double *Lnew_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&Lnew_vals, sizeof(double) * Ld.nnz));

    int threads = 128;
    int blocks = (Ld.num_rows + threads - 1) / threads;

    parict_fixed_point_sweep_kernel<<<blocks, threads>>>(
        Ad.row_ptr, Ad.col_ind, Ad.values,
        Ld.row_ptr, Ld.col_ind,
        Ld.values, Lnew_vals,
        Ld.num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back updated values
    ichol::CSR<double> Lout = Lhost;
    CUDA_CHECK(cudaMemcpy(Lout.values.data(), Lnew_vals, sizeof(double) * Ld.nnz, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(Lnew_vals));
    csr_free(Ad);
    csr_free(Ld);

    return Lout;
}

// Compute candidate residuals on GPU and return residual values on host plus norm estimate.
static void compute_candidate_residuals_and_norm_gpu(const ichol::CSR<double> &Ahost, const ichol::CSR<double> &Lhost,
                                                     const std::vector<CandidateIJ> &cand,
                                                     std::vector<double> &rvals_host,
                                                     double &rnorm_est_out)
{
    rvals_host.clear();
    rnorm_est_out = 0.0;
    if (cand.empty())
        return;

    // Device matrices
    CsrDevice Ad = csr_to_device(Ahost);
    CsrDevice Ld = csr_to_device(Lhost);

    // Device candidate array
    CandidateIJ *d_cand = nullptr;
    double *d_rvals = nullptr;
    int nc = (int)cand.size();

    CUDA_CHECK(cudaMalloc(&d_cand, sizeof(CandidateIJ) * nc));
    CUDA_CHECK(cudaMalloc(&d_rvals, sizeof(double) * nc));
    CUDA_CHECK(cudaMemcpy(d_cand, cand.data(), sizeof(CandidateIJ) * nc, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (nc + threads - 1) / threads;

    compute_candidates_residual_kernel<<<blocks, threads>>>(
        Ad.row_ptr, Ad.col_ind, Ad.values,
        Ld.row_ptr, Ld.col_ind, Ld.values,
        d_cand, d_rvals, nc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy residuals back
    rvals_host.resize(nc);
    CUDA_CHECK(cudaMemcpy(rvals_host.data(), d_rvals, sizeof(double) * nc, cudaMemcpyDeviceToHost));

    // Estimate residual norm using candidate residuals:
    // Algorithm 2 line: "Estimate ILU residual norm" approximated by ||(R)Sc||_F. :contentReference[oaicite:6]{index=6}
    thrust::device_ptr<double> rp(d_rvals);
    double sumsq = thrust::transform_reduce(
        rp, rp + nc,
        [] __host__ __device__(double x)
        { return x * x; },
        0.0, thrust::plus<double>());
    rnorm_est_out = std::sqrt(sumsq);

    CUDA_CHECK(cudaFree(d_cand));
    CUDA_CHECK(cudaFree(d_rvals));
    csr_free(Ad);
    csr_free(Ld);
}

// ------------------------------
// ParICT driver following Algorithm 2 line-by-line
// ------------------------------

namespace ichol
{
    ichol::CSR<double> parict_factorize(const ichol::CSR<double> &Ahost, const ichol::ParICT_Params &params)
    {
        // Initial L = lower triangular part of A
        ichol::CSR<double> L = init_L_from_A_lower(Ahost);

        for (int step = 0; step < params.max_steps; ++step)
        {

            // ------------------------------------------------------------
            // Algorithm 2 line 1: Identify candidate locations
            // ------------------------------------------------------------
            std::vector<CandidateIJ> candidates = identify_candidates_host(Ahost, L);

            // ------------------------------------------------------------
            // Algorithm 2 line 2: Compute ILU residual at candidate locations
            //  (ParICT: residual of A - L L^T at candidate locations)
            // ------------------------------------------------------------
            std::vector<double> rvals;
            double rnorm_est = 0.0;
            compute_candidate_residuals_and_norm_gpu(Ahost, L, candidates, rvals, rnorm_est);

            // ------------------------------------------------------------
            // Algorithm 2 line 3: Estimate ILU residual norm
            //  (we already computed rnorm_est from candidate residuals)
            // ------------------------------------------------------------
            (void)rnorm_est; // available for convergence logic if desired

            // ------------------------------------------------------------
            // Algorithm 2 line 4: Add mL nonzeros to L
            //  (ParICT adaptation: add all candidates; init value l_ij = r_ij)
            // ------------------------------------------------------------
            int mL_added = 0;
            L = add_candidates_to_L_host(L, candidates, rvals, mL_added);

            // ------------------------------------------------------------
            // Algorithm 2 line 5: Do one sweep of the fixed-point ILU algorithm
            //  (ParICT: one fixed-point incomplete Cholesky-style sweep)
            // ------------------------------------------------------------
            L = parict_sweep_gpu(Ahost, L);

            // ------------------------------------------------------------
            // Algorithm 2 line 6: Remove the mL smallest magnitude elements from L
            //  (keeping diagonal)
            // ------------------------------------------------------------
            L = remove_smallest_magnitude_L_gpu(L, mL_added);

            // ------------------------------------------------------------
            // Algorithm 2 line 7: Do one sweep of the fixed-point ILU algorithm
            //  (ParICT: second sweep after removal)
            // ------------------------------------------------------------
            L = parict_sweep_gpu(Ahost, L);

            // ------------------------------------------------------------
            // Algorithm 2 line 8: until (convergence)
            //  (Here: fixed step count; convergence check can use rnorm_est)
            // ------------------------------------------------------------
        }

        return L;
    }

} // namespace ichol

// Notes on faithfulness to the paper:
//
// - The step structure exactly matches Algorithm 2 ordering:
//   add -> sweep -> remove -> sweep. :contentReference[oaicite:7]{index=7}
// - Candidate definition follows the paper’s union of nz(A) and nz(LU) not in S,
//   adapted to SPD by using L L^T and lower triangle only. :contentReference[oaicite:8]{index=8}
// - New entries are initialized using residual values (paper’s formula (5) choice). :contentReference[oaicite:9]{index=9}
// - Removal keeps the number of nonzeros roughly constant by removing mL entries.
//
// This file is a CUDA-structured reference implementation; performance-critical
// candidate search and pattern operations are shown in a simple host form consistent
// with the algorithmic description.
