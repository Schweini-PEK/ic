// ictp_rowwise_v3.cu
//
// Row-wise bounded ICTP-like kernel with ILUTP-style philosophy:
//   - Form a tentative row using all A(i,j), j<i (bounded by MAX_CAND).
//   - Compute l_ij in increasing j order WITHOUT truncating workspace.
//   - After numeric row is formed, drop to (cap-1) by |l_ij|.
//   - Pivot check with pivotTol.
//   - Fixed-cap row storage on device.
//
// This avoids the IC(0)-like behavior from "top-|aij| seeds only"
// and avoids the correctness bug of dynamic top-k during recurrence.

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

// Row-wise kernel: single thread computes one row i
template <typename T, int MAX_CAP, int MAX_CAND>
__global__ void ictp_row_kernel_v3(
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

    // 1) Collect lower A(i,j) into candidates (bounded by MAX_CAND).
    //    This is intentionally NOT pre-truncated to keep_max.
    int cand_j[MAX_CAND];
    T cand_a[MAX_CAND];
    int candSz = 0;

    T a_ii = T(0);

    for (int p = rowStartA; p < rowEndA; ++p)
    {
        int j = colIndA[p];
        T v = valA[p];

        if (j == i)
            a_ii = v;

        if (j < i)
        {
            if (candSz < MAX_CAND)
            {
                cand_j[candSz] = j;
                cand_a[candSz] = v;
                ++candSz;
            }
            else
            {
                int minpos = 0;
                double minabs = absd(cand_a[0]);
                for (int t = 1; t < candSz; ++t)
                {
                    double a = absd(cand_a[t]);
                    if (a < minabs)
                    {
                        minabs = a;
                        minpos = t;
                    }
                }
                if (absd(v) > minabs)
                {
                    cand_j[minpos] = j;
                    cand_a[minpos] = v;
                }
            }
        }
    }

    sort_by_j(cand_j, cand_a, candSz);

    // 2) Workspace for ALL computed l_ij for candidates (no truncation here)
    int wrk_j[MAX_CAND];
    T wrk_l[MAX_CAND];
    int wrkSz = 0;

    for (int s = 0; s < candSz; ++s)
    {
        int j = cand_j[s];
        T aij = cand_a[s];

        T ljj = diagL[j];
        if (ljj == T(0))
            continue;

        T dot = T(0);

        const int jBase = j * cap;
        const int jCount = rowCountL[j];

        // dot = sum_{k<j} l_ik * l_jk
        for (int t = 0; t < wrkSz; ++t)
        {
            int k = wrk_j[t];
            if (k >= j)
                continue;

            T lik = wrk_l[t];

            // Find ljk in row j (bounded scan)
            T ljk = T(0);
            for (int q = 1; q < jCount; ++q)
            {
                int colq = colIndL[jBase + q];
                if (colq == k)
                {
                    ljk = valL[jBase + q];
                    break;
                }
            }

            if (ljk != T(0))
                dot += lik * ljk;
        }

        T lij = (aij - dot) / ljj;

        if (dropTol > T(0) && absd(lij) < (double)dropTol)
            continue;

        if (wrkSz < MAX_CAND)
        {
            wrk_j[wrkSz] = j;
            wrk_l[wrkSz] = lij;
            ++wrkSz;
        }
        else
        {
            // Hard bound: ignore any extra to avoid overflow
            // (rare if MAX_CAND sized sanely).
            break;
        }
    }

    // 3) Apply capacity-based dropping AFTER the row is formed
    int sel_j[MAX_CAP];
    T sel_l[MAX_CAP];
    int selSz = 0;

    for (int t = 0; t < wrkSz; ++t)
    {
        topk_insert_by_abs<T, MAX_CAP>(sel_j, sel_l, selSz, keep_max, wrk_j[t], wrk_l[t]);
    }

    // 4) Pivot using kept entries
    T sumsq = T(0);
    for (int t = 0; t < selSz; ++t)
        sumsq += sel_l[t] * sel_l[t];

    T pivot = a_ii - sumsq;

    if (pivot <= pivotTol)
    {
        *status = 1;
        *fail_row = i;
        *fail_pivot = pivot;
        return;
    }

    T lii = (T)sqrt((double)pivot);
    diagL[i] = lii;

    // 5) Write row i into fixed-cap storage
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

template <typename T>
static bool ictp_rowwise_gpu_v3(
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

    constexpr int MAX_CAP = 64;
    constexpr int MAX_CAND = 128; // candidate workspace > storage cap

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

    int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
    T *d_valA = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrA, Ahost.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIndA, Ahost.col_ind.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valA, Ahost.values.data(), nnzA * sizeof(T), cudaMemcpyHostToDevice));

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
        ictp_row_kernel_v3<T, MAX_CAP, MAX_CAND>
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

                std::fprintf(stderr, "ICTP breakdown at row %d, pivot=%g, pivotTol=%g\n",
                             fr, (double)fp, (double)pivotTol);

                info->code = IC_Breakdown::B1_SmallOrNegativePivot;
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

        // sort per row
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
    cudaFree(d_rowCountL);
    cudaFree(d_colIndL);
    cudaFree(d_valL);
    cudaFree(d_diagL);
    cudaFree(d_status);
    cudaFree(d_fail_row);
    cudaFree(d_fail_pivot);

    return true;
}

// Public API wrapper
namespace ichol
{
    template <class T>
    CSR<T> ictp(const CSR<T> &Ahost,
                const ICTP_Params &row_params,
                const IC_Attempt_Params &fparams,
                ICTP_Factor_Info *info)
    {
        CSR<T> L;
        L.num_rows = Ahost.num_rows;
        L.num_cols = Ahost.num_cols;
        L.row_ptr.assign(Ahost.num_rows + 1, 0);

        bool ok = ictp_rowwise_gpu_v3<T>(Ahost, row_params, fparams, L, info);
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
                                      ICTP_Factor_Info *);

    template CSR<float> ictp<float>(const CSR<float> &,
                                    const ICTP_Params &,
                                    const IC_Attempt_Params &,
                                    ICTP_Factor_Info *);
}
