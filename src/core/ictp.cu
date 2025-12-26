#include <cuda_runtime.h>

#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>

#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <iostream>
#include <algorithm>

#include "ichol/ictp.hpp"
#include "ichol/symbolic.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/fact.hpp"
#include "ichol/half.hpp"
#include "ichol/cuda_utils.hpp"

// -------------------------
// GPU scalar type mapping
// -------------------------
template <class T>
struct gpu_type
{
    using type = T;
};

template <>
struct gpu_type<half_float::half>
{
    using type = __half;
};

// -------------------------
// Magnitude type (no double for fp16/fp32)
//   NOTE: for __half, keep magnitude in __half to avoid explicit half<->float conversions.
// -------------------------
template <class G>
struct mag_type
{
    using type = float;
};
template <>
struct mag_type<double>
{
    using type = double;
};
template <>
struct mag_type<__half>
{
    using type = __half;
};
template <class G>
using mag_t = typename mag_type<G>::type;

template <class G>
__host__ __device__ __forceinline__ mag_t<G> to_mag(G x) { return (mag_t<G>)x; }

template <>
__host__ __device__ __forceinline__ __half to_mag<__half>(__half x) { return x; }

// -------------------------
// Half helpers (bit-level abs/zero + explicit half intrinsics)
// -------------------------
__host__ __device__ __forceinline__ __half hzero()
{
    return CUDART_ZERO_FP16;
}

template <class M>
__device__ __forceinline__ M mag_zero()
{
    return (M)0;
}
template <>
__device__ __forceinline__ __half mag_zero<__half>()
{
    return hzero();
}

template <class M>
__device__ __forceinline__ bool mag_lt(M a, M b) { return a < b; }
template <class M>
__device__ __forceinline__ bool mag_le(M a, M b) { return a <= b; }
template <class M>
__device__ __forceinline__ bool mag_gt(M a, M b) { return a > b; }

template <>
__device__ __forceinline__ bool mag_lt<__half>(__half a, __half b) { return __hlt(a, b); }
template <>
__device__ __forceinline__ bool mag_le<__half>(__half a, __half b) { return __hle(a, b); }
template <>
__device__ __forceinline__ bool mag_gt<__half>(__half a, __half b) { return __hgt(a, b); }

template <class G>
__device__ __forceinline__ G g_zero()
{
    return (G)0;
}
template <>
__device__ __forceinline__ __half g_zero<__half>()
{
    return hzero();
}

template <class G>
__device__ __forceinline__ G g_add(G a, G b) { return a + b; }
template <class G>
__device__ __forceinline__ G g_sub(G a, G b) { return a - b; }
template <class G>
__device__ __forceinline__ G g_mul(G a, G b) { return a * b; }
template <class G>
__device__ __forceinline__ G g_div(G a, G b) { return a / b; }
template <class G>
__device__ __forceinline__ G g_neg(G a) { return -a; }

template <>
__device__ __forceinline__ __half g_add<__half>(__half a, __half b) { return __hadd_rn(a, b); }
template <>
__device__ __forceinline__ __half g_sub<__half>(__half a, __half b) { return __hsub_rn(a, b); }
template <>
__device__ __forceinline__ __half g_mul<__half>(__half a, __half b) { return __hmul_rn(a, b); }
template <>
__device__ __forceinline__ __half g_div<__half>(__half a, __half b) { return __hdiv(a, b); }
template <>
__device__ __forceinline__ __half g_neg<__half>(__half a) { return __hneg(a); }

template <class G>
__device__ __forceinline__ mag_t<G> mag_abs(G x);

template <>
__device__ __forceinline__ float mag_abs<float>(float x) { return fabsf(x); }

template <>
__device__ __forceinline__ double mag_abs<double>(double x) { return fabs(x); }

template <>
__device__ __forceinline__ __half mag_abs<__half>(__half x)
{
    unsigned short bits = __half_as_ushort(x);
    bits &= (unsigned short)0x7FFFu;
    return __ushort_as_half(bits);
}

// sqrt in "same working precision"
template <class G>
__device__ __forceinline__ G sqrt_g(G x);

template <>
__device__ __forceinline__ float sqrt_g<float>(float x) { return sqrtf(x); }

template <>
__device__ __forceinline__ double sqrt_g<double>(double x) { return sqrt(x); }

template <>
__device__ __forceinline__ __half sqrt_g<__half>(__half x)
{
    return hsqrt(x);
}

template <class G>
__host__ __device__ __forceinline__ bool is_zero(G x) { return x == G(0); }

template <>
__host__ __device__ __forceinline__ bool is_zero<__half>(__half x)
{
    unsigned short bits = __half_as_ushort(x);
    return (bits & (unsigned short)0x7FFFu) == (unsigned short)0;
}

template <typename U>
__host__ __device__ __forceinline__ void swap_dev(U &a, U &b)
{
    U t = a;
    a = b;
    b = t;
}

template <typename G, int MAXK>
__device__ __forceinline__ void topk_insert_by_abs(
    int *js, G *vs, int &sz, int limit, int j, G v)
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
    mag_t<G> minabs = mag_abs(vs[0]);
    for (int t = 1; t < sz; ++t)
    {
        mag_t<G> a = mag_abs(vs[t]);
        if (mag_lt(a, minabs))
        {
            minabs = a;
            minpos = t;
        }
    }

    if (mag_gt(mag_abs(v), minabs))
    {
        js[minpos] = j;
        vs[minpos] = v;
    }
}

template <typename G>
__device__ __forceinline__ void sort_by_j(int *js, G *vs, int sz)
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

namespace ichol
{
    struct IC_Symbolic
    {
        int n;
        std::vector<int> row_ptr_L;
        std::vector<int> col_ind_L;
    };
}

// -------------------------
// Bounded workspace update
// -------------------------
template <typename G, int MAX_WORK>
__device__ __forceinline__ void w_add_bounded(
    int i,
    int *w_col, G *w_val, int &wSz,
    int col, G delta,
    G dropTol)
{
    if (col < 0 || col >= i)
        return;
    if (is_zero(delta))
        return;

    for (int t = 0; t < wSz; ++t)
    {
        if (w_col[t] == col)
        {
            w_val[t] = g_add<G>(w_val[t], delta);
            return;
        }
    }

    const mag_t<G> dropTolM = to_mag(dropTol);
    if (mag_gt(dropTolM, mag_zero<mag_t<G>>()) && mag_lt(mag_abs(delta), dropTolM))
        return;

    if (wSz < MAX_WORK)
    {
        w_col[wSz] = col;
        w_val[wSz] = delta;
        ++wSz;
        return;
    }

    int minpos = 0;
    mag_t<G> minabs = mag_abs(w_val[0]);
    for (int t = 1; t < wSz; ++t)
    {
        mag_t<G> a = mag_abs(w_val[t]);
        if (mag_lt(a, minabs))
        {
            minabs = a;
            minpos = t;
        }
    }

    if (mag_le(mag_abs(delta), minabs))
        return;

    w_col[minpos] = col;
    w_val[minpos] = delta;
}

// -------------------------
// Shared-memory layout helpers (alignment-safe)
// -------------------------
__device__ __forceinline__ size_t align_up_dev(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

// -------------------------
// One-row ICTP body (no kernel launch; expects shared work buffers)
// Changes vs original:
//   - w_* and sel_* live in shared memory (not local spill)
//   - Ljk lookup is O(1) using (row,pos) stored in column lists
//   - nodeCounter/head updates are non-atomic (single-thread execution)
// -------------------------
template <typename G, int MAX_CAP, int MAX_WORK>
__device__ __forceinline__ void ictp_process_row(
    int n,
    const int *__restrict__ rowPtrA,
    const int *__restrict__ colIndA,
    const G *__restrict__ valA,
    int cap,
    G dropTol,
    G pivotTol,
    int *__restrict__ rowCountL,
    int *__restrict__ colIndL,
    G *__restrict__ valL,
    G *__restrict__ diagL,
    int *__restrict__ colHead,
    int *__restrict__ colNext,
    int *__restrict__ colRow,
    int *__restrict__ colPos, // NEW: position within row (1..rowCount-1)
    int &nodeCounterLocal,    // NEW: non-atomic local counter
    int maxNodes,
    int i,
    int *__restrict__ status,
    int *__restrict__ fail_row,
    G *__restrict__ fail_pivot,
    int *w_col, G *w_val, int *sel_j, G *sel_l)
{
    if (*status != 0)
        return;

    const int keep_max = cap - 1;

    const int rowStartA = rowPtrA[i];
    const int rowEndA = rowPtrA[i + 1];

    G a_ii = g_zero<G>();
    if (rowEndA > rowStartA)
    {
        int last = rowEndA - 1;
        if (colIndA[last] == i)
            a_ii = valA[last];
        else
        {
            for (int p = rowStartA; p < rowEndA; ++p)
            {
                if (colIndA[p] == i)
                {
                    a_ii = valA[p];
                    break;
                }
            }
        }
    }

    int wSz = 0;

    for (int p = rowStartA; p < rowEndA; ++p)
    {
        int j = colIndA[p];
        if (j < 0)
            continue;
        if (j >= i)
            continue;
        w_add_bounded<G, MAX_WORK>(i, w_col, w_val, wSz, j, valA[p], dropTol);
    }

    G w_ii = a_ii;

    int selSz = 0;

    const mag_t<G> dropTolM = to_mag(dropTol);
    const mag_t<G> pivotTolM = to_mag(pivotTol);

    while (true)
    {
        int idxMin = -1;
        int kMin = n;

        for (int t = 0; t < wSz; ++t)
        {
            int k = w_col[t];
            G v = w_val[t];
            if (is_zero(v))
                continue;
            if (k < kMin)
            {
                kMin = k;
                idxMin = t;
            }
        }

        if (idxMin < 0)
            break;

        const int k = kMin;
        const G wk = w_val[idxMin];

        --wSz;
        w_col[idxMin] = w_col[wSz];
        w_val[idxMin] = w_val[wSz];

        if (is_zero(wk))
            continue;

        G Lkk = diagL[k];
        if (is_zero(Lkk))
            continue;

        G lik = g_div<G>(wk, Lkk);

        if (mag_gt(dropTolM, mag_zero<mag_t<G>>()) && mag_lt(mag_abs(lik), dropTolM))
            continue;

        w_ii = g_sub<G>(w_ii, g_mul<G>(lik, lik));

        // Traverse existing rows j that have L(j,k) via colHead[k]
        for (int node = colHead[k]; node != -1; node = colNext[node])
        {
            int j = colRow[node];
            if (j <= k)
                continue;
            if (j >= i)
                continue;

            // O(1) Ljk load using stored position
            int pos = colPos[node];
            G Ljk = valL[j * cap + pos];

            if (!is_zero(Ljk))
            {
                G delta = g_neg<G>(g_mul<G>(lik, Ljk));
                w_add_bounded<G, MAX_WORK>(i, w_col, w_val, wSz, j, delta, dropTol);
            }
        }

        topk_insert_by_abs<G, MAX_CAP>(sel_j, sel_l, selSz, keep_max, k, lik);
    }

    const mag_t<G> pivotM = to_mag(w_ii);
    if (mag_le(pivotM, pivotTolM))
    {
        *status = 1;
        *fail_row = i;
        *fail_pivot = w_ii;
        return;
    }

    G lii = sqrt_g<G>(w_ii);
    diagL[i] = lii;

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

    // Insert nodes into column lists (non-atomic: single-thread factorization)
    for (int t = 0; t < selSz; ++t)
    {
        int kk = sel_j[t];
        int node = nodeCounterLocal++;
        if (node >= maxNodes)
        {
            *status = 3;
            *fail_row = i;
            *fail_pivot = g_zero<G>();
            return;
        }
        colRow[node] = i;
        colPos[node] = 1 + t; // NEW: exact position in row i
        colNext[node] = colHead[kk];
        colHead[kk] = node;
    }
}

// -------------------------
// Persistent factor kernel (single launch; single thread)
// -------------------------
template <typename G, int MAX_CAP, int MAX_WORK>
__global__ void ictp_factor_kernel_persistent(
    int n,
    const int *__restrict__ rowPtrA,
    const int *__restrict__ colIndA,
    const G *__restrict__ valA,
    int cap,
    G dropTol,
    G pivotTol,
    int *__restrict__ rowCountL,
    int *__restrict__ colIndL,
    G *__restrict__ valL,
    G *__restrict__ diagL,
    int *__restrict__ colHead,
    int *__restrict__ colNext,
    int *__restrict__ colRow,
    int *__restrict__ colPos, // NEW
    int *__restrict__ status,
    int *__restrict__ fail_row,
    G *__restrict__ fail_pivot)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    if (cap < 1 || cap > MAX_CAP)
        return;

    extern __shared__ unsigned char smem_raw[];

    size_t off = 0;
    off = align_up_dev(off, alignof(int));
    int *w_col = (int *)(smem_raw + off);
    off += (size_t)MAX_WORK * sizeof(int);

    off = align_up_dev(off, alignof(G));
    G *w_val = (G *)(smem_raw + off);
    off += (size_t)MAX_WORK * sizeof(G);

    off = align_up_dev(off, alignof(int));
    int *sel_j = (int *)(smem_raw + off);
    off += (size_t)MAX_CAP * sizeof(int);

    off = align_up_dev(off, alignof(G));
    G *sel_l = (G *)(smem_raw + off);
    off += (size_t)MAX_CAP * sizeof(G);

    int nodeCounterLocal = 0;
    const int maxNodes = n * (cap - 1);

    for (int i = 0; i < n; ++i)
    {
        if (*status != 0)
            return;

        ictp_process_row<G, MAX_CAP, MAX_WORK>(
            n,
            rowPtrA, colIndA, valA,
            cap,
            dropTol,
            pivotTol,
            rowCountL,
            colIndL,
            valL,
            diagL,
            colHead,
            colNext,
            colRow,
            colPos,
            nodeCounterLocal,
            maxNodes,
            i,
            status,
            fail_row,
            fail_pivot,
            w_col, w_val, sel_j, sel_l);
    }
}

// -------------------------
// Host utilities
// -------------------------
template <class T>
static void validate_csr(const ichol::matrix::CsrMatrix<T> &A)
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

template <class G>
static __host__ __forceinline__ G host_cast(double x) { return (G)x; }

template <>
__host__ __forceinline__ __half host_cast<__half>(double x) { return __float2half_rn((float)x); }

template <class G>
static __host__ __forceinline__ double host_to_double(G x) { return (double)x; }

template <>
__host__ __forceinline__ double host_to_double<__half>(__half x) { return (double)__half2float(x); }

template <typename G, int MAX_CAP, int MAX_WORK>
static size_t shared_bytes_needed()
{
    auto align_up = [](size_t x, size_t a)
    { return (x + (a - 1)) & ~(a - 1); };

    size_t off = 0;
    off = align_up(off, alignof(int));
    off += (size_t)MAX_WORK * sizeof(int);

    off = align_up(off, alignof(G));
    off += (size_t)MAX_WORK * sizeof(G);

    off = align_up(off, alignof(int));
    off += (size_t)MAX_CAP * sizeof(int);

    off = align_up(off, alignof(G));
    off += (size_t)MAX_CAP * sizeof(G);

    return off;
}

template <typename T>
static bool ictp_rowwise_gpu_dynamic(
    const ichol::matrix::CsrMatrix<T> &Ahost,
    const ICTP_Params &row_params,
    const IC_Attempt_Params &attempt_params,
    ichol::matrix::CsrMatrix<T> &Lhost_out,
    ICTP_Factor_Info *info)
{
    using G = typename gpu_type<T>::type;

    const int n = Ahost.num_rows;
    const int cap = row_params.lfil_per_row;

    if (info)
        *info = ICTP_Factor_Info{};

    validate_csr(Ahost);
    if (cap < 1)
        return false;

    constexpr int MAX_CAP = 1000;
    constexpr int MAX_WORK = 2000;

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
    G *d_valA = nullptr;

    int *d_rowCountL = nullptr;
    int *d_colIndL = nullptr;
    G *d_valL = nullptr;
    G *d_diagL = nullptr;

    int *d_colHead = nullptr;
    int *d_colNext = nullptr;
    int *d_colRow = nullptr;
    int *d_colPos = nullptr; // NEW

    int *d_status = nullptr;
    int *d_fail_row = nullptr;
    G *d_fail_pivot = nullptr;

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
        cudaFree(d_colPos);

        cudaFree(d_status);
        cudaFree(d_fail_row);
        cudaFree(d_fail_pivot);
    };

    CUDA_CHECK(cudaMalloc(&d_rowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(G)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrA, Ahost.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIndA, Ahost.col_ind.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, G>)
    {
        CUDA_CHECK(cudaMemcpy(d_valA, Ahost.values.data(), nnzA * sizeof(G), cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<G> h_valA(nnzA);
        for (size_t p = 0; p < nnzA; ++p)
            h_valA[p] = host_cast<G>((double)Ahost.values[p]);
        CUDA_CHECK(cudaMemcpy(d_valA, h_valA.data(), nnzA * sizeof(G), cudaMemcpyHostToDevice));
    }

    const size_t maxNnzL = (size_t)n * (size_t)cap;

    CUDA_CHECK(cudaMalloc(&d_rowCountL, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIndL, maxNnzL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valL, maxNnzL * sizeof(G)));
    CUDA_CHECK(cudaMalloc(&d_diagL, n * sizeof(G)));

    CUDA_CHECK(cudaMemset(d_rowCountL, 0, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_diagL, 0, n * sizeof(G)));

    CUDA_CHECK(cudaMalloc(&d_colHead, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_colHead, 0xFF, n * sizeof(int))); // -1

    const int maxNodes = n * (cap - 1);
    if (maxNodes > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_colNext, (size_t)maxNodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colRow, (size_t)maxNodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colPos, (size_t)maxNodes * sizeof(int)));
    }
    else
    {
        d_colNext = nullptr;
        d_colRow = nullptr;
        d_colPos = nullptr;
    }

    CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_row, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_pivot, sizeof(G)));

    CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fail_row, -1, sizeof(int)));

    const G dropTol = host_cast<G>(row_params.drop_tol);
    const G pivotTol = host_cast<G>(attempt_params.pivot_tol);

    // Single persistent kernel launch (no per-row sync/copy)
    const size_t shmem = shared_bytes_needed<G, MAX_CAP, MAX_WORK>();
    ictp_factor_kernel_persistent<G, MAX_CAP, MAX_WORK>
        <<<1, 1, shmem>>>(
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
            d_colPos,
            d_status,
            d_fail_row,
            d_fail_pivot);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int host_status = 0;
    CUDA_CHECK(cudaMemcpy(&host_status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
    if (host_status != 0)
    {
        if (info)
        {
            int fr = -1;
            G fp = (G)0;
            CUDA_CHECK(cudaMemcpy(&fr, d_fail_row, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&fp, d_fail_pivot, sizeof(G), cudaMemcpyDeviceToHost));

            std::fprintf(stderr,
                         "ICTP failure at row %d, pivot=%g, pivotTol=%g, status=%d\n",
                         fr, host_to_double(fp), host_to_double(pivotTol), host_status);

            if (host_status == 1)
                info->code = IC_Breakdown::B1_SmallOrNegativePivot;
            else
                info->code = IC_Breakdown::OtherNumericalIssue;

            info->step = fr;
            info->pivot_value = host_to_double(fp);
        }

        cleanup();
        return false;
    }

    std::vector<int> rowCountL(n);
    std::vector<int> colIndL(maxNnzL);

    CUDA_CHECK(cudaMemcpy(rowCountL.data(), d_rowCountL, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colIndL.data(), d_colIndL, maxNnzL * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<T> valL(maxNnzL);
    if constexpr (std::is_same_v<T, G>)
    {
        CUDA_CHECK(cudaMemcpy(valL.data(), d_valL, maxNnzL * sizeof(G), cudaMemcpyDeviceToHost));
    }
    else
    {
        std::vector<G> h_valL(maxNnzL);
        CUDA_CHECK(cudaMemcpy(h_valL.data(), d_valL, maxNnzL * sizeof(G), cudaMemcpyDeviceToHost));
        for (size_t p = 0; p < maxNnzL; ++p)
            valL[p] = (T)host_to_double(h_valL[p]);
    }

    ichol::matrix::CsrMatrix<T> L;
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

        // Already sorted in-kernel; no host re-sort
        for (int t = 0; t < cnt; ++t)
        {
            L.col_ind[dst + t] = colIndL[base + t];
            L.values[dst + t] = valL[base + t];
        }
    }

    Lhost_out = std::move(L);
    cleanup();
    return true;
}

namespace ichol
{
    template <class T>
    matrix::CsrMatrix<T> ictp(const matrix::CsrMatrix<T> &Ahost,
                              const ICTP_Params &row_params,
                              const IC_Attempt_Params &fparams,
                              const ichol::core::IC_Symbolic &Sym,
                              ICTP_Factor_Info *info)
    {
        (void)Sym; // dynamic path ignores symbolic

        ichol::matrix::CsrMatrix<T> L;
        L.num_rows = Ahost.num_rows;
        L.num_cols = Ahost.num_cols;
        L.row_ptr.assign(Ahost.num_rows + 1, 0);

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

    template matrix::CsrMatrix<double> ictp<double>(const matrix::CsrMatrix<double> &,
                                                    const ICTP_Params &,
                                                    const IC_Attempt_Params &,
                                                    const ichol::core::IC_Symbolic &,
                                                    ICTP_Factor_Info *);

    template matrix::CsrMatrix<float> ictp<float>(const matrix::CsrMatrix<float> &,
                                                  const ICTP_Params &,
                                                  const IC_Attempt_Params &,
                                                  const ichol::core::IC_Symbolic &,
                                                  ICTP_Factor_Info *);

    template matrix::CsrMatrix<half_float::half> ictp<half_float::half>(const matrix::CsrMatrix<half_float::half> &,
                                                                        const ICTP_Params &,
                                                                        const IC_Attempt_Params &,
                                                                        const ichol::core::IC_Symbolic &,
                                                                        ICTP_Factor_Info *);
} // namespace ichol
