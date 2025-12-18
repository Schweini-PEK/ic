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
#include "ichol/matrix_formats.hpp"
#include "ichol/fact.hpp"
#include "ichol/half.hpp"
#include "ichol/cuda_utils.hpp"

// #define CUDA_CHECK(call)                                                 \
//     do                                                                   \
//     {                                                                    \
//         cudaError_t err = call;                                          \
//         if (err != cudaSuccess)                                          \
//         {                                                                \
//             std::cerr << "CUDA error in " << __FILE__ << ':' << __LINE__ \
//                       << " " << cudaGetErrorString(err) << std::endl;    \
//             std::exit(EXIT_FAILURE);                                     \
//         }                                                                \
//     } while (0)

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
    // abs(x) by clearing sign bit, no half<->float conversions
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
    // explicit half math function (may internally use wider precision, but API is half)
    return hsqrt(x);
}

template <class G>
__host__ __device__ __forceinline__ bool is_zero(G x) { return x == G(0); }

template <>
__host__ __device__ __forceinline__ bool is_zero<__half>(__half x)
{
    // treat +0 and -0 as zero, no half<->float conversions
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

    // accumulate if present
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

template <typename G>
__device__ __forceinline__ G get_Ljk_rowwise(
    int j, int k, int cap,
    const int *__restrict__ rowCountL,
    const int *__restrict__ colIndL,
    const G *__restrict__ valL)
{
    const int jBase = j * cap;
    const int jCount = rowCountL[j];
    for (int q = 1; q < jCount; ++q)
    {
        int colq = colIndL[jBase + q];
        if (colq == k)
            return valL[jBase + q];
        if (colq > k)
            break;
    }
    return g_zero<G>();
}

// -------------------------
// Dynamic ICTP row kernel (GPU scalar = G)
// -------------------------
template <typename G, int MAX_CAP, int MAX_WORK>
__global__ void ictp_row_kernel_dynamic(
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
    int *__restrict__ nodeCounter,
    int maxNodes,
    int i,
    int *__restrict__ status,
    int *__restrict__ fail_row,
    G *__restrict__ fail_pivot)
{
    if (threadIdx.x != 0)
        return;
    if (cap < 1 || cap > MAX_CAP)
        return;

    const int keep_max = cap - 1;

    const int rowStartA = rowPtrA[i];
    const int rowEndA = rowPtrA[i + 1];

    // robust diagonal extract
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

    int w_col[MAX_WORK];
    G w_val[MAX_WORK];
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

    int sel_j[MAX_CAP];
    G sel_l[MAX_CAP];
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

        // lik = wk / Lkk (explicit half division when G=__half)
        G lik = g_div<G>(wk, Lkk);

        if (mag_gt(dropTolM, mag_zero<mag_t<G>>()) && mag_lt(mag_abs(lik), dropTolM))
            continue;

        // w_ii = w_ii - lik*lik
        w_ii = g_sub<G>(w_ii, g_mul<G>(lik, lik));

        for (int node = colHead[k]; node != -1; node = colNext[node])
        {
            int j = colRow[node];
            if (j <= k)
                continue;
            if (j >= i)
                continue;

            G Ljk = get_Ljk_rowwise<G>(j, k, cap, rowCountL, colIndL, valL);
            if (!is_zero(Ljk))
            {
                // delta = -lik * Ljk
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

    for (int t = 0; t < selSz; ++t)
    {
        int kk = sel_j[t];
        int node = atomicAdd(nodeCounter, 1);
        if (node >= maxNodes)
        {
            *status = 3;
            *fail_row = i;
            *fail_pivot = g_zero<G>();
            return;
        }
        colRow[node] = i;
        colNext[node] = atomicExch(&colHead[kk], node);
    }
}

// -------------------------
// Host utilities
// -------------------------
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

template <class G>
static __host__ __forceinline__ G host_cast(double x) { return (G)x; }

template <>
__host__ __forceinline__ __half host_cast<__half>(double x) { return __float2half_rn((float)x); }

template <class G>
static __host__ __forceinline__ double host_to_double(G x) { return (double)x; }

template <>
__host__ __forceinline__ double host_to_double<__half>(__half x) { return (double)__half2float(x); }

template <typename T>
static bool ictp_rowwise_gpu_dynamic(
    const ichol::CSR<T> &Ahost,
    const ICTP_Params &row_params,
    const IC_Attempt_Params &attempt_params,
    ichol::CSR<T> &Lhost_out,
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
    int *d_nodeCounter = nullptr;

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
        cudaFree(d_nodeCounter);

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

    CUDA_CHECK(cudaMalloc(&d_nodeCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nodeCounter, 0, sizeof(int)));

    const int maxNodes = n * (cap - 1);
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

    CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_row, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fail_pivot, sizeof(G)));

    CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fail_row, -1, sizeof(int)));

    const G dropTol = host_cast<G>(row_params.drop_tol);
    const G pivotTol = host_cast<G>(attempt_params.pivot_tol);

    int host_status = 0;

    for (int i = 0; i < n; ++i)
    {
        ictp_row_kernel_dynamic<G, MAX_CAP, MAX_WORK>
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

namespace ichol
{
    template <class T>
    CSR<T> ictp(const CSR<T> &Ahost,
                const ICTP_Params &row_params,
                const IC_Attempt_Params &fparams,
                const core::IC_Symbolic &Sym,
                ICTP_Factor_Info *info)
    {
        (void)Sym; // dynamic path ignores symbolic

        CSR<T> L;
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

    template CSR<half_float::half> ictp<half_float::half>(const CSR<half_float::half> &,
                                                          const ICTP_Params &,
                                                          const IC_Attempt_Params &,
                                                          const core::IC_Symbolic &,
                                                          ICTP_Factor_Info *);
} // namespace ichol
