// test_fp16_sptrsv_gtest.cu
//
// Unit tests for an fp16 (device __half) sparse triangular solve intended to
// replace cusparseSpSV_solve for IC(0)/IC-like preconditioners.
//
// Assumes you provide (and link) this function:
//
//   cudaError_t MySpSVSolveFp16CsrLower(
//       int n,
//       const int* d_rowPtr, const int* d_colInd, const __half* d_vals, // CSR of LOWER-triangular L (includes diag)
//       bool transpose,                                                 // false: solve L x = rhs, true: solve L^T x = rhs
//       const __half* d_rhs,
//       __half* d_x,
//       cudaStream_t stream);
//
// Host-side uses half.hpp (half_float::half) to emulate fp16 arithmetic.
//
// Build example (adjust include/lib paths):
//   nvcc -std=c++17 --expt-relaxed-constexpr -I<gtest_include> test_fp16_sptrsv_gtest.cu \
//        -L<gtest_lib> -lgtest -lgtest_main -lpthread -o test_fp16_sptrsv
//
// Notes:
// - CSR rows are generated sorted by col_ind ascending; only lower-tri + diag are stored.
// - Reference solve uses half_float::half operations (round-to-half at each op).
// - Tolerances are set to allow implementation differences (ordering/rounding).

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ichol/half.hpp"
#include "ichol/matrix_formats.hpp"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

// ---------------------- Declaration of the tested function ----------------------
extern cudaError_t MySpSVSolveFp16CsrLower(
    int n,
    const int *d_rowPtr, const int *d_colInd, const __half *d_vals,
    bool transpose,
    const __half *d_rhs,
    __half *d_x,
    cudaStream_t stream);

// ---------------------- CUDA helpers ----------------------
#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t _e = (call);                              \
        ASSERT_EQ(_e, cudaSuccess) << cudaGetErrorString(_e); \
    } while (0)

static inline __half to_device_half(half_float::half x)
{
    return __float2half_rn(static_cast<float>(x));
}
static inline half_float::half to_host_half(__half x)
{
    return half_float::half(__half2float(x));
}

// ---------------------- CSR builders (lower-tri + diag, sorted rows) ----------------------
static ichol::matrix::CsrMatrix<half_float::half> DenseLowerToCsrHalf(const std::vector<float> &Ldense, int n)
{
    ichol::matrix::CsrMatrix<half_float::half> csr;
    csr.num_rows = n;
    csr.num_cols = n;
    csr.row_ptr.assign(n + 1, 0);

    // count nnz (store only j<=i)
    for (int i = 0; i < n; ++i)
    {
        int cnt = 0;
        for (int j = 0; j <= i; ++j)
        {
            float v = Ldense[i * n + j];
            if (v != 0.0f)
                cnt++;
        }
        csr.row_ptr[i + 1] = cnt;
    }
    for (int i = 0; i < n; ++i)
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    csr.nnz = csr.row_ptr[n];

    csr.col_ind.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    int k = 0;
    for (int i = 0; i < n; ++i)
    {
        // j ascending => col_ind sorted in each row
        for (int j = 0; j <= i; ++j)
        {
            float v = Ldense[i * n + j];
            if (v != 0.0f)
            {
                csr.col_ind[k] = j;
                csr.values[k] = half_float::half(v);
                k++;
            }
        }
    }
    return csr;
}

// Simple CSC for reference transpose solve (L^T x = rhs) using column access of L.
template <typename T>
struct CscMatrix
{
    std::vector<int> col_ptr; // size n+1
    std::vector<int> row_ind; // size nnz
    std::vector<T> values;    // size nnz
    int n = 0;
    int nnz = 0;
};

static CscMatrix<half_float::half> BuildCscFromCsr(const ichol::matrix::CsrMatrix<half_float::half> &csr)
{
    const int n = csr.num_rows;

    CscMatrix<half_float::half> csc;
    csc.n = n;
    csc.nnz = csr.nnz;
    csc.col_ptr.assign(n + 1, 0);
    csc.row_ind.resize(csr.nnz);
    csc.values.resize(csr.nnz);

    // count nnz per col
    for (int k = 0; k < csr.nnz; ++k)
    {
        int c = csr.col_ind[k];
        csc.col_ptr[c + 1]++;
    }
    // prefix sum
    for (int c = 0; c < n; ++c)
        csc.col_ptr[c + 1] += csc.col_ptr[c];

    std::vector<int> next = csc.col_ptr; // write heads

    for (int r = 0; r < n; ++r)
    {
        for (int kk = csr.row_ptr[r]; kk < csr.row_ptr[r + 1]; ++kk)
        {
            int c = csr.col_ind[kk];
            int dst = next[c]++;
            csc.row_ind[dst] = r;
            csc.values[dst] = csr.values[kk];
        }
    }

    // Optional: sort row_ind within each column to make reference deterministic.
    for (int c = 0; c < n; ++c)
    {
        int b = csc.col_ptr[c];
        int e = csc.col_ptr[c + 1];
        auto begin = b;
        auto end = e;

        // sort indices [b,e) by row_ind
        std::vector<int> perm(e - b);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](int a, int b2)
                  { return csc.row_ind[begin + a] < csc.row_ind[begin + b2]; });

        std::vector<int> rows_tmp(e - b);
        std::vector<half_float::half> vals_tmp(e - b);
        for (int i = 0; i < (int)perm.size(); ++i)
        {
            rows_tmp[i] = csc.row_ind[begin + perm[i]];
            vals_tmp[i] = csc.values[begin + perm[i]];
        }
        for (int i = 0; i < (int)perm.size(); ++i)
        {
            csc.row_ind[begin + i] = rows_tmp[i];
            csc.values[begin + i] = vals_tmp[i];
        }
    }

    return csc;
}

// ---------------------- Host reference (fp16 via half.hpp) ----------------------
static std::vector<half_float::half> HostSolveLowerCsr_fp16(const ichol::matrix::CsrMatrix<half_float::half> &L,
                                                            const std::vector<half_float::half> &rhs)
{
    const int n = L.num_rows;
    std::vector<half_float::half> x(n, half_float::half(0));

    for (int i = 0; i < n; ++i)
    {
        half_float::half sum = half_float::half(0);
        half_float::half diag = half_float::half(0);
        bool have_diag = false;

        for (int kk = L.row_ptr[i]; kk < L.row_ptr[i + 1]; ++kk)
        {
            int j = L.col_ind[kk];
            half_float::half a = L.values[kk];
            if (j < i)
            {
                sum = sum + a * x[j]; // half arithmetic
            }
            else if (j == i)
            {
                diag = a;
                have_diag = true;
            }
            else
            {
                // should not happen for lower-tri storage
            }
        }
        EXPECT_TRUE(have_diag) << "Missing diagonal at row " << i;
        x[i] = (rhs[i] - sum) / diag;
    }
    return x;
}

static std::vector<half_float::half> HostSolveTransposeLower_fp16(const ichol::matrix::CsrMatrix<half_float::half> &L,
                                                                  const std::vector<half_float::half> &rhs)
{
    // Solve (L^T) x = rhs using CSC of L:
    // row i of L^T corresponds to column i of L.
    const int n = L.num_rows;
    CscMatrix<half_float::half> csc = BuildCscFromCsr(L);

    std::vector<half_float::half> x(n, half_float::half(0));

    for (int i = n - 1; i >= 0; --i)
    {
        half_float::half sum = half_float::half(0);
        half_float::half diag = half_float::half(0);
        bool have_diag = false;

        int b = csc.col_ptr[i];
        int e = csc.col_ptr[i + 1];
        for (int p = b; p < e; ++p)
        {
            int r = csc.row_ind[p];             // row in L, hence col in L^T
            half_float::half a = csc.values[p]; // a = L[r,i]
            if (r == i)
            {
                diag = a;
                have_diag = true;
            }
            else if (r > i)
            {
                // contributes to sum over j>i: L[j,i] * x[j]
                sum = sum + a * x[r];
            }
            else
            {
                // r<i shouldn't appear for lower-tri, but harmless to ignore.
            }
        }
        EXPECT_TRUE(have_diag) << "Missing diagonal at col/row " << i;
        x[i] = (rhs[i] - sum) / diag;
    }
    return x;
}

// ---------------------- Device upload/download helpers ----------------------
struct DeviceCsr
{
    int n = 0;
    int nnz = 0;
    int *d_row_ptr = nullptr;
    int *d_col_ind = nullptr;
    __half *d_vals = nullptr;
};

static DeviceCsr UploadCsrToDevice(const ichol::matrix::CsrMatrix<half_float::half> &h, cudaStream_t stream)
{
    DeviceCsr d;
    d.n = h.num_rows;
    d.nnz = h.nnz;

    cudaMalloc(&d.d_row_ptr, sizeof(int) * (h.num_rows + 1));
    cudaMalloc(&d.d_col_ind, sizeof(int) * h.nnz);
    cudaMalloc(&d.d_vals, sizeof(__half) * h.nnz);

    std::vector<__half> vals_dev(h.nnz);
    for (int k = 0; k < h.nnz; ++k)
        vals_dev[k] = to_device_half(h.values[k]);

    cudaMemcpyAsync(d.d_row_ptr, h.row_ptr.data(), sizeof(int) * (h.num_rows + 1),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d.d_col_ind, h.col_ind.data(), sizeof(int) * h.nnz,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d.d_vals, vals_dev.data(), sizeof(__half) * h.nnz,
                    cudaMemcpyHostToDevice, stream);
    return d;
}

static void FreeDeviceCsr(DeviceCsr &d)
{
    if (d.d_row_ptr)
        cudaFree(d.d_row_ptr);
    if (d.d_col_ind)
        cudaFree(d.d_col_ind);
    if (d.d_vals)
        cudaFree(d.d_vals);
    d = {};
}

static void AssertAllCloseHalfAsFloat(const std::vector<half_float::half> &got,
                                      const std::vector<half_float::half> &ref,
                                      float atol, float rtol)
{
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        float a = static_cast<float>(got[i]);
        float b = static_cast<float>(ref[i]);
        float diff = std::fabs(a - b);
        float tol = atol + rtol * std::fabs(b);
        ASSERT_LE(diff, tol) << "i=" << i << " got=" << a << " ref=" << b
                             << " diff=" << diff << " tol=" << tol;
    }
}

// ---------------------- Test fixture ----------------------
class Fp16SpSVTest : public ::testing::Test
{
protected:
    void SetUp() override { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    void TearDown() override
    {
        if (stream_)
            cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    cudaStream_t stream_ = nullptr;
};

// ---------------------- Tests ----------------------

TEST_F(Fp16SpSVTest, IdentityMatrix_ForwardAndBackward)
{
    const int n = 16;

    std::vector<float> Ldense(n * n, 0.0f);
    for (int i = 0; i < n; ++i)
        Ldense[i * n + i] = 1.0f;

    ichol::matrix::CsrMatrix<half_float::half> L = DenseLowerToCsrHalf(Ldense, n);
    DeviceCsr dL = UploadCsrToDevice(L, stream_);

    std::vector<half_float::half> r(n);
    for (int i = 0; i < n; ++i)
        r[i] = half_float::half(0.25f * (i - 7));

    std::vector<__half> r_dev(n), w_dev(n), z_dev(n);
    for (int i = 0; i < n; ++i)
        r_dev[i] = to_device_half(r[i]);

    __half *d_r = nullptr, *d_w = nullptr, *d_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_r, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_z, sizeof(__half) * n));
    CUDA_CHECK(cudaMemcpyAsync(d_r, r_dev.data(), sizeof(__half) * n, cudaMemcpyHostToDevice, stream_));

    // forward: L w = r
    CUDA_CHECK(MySpSVSolveFp16CsrLower(n, dL.d_row_ptr, dL.d_col_ind, dL.d_vals,
                                       /*transpose=*/false, d_r, d_w, stream_));
    // backward: L^T z = w
    CUDA_CHECK(MySpSVSolveFp16CsrLower(n, dL.d_row_ptr, dL.d_col_ind, dL.d_vals,
                                       /*transpose=*/true, d_w, d_z, stream_));

    CUDA_CHECK(cudaMemcpyAsync(w_dev.data(), d_w, sizeof(__half) * n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(z_dev.data(), d_z, sizeof(__half) * n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<half_float::half> w(n), z(n);
    for (int i = 0; i < n; ++i)
    {
        w[i] = to_host_half(w_dev[i]);
        z[i] = to_host_half(z_dev[i]);
    }

    // Reference: identity => w=r, z=r exactly in half arithmetic
    AssertAllCloseHalfAsFloat(w, r, 0.0f, 0.0f);
    AssertAllCloseHalfAsFloat(z, r, 0.0f, 0.0f);

    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_z);
    FreeDeviceCsr(dL);
}

TEST_F(Fp16SpSVTest, SmallKnownLowerTriangular_TwoStageApply)
{
    // L =
    // [ 1      0      0 ]
    // [ 0.25   1      0 ]
    // [ -0.5   0.125  1 ]
    const int n = 3;
    std::vector<float> Ldense = {
        1.0f, 0.0f, 0.0f,
        0.25f, 1.0f, 0.0f,
        -0.5f, 0.125f, 1.0f};

    ichol::matrix::CsrMatrix<half_float::half> L = DenseLowerToCsrHalf(Ldense, n);
    DeviceCsr dL = UploadCsrToDevice(L, stream_);

    std::vector<half_float::half> r = {half_float::half(1.0f), half_float::half(-2.0f), half_float::half(0.5f)};

    std::vector<__half> r_dev(n), w_dev(n), z_dev(n);
    for (int i = 0; i < n; ++i)
        r_dev[i] = to_device_half(r[i]);

    __half *d_r = nullptr, *d_w = nullptr, *d_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_r, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_z, sizeof(__half) * n));
    CUDA_CHECK(cudaMemcpyAsync(d_r, r_dev.data(), sizeof(__half) * n, cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(MySpSVSolveFp16CsrLower(n, dL.d_row_ptr, dL.d_col_ind, dL.d_vals,
                                       false, d_r, d_w, stream_));
    CUDA_CHECK(MySpSVSolveFp16CsrLower(n, dL.d_row_ptr, dL.d_col_ind, dL.d_vals,
                                       true, d_w, d_z, stream_));

    CUDA_CHECK(cudaMemcpyAsync(w_dev.data(), d_w, sizeof(__half) * n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(z_dev.data(), d_z, sizeof(__half) * n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<half_float::half> w(n), z(n);
    for (int i = 0; i < n; ++i)
    {
        w[i] = to_host_half(w_dev[i]);
        z[i] = to_host_half(z_dev[i]);
    }

    // Reference (half arithmetic):
    std::vector<half_float::half> w_ref = HostSolveLowerCsr_fp16(L, r);
    std::vector<half_float::half> z_ref = HostSolveTransposeLower_fp16(L, w_ref);

    // Tight-ish tolerance; still allow small differences from GPU op ordering.
    AssertAllCloseHalfAsFloat(w, w_ref, /*atol=*/2e-2f, /*rtol=*/2e-2f);
    AssertAllCloseHalfAsFloat(z, z_ref, /*atol=*/2e-2f, /*rtol=*/2e-2f);

    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_z);
    FreeDeviceCsr(dL);
}

TEST_F(Fp16SpSVTest, RandomWellScaledLowerTriangular_TwoStageApply)
{
    const int n = 64;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> offdiag(-0.05f, 0.05f);
    std::uniform_real_distribution<float> rhsdist(-0.5f, 0.5f);

    // Dense lower-tri with diag=1, small offdiag; sparsify; rows end up sorted in CSR builder.
    std::vector<float> Ldense(n * n, 0.0f);
    for (int i = 0; i < n; ++i)
    {
        Ldense[i * n + i] = 1.0f;
        for (int j = 0; j < i; ++j)
        {
            float v = offdiag(rng);
            if ((i + 3 * j) % 7 == 0)
                v = 0.0f; // sparsify
            Ldense[i * n + j] = v;
        }
    }

    ichol::matrix::CsrMatrix<half_float::half> L = DenseLowerToCsrHalf(Ldense, n);
    DeviceCsr dL = UploadCsrToDevice(L, stream_);

    std::vector<half_float::half> r(n);
    for (int i = 0; i < n; ++i)
        r[i] = half_float::half(rhsdist(rng));

    std::vector<__half> r_dev(n), w_dev(n), z_dev(n);
    for (int i = 0; i < n; ++i)
        r_dev[i] = to_device_half(r[i]);

    __half *d_r = nullptr, *d_w = nullptr, *d_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_r, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_z, sizeof(__half) * n));
    CUDA_CHECK(cudaMemcpyAsync(d_r, r_dev.data(), sizeof(__half) * n, cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(MySpSVSolveFp16CsrLower(n, dL.d_row_ptr, dL.d_col_ind, dL.d_vals,
                                       false, d_r, d_w, stream_));
    CUDA_CHECK(MySpSVSolveFp16CsrLower(n, dL.d_row_ptr, dL.d_col_ind, dL.d_vals,
                                       true, d_w, d_z, stream_));

    CUDA_CHECK(cudaMemcpyAsync(w_dev.data(), d_w, sizeof(__half) * n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(z_dev.data(), d_z, sizeof(__half) * n, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<half_float::half> w(n), z(n);
    for (int i = 0; i < n; ++i)
    {
        w[i] = to_host_half(w_dev[i]);
        z[i] = to_host_half(z_dev[i]);
    }

    std::vector<half_float::half> w_ref = HostSolveLowerCsr_fp16(L, r);
    std::vector<half_float::half> z_ref = HostSolveTransposeLower_fp16(L, w_ref);

    // Looser tolerance: fp16 triangular solve is sensitive to ordering and dependency depth.
    AssertAllCloseHalfAsFloat(w, w_ref, /*atol=*/8e-2f, /*rtol=*/8e-2f);
    AssertAllCloseHalfAsFloat(z, z_ref, /*atol=*/8e-2f, /*rtol=*/8e-2f);

    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_z);
    FreeDeviceCsr(dL);
}
