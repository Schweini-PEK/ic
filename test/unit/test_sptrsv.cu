// sptrsv_levelsets_test.cu

#include "solve/sptrsv/cuda/sptrsv_level.cuh"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>

template <typename T>
constexpr bool sptrsv_test_supported_v = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, __half>;

static void skip_if_no_cuda_device()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0)
        GTEST_SKIP() << "No CUDA device available for SpTRSV levelsets tests.";
}

/* Compute max absolute difference between two vectors. */
template <typename T>
static double max_abs_diff(const std::vector<T> &a, const std::vector<T> &b)
{
    EXPECT_EQ(a.size(), b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = std::abs((double)a[i] - (double)b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

/*
CPU reference: forward substitution for lower-triangular solve in CSR.
Interprets CSR as A, computes x from Ax=b in increasing row order.
*/
template <typename T>
static std::vector<T> cpu_trsv_csr_forward(
    int n,
    const std::vector<int> &row_ptr,
    const std::vector<int> &col_ind,
    const std::vector<T> &val,
    const std::vector<T> &b,
    bool unit_diag)
{
    std::vector<T> x(n, T(0));
    for (int i = 0; i < n; ++i)
    {
        T s = b[i];
        T diag = T(1);
        bool diag_found = unit_diag;

        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            int j = col_ind[k];
            T a = val[k];

            if (j == i)
            {
                if (!unit_diag)
                {
                    diag = a;
                    diag_found = true;
                }
                continue;
            }
            s = s - a * x[j];
        }

        if (!unit_diag)
        {
            EXPECT_TRUE(diag_found);
            s = s / diag;
        }
        x[i] = s;
    }
    return x;
}

/* Build trivial level sets: one row per level, forward order. */
static ichol::symbolic::LevelSets make_levelsets_sequential(int n)
{
    ichol::symbolic::LevelSets ls;
    ls.level_ptr.resize(n + 1);
    ls.levels.resize(n);

    for (int i = 0; i <= n; ++i)
        ls.level_ptr[i] = i;

    for (int i = 0; i < n; ++i)
        ls.levels[i] = i;

    return ls;
}

/*
Test: NON_TRANSPOSE on a small lower-triangular matrix with diagonal present.
*/
template <typename ValueT>
static void run_test_lower_nontranspose()
{
    if constexpr (!sptrsv_test_supported_v<ValueT>)
    {
        GTEST_SKIP() << "SpTRSV levelsets tests currently support float/double only.";
    }
    else
    {
        skip_if_no_cuda_device();
        using IndexT = int;

    int n = 4;
    std::vector<int> h_rowPtr = {0, 1, 3, 6, 10};
    std::vector<int> h_colInd = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
    std::vector<ValueT> h_val = {ValueT(2), ValueT(3), ValueT(1),
                                 ValueT(-1), ValueT(4), ValueT(5),
                                 ValueT(2), ValueT(0.5), ValueT(-3), ValueT(1)};
    std::vector<ValueT> h_b = {ValueT(1), ValueT(2), ValueT(-1), ValueT(4)};

    auto x_ref_fwd = cpu_trsv_csr_forward<ValueT>(n, h_rowPtr, h_colInd, h_val, h_b, /*unit_diag=*/false);

    cudaStream_t stream;
    cudaError_t stream_err = cudaStreamCreate(&stream);
    if (stream_err != cudaSuccess)
        GTEST_SKIP() << "CUDA stream creation failed: " << cudaGetErrorString(stream_err);

    IndexT *d_rowPtr = nullptr, *d_colInd = nullptr;
    ValueT *d_val = nullptr, *d_b = nullptr, *d_x = nullptr;

    ASSERT_EQ(cudaMalloc((void **)&d_rowPtr, sizeof(IndexT) * (n + 1)), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_colInd, sizeof(IndexT) * h_colInd.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_val, sizeof(ValueT) * h_val.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_b, sizeof(ValueT) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_x, sizeof(ValueT) * n), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_rowPtr, h_rowPtr.data(), sizeof(IndexT) * (n + 1), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_colInd, h_colInd.data(), sizeof(IndexT) * h_colInd.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_val, h_val.data(), sizeof(ValueT) * h_val.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_b, h_b.data(), sizeof(ValueT) * n, cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ichol::symbolic::LevelSets ls = make_levelsets_sequential(n);
    int rc = SpTRSV_solve_levelsets<IndexT, ValueT>(
        n, d_rowPtr, d_colInd, d_val, d_b, d_x,
        FillMode::LOWER, /*unit_diag=*/false,
        ls, stream);
    ASSERT_EQ(rc, 0);

    std::vector<ValueT> h_x(n);
    ASSERT_EQ(cudaMemcpyAsync(h_x.data(), d_x, sizeof(ValueT) * n, cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    double err = max_abs_diff(h_x, x_ref_fwd);
    double tol = std::is_same_v<ValueT, float> ? 1e-5 : 1e-12;
    ASSERT_LT(err, tol);

    ASSERT_EQ(cudaFree(d_x), cudaSuccess);
    ASSERT_EQ(cudaFree(d_b), cudaSuccess);
    ASSERT_EQ(cudaFree(d_val), cudaSuccess);
    ASSERT_EQ(cudaFree(d_colInd), cudaSuccess);
    ASSERT_EQ(cudaFree(d_rowPtr), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

/*
Test: unit diagonal with diagonal absent in CSR.
Ensures unit_diag=true path does not require diagonal entries.
*/
template <typename ValueT>
static void run_test_unit_diag_absent()
{
    if constexpr (!sptrsv_test_supported_v<ValueT>)
    {
        GTEST_SKIP() << "SpTRSV levelsets tests currently support float/double only.";
    }
    else
    {
        skip_if_no_cuda_device();
        using IndexT = int;

    int n = 3;
    std::vector<int> h_rowPtr = {0, 0, 1, 3};
    std::vector<int> h_colInd = {0, 0, 1};
    std::vector<ValueT> h_val = {ValueT(2), ValueT(-1), ValueT(3)};
    std::vector<ValueT> h_b = {ValueT(1), ValueT(2), ValueT(3)};

    auto x_ref = cpu_trsv_csr_forward<ValueT>(n, h_rowPtr, h_colInd, h_val, h_b, /*unit_diag=*/true);

    cudaStream_t stream;
    cudaError_t stream_err = cudaStreamCreate(&stream);
    if (stream_err != cudaSuccess)
        GTEST_SKIP() << "CUDA stream creation failed: " << cudaGetErrorString(stream_err);

    IndexT *d_rowPtr = nullptr, *d_colInd = nullptr;
    ValueT *d_val = nullptr, *d_b = nullptr, *d_x = nullptr;

    ASSERT_EQ(cudaMalloc((void **)&d_rowPtr, sizeof(IndexT) * (n + 1)), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_colInd, sizeof(IndexT) * h_colInd.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_val, sizeof(ValueT) * h_val.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_b, sizeof(ValueT) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_x, sizeof(ValueT) * n), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_rowPtr, h_rowPtr.data(), sizeof(IndexT) * (n + 1), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_colInd, h_colInd.data(), sizeof(IndexT) * h_colInd.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_val, h_val.data(), sizeof(ValueT) * h_val.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_b, h_b.data(), sizeof(ValueT) * n, cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ichol::symbolic::LevelSets ls = make_levelsets_sequential(n);
    int rc = SpTRSV_solve_levelsets<IndexT, ValueT>(
        n, d_rowPtr, d_colInd, d_val, d_b, d_x,
        FillMode::LOWER, /*unit_diag=*/true,
        ls, stream);
    ASSERT_EQ(rc, 0);

    std::vector<ValueT> h_x(n);
    ASSERT_EQ(cudaMemcpyAsync(h_x.data(), d_x, sizeof(ValueT) * n, cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    double err = max_abs_diff(h_x, x_ref);
    double tol = std::is_same_v<ValueT, float> ? 1e-5 : 1e-12;
    ASSERT_LT(err, tol);

    ASSERT_EQ(cudaFree(d_x), cudaSuccess);
    ASSERT_EQ(cudaFree(d_b), cudaSuccess);
    ASSERT_EQ(cudaFree(d_val), cudaSuccess);
    ASSERT_EQ(cudaFree(d_colInd), cudaSuccess);
    ASSERT_EQ(cudaFree(d_rowPtr), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

/*
Test: numerical safeguard triggers when diagonal is zero and unit_diag=false.
Expect return code -2.
*/
template <typename ValueT>
static void run_test_nan_inf_detection()
{
    if constexpr (!sptrsv_test_supported_v<ValueT>)
    {
        GTEST_SKIP() << "SpTRSV levelsets tests currently support float/double only.";
    }
    else
    {
        skip_if_no_cuda_device();
        using IndexT = int;

    int n = 2;
    std::vector<int> h_rowPtr = {0, 1, 3};
    std::vector<int> h_colInd = {0, 0, 1};
    std::vector<ValueT> h_val = {ValueT(0), ValueT(1), ValueT(1)};
    std::vector<ValueT> h_b = {ValueT(1), ValueT(2)};

    cudaStream_t stream;
    cudaError_t stream_err = cudaStreamCreate(&stream);
    if (stream_err != cudaSuccess)
        GTEST_SKIP() << "CUDA stream creation failed: " << cudaGetErrorString(stream_err);

    IndexT *d_rowPtr = nullptr, *d_colInd = nullptr;
    ValueT *d_val = nullptr, *d_b = nullptr, *d_x = nullptr;

    ASSERT_EQ(cudaMalloc((void **)&d_rowPtr, sizeof(IndexT) * (n + 1)), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_colInd, sizeof(IndexT) * h_colInd.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_val, sizeof(ValueT) * h_val.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_b, sizeof(ValueT) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_x, sizeof(ValueT) * n), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_rowPtr, h_rowPtr.data(), sizeof(IndexT) * (n + 1), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_colInd, h_colInd.data(), sizeof(IndexT) * h_colInd.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_val, h_val.data(), sizeof(ValueT) * h_val.size(), cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_b, h_b.data(), sizeof(ValueT) * n, cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ichol::symbolic::LevelSets ls = make_levelsets_sequential(n);
    int rc = SpTRSV_solve_levelsets<IndexT, ValueT>(
        n, d_rowPtr, d_colInd, d_val, d_b, d_x,
        FillMode::LOWER, /*unit_diag=*/false,
        ls, stream);
    ASSERT_EQ(rc, -2);

    ASSERT_EQ(cudaFree(d_x), cudaSuccess);
    ASSERT_EQ(cudaFree(d_b), cudaSuccess);
    ASSERT_EQ(cudaFree(d_val), cudaSuccess);
    ASSERT_EQ(cudaFree(d_colInd), cudaSuccess);
    ASSERT_EQ(cudaFree(d_rowPtr), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

TEST(SpTRSVLevelSets, LowerNonTranspose_Float)
{
    run_test_lower_nontranspose<float>();
}

// TEST(SpTRSVLevelSets, UnitDiagAbsent_Float)
// {
//     run_test_unit_diag_absent<float>();
// }

TEST(SpTRSVLevelSets, NanInfDetection_Float)
{
    run_test_nan_inf_detection<float>();
}

TEST(SpTRSVLevelSets, LowerNonTranspose_Double)
{
    run_test_lower_nontranspose<double>();
}

TEST(SpTRSVLevelSets, LowerNonTranspose_Half)
{
    run_test_lower_nontranspose<__half>();
}

// TEST(SpTRSVLevelSets, UnitDiagAbsent_Double)
// {
//     run_test_unit_diag_absent<double>();
// }
//
// TEST(SpTRSVLevelSets, UnitDiagAbsent_Half)
// {
//     run_test_unit_diag_absent<__half>();
// }

TEST(SpTRSVLevelSets, NanInfDetection_Double)
{
    run_test_nan_inf_detection<double>();
}

TEST(SpTRSVLevelSets, NanInfDetection_Half)
{
    run_test_nan_inf_detection<__half>();
}
