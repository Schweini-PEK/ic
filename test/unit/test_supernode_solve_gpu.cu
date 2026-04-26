#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "factor/supernodal_solve.hpp"
#include "factor/supernodal_solve_cuda.cuh"
#include "factor/supernodal_storage.hpp"
#include "test_utils.hpp"

namespace
{
    bool has_cuda_device()
    {
        int count = 0;
        const cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        return count > 0;
    }

    template <typename ValueT>
    std::vector<ValueT> apply_lower_csr_diag_last(
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<ValueT> &values,
        const std::vector<ValueT> &x)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        std::vector<ValueT> b((size_t)n, ValueT{});
        for (int i = 0; i < n; ++i)
        {
            ValueT sum = ValueT{};
            for (int p = row_ptr[(size_t)i]; p < row_ptr[(size_t)i + 1]; ++p)
                sum += values[(size_t)p] * x[(size_t)col_ind[(size_t)p]];
            b[(size_t)i] = sum;
        }
        return b;
    }

    template <typename ValueT>
    std::vector<ValueT> apply_lower_transpose_csr(
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<ValueT> &values,
        const std::vector<ValueT> &x)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        std::vector<ValueT> b((size_t)n, ValueT{});
        for (int i = 0; i < n; ++i)
        {
            const ValueT xi = x[(size_t)i];
            for (int p = row_ptr[(size_t)i]; p < row_ptr[(size_t)i + 1]; ++p)
                b[(size_t)col_ind[(size_t)p]] += values[(size_t)p] * xi;
        }
        return b;
    }

    template <typename ValueT>
    void expect_vector_near(
        const std::vector<ValueT> &actual,
        const std::vector<ValueT> &expected,
        double tol,
        const char *label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label << " size mismatch";
        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(static_cast<double>(actual[i]), static_cast<double>(expected[i]), tol)
                << label << " mismatch at index " << i;
        }
    }
} // namespace

TEST(SupernodeSolveGpu, DoubleDensePanelsMatchCpuReference)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device is not available";

    const std::vector<int> row_ptr = {0, 1, 3, 6, 10, 12};
    const std::vector<int> col_ind = {
        0,
        0, 1,
        0, 1, 2,
        0, 1, 2, 3,
        3, 4};
    const std::vector<double> values = {
        10.0,
        20.0, 21.0,
        30.0, 31.0, 32.0,
        40.0, 41.0, 42.0, 43.0,
        50.0, 51.0};
    const std::vector<double> x_true = {1.25, -0.5, 2.0, -1.0, 0.75};

    ichol::testutil::assert_lower_only_csr(row_ptr, col_ind);
    ichol::testutil::assert_diag_last_csr(row_ptr, col_ind);
    ichol::testutil::assert_cols_sorted_unique(row_ptr, col_ind);
    ichol::testutil::assert_diag_positive_csr(row_ptr, col_ind, values);

    const int n = static_cast<int>(row_ptr.size()) - 1;
    const auto super = ichol::supernodal::detect_supernode_boundaries_from_csr_l(n, row_ptr, col_ind);
    const auto sym = ichol::supernodal::build_super_sym_from_csr_l(n, row_ptr, col_ind, super);
    const auto buckets = ichol::supernodal::build_forward_solve_buckets_from_sym(sym);
    const auto packed = ichol::supernodal::pack_supernode_values_from_csr_l<double>(n, row_ptr, col_ind, values, sym);

    const auto b_forward = apply_lower_csr_diag_last(row_ptr, col_ind, values, x_true);
    const auto x_forward_scalar = ichol::supernodal::solve_lower_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_forward);
    const auto x_forward_gpu = ichol::supernodal::cuda_reference::solve_lower(
        sym, packed, buckets, b_forward);

    expect_vector_near(x_forward_gpu, x_forward_scalar, 1e-12, "double GPU forward vs scalar");
    expect_vector_near(x_forward_gpu, x_true, 1e-12, "double GPU forward vs truth");

    const auto b_backward = apply_lower_transpose_csr(row_ptr, col_ind, values, x_true);
    const auto x_backward_scalar = ichol::supernodal::solve_lower_transpose_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_backward);
    const auto x_backward_gpu = ichol::supernodal::cuda_reference::solve_lower_transpose(
        sym, packed, buckets, b_backward);

    expect_vector_near(x_backward_gpu, x_backward_scalar, 1e-12, "double GPU backward vs scalar");
    expect_vector_near(x_backward_gpu, x_true, 1e-12, "double GPU backward vs truth");
}

TEST(SupernodeSolveGpu, FloatForwardHandlesSharedUpdateRowWithAtomicAdd)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device is not available";

    const std::vector<int> row_ptr = {0, 1, 2, 5};
    const std::vector<int> col_ind = {
        0,
        1,
        0, 1, 2};
    const std::vector<float> values = {
        2.0f,
        3.0f,
        5.0f, 7.0f, 11.0f};
    const std::vector<double> values_double(values.begin(), values.end());
    const std::vector<float> x_true = {1.0f, -2.0f, 0.5f};

    ichol::testutil::assert_lower_only_csr(row_ptr, col_ind);
    ichol::testutil::assert_diag_last_csr(row_ptr, col_ind);
    ichol::testutil::assert_cols_sorted_unique(row_ptr, col_ind);
    ichol::testutil::assert_diag_positive_csr(row_ptr, col_ind, values_double);

    const int n = static_cast<int>(row_ptr.size()) - 1;
    const std::vector<int> singleton_super = {0, 1, 2, 3};
    const auto sym = ichol::supernodal::build_super_sym_from_csr_l(n, row_ptr, col_ind, singleton_super);
    const auto buckets = ichol::supernodal::build_forward_solve_buckets_from_sym(sym);
    ASSERT_EQ(buckets.size(), 2u);
    ASSERT_EQ(buckets[0].size(), 2u);

    const auto packed = ichol::supernodal::pack_supernode_values_from_csr_l<float>(n, row_ptr, col_ind, values, sym);

    const auto b_forward = apply_lower_csr_diag_last(row_ptr, col_ind, values, x_true);
    const auto x_forward_scalar = ichol::supernodal::solve_lower_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_forward);
    const auto x_forward_gpu = ichol::supernodal::cuda_reference::solve_lower(
        sym, packed, buckets, b_forward);

    expect_vector_near(x_forward_gpu, x_forward_scalar, 1e-5, "float GPU forward vs scalar");
    expect_vector_near(x_forward_gpu, x_true, 1e-5, "float GPU forward vs truth");

    const auto b_backward = apply_lower_transpose_csr(row_ptr, col_ind, values, x_true);
    const auto x_backward_scalar = ichol::supernodal::solve_lower_transpose_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_backward);
    const auto x_backward_gpu = ichol::supernodal::cuda_reference::solve_lower_transpose(
        sym, packed, buckets, b_backward);

    expect_vector_near(x_backward_gpu, x_backward_scalar, 1e-5, "float GPU backward vs scalar");
    expect_vector_near(x_backward_gpu, x_true, 1e-5, "float GPU backward vs truth");
}
