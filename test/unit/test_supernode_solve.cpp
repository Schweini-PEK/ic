#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/supernodal_solve.hpp"
#include "factor/supernodal_storage.hpp"
#include "test_utils.hpp"

namespace
{
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

TEST(SupernodeSolve, ForwardAndBackwardMatchScalarAndKnownSolution)
{
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
    const auto plan = ichol::symbolic::ll_plan_from_sym(sym, n);
    const auto packed = ichol::supernodal::pack_supernode_values_from_csr_l<double>(n, row_ptr, col_ind, values, sym);

    const auto b_forward = apply_lower_csr_diag_last(row_ptr, col_ind, values, x_true);
    const auto x_forward_scalar = ichol::supernodal::solve_lower_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_forward);
    const auto x_forward_super = ichol::supernodal::solve_lower_supernodal(sym, packed, b_forward);
    const auto x_forward_bucketed = ichol::supernodal::solve_lower_supernodal_bucketed(plan, packed, b_forward);

    expect_vector_near(x_forward_scalar, x_true, 1e-12, "forward scalar vs truth");
    expect_vector_near(x_forward_super, x_true, 1e-12, "forward super vs truth");
    expect_vector_near(x_forward_super, x_forward_scalar, 1e-12, "forward super vs scalar");
    expect_vector_near(x_forward_bucketed, x_true, 1e-12, "forward bucketed vs truth");
    expect_vector_near(x_forward_bucketed, x_forward_super, 1e-12, "forward bucketed vs serial super");

    const auto b_backward = apply_lower_transpose_csr(row_ptr, col_ind, values, x_true);
    const auto x_backward_scalar = ichol::supernodal::solve_lower_transpose_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_backward);
    const auto x_backward_super = ichol::supernodal::solve_lower_transpose_supernodal(sym, packed, b_backward);
    const auto x_backward_bucketed = ichol::supernodal::solve_lower_transpose_supernodal_bucketed(plan, packed, b_backward);

    expect_vector_near(x_backward_scalar, x_true, 1e-12, "backward scalar vs truth");
    expect_vector_near(x_backward_super, x_true, 1e-12, "backward super vs truth");
    expect_vector_near(x_backward_super, x_backward_scalar, 1e-12, "backward super vs scalar");
    expect_vector_near(x_backward_bucketed, x_true, 1e-12, "backward bucketed vs truth");
    expect_vector_near(x_backward_bucketed, x_backward_super, 1e-12, "backward bucketed vs serial super");
}

TEST(SupernodeSolve, FloatBucketedForwardAndBackwardMatchScalar)
{
    const std::vector<int> row_ptr = {0, 1, 3, 6, 10, 12};
    const std::vector<int> col_ind = {
        0,
        0, 1,
        0, 1, 2,
        0, 1, 2, 3,
        3, 4};
    const std::vector<float> values = {
        10.0f,
        20.0f, 21.0f,
        30.0f, 31.0f, 32.0f,
        40.0f, 41.0f, 42.0f, 43.0f,
        50.0f, 51.0f};
    const std::vector<float> x_true = {1.25f, -0.5f, 2.0f, -1.0f, 0.75f};

    const int n = static_cast<int>(row_ptr.size()) - 1;
    const auto super = ichol::supernodal::detect_supernode_boundaries_from_csr_l(n, row_ptr, col_ind);
    const auto sym = ichol::supernodal::build_super_sym_from_csr_l(n, row_ptr, col_ind, super);
    const auto plan = ichol::symbolic::ll_plan_from_sym(sym, n);
    const auto packed = ichol::supernodal::pack_supernode_values_from_csr_l<float>(n, row_ptr, col_ind, values, sym);

    const auto b_forward = apply_lower_csr_diag_last(row_ptr, col_ind, values, x_true);
    const auto x_forward_scalar = ichol::supernodal::solve_lower_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_forward);
    const auto x_forward_bucketed = ichol::supernodal::solve_lower_supernodal_bucketed(plan, packed, b_forward);

    expect_vector_near(x_forward_bucketed, x_forward_scalar, 1e-5, "float forward bucketed vs scalar");

    const auto b_backward = apply_lower_transpose_csr(row_ptr, col_ind, values, x_true);
    const auto x_backward_scalar = ichol::supernodal::solve_lower_transpose_scalar_csr_diag_last(
        n, row_ptr, col_ind, values, b_backward);
    const auto x_backward_bucketed = ichol::supernodal::solve_lower_transpose_supernodal_bucketed(plan, packed, b_backward);

    expect_vector_near(x_backward_bucketed, x_backward_scalar, 1e-5, "float backward bucketed vs scalar");
}
