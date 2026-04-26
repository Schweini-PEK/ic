#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "factor/supernodal_storage.hpp"
#include "test_utils.hpp"

namespace
{
    using Entry = std::tuple<int, int, double>;

    std::vector<Entry> csr_entries(
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<double> &values)
    {
        std::vector<Entry> out;
        const int n = static_cast<int>(row_ptr.size()) - 1;
        out.reserve(values.size());

        for (int i = 0; i < n; ++i)
        {
            for (int p = row_ptr[(size_t)i]; p < row_ptr[(size_t)i + 1]; ++p)
                out.emplace_back(i, col_ind[(size_t)p], values[(size_t)p]);
        }
        return out;
    }

    std::vector<Entry> coo_entries(const ichol::matrix::CooMatrix<double> &coo)
    {
        std::vector<Entry> out;
        out.reserve(coo.values.size());
        for (size_t k = 0; k < coo.values.size(); ++k)
            out.emplace_back(coo.row_ind[k], coo.col_ind[k], coo.values[k]);
        return out;
    }

    void sort_entries(std::vector<Entry> &entries)
    {
        std::sort(entries.begin(), entries.end(),
                  [](const Entry &a, const Entry &b)
                  {
                      if (std::get<0>(a) != std::get<0>(b))
                          return std::get<0>(a) < std::get<0>(b);
                      if (std::get<1>(a) != std::get<1>(b))
                          return std::get<1>(a) < std::get<1>(b);
                      return std::get<2>(a) < std::get<2>(b);
                  });
    }
} // namespace

TEST(SupernodeStorage, PackAndExpandRoundTripMatchesOriginalCsr)
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

    ichol::testutil::assert_lower_only_csr(row_ptr, col_ind);
    ichol::testutil::assert_diag_last_csr(row_ptr, col_ind);
    ichol::testutil::assert_cols_sorted_unique(row_ptr, col_ind);

    const int n = static_cast<int>(row_ptr.size()) - 1;
    const auto super = ichol::supernodal::detect_supernode_boundaries_from_csr_l(n, row_ptr, col_ind);
    const std::vector<int> expected_super = {0, 3, 5};
    EXPECT_EQ(super, expected_super);

    const auto sym = ichol::supernodal::build_super_sym_from_csr_l(n, row_ptr, col_ind, super);
    const std::vector<int> expected_pi = {0, 4, 6};
    const std::vector<int> expected_px = {0, 12, 16};
    const std::vector<int> expected_s = {0, 1, 2, 3, 3, 4};
    EXPECT_EQ(sym.pi, expected_pi);
    EXPECT_EQ(sym.px, expected_px);
    EXPECT_EQ(sym.s, expected_s);

    const auto packed = ichol::supernodal::pack_supernode_values_from_csr_l<double>(n, row_ptr, col_ind, values, sym);
    ASSERT_EQ(packed.size(), 16u);
    EXPECT_DOUBLE_EQ(packed[0], 10.0);
    EXPECT_DOUBLE_EQ(packed[1], 20.0);
    EXPECT_DOUBLE_EQ(packed[2], 30.0);
    EXPECT_DOUBLE_EQ(packed[3], 40.0);
    EXPECT_DOUBLE_EQ(packed[4], 0.0);
    EXPECT_DOUBLE_EQ(packed[5], 21.0);
    EXPECT_DOUBLE_EQ(packed[6], 31.0);
    EXPECT_DOUBLE_EQ(packed[7], 41.0);
    EXPECT_DOUBLE_EQ(packed[8], 0.0);
    EXPECT_DOUBLE_EQ(packed[9], 0.0);
    EXPECT_DOUBLE_EQ(packed[10], 32.0);
    EXPECT_DOUBLE_EQ(packed[11], 42.0);
    EXPECT_DOUBLE_EQ(packed[12], 43.0);
    EXPECT_DOUBLE_EQ(packed[13], 50.0);
    EXPECT_DOUBLE_EQ(packed[14], 0.0);
    EXPECT_DOUBLE_EQ(packed[15], 51.0);

    const auto expanded = ichol::supernodal::expand_supernode_values_to_coo(sym, packed);
    ASSERT_EQ(expanded.nnz, static_cast<int>(values.size()));

    auto expected = csr_entries(row_ptr, col_ind, values);
    auto actual = coo_entries(expanded);
    sort_entries(expected);
    sort_entries(actual);

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(std::get<0>(actual[i]), std::get<0>(expected[i])) << "row mismatch at entry " << i;
        EXPECT_EQ(std::get<1>(actual[i]), std::get<1>(expected[i])) << "col mismatch at entry " << i;
        EXPECT_DOUBLE_EQ(std::get<2>(actual[i]), std::get<2>(expected[i])) << "value mismatch at entry " << i;
    }
}
