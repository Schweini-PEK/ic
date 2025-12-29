// // test_supernodal.cpp
//
#include <gtest/gtest.h>
//
// // #include "factor/symbolic/symbolic.hpp"
// // #include "ichol/matrix_formats.hpp"
// // #include "factor/symbolic/symbolic.hpp"
// // namespace test_checks
// // {
// //     // (No additional checks needed here for supernodal tests yet)
// // } // namespace test_checks
// //
// // TEST(Supernodal, PlaceholderTest)
// // {
// //     std::string path = "test/data/nasa2146.mtx";
// //
// //     ASSERT_TRUE(true);
// // }
//
// #include <algorithm>
//
#include <vector>

#include <utility>

#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/detail/symbolic_plan.hpp"
template <typename T>
static ichol::matrix::CscMatrix<T> vector_to_csc(
    int n,
    std::initializer_list<std::tuple<int, int, T>> undirected_edges,
    T diag_val = T(10))
{
    std::vector<std::vector<std::pair<int, T>>> cols(n);
    for (int i = 0; i < n; ++i)
        cols[i].push_back({i, diag_val});

    for (const auto &e : undirected_edges)
    {
        int a, b;
        T v;
        std::tie(a, b, v) = e;
        if (a < 0 || a >= n || b < 0 || b >= n)
            continue;

        if (a >= b)
            cols[b].push_back({a, v});
        else
            cols[a].push_back({b, v});
    }

    for (int j = 0; j < n; ++j)
    {
        auto &c = cols[j];
        std::sort(c.begin(), c.end(),
                  [](const auto &x, const auto &y) { return x.first < y.first; });
        c.erase(std::unique(c.begin(), c.end(),
                            [](const auto &x, const auto &y)
                            { return x.first == y.first; }),
                c.end());
    }

    ichol::matrix::CscMatrix<T> A;
    A.num_rows = n;
    A.num_cols = n;

    A.col_ptr.resize(n + 1, 0);
    for (int j = 0; j < n; ++j)
        A.col_ptr[j + 1] = A.col_ptr[j] + (int)cols[j].size();

    A.nnz = A.col_ptr[n];
    A.row_ind.reserve(A.nnz);
    A.values.reserve(A.nnz);

    for (int j = 0; j < n; ++j)
        for (auto &p : cols[j])
        {
            A.row_ind.push_back(p.first);
            A.values.push_back(p.second);
        }

    return A;
}
TEST(SupernodalCSC, ConservativeDetectionAndLevels)
{
    using T = double;
    const int n = 5;

    // Chain: 0-1-2-3-4
    auto A = vector_to_csc<T>(
        n,
        {
            {0,1,-1},
            {1,2,-1},
            {2,3,-1},
            {3,4,-1},
        });

    auto etree = ichol::symbolic::build_etree(A);
    auto fp    = ichol::symbolic::compute_complete_cholesky_pattern(A, etree);

    auto col_ls = ichol::symbolic::build_level_sets(fp, ichol::SymbolicOptions{});

    auto snodes = ichol::symbolic::detect_supernodes(fp, etree);
    auto sn_res = ichol::symbolic::build_snode_level_sets(col_ls, snodes);

    const auto &snode_level = sn_res.snode_level;
    const auto &snode_ls    = sn_res.level_sets;

    ASSERT_FALSE(snodes.empty());
    ASSERT_EQ((int)snode_level.size(), (int)snodes.size());

    // Each column must belong to exactly one supernode
    std::vector<int> col_count(n, 0);
    for (auto &pr : snodes)
        for (int c = pr.first; c < pr.second; ++c)
            col_count[c]++;

    for (int c = 0; c < n; ++c)
        EXPECT_EQ(col_count[c], 1);

    // snode level must be >= level of all columns it covers
    std::vector<int> col_level(n, -1);
    for (int L = 0; L + 1 < (int)col_ls.level_ptr.size(); ++L)
        for (int p = col_ls.level_ptr[L]; p < col_ls.level_ptr[L+1]; ++p)
            col_level[col_ls.levels[p]] = L;

    for (size_t id = 0; id < snodes.size(); ++id)
    {
        int s = snodes[id].first;
        int e = snodes[id].second;
        int lv = snode_level[id];
        for (int c = s; c < e; ++c)
            EXPECT_GE(lv, col_level[c]);
    }

    // LevelSets sanity
    EXPECT_EQ((int)snode_ls.levels.size(), (int)snodes.size());
    EXPECT_EQ(snode_ls.level_ptr.back(), (int)snodes.size());
}


TEST(SupernodalCSC, ApproximateEqualsConservativeAtOne)
{
    using T = double;
    const int n = 6;

    auto A = vector_to_csc<T>(
        n,
        {
            {0,1,-1},
            {0,2,-1},
            {1,2,-1},
            {2,3,-1},
            {3,4,-1},
            {4,5,-1},
        });

    auto etree = ichol::symbolic::build_etree(A);
    auto fp    = ichol::symbolic::compute_complete_cholesky_pattern(A, etree);

    auto sn1 = ichol::symbolic::detect_supernodes(fp, etree);
    auto sn2 = ichol::symbolic::detect_supernodes_approx(fp, etree, 1.0);

    ASSERT_EQ(sn1.size(), sn2.size());
    for (size_t i = 0; i < sn1.size(); ++i)
    {
        EXPECT_EQ(sn1[i].first,  sn2[i].first);
        EXPECT_EQ(sn1[i].second, sn2[i].second);
    }
}
