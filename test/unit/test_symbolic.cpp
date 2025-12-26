#include <gtest/gtest.h>

#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/detail/symbolic_plan.hpp"

template <typename T>
static ichol::CsrMatrix<T> vector_to_csr(
    int n,
    std::initializer_list<std::tuple<int, int, T>> undirected_edges,
    T diag_val = T(10))
{
    std::vector<std::vector<std::pair<int, T>>> rows(n);
    for (int i = 0; i < n; ++i)
        rows[i].push_back({i, diag_val});

    for (const auto &e : undirected_edges)
    {
        int a, b;
        T v;
        std::tie(a, b, v) = e;
        if (a < 0 || a >= n || b < 0 || b >= n)
            continue;
        // store in lower-triangular convention: row = max(a,b), col = min(a,b)
        if (a <= b)
            rows[b].push_back({a, v});
        else
            rows[a].push_back({b, v});
    }

    for (int i = 0; i < n; ++i)
    {
        auto &r = rows[i];
        std::sort(r.begin(), r.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        r.erase(std::unique(r.begin(), r.end(),
                            [](const auto &a, const auto &b)
                            { return a.first == b.first; }),
                r.end());
    }

    ichol::CsrMatrix<T> A;
    A.num_rows = n;
    A.num_cols = n;

    A.row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i)
        A.row_ptr[i + 1] = A.row_ptr[i] + static_cast<int>(rows[i].size());

    A.nnz = A.row_ptr[n];
    A.col_ind.reserve(A.nnz);
    A.values.reserve(A.nnz);

    for (int i = 0; i < n; ++i)
    {
        for (const auto &p : rows[i])
        {
            A.col_ind.push_back(p.first);
            A.values.push_back(p.second);
        }
    }

    return A;
}

static void ExpectValidPattern(const ichol::symbolic::FactorPattern &fp, int n)
{
    ASSERT_EQ((int)fp.row_ptr_L.size(), n + 1);
    ASSERT_FALSE(fp.row_ptr_L.empty());
    EXPECT_EQ(fp.row_ptr_L[0], 0);
    EXPECT_EQ(fp.row_ptr_L.back(), (int)fp.col_ind_L.size());

    for (int i = 0; i < n; ++i)
    {
        int b = fp.row_ptr_L[i];
        int e = fp.row_ptr_L[i + 1];
        ASSERT_LE(b, e);
        ASSERT_LE(e, (int)fp.col_ind_L.size());

        bool has_diag = false;
        int prev = -1;
        for (int p = b; p < e; ++p)
        {
            int c = fp.col_ind_L[p];
            EXPECT_LT(prev, c); // strictly increasing
            EXPECT_LE(c, i);    // lower-triangular convention
            if (c == i)
                has_diag = true;
            prev = c;
        }
        EXPECT_TRUE(has_diag);
    }
}

static void ExpectSubsetPerRow(const ichol::symbolic::FactorPattern &a,
                               const ichol::symbolic::FactorPattern &b,
                               int n)
{
    // require sorted rows; then linear subset check
    for (int i = 0; i < n; ++i)
    {
        int ab = a.row_ptr_L[i], ae = a.row_ptr_L[i + 1];
        int bb = b.row_ptr_L[i], be = b.row_ptr_L[i + 1];

        int p = ab, q = bb;
        while (p < ae && q < be)
        {
            int x = a.col_ind_L[p], y = b.col_ind_L[q];
            if (x == y)
            {
                ++p;
                ++q;
            }
            else if (x > y)
            {
                ++q;
            }
            else
            {
                FAIL() << "Row " << i << " missing col " << x;
            }
        }
        if (p != ae)
            FAIL() << "Row " << i << " missing trailing cols";
    }
}

TEST(ICSymbolic, InvariantsAndMonotonicity)
{
    using T = double;
    const int n = 4;

    auto A = vector_to_csr<T>(n, {{0, 1, T(-1)}, {0, 2, T(-1)}, {1, 3, T(-1)}, {2, 3, T(-1)}});

    auto fp0 = ichol::symbolic::compute_ic_factor_pattern(A, 0);
    auto fp1 = ichol::symbolic::compute_ic_factor_pattern(A, 1);
    auto fp2 = ichol::symbolic::compute_ic_factor_pattern(A, 2);

    ExpectValidPattern(fp0, n);
    ExpectValidPattern(fp1, n);
    ExpectValidPattern(fp2, n);

    ExpectSubsetPerRow(fp0, fp1, n);
    ExpectSubsetPerRow(fp1, fp2, n);
}
