// etree.cpp
#include <cassert>
#include <vector>
#include <algorithm>

#include "symbolic.hpp"

namespace ichol::symbolic
{
    //------------------------------------------------------------------------------
    // update_etree: identical to CHOLMOD/Cholesky/cholmod_etree.c (symmetric case)
    //------------------------------------------------------------------------------
    static inline void update_etree(
        int k,
        int i,
        std::vector<int> &Parent,
        std::vector<int> &Ancestor)
    {
        for (;;)
        {
            const int a = Ancestor[k];
            if (a == i)
            {
                return;
            }
            Ancestor[k] = i;
            if (a == -1)
            {
                Parent[k] = i;
                return;
            }
            k = a;
        }
    }

    static inline void sort_unique(std::vector<int> &v)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    //------------------------------------------------------------------------------
    // Build strict upper adjacency U of A: U[j] = { i | i < j and A(i,j) != 0 }.
    // This matches CHOLMOD(etree) symmetric case, which uses ONLY triu(A).
    //------------------------------------------------------------------------------
    template <class T>
    static std::vector<std::vector<int>> build_strict_upper_adjacency(
        const ichol::matrix::CscMatrix<T> &A)
    {
        const int n = A.num_cols;
        std::vector<std::vector<int>> U(n);

        for (int j = 0; j < n; ++j)
        {
            for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p)
            {
                const int i = A.row_ind[p];
                if (i == j) continue;

                // 关键：不管输入是上三角还是下三角，一律映射到上三角 (u<v)
                const int u = (i < j) ? i : j;
                const int v = (i < j) ? j : i;

                U[v].push_back(u);
            }
        }

        for (int j = 0; j < n; ++j)
            sort_unique(U[j]);

        return U;
    }

    template <class T>
    static std::vector<std::vector<int>> build_strict_upper_adjacency(
        const ichol::matrix::CsrMatrix<T> &A)
    {
        const int n = A.num_rows;
        std::vector<std::vector<int>> U(n);
        for (int r = 0; r < n; ++r)
        {
            for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p)
            {
                const int c = A.col_ind[p];
                if (r < c)
                {
                    U[c].push_back(r);
                }
            }
        }
        for (int j = 0; j < n; ++j)
            sort_unique(U[j]);
        return U;
    }

    //------------------------------------------------------------------------------
    // etree from strict upper adjacency (CHOLMOD(etree), symmetric case)
    //------------------------------------------------------------------------------
    static std::vector<int> cholmod_etree_symmetric_upper(
        const std::vector<std::vector<int>> &U)
    {
        const int n = (int)U.size();
        std::vector<int> Parent(n, -1);
        std::vector<int> Ancestor(n, -1);
        for (int j = 0; j < n; ++j)
        {
            for (int i : U[j])
            {
                // i < j
                update_etree(i, j, Parent, Ancestor);
            }
        }
        return Parent;
    }

    //------------------------------------------------------------------------------
    // postorder of a forest/tree (CHOLMOD(postorder) with Weight == NULL)
    //------------------------------------------------------------------------------
    static std::vector<int> cholmod_postorder(const std::vector<int> &Parent)
    {
        const int n = (int)Parent.size();
        std::vector<int> Head(n, -1);
        std::vector<int> Next(n, -1);

        // build child lists, reverse order so children in ascending order
        for (int j = n - 1; j >= 0; --j)
        {
            const int p = Parent[j];
            if (p >= 0 && p < n)
            {
                Next[j] = Head[p];
                Head[p] = j;
            }
        }

        std::vector<int> Post;
        Post.reserve(n);
        std::vector<int> Pstack(n);

        auto dfs = [&](int root)
        {
            int phead = 0;
            Pstack[0] = root;
            while (phead >= 0)
            {
                const int p = Pstack[phead];
                const int j = Head[p];
                if (j == -1)
                {
                    // done with p
                    --phead;
                    Post.push_back(p);
                }
                else
                {
                    // visit child j
                    Head[p] = Next[j];
                    Pstack[++phead] = j;
                }
            }
        };

        // roots in ascending order
        for (int j = 0; j < n; ++j)
        {
            if (Parent[j] == -1)
                dfs(j);
        }
        assert((int)Post.size() == n);
        return Post;
    }

    //------------------------------------------------------------------------------
    // Build strict lower CSC (transpose of strict upper):
    //   Lcol[j] = { i | i > j and A(j,i) is in strict upper }
    // This matches cholmod_rowcolcounts symmetric input convention.
    //------------------------------------------------------------------------------
    static void build_strict_lower_csc_from_upper(
        const std::vector<std::vector<int>> &U,
        std::vector<int> &Ap,
        std::vector<int> &Ai)
    {
        const int n = (int)U.size();
        std::vector<int> col_nnz(n, 0);
        for (int c = 0; c < n; ++c)
        {
            for (int r : U[c])
            {
                // upper edge (r,c) => lower edge in column r with row c
                col_nnz[r]++;
            }
        }

        Ap.assign(n + 1, 0);
        for (int j = 0; j < n; ++j)
            Ap[j + 1] = Ap[j] + col_nnz[j];
        Ai.assign(Ap[n], -1);

        std::vector<int> next = Ap;
        for (int c = 0; c < n; ++c)
        {
            for (int r : U[c])
            {
                Ai[next[r]++] = c; // c > r
            }
        }

        // Optional: sort each column for determinism (CHOLMOD does not require)
        for (int j = 0; j < n; ++j)
        {
            const int p0 = Ap[j];
            const int p1 = Ap[j + 1];
            std::sort(Ai.begin() + p0, Ai.begin() + p1);
        }
    }

    //------------------------------------------------------------------------------
    // CHOLMOD(rowcolcounts) symmetric case (only ColCount is returned)
    //   - Parent and Post must correspond to etree(A) where A uses triu(A)
    //   - The matrix passed here is the TRANSPOSE representation: strictly-lower
    //     of A (i>j entries in each column j).
    //------------------------------------------------------------------------------
    static std::vector<int> cholmod_colcount_symmetric(
        const std::vector<int> &Ap,
        const std::vector<int> &Ai,
        const std::vector<int> &Parent,
        const std::vector<int> &Post)
    {
        const int n = (int)Parent.size();
        assert((int)Post.size() == n);
        assert((int)Ap.size() == n + 1);

        std::vector<int> ColCount(n, 0);
        std::vector<int> First(n, -1);
        std::vector<int> Level(n, 0);
        std::vector<int> PrevNbr(n, -1);
        std::vector<int> PrevLeaf(n, -1);
        std::vector<int> SetParent(n, 0);

        // find First and Level; initialize ColCount leaf weights
        for (int k = 0; k < n; ++k)
        {
            const int i = Post[k];
            ColCount[i] = (First[i] == -1) ? 1 : 0;

            int len = 0;
            int r;
            for (r = i; (r != -1) && (First[r] == -1); r = Parent[r])
            {
                First[r] = k;
                len++;
            }
            if (r == -1)
            {
                len--;
            }
            else
            {
                len += Level[r];
            }
            for (int s = i; s != r; s = Parent[s])
            {
                Level[s] = len--;
            }
        }

        for (int i = 0; i < n; ++i)
        {
            PrevLeaf[i] = -1;
            PrevNbr[i] = -1;
            SetParent[i] = i;
        }

        auto initialize_node = [&](int k) -> int
        {
            const int p = Post[k];
            const int parent = Parent[p];
            if (parent != -1)
            {
                ColCount[parent]--;
            }
            PrevNbr[p] = k;
            return p;
        };

        auto find_set = [&](int x) -> int
        {
            int q = x;
            while (q != SetParent[q])
                q = SetParent[q];
            // path compression
            while (x != q)
            {
                int xp = SetParent[x];
                SetParent[x] = q;
                x = xp;
            }
            return q;
        };

        auto process_edge = [&](int p, int u, int k)
        {
            // if First[p] > PrevNbr[u], then p is a leaf of subtree of u
            if (First[p] > PrevNbr[u])
            {
                ColCount[p]++;
                const int prevleaf = PrevLeaf[u];
                if (prevleaf != -1)
                {
                    const int q = find_set(prevleaf);
                    ColCount[q]--;
                }
                PrevLeaf[u] = p;
            }
            PrevNbr[u] = k;
        };

        auto finalize_node = [&](int p)
        {
            const int parent = Parent[p];
            if (parent != -1)
                SetParent[p] = parent;
        };

        // symmetric case main loop
        for (int k = 0; k < n; ++k)
        {
            const int j = initialize_node(k);
            for (int p = Ap[j]; p < Ap[j + 1]; ++p)
            {
                const int i = Ai[p];
                if (i > j)
                {
                    process_edge(j, i, k);
                }
            }
            finalize_node(j);
        }

        // finish ColCount: accumulate to parents
        for (int j = 0; j < n; ++j)
        {
            const int parent = Parent[j];
            if (parent != -1)
                ColCount[parent] += ColCount[j];
        }

        return ColCount;
    }

    //------------------------------------------------------------------------------
    // Public API: build_etree for CSC / CSR
    // Computes:
    //   - etree.parent: CHOLMOD(etree) symmetric semantics (uses strict upper)
    //   - etree.colcount: CHOLMOD(rowcolcounts) symmetric semantics
    //------------------------------------------------------------------------------
    template <class T>
    ETree build_etree(const ichol::matrix::CscMatrix<T> &A)
    {
        const int n = A.num_cols;
        ETree e;
        e.parent.assign(n, -1);
        e.colcount.assign(n, 1);

        auto U = build_strict_upper_adjacency(A);
        e.parent = cholmod_etree_symmetric_upper(U);
        const auto Post = cholmod_postorder(e.parent);

        std::vector<int> Ap, Ai;
        build_strict_lower_csc_from_upper(U, Ap, Ai);
        e.colcount = cholmod_colcount_symmetric(Ap, Ai, e.parent, Post);
        return e;
    }

    template <typename T>
    ETree build_etree(const ichol::matrix::CsrMatrix<T> &A)
    {
        const int n = A.num_rows;
        ETree e;
        e.parent.assign(n, -1);
        e.colcount.assign(n, 1);

        auto U = build_strict_upper_adjacency(A);
        e.parent = cholmod_etree_symmetric_upper(U);
        const auto Post = cholmod_postorder(e.parent);

        std::vector<int> Ap, Ai;
        build_strict_lower_csc_from_upper(U, Ap, Ai);
        e.colcount = cholmod_colcount_symmetric(Ap, Ai, e.parent, Post);
        return e;
    }

    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CsrMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CscMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CsrMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CscMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CscMatrix<half_float::half> &A);
}
