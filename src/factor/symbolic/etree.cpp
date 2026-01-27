// etree.cpp
#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>

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

    
//------------------------------------------------------------------------------
// Strict upper adjacency U of A in compressed form:
//   U[j] = { i | i < j and A(i,j) != 0 }.
//
// We treat A as a symmetric pattern: each nonzero (i,j) contributes to the
// undirected edge between i and j. We map it to the strict-upper entry
// (v=min(i,j), u=max(i,j)) and store v in column u. Duplicates (from symmetric
// storage) are removed per column using a stamp array (linear-time, no sorting).
//------------------------------------------------------------------------------
struct UpperAdj {
    std::vector<int> ptr;  // size n+1
    std::vector<int> ind;  // concatenated indices (each < its column)
};

template <class T>
static UpperAdj build_strict_upper_adjacency(const ichol::matrix::CscMatrix<T> &A)
{
    const int n = A.num_cols;
    std::vector<int> cnt((size_t)n, 0);

    for (int j = 0; j < n; ++j) {
        const int p0 = A.col_ptr[(size_t)j];
        const int p1 = A.col_ptr[(size_t)j + 1];
        for (int p = p0; p < p1; ++p) {
            const int i = A.row_ind[(size_t)p];
            if (i == j) continue;
            const int u = (i > j) ? i : j;
            ++cnt[(size_t)u];
        }
    }

    UpperAdj U;
    U.ptr.assign((size_t)n + 1, 0);
    for (int k = 0; k < n; ++k) U.ptr[(size_t)k + 1] = U.ptr[(size_t)k] + cnt[(size_t)k];
    U.ind.assign((size_t)U.ptr[(size_t)n], 0);

    std::vector<int> next = U.ptr;
    for (int j = 0; j < n; ++j) {
        const int p0 = A.col_ptr[(size_t)j];
        const int p1 = A.col_ptr[(size_t)j + 1];
        for (int p = p0; p < p1; ++p) {
            const int i = A.row_ind[(size_t)p];
            if (i == j) continue;
            const int u = (i > j) ? i : j;
            const int v = (i > j) ? j : i;
            U.ind[(size_t)next[(size_t)u]++] = v;
        }
    }

    // Deduplicate per column (stable, linear time).
    std::vector<int> mark((size_t)n, -1);
    std::vector<int> new_ptr((size_t)n + 1, 0);

    int nnz = 0;
    new_ptr[0] = 0;
    for (int k = 0; k < n; ++k) {
        const int a = U.ptr[(size_t)k];
        const int b = U.ptr[(size_t)k + 1];
        for (int p = a; p < b; ++p) {
            const int v = U.ind[(size_t)p];
            if ((unsigned)v >= (unsigned)n) continue;
            if (mark[(size_t)v] == k) continue;
            mark[(size_t)v] = k;
            U.ind[(size_t)nnz++] = v;
        }
        new_ptr[(size_t)k + 1] = nnz;
    }

    U.ptr.swap(new_ptr);
    U.ind.resize((size_t)nnz);
    return U;
}

template <class T>
static UpperAdj build_strict_upper_adjacency(const ichol::matrix::CsrMatrix<T> &A)
{
    const int n = A.num_rows;
    std::vector<int> cnt((size_t)n, 0);

    for (int r = 0; r < n; ++r) {
        const int p0 = A.row_ptr[(size_t)r];
        const int p1 = A.row_ptr[(size_t)r + 1];
        for (int p = p0; p < p1; ++p) {
            const int c = A.col_ind[(size_t)p];
            if (c == r) continue;
            const int u = (c > r) ? c : r;
            ++cnt[(size_t)u];
        }
    }

    UpperAdj U;
    U.ptr.assign((size_t)n + 1, 0);
    for (int k = 0; k < n; ++k) U.ptr[(size_t)k + 1] = U.ptr[(size_t)k] + cnt[(size_t)k];
    U.ind.assign((size_t)U.ptr[(size_t)n], 0);

    std::vector<int> next = U.ptr;
    for (int r = 0; r < n; ++r) {
        const int p0 = A.row_ptr[(size_t)r];
        const int p1 = A.row_ptr[(size_t)r + 1];
        for (int p = p0; p < p1; ++p) {
            const int c = A.col_ind[(size_t)p];
            if (c == r) continue;
            const int u = (c > r) ? c : r;
            const int v = (c > r) ? r : c;
            U.ind[(size_t)next[(size_t)u]++] = v;
        }
    }

    std::vector<int> mark((size_t)n, -1);
    std::vector<int> new_ptr((size_t)n + 1, 0);

    int nnz = 0;
    new_ptr[0] = 0;
    for (int k = 0; k < n; ++k) {
        const int a = U.ptr[(size_t)k];
        const int b = U.ptr[(size_t)k + 1];
        for (int p = a; p < b; ++p) {
            const int v = U.ind[(size_t)p];
            if ((unsigned)v >= (unsigned)n) continue;
            if (mark[(size_t)v] == k) continue;
            mark[(size_t)v] = k;
            U.ind[(size_t)nnz++] = v;
        }
        new_ptr[(size_t)k + 1] = nnz;
    }

    U.ptr.swap(new_ptr);
    U.ind.resize((size_t)nnz);
    return U;
}

//------------------------------------------------------------------------------
    // etree from strict upper adjacency (CHOLMOD(etree), symmetric case)
    //------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------
// etree from strict upper adjacency (CHOLMOD(etree), symmetric case)
//------------------------------------------------------------------------------
static std::vector<int> cholmod_etree_symmetric_upper(const UpperAdj &U)
{
    const int n = (int)U.ptr.size() - 1;
    std::vector<int> Parent(n, -1);
    std::vector<int> Ancestor(n, -1);
    for (int j = 0; j < n; ++j)
    {
        const int a = U.ptr[(size_t)j];
        const int b = U.ptr[(size_t)j + 1];
        for (int p = a; p < b; ++p)
        {
            const int i = U.ind[(size_t)p]; // i < j by construction
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
    
//------------------------------------------------------------------------------
// Build strict lower CSC (transpose of strict upper):
//   Lcol[j] = { i | i > j and (j,i) is in strict upper }.
// This matches cholmod_rowcolcounts symmetric input convention.
//
// Note: ordering within each column is irrelevant for cholmod_colcount_symmetric,
// so we do NOT sort Ai (saves time).
//------------------------------------------------------------------------------
static void build_strict_lower_csc_from_upper(
    const UpperAdj &U,
    std::vector<int> &Ap,
    std::vector<int> &Ai)
{
    const int n = (int)U.ptr.size() - 1;
    std::vector<int> col_nnz((size_t)n, 0);

    // Each strict-upper entry (v,u) contributes to strict-lower entry (u,v)
    // stored in column v with row u.
    for (int u = 0; u < n; ++u)
    {
        const int a = U.ptr[(size_t)u];
        const int b = U.ptr[(size_t)u + 1];
        for (int p = a; p < b; ++p)
        {
            const int v = U.ind[(size_t)p]; // v < u
            if ((unsigned)v < (unsigned)n) col_nnz[(size_t)v]++;
        }
    }

    Ap.assign((size_t)n + 1, 0);
    for (int j = 0; j < n; ++j) Ap[(size_t)j + 1] = Ap[(size_t)j] + col_nnz[(size_t)j];
    Ai.assign((size_t)Ap[(size_t)n], -1);

    std::vector<int> next = Ap;
    for (int u = 0; u < n; ++u)
    {
        const int a = U.ptr[(size_t)u];
        const int b = U.ptr[(size_t)u + 1];
        for (int p = a; p < b; ++p)
        {
            const int v = U.ind[(size_t)p];
            if ((unsigned)v >= (unsigned)n) continue;
            Ai[(size_t)next[(size_t)v]++] = u; // u > v
        }
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

        UpperAdj U = build_strict_upper_adjacency(A);
        e.parent = cholmod_etree_symmetric_upper(U);
        const auto Post = cholmod_postorder(e.parent);

        std::vector<int> Ap, Ai;
        build_strict_lower_csc_from_upper(U, Ap, Ai);
        e.colcount = cholmod_colcount_symmetric(Ap, Ai, e.parent, Post);

        // Cache strict-upper adjacency for downstream symbolic stages (optional).
        e.upper_ptr = std::move(U.ptr);
        e.upper_ind = std::move(U.ind);
        return e;
    }

    template <typename T>
    ETree build_etree(const ichol::matrix::CsrMatrix<T> &A)
    {
        const int n = A.num_rows;
        ETree e;
        e.parent.assign(n, -1);
        e.colcount.assign(n, 1);

        UpperAdj U = build_strict_upper_adjacency(A);
        e.parent = cholmod_etree_symmetric_upper(U);
        const auto Post = cholmod_postorder(e.parent);

        std::vector<int> Ap, Ai;
        build_strict_lower_csc_from_upper(U, Ap, Ai);
        e.colcount = cholmod_colcount_symmetric(Ap, Ai, e.parent, Post);

        // Cache strict-upper adjacency for downstream symbolic stages (optional).
        e.upper_ptr = std::move(U.ptr);
        e.upper_ind = std::move(U.ind);
        return e;
    }

    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CsrMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CscMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CsrMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CscMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CscMatrix<half_float::half> &A);
}
