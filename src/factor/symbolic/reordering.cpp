#include "symbolic.hpp"

#include <numeric>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>

// PETSc (optional in practice, but this project exposes rcm/nd orderings)
// NOTE: Never include PETSc headers inside extern "C".
// PETSc headers intentionally contain C++ constructs (templates, overloads, etc.).
#include <petscmat.h>
#include <petscis.h>

extern "C" {
#include <amd.h>
}

// SuiteSparse / CHOLMOD (used for symmetric CSC permutation to avoid reindexing code)
extern "C" {
#include <cholmod.h>
}

namespace ichol::symbolic
{

    namespace detail {
        // Build symmetric graph CSC for AMD from a lower-triangular CSC (incl diag).
        inline void symmetric_graph_from_lower_csc(
            int n,
            const std::vector<int>& Lp,
            const std::vector<int>& Li,
            std::vector<int>& Ap,
            std::vector<int>& Ai)
        {
            if ((int)Lp.size() != n + 1) throw std::runtime_error("amd_from_csc: col_ptr size mismatch");
            const int nnzL = (int)Li.size();
            if (Lp.back() != nnzL) throw std::runtime_error("amd_from_csc: col_ptr.back mismatch");

            std::vector<int> counts((std::size_t)n, 0);
            std::vector<char> has_diag((std::size_t)n, 0);

            for (int j = 0; j < n; ++j) {
                for (int p = Lp[(std::size_t)j]; p < Lp[(std::size_t)j + 1]; ++p) {
                    const int i = Li[(std::size_t)p];
                    if (i < 0 || i >= n) throw std::runtime_error("amd_from_csc: row_ind out of range");
                    ++counts[(std::size_t)j];
                    if (i != j) ++counts[(std::size_t)i];
                    if (i == j) has_diag[(std::size_t)j] = 1;
                }
            }
            for (int j = 0; j < n; ++j) {
                if (!has_diag[(std::size_t)j]) ++counts[(std::size_t)j];
            }

            Ap.assign((std::size_t)n + 1, 0);
            for (int j = 0; j < n; ++j) Ap[(std::size_t)j + 1] = Ap[(std::size_t)j] + counts[(std::size_t)j];
            Ai.assign((std::size_t)Ap.back(), 0);
            std::vector<int> next = Ap;

            for (int j = 0; j < n; ++j) {
                for (int p = Lp[(std::size_t)j]; p < Lp[(std::size_t)j + 1]; ++p) {
                    const int i = Li[(std::size_t)p];
                    Ai[(std::size_t)next[(std::size_t)j]++] = i;
                    if (i != j) Ai[(std::size_t)next[(std::size_t)i]++] = j;
                }
            }
            for (int j = 0; j < n; ++j) {
                if (!has_diag[(std::size_t)j]) Ai[(std::size_t)next[(std::size_t)j]++] = j;
            }

            // Sort + unique per column, then repack to a tight CSC.
            std::vector<int> new_counts((std::size_t)n, 0);
            for (int j = 0; j < n; ++j) {
                const int b = Ap[(std::size_t)j];
                const int e = Ap[(std::size_t)j + 1];
                auto first = Ai.begin() + (std::ptrdiff_t)b;
                auto last  = Ai.begin() + (std::ptrdiff_t)e;
                std::sort(first, last);
                last = std::unique(first, last);
                new_counts[(std::size_t)j] = (int)(last - first);
            }
            std::vector<int> Ap2((std::size_t)n + 1, 0);
            for (int j = 0; j < n; ++j) Ap2[(std::size_t)j + 1] = Ap2[(std::size_t)j] + new_counts[(std::size_t)j];
            std::vector<int> Ai2((std::size_t)Ap2.back(), 0);
            for (int j = 0; j < n; ++j) {
                const int b = Ap[(std::size_t)j];
                const int e = Ap[(std::size_t)j + 1];
                auto first = Ai.begin() + (std::ptrdiff_t)b;
                auto last  = Ai.begin() + (std::ptrdiff_t)e;
                std::sort(first, last);
                last = std::unique(first, last);
                int out = Ap2[(std::size_t)j];
                for (auto it = first; it != last; ++it) Ai2[(std::size_t)out++] = *it;
            }
            Ap.swap(Ap2);
            Ai.swap(Ai2);
        }
    
// Ensure PETSc is initialized before calling MatGetOrdering.
inline void ensure_petsc_initialized()
{
    PetscBool inited = PETSC_FALSE;
    PetscInitialized(&inited);
    if (!inited)
    {
        int argc = 0;
        char **argv = nullptr;
        PetscInitialize(&argc, &argv, nullptr, nullptr);
    }
}

// Build a PETSc SeqSBAIJ(1) matrix that represents the UPPER triangle pattern
// corresponding to a LOWER-triangular CSC pattern (incl. diagonal).
//
// PETSc's MatGetOrdering expects an assembled Mat. We keep everything local
// and destroy the Mat right after extracting the permutation.
inline Mat make_seq_sbaij_from_lower_csc_pattern(
    int n,
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_ind,
    std::vector<PetscInt>& iptr_upper,
    std::vector<PetscInt>& jind_upper,
    std::vector<PetscScalar>& aval_upper)
{
    if ((int)col_ptr.size() != n + 1) throw std::runtime_error("make_seq_sbaij_from_lower_csc_pattern: col_ptr size mismatch");
    if (col_ptr.back() != (int)row_ind.size()) throw std::runtime_error("make_seq_sbaij_from_lower_csc_pattern: nnz mismatch");

    // Count entries in the UPPER triangle CSR by mapping each lower entry (i>=j)
    // at column j to upper entry (row=j, col=i).
    std::vector<PetscInt> counts((std::size_t)n, 0);
    std::vector<char> diag_present((std::size_t)n, 0);

    for (int j = 0; j < n; ++j)
    {
        for (int p = col_ptr[(std::size_t)j]; p < col_ptr[(std::size_t)j + 1]; ++p)
        {
            const int i = row_ind[(std::size_t)p];
            if (i < j) continue; // be defensive
            counts[(std::size_t)j] += 1;
            if (i == j) diag_present[(std::size_t)j] = 1;
        }
    }
    for (int r = 0; r < n; ++r)
    {
        if (!diag_present[(std::size_t)r]) counts[(std::size_t)r] += 1; // enforce diagonal
    }

    iptr_upper.assign((std::size_t)n + 1, 0);
    for (int r = 0; r < n; ++r)
    {
        iptr_upper[(std::size_t)r + 1] = iptr_upper[(std::size_t)r] + counts[(std::size_t)r];
    }

    const PetscInt nnzU = iptr_upper[(std::size_t)n];
    jind_upper.assign((std::size_t)nnzU, 0);
    aval_upper.assign((std::size_t)nnzU, (PetscScalar)1.0);

    std::vector<PetscInt> next = iptr_upper;

    // Fill from mapped lower entries
    for (int j = 0; j < n; ++j)
    {
        for (int p = col_ptr[(std::size_t)j]; p < col_ptr[(std::size_t)j + 1]; ++p)
        {
            const int i = row_ind[(std::size_t)p];
            if (i < j) continue;
            const PetscInt row = (PetscInt)j;
            const PetscInt col = (PetscInt)i;
            jind_upper[(std::size_t)next[(std::size_t)row]++] = col;
        }
    }
    // Add missing diagonal if needed
    for (int r = 0; r < n; ++r)
    {
        if (!diag_present[(std::size_t)r])
        {
            const PetscInt row = (PetscInt)r;
            jind_upper[(std::size_t)next[(std::size_t)row]++] = (PetscInt)r;
        }
    }

    // Sort & unique per row (PETSc likes sorted cols)
    for (int r = 0; r < n; ++r)
    {
        const PetscInt b = iptr_upper[(std::size_t)r];
        const PetscInt e = iptr_upper[(std::size_t)r + 1];
        auto first = jind_upper.begin() + (std::ptrdiff_t)b;
        auto last  = jind_upper.begin() + (std::ptrdiff_t)e;
        std::sort(first, last);
        last = std::unique(first, last);
        // If we removed duplicates, we keep them but PETSc needs consistent rowptr.
        // Duplicates should be rare; enforce no duplicates by shifting (compact).
        const PetscInt new_len = (PetscInt)std::distance(first, last);
        if (new_len != (e - b))
        {
            // Compact globally: rebuild CSR arrays
            std::vector<PetscInt> ip2((std::size_t)n + 1, 0);
            for (int rr = 0; rr < n; ++rr)
            {
                const PetscInt bb = iptr_upper[(std::size_t)rr];
                const PetscInt ee = iptr_upper[(std::size_t)rr + 1];
                auto f = jind_upper.begin() + (std::ptrdiff_t)bb;
                auto l = jind_upper.begin() + (std::ptrdiff_t)ee;
                std::sort(f, l);
                l = std::unique(f, l);
                ip2[(std::size_t)rr + 1] = ip2[(std::size_t)rr] + (PetscInt)std::distance(f, l);
            }
            std::vector<PetscInt> ji2((std::size_t)ip2[(std::size_t)n], 0);
            std::vector<PetscScalar> av2((std::size_t)ip2[(std::size_t)n], (PetscScalar)1.0);
            for (int rr = 0; rr < n; ++rr)
            {
                const PetscInt bb = iptr_upper[(std::size_t)rr];
                const PetscInt ee = iptr_upper[(std::size_t)rr + 1];
                auto f = jind_upper.begin() + (std::ptrdiff_t)bb;
                auto l = jind_upper.begin() + (std::ptrdiff_t)ee;
                std::sort(f, l);
                l = std::unique(f, l);
                PetscInt out = ip2[(std::size_t)rr];
                for (auto it = f; it != l; ++it) ji2[(std::size_t)out++] = *it;
            }
            iptr_upper.swap(ip2);
            jind_upper.swap(ji2);
            aval_upper.swap(av2);
            break;
        }
    }

    Mat A = nullptr;
    ensure_petsc_initialized();
    PetscErrorCode ierr = MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,
                                                     /*bs=*/1,
                                                     (PetscInt)n, (PetscInt)n,
                                                     iptr_upper.data(),
                                                     jind_upper.data(),
                                                     aval_upper.data(),
                                                     &A);
    if (ierr) throw std::runtime_error("MatCreateSeqSBAIJWithArrays failed");
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    if (ierr) throw std::runtime_error("MatAssemblyBegin failed");
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    if (ierr) throw std::runtime_error("MatAssemblyEnd failed");
    return A;
}

inline ichol::symbolic::Permutation ordering_from_mat(Mat A, const char* ordering_type, int n)
{
    ensure_petsc_initialized();

    IS row = nullptr, col = nullptr;
    PetscErrorCode ierr = MatGetOrdering(A, ordering_type, &row, &col);
    if (ierr) throw std::runtime_error("MatGetOrdering failed");

    const PetscInt* idx = nullptr;
    ierr = ISGetIndices(row, &idx);
    if (ierr) throw std::runtime_error("ISGetIndices failed");

    ichol::symbolic::Permutation P;
    P.perm.assign((std::size_t)n, 0);
    P.inv_perm.assign((std::size_t)n, 0);

    for (int k = 0; k < n; ++k)
    {
        const int pk = (int)idx[(std::size_t)k];
        P.perm[(std::size_t)k] = pk;
    }
    ierr = ISRestoreIndices(row, &idx);
    if (ierr) throw std::runtime_error("ISRestoreIndices failed");

    for (int k = 0; k < n; ++k)
    {
        const int pk = P.perm[(std::size_t)k];
        if ((unsigned)pk >= (unsigned)n) throw std::runtime_error("MatGetOrdering produced out-of-range index");
        P.inv_perm[(std::size_t)pk] = k;
    }

    ISDestroy(&row);
    ISDestroy(&col);
    return P;
}

} // namespace detail
    ichol::symbolic::Permutation amd_from_csr(int n,
                                              const std::vector<int> &row_ptr,
                                              const std::vector<int> &col_ind)
    {
        std::vector<int> Ap, Ai;
        std::vector<int> col_csr_pos;
        ichol::matrix::csr_to_csc_pattern_only(n,
                                               row_ptr,
                                               col_ind,
                                               Ap,
                                               Ai,
                                               col_csr_pos);

        ichol::symbolic::Permutation P;
        P.perm.assign(n, 0);
        P.inv_perm.assign(n, 0);

        double Info[AMD_INFO];
        const int status = amd_order(n,
                                     Ap.data(),
                                     Ai.data(),
                                     P.perm.data(),
                                     nullptr,
                                     Info);

        if (status != AMD_OK)
        {
            throw std::runtime_error("amd_order failed (status=" + std::to_string(status) + ").");
        }

        for (int k = 0; k < n; ++k)
        {
            const int orig = P.perm[k];
            if (orig < 0 || orig >= n)
                throw std::runtime_error("amd_order returned invalid permutation.");
            P.inv_perm[orig] = k;
        }

        return P;
    }

    ichol::symbolic::Permutation amd_from_csc(int n,
                                             const std::vector<int> &col_ptr,
                                             const std::vector<int> &row_ind)
    {
        // AMD expects the graph of A+A'. Our supernodal pipeline typically stores
        // only the lower triangle (incl diag) in CSC, so we explicitly symmetrize.
        std::vector<int> Ap, Ai;
        detail::symmetric_graph_from_lower_csc(n, col_ptr, row_ind, Ap, Ai);

        ichol::symbolic::Permutation P;
        P.perm.assign(n, 0);
        P.inv_perm.assign(n, 0);

        double Info[AMD_INFO];
        const int status = amd_order(n, Ap.data(), Ai.data(), P.perm.data(), nullptr, Info);
        if (status != AMD_OK)
            throw std::runtime_error("amd_order failed (status=" + std::to_string(status) + ").");

        for (int k = 0; k < n; ++k)
        {
            const int orig = P.perm[k];
            if (orig < 0 || orig >= n)
                throw std::runtime_error("amd_order returned invalid permutation.");
            P.inv_perm[orig] = k;
        }
        return P;
    }

    
ichol::symbolic::Permutation rcm_from_csc(int n,
                                          const std::vector<int> &col_ptr,
                                          const std::vector<int> &row_ind)
{
    if (n <= 0)
        return ichol::symbolic::Permutation{};

    std::vector<PetscInt> iptr_upper, jind_upper;
    std::vector<PetscScalar> aval_upper;

    Mat A = detail::make_seq_sbaij_from_lower_csc_pattern(n, col_ptr, row_ind,
                                                          iptr_upper, jind_upper, aval_upper);

    ichol::symbolic::Permutation P = detail::ordering_from_mat(A, MATORDERINGRCM, n);

    MatDestroy(&A);
    return P;
}


ichol::symbolic::Permutation nd_from_csc(int n,
                                         const std::vector<int> &col_ptr,
                                         const std::vector<int> &row_ind)
{
    if (n <= 0)
        return ichol::symbolic::Permutation{};

    std::vector<PetscInt> iptr_upper, jind_upper;
    std::vector<PetscScalar> aval_upper;

    Mat A = detail::make_seq_sbaij_from_lower_csc_pattern(n, col_ptr, row_ind,
                                                          iptr_upper, jind_upper, aval_upper);

    ichol::symbolic::Permutation P = detail::ordering_from_mat(A, MATORDERINGND, n);

    MatDestroy(&A);
    return P;
}

// ------------------------------------------------------------------------
// Apply symmetric permutation to a LOWER-triangular CSC matrix (incl diag):
//   A := A(P,P), where P is CHOLMOD-style "perm[new] = old".
//
// IMPORTANT:
// - This permutation is required because we import CHOLMOD's supernodal
//   symbolic (super/pi/px/s) which is defined on the permuted ordering.
// - We implement the permutation ourselves (instead of calling CHOLMOD
//   ptranspose) to avoid long runtimes / apparent hangs on large problems
//   due to CHOLMOD's internal sorting/workspace behavior in some builds.
//
// Implementation notes:
// - We map each stored entry (i_old, j_old) with i_old>=j_old to
//     (i_new, j_new) = (inv_perm[i_old], inv_perm[j_old]).
// - We then enforce lower-triangular storage by swapping if i_new<j_new.
// - Because the permutation is bijective, duplicates cannot be created.
// - We sort rows within each column to keep CSC "sorted" invariant.
// ------------------------------------------------------------------------

namespace detail
{
    template <typename T>
    static inline void ensure_inv_perm(const Permutation &P, std::vector<int> &inv_out)
    {
        const int n = (int)P.perm.size();
        inv_out.assign((size_t)n, 0);
        if (!P.inv_perm.empty())
        {
            for (int i = 0; i < n; ++i) inv_out[(size_t)i] = (int)P.inv_perm[(size_t)i];
            return;
        }
        // Build inverse from perm[new]=old
        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = (int)P.perm[(size_t)new_i];
            inv_out[(size_t)old_i] = new_i;
        }
    }

    template <typename T>
    static void apply_sym_perm_lower_csc_inplace(ichol::matrix::CscMatrix<T> &A, const Permutation &P)
    {
        const int n = A.num_cols;
        if (n <= 0) return;
        if (A.num_rows != n) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: A must be square");
        if ((int)P.perm.size() != n) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: perm size mismatch");
        if ((int)A.col_ptr.size() != n + 1) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: bad col_ptr size");

        const int nnz = (int)A.row_ind.size();
        if ((int)A.values.size() != nnz) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: values size mismatch");
        if (A.col_ptr.back() != nnz) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: col_ptr.back() != nnz");

        std::vector<int> inv;
        ensure_inv_perm<T>(P, inv);

        // 1) Count nnz per new column
        std::vector<int> col_counts((size_t)n, 0);
        int max_col_nnz = 0;

        for (int j_old = 0; j_old < n; ++j_old)
        {
            const int j_new = inv[(size_t)j_old];
            const int p0 = A.col_ptr[(size_t)j_old];
            const int p1 = A.col_ptr[(size_t)j_old + 1];

            for (int p = p0; p < p1; ++p)
            {
                const int i_old = A.row_ind[(size_t)p];
                const int i_new = inv[(size_t)i_old];

                int col = j_new;
                if (i_new < j_new) col = i_new; // store in lower triangle
                col_counts[(size_t)col] += 1;
            }
        }

        for (int j = 0; j < n; ++j) max_col_nnz = std::max(max_col_nnz, col_counts[(size_t)j]);

        // 2) Prefix sum -> new col_ptr
        std::vector<int> new_col_ptr((size_t)n + 1, 0);
        for (int j = 0; j < n; ++j)
        {
            new_col_ptr[(size_t)j + 1] = new_col_ptr[(size_t)j] + col_counts[(size_t)j];
        }

        const int new_nnz = new_col_ptr[(size_t)n];
        if (new_nnz != nnz)
            throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: nnz mismatch after permutation");

        std::vector<int> new_row_ind((size_t)new_nnz, 0);
        std::vector<T>   new_values((size_t)new_nnz);

        // 3) Fill
        std::vector<int> next = new_col_ptr;

        for (int j_old = 0; j_old < n; ++j_old)
        {
            const int j_new = inv[(size_t)j_old];
            const int p0 = A.col_ptr[(size_t)j_old];
            const int p1 = A.col_ptr[(size_t)j_old + 1];

            for (int p = p0; p < p1; ++p)
            {
                const int i_old = A.row_ind[(size_t)p];
                const int i_new = inv[(size_t)i_old];
                const T   aij   = A.values[(size_t)p];

                int col = j_new;
                int row = i_new;
                if (row < col) { std::swap(row, col); } // enforce lower

                const int dst = next[(size_t)col]++;
                new_row_ind[(size_t)dst] = row;
                new_values[(size_t)dst]  = aij;
            }
        }

        // 4) Sort rows within each column (keep values aligned)
        struct RowVal { int r; T v; };
        std::vector<RowVal> buf;
        buf.reserve((size_t)max_col_nnz);

        for (int j = 0; j < n; ++j)
        {
            const int b0 = new_col_ptr[(size_t)j];
            const int b1 = new_col_ptr[(size_t)j + 1];
            const int len = b1 - b0;
            if (len <= 1) continue;

            buf.clear();
            for (int t = 0; t < len; ++t)
            {
                buf.push_back(RowVal{ new_row_ind[(size_t)(b0 + t)], new_values[(size_t)(b0 + t)] });
            }

            std::sort(buf.begin(), buf.end(), [](const RowVal &a, const RowVal &b) { return a.r < b.r; });

            for (int t = 0; t < len; ++t)
            {
                new_row_ind[(size_t)(b0 + t)] = buf[(size_t)t].r;
                new_values[(size_t)(b0 + t)]  = buf[(size_t)t].v;
            }
        }

        // 5) Swap into A
        A.col_ptr.swap(new_col_ptr);
        A.row_ind.swap(new_row_ind);
        A.values.swap(new_values);
        A.nnz = nnz;
    }
} // namespace detail

void apply_symmetric_permutation_csc_lower_inplace(ichol::matrix::CscMatrix<double> &A,
                                                   const Permutation &P)
{
    detail::apply_sym_perm_lower_csc_inplace<double>(A, P);
}

void apply_symmetric_permutation_csc_lower_inplace(ichol::matrix::CscMatrix<float> &A,
                                                   const Permutation &P)
{
    detail::apply_sym_perm_lower_csc_inplace<float>(A, P);
}

    namespace detail
    {
        inline void petsc_check(PetscErrorCode ierr, const char *where)
        {
            if (ierr)
                throw std::runtime_error(std::string(where) + " failed (PetscErrorCode=" + std::to_string((int)ierr) + ").");
        }

// Convert LOWER-triangular (incl diag) CSC to UPPER-triangular CSR (incl diag) for SeqSBAIJ(bs=1),
// without forming an intermediate CSR of the lower triangle.
// For each stored (i,j) with i>j in column j, we emit (j,i) into UPPER row j. Diagonal stays (i,i).
// Also ensures every row has a diagonal entry (adds (i,i) if missing).
inline void lower_csc_to_upper_csr_for_sbaij(int n,
                                             const std::vector<int> &l_col_ptr,
                                             const std::vector<int> &l_row_ind,
                                             std::vector<PetscInt> &u_iptr,
                                             std::vector<PetscInt> &u_jind)
{
    if ((int)l_col_ptr.size() != n + 1)
        throw std::runtime_error("CSC col_ptr size must be n+1.");
    if (l_col_ptr.back() != (int)l_row_ind.size())
        throw std::runtime_error("CSC col_ptr.back() must equal row_ind.size().");

    std::vector<PetscInt> row_counts((std::size_t)n, 0);
    std::vector<char> has_diag((std::size_t)n, 0);

    // Count nnz per upper row
    for (int j = 0; j < n; ++j)
    {
        for (int p = l_col_ptr[(std::size_t)j]; p < l_col_ptr[(std::size_t)j + 1]; ++p)
        {
            const int i = l_row_ind[(std::size_t)p];
            if (i < 0 || i >= n)
                throw std::runtime_error("row_ind out of range.");
            if (i < j)
                throw std::runtime_error("Input pattern is not lower-triangular+diag (found row < col).");

            if (i == j)
            {
                has_diag[(std::size_t)j] = 1;
                row_counts[(std::size_t)j] += 1;
            }
            else
            {
                // (i,j) -> (j,i) in upper
                row_counts[(std::size_t)j] += 1;
            }
        }
    }
    for (int i = 0; i < n; ++i)
    {
        if (!has_diag[(std::size_t)i])
            row_counts[(std::size_t)i] += 1; // add missing diagonal
    }

    u_iptr.assign((std::size_t)n + 1, 0);
    for (int i = 0; i < n; ++i)
        u_iptr[(std::size_t)i + 1] = u_iptr[(std::size_t)i] + row_counts[(std::size_t)i];

    const PetscInt nnzU = u_iptr[(std::size_t)n];
    u_jind.assign((std::size_t)nnzU, 0);

    std::vector<PetscInt> cursor = u_iptr;

    // Fill columns
    for (int j = 0; j < n; ++j)
    {
        bool diag_in_col = false;
        for (int p = l_col_ptr[(std::size_t)j]; p < l_col_ptr[(std::size_t)j + 1]; ++p)
        {
            const int i = l_row_ind[(std::size_t)p];
            if (i == j)
            {
                diag_in_col = true;
                u_jind[(std::size_t)cursor[(std::size_t)j]++] = (PetscInt)j;
            }
            else
            {
                // (i,j) -> (j,i)
                u_jind[(std::size_t)cursor[(std::size_t)j]++] = (PetscInt)i;
            }
        }
        if (!diag_in_col)
            u_jind[(std::size_t)cursor[(std::size_t)j]++] = (PetscInt)j;
    }

    // Sort per row and reject duplicates
    for (int i = 0; i < n; ++i)
    {
        const PetscInt b = u_iptr[(std::size_t)i];
        const PetscInt e = u_iptr[(std::size_t)i + 1];
        auto first = u_jind.begin() + (std::ptrdiff_t)b;
        auto last = u_jind.begin() + (std::ptrdiff_t)e;

        std::sort(first, last);
        if (std::adjacent_find(first, last) != last)
            throw std::runtime_error("Duplicate column index in constructed upper-triangular pattern.");
    }
}

// Creates a SEQUENTIAL SBAIJ matrix (bs=1) from LOWER-triangular CSC (incl diag), without building CSR(A).
// Uses MatCreateSeqSBAIJWithArrays => PETSc does NOT copy i/j/a; vectors must outlive MatDestroy.

        // Convert LOWER-triangular (incl diag) CSR to UPPER-triangular CSR (incl diag) for SeqSBAIJ(bs=1).
        // For each stored (i,j) with j<i, we emit (j,i) into row j. Diagonal stays (i,i).
        // Also ensures every row has a diagonal entry (adds (i,i) if missing).
        inline void lower_csr_to_upper_csr_for_sbaij(int n,
                                                     const std::vector<int> &l_row_ptr,
                                                     const std::vector<int> &l_col_ind,
                                                     std::vector<PetscInt> &u_iptr,
                                                     std::vector<PetscInt> &u_jind)
        {
            if ((int)l_row_ptr.size() != n + 1)
                throw std::runtime_error("CSR row_ptr size must be n+1.");
            if (l_row_ptr.back() != (int)l_col_ind.size())
                throw std::runtime_error("CSR row_ptr.back() must equal col_ind.size().");

            std::vector<PetscInt> row_counts((std::size_t)n, 0);

            // Count nnz per upper row
            for (int i = 0; i < n; ++i)
            {
                bool has_diag = false;
                for (int p = l_row_ptr[i]; p < l_row_ptr[i + 1]; ++p)
                {
                    const int j = l_col_ind[p];
                    if (j < 0 || j >= n)
                        throw std::runtime_error("col_ind out of range.");
                    if (j > i)
                        throw std::runtime_error("Input pattern is not lower-triangular+diag (found col > row).");

                    if (j == i)
                    {
                        has_diag = true;
                        row_counts[i] += 1;
                    }
                    else
                    {
                        // (i,j) -> (j,i) in upper
                        row_counts[j] += 1;
                    }
                }
                if (!has_diag)
                    row_counts[i] += 1; // add missing diagonal
            }

            u_iptr.assign((std::size_t)n + 1, 0);
            for (int i = 0; i < n; ++i)
                u_iptr[i + 1] = u_iptr[i] + row_counts[i];

            const PetscInt nnzU = u_iptr[n];
            u_jind.assign((std::size_t)nnzU, 0);

            std::vector<PetscInt> cursor = u_iptr;

            // Fill columns
            for (int i = 0; i < n; ++i)
            {
                bool has_diag = false;
                for (int p = l_row_ptr[i]; p < l_row_ptr[i + 1]; ++p)
                {
                    const int j = l_col_ind[p];
                    if (j == i)
                    {
                        has_diag = true;
                        u_jind[(std::size_t)cursor[i]++] = (PetscInt)i;
                    }
                    else
                    {
                        // (i,j) -> (j,i)
                        u_jind[(std::size_t)cursor[j]++] = (PetscInt)i;
                    }
                }
                if (!has_diag)
                    u_jind[(std::size_t)cursor[i]++] = (PetscInt)i;
            }

            // Sort per row and reject duplicates (PETSc requires sorted; duplicates are almost always a bug for patterns)
            for (int i = 0; i < n; ++i)
            {
                const PetscInt b = u_iptr[i];
                const PetscInt e = u_iptr[i + 1];
                auto first = u_jind.begin() + (std::ptrdiff_t)b;
                auto last = u_jind.begin() + (std::ptrdiff_t)e;

                std::sort(first, last);
                if (std::adjacent_find(first, last) != last)
                    throw std::runtime_error("Duplicate column index in constructed upper-triangular pattern.");
            }
        }

        // Creates a SEQUENTIAL SBAIJ matrix (bs=1) from LOWER-triangular CSR (incl diag), without building full A.
        // Uses MatCreateSeqSBAIJWithArrays => PETSc does NOT copy i/j/a; vectors must outlive MatDestroy.
        inline Mat make_seq_sbaij_from_lower_csr_pattern(int n,
                                                         const std::vector<int> &row_ptr,
                                                         const std::vector<int> &col_ind,
                                                         std::vector<PetscInt> &iptr_upper,
                                                         std::vector<PetscInt> &jind_upper,
                                                         std::vector<PetscScalar> &aval_upper)
        {
            if (n <= 0)
                return nullptr;

            lower_csr_to_upper_csr_for_sbaij(n, row_ptr, col_ind, iptr_upper, jind_upper);

            aval_upper.assign(jind_upper.size(), (PetscScalar)1.0); // dummy values; ordering uses graph

            Mat A = nullptr;
            PetscErrorCode ierr = MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,
                                                              (PetscInt)1, (PetscInt)n, (PetscInt)n,
                                                              iptr_upper.data(),
                                                              jind_upper.data(),
                                                              aval_upper.data(),
                                                              &A);
            petsc_check(ierr, "MatCreateSeqSBAIJWithArrays");

            ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
            petsc_check(ierr, "MatSetOption(MAT_SYMMETRIC)");
            ierr = MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE);
            petsc_check(ierr, "MatSetOption(MAT_STRUCTURALLY_SYMMETRIC)");

            return A;
        }

    } // namespace detail

    ichol::symbolic::Permutation rcm_from_csr(int n,
                                              const std::vector<int> &row_ptr,
                                              const std::vector<int> &col_ind)
    {
        if (n <= 0)
            return ichol::symbolic::Permutation{};

        std::vector<PetscInt> iptr_upper, jind_upper;
        std::vector<PetscScalar> aval_upper;

        Mat A = detail::make_seq_sbaij_from_lower_csr_pattern(n, row_ptr, col_ind,
                                                              iptr_upper, jind_upper, aval_upper);

        ichol::symbolic::Permutation P = detail::ordering_from_mat(A, MATORDERINGRCM, n);

        MatDestroy(&A);
        return P;
    }

    ichol::symbolic::Permutation nd_from_csr(int n,
                                             const std::vector<int> &row_ptr,
                                             const std::vector<int> &col_ind)
    {
        if (n <= 0)
            return ichol::symbolic::Permutation{};

        std::vector<PetscInt> iptr_upper, jind_upper;
        std::vector<PetscScalar> aval_upper;

        Mat A = detail::make_seq_sbaij_from_lower_csr_pattern(n, row_ptr, col_ind,
                                                              iptr_upper, jind_upper, aval_upper);

        ichol::symbolic::Permutation P = detail::ordering_from_mat(A, MATORDERINGND, n);

        MatDestroy(&A);
        return P;
    }

    // -----------------------------------------------------------------------------
    // Baseline utilities (used by the non-supernode symbolic pipeline).
    // -----------------------------------------------------------------------------

    Permutation identity_permutation(int n)
    {
        if (n <= 0) return Permutation{};
        Permutation P;
        P.perm.resize((std::size_t)n);
        P.inv_perm.resize((std::size_t)n);
        for (int i = 0; i < n; ++i)
        {
            P.perm[(std::size_t)i] = i;
            P.inv_perm[(std::size_t)i] = i;
        }
        return P;
    }

    template <typename T>
    void apply_permutation_csr(ichol::matrix::CsrMatrix<T> &A, const Permutation &P)
    {
        const int n = A.num_rows;
        if (n <= 0) return;
        if ((int)P.perm.size() != n || (int)P.inv_perm.size() != n)
            throw std::runtime_error("apply_permutation_csr: permutation size mismatch");

        // Build new CSR for A_new = P*A*P^T, where perm[new] = old.
        std::vector<int> new_row_ptr((std::size_t)n + 1, 0);
        std::vector<int> new_col_ind;
        std::vector<T>   new_vals;
        new_col_ind.reserve(A.col_ind.size());
        new_vals.reserve(A.values.size());

        for (int i_new = 0; i_new < n; ++i_new)
        {
            const int i_old = P.perm[(std::size_t)i_new];
            int cnt = A.row_ptr[(std::size_t)i_old + 1] - A.row_ptr[(std::size_t)i_old];
            new_row_ptr[(std::size_t)i_new + 1] = cnt;
        }
        for (int i = 0; i < n; ++i) new_row_ptr[(std::size_t)i + 1] += new_row_ptr[(std::size_t)i];

        const int nnz = new_row_ptr.back();
        new_col_ind.resize((std::size_t)nnz);
        new_vals.resize((std::size_t)nnz);

        std::vector<int> next = new_row_ptr;

        for (int i_new = 0; i_new < n; ++i_new)
        {
            const int i_old = P.perm[(std::size_t)i_new];
            for (int p = A.row_ptr[(std::size_t)i_old]; p < A.row_ptr[(std::size_t)i_old + 1]; ++p)
            {
                const int j_old = A.col_ind[(std::size_t)p];
                const int j_new = P.inv_perm[(std::size_t)j_old];

                const int dest = next[(std::size_t)i_new]++;
                new_col_ind[(std::size_t)dest] = j_new;
                new_vals[(std::size_t)dest] = A.values[(std::size_t)p];
            }
        }

        for (int i = 0; i < n; ++i)
        {
            const int b = new_row_ptr[(std::size_t)i];
            const int e = new_row_ptr[(std::size_t)i + 1];
            const int len = e - b;
            if (len <= 1) continue;

            std::vector<int> idx((std::size_t)len);
            for (int k = 0; k < len; ++k) idx[(std::size_t)k] = k;

            std::sort(idx.begin(), idx.end(), [&](int a, int b2)
            {
                return new_col_ind[(std::size_t)b + (std::size_t)a] < new_col_ind[(std::size_t)b + (std::size_t)b2];
            });

            std::vector<int> cols_sorted((std::size_t)len);
            std::vector<T>   vals_sorted((std::size_t)len);
            for (int k = 0; k < len; ++k)
            {
                cols_sorted[(std::size_t)k] = new_col_ind[(std::size_t)b + (std::size_t)idx[(std::size_t)k]];
                vals_sorted[(std::size_t)k] = new_vals[(std::size_t)b + (std::size_t)idx[(std::size_t)k]];
            }
            for (int k = 0; k < len; ++k)
            {
                new_col_ind[(std::size_t)b + (std::size_t)k] = cols_sorted[(std::size_t)k];
                new_vals[(std::size_t)b + (std::size_t)k] = vals_sorted[(std::size_t)k];
            }
        }

        A.row_ptr.swap(new_row_ptr);
        A.col_ind.swap(new_col_ind);
        A.values.swap(new_vals);
        A.nnz = nnz;
        A.num_cols = n;
        A.num_rows = n;
    }

    template void apply_permutation_csr<double>(ichol::matrix::CsrMatrix<double> &, const Permutation &);
    template void apply_permutation_csr<float>(ichol::matrix::CsrMatrix<float> &, const Permutation &);
    template void apply_permutation_csr<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &, const Permutation &);


} // namespace ichol::symbolic
