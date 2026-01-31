#include "symbolic.hpp"

#include <numeric>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>

// PETSc (optional in practice, but this project exposes rcm/nd orderings)
extern "C" {
#include <petscmat.h>
#include <petscis.h>
}

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

    void apply_symmetric_permutation_csc_lower_inplace(ichol::matrix::CscMatrix<double> &A,
                                                       const Permutation &P)
    {
        // Prefer CHOLMOD for symmetric permutation to avoid maintaining our own
        // CSC reindexing + sorting logic.
        const int n = A.num_cols;
        if (n <= 0) return;
        if (A.num_rows != n) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: A must be square");
        if ((int)P.perm.size() != n) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: perm size mismatch");

        cholmod_common cc;
        cholmod_start(&cc);
        cc.itype = CHOLMOD_LONG;
        cc.dtype = CHOLMOD_DOUBLE;

        const int nnz = (int)A.row_ind.size();
        cholmod_sparse* S = cholmod_allocate_sparse(
            (size_t)n, (size_t)n, (size_t)nnz,
            /*sorted=*/1,
            /*packed=*/1,
            /*stype=*/-1,
            CHOLMOD_REAL,
            &cc);
        if (!S)
        {
            cholmod_finish(&cc);
            throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: cholmod_allocate_sparse failed");
        }

        auto* Sp = reinterpret_cast<SuiteSparse_long*>(S->p);
        auto* Si = reinterpret_cast<SuiteSparse_long*>(S->i);
        auto* Sx = reinterpret_cast<double*>(S->x);
        for (int j = 0; j < n + 1; ++j) Sp[(std::size_t)j] = (SuiteSparse_long)A.col_ptr[(std::size_t)j];
        for (int p = 0; p < nnz; ++p)
        {
            Si[(std::size_t)p] = (SuiteSparse_long)A.row_ind[(std::size_t)p];
            Sx[(std::size_t)p] = A.values[(std::size_t)p];
        }

        std::vector<SuiteSparse_long> perm_long((std::size_t)n);
        for (int k = 0; k < n; ++k) perm_long[(std::size_t)k] = (SuiteSparse_long)P.perm[(std::size_t)k];

        // For symmetric A stored as tril(A), CHOLMOD's ptranspose returns A(P,P)'.
        // Since A is symmetric, this equals A(P,P). stype=-1 keeps lower triangle.
        cholmod_sparse* Spm = cholmod_l_ptranspose(S, /*values=*/1,
                                                 perm_long.data(),
                                                 /*fset=*/nullptr, /*fsize=*/0, &cc);
        if (!Spm)
        {
            cholmod_free_sparse(&S, &cc);
            cholmod_finish(&cc);
            throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: cholmod_ptranspose failed");
        }

        const auto* Pp = reinterpret_cast<const SuiteSparse_long*>(Spm->p);
        const auto* Pi = reinterpret_cast<const SuiteSparse_long*>(Spm->i);
        const auto* Px = reinterpret_cast<const double*>(Spm->x);
        const int new_nnz = (int)Pp[(std::size_t)n];

        A.col_ptr.assign((std::size_t)n + 1, 0);
        A.row_ind.assign((std::size_t)new_nnz, 0);
        A.values.assign((std::size_t)new_nnz, 0.0);
        A.nnz = new_nnz;
        for (int j = 0; j < n + 1; ++j) A.col_ptr[(std::size_t)j] = (int)Pp[(std::size_t)j];
        for (int p = 0; p < new_nnz; ++p)
        {
            A.row_ind[(std::size_t)p] = (int)Pi[(std::size_t)p];
            A.values[(std::size_t)p]  = Px[(std::size_t)p];
        }

        cholmod_free_sparse(&Spm, &cc);
        cholmod_free_sparse(&S, &cc);
        cholmod_finish(&cc);
    }

    void apply_symmetric_permutation_csc_lower_inplace(ichol::matrix::CscMatrix<float> &A,
                                                       const Permutation &P)
    {
        // Same as the double variant, but using CHOLMOD single-precision payload.
        const int n = A.num_cols;
        if (n <= 0) return;
        if (A.num_rows != n) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: A must be square");
        if ((int)P.perm.size() != n) throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: perm size mismatch");

        cholmod_common cc;
        cholmod_start(&cc);
        cc.itype = CHOLMOD_LONG;
        cc.dtype = CHOLMOD_SINGLE;

        const int nnz = (int)A.row_ind.size();
        cholmod_sparse* S = cholmod_allocate_sparse(
            (size_t)n, (size_t)n, (size_t)nnz,
            /*sorted=*/1,
            /*packed=*/1,
            /*stype=*/-1,
            CHOLMOD_REAL,
            &cc);
        if (!S)
        {
            cholmod_finish(&cc);
            throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: cholmod_allocate_sparse failed");
        }

        auto* Sp = reinterpret_cast<SuiteSparse_long*>(S->p);
        auto* Si = reinterpret_cast<SuiteSparse_long*>(S->i);
        auto* Sx = reinterpret_cast<float*>(S->x);
        for (int j = 0; j < n + 1; ++j) Sp[(std::size_t)j] = (SuiteSparse_long)A.col_ptr[(std::size_t)j];
        for (int p = 0; p < nnz; ++p)
        {
            Si[(std::size_t)p] = (SuiteSparse_long)A.row_ind[(std::size_t)p];
            Sx[(std::size_t)p] = A.values[(std::size_t)p];
        }

        std::vector<SuiteSparse_long> perm_long((std::size_t)n);
        for (int k = 0; k < n; ++k) perm_long[(std::size_t)k] = (SuiteSparse_long)P.perm[(std::size_t)k];

        cholmod_sparse* Spm = cholmod_l_ptranspose(S, /*values=*/1,
                                                 perm_long.data(),
                                                 /*fset=*/nullptr, /*fsize=*/0, &cc);
        if (!Spm)
        {
            cholmod_free_sparse(&S, &cc);
            cholmod_finish(&cc);
            throw std::runtime_error("apply_symmetric_permutation_csc_lower_inplace: cholmod_ptranspose failed");
        }

        const auto* Pp = reinterpret_cast<const SuiteSparse_long*>(Spm->p);
        const auto* Pi = reinterpret_cast<const SuiteSparse_long*>(Spm->i);
        const auto* Px = reinterpret_cast<const float*>(Spm->x);
        const int new_nnz = (int)Pp[(std::size_t)n];

        A.col_ptr.assign((std::size_t)n + 1, 0);
        A.row_ind.assign((std::size_t)new_nnz, 0);
        A.values.assign((std::size_t)new_nnz, 0.0f);
        A.nnz = new_nnz;
        for (int j = 0; j < n + 1; ++j) A.col_ptr[(std::size_t)j] = (int)Pp[(std::size_t)j];
        for (int p = 0; p < new_nnz; ++p)
        {
            A.row_ind[(std::size_t)p] = (int)Pi[(std::size_t)p];
            A.values[(std::size_t)p]  = Px[(std::size_t)p];
        }

        cholmod_free_sparse(&Spm, &cc);
        cholmod_free_sparse(&S, &cc);
        cholmod_finish(&cc);
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
inline Mat make_seq_sbaij_from_lower_csc_pattern(int n,
                                                 const std::vector<int> &col_ptr,
                                                 const std::vector<int> &row_ind,
                                                 std::vector<PetscInt> &iptr_upper,
                                                 std::vector<PetscInt> &jind_upper,
                                                 std::vector<PetscScalar> &aval_upper)
{
    if (n <= 0)
        return nullptr;

    lower_csc_to_upper_csr_for_sbaij(n, col_ptr, row_ind, iptr_upper, jind_upper);

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

        inline ichol::symbolic::Permutation ordering_from_mat(Mat A, MatOrderingType ord, int n)
        {
            IS rperm = nullptr, cperm = nullptr;
            PetscErrorCode ierr = MatGetOrdering(A, ord, &rperm, &cperm);
            petsc_check(ierr, "MatGetOrdering");

            const PetscInt *idx = nullptr;
            ierr = ISGetIndices(rperm, &idx);
            petsc_check(ierr, "ISGetIndices");

            ichol::symbolic::Permutation P;
            P.perm.assign(n, 0);
            P.inv_perm.assign(n, 0);

            for (int k = 0; k < n; ++k)
            {
                const PetscInt orig_pi = idx[k];
                if (orig_pi < 0 || orig_pi >= (PetscInt)n)
                    throw std::runtime_error("PETSc ordering returned invalid permutation entry.");
                if (orig_pi > (PetscInt)std::numeric_limits<int>::max())
                    throw std::runtime_error("PETSc ordering entry does not fit in int.");

                const int orig = (int)orig_pi;
                P.perm[k] = orig;
                P.inv_perm[orig] = k;
            }

            ierr = ISRestoreIndices(rperm, &idx);
            petsc_check(ierr, "ISRestoreIndices");

            ISDestroy(&rperm);
            ISDestroy(&cperm);
            return P;
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

} // namespace ichol::symbolic
