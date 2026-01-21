#include "symbolic.hpp"

#include <numeric>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace ichol::symbolic
{
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

    namespace detail
    {
        inline void petsc_check(PetscErrorCode ierr, const char *where)
        {
            if (ierr)
                throw std::runtime_error(std::string(where) + " failed (PetscErrorCode=" + std::to_string((int)ierr) + ").");
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
