#include "symbolic.hpp"

#include <numeric>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include <petscmat.h>
#include <petscis.h>

extern "C"
{
#include <amd.h>
#include <cholmod.h>
}

namespace
{
    void init_petsc()
    {
        // Step 1: query initialization state.
        PetscBool inited = PETSC_FALSE;
        PetscInitialized(&inited);
        if (!inited)
        {
            // Step 2: initialize PETSc with empty command line.
            int argc = 0;
            char **argv = nullptr;
            PetscInitialize(&argc, &argv, nullptr, nullptr);
        }
    }

    /**
     * @brief Convert lower CSR pattern to upper CSR pattern for PETSc SBAIJ.
     *
     * @param n Matrix size.
     * @param l_row_ptr Lower CSR row pointers.
     * @param l_col_ind Lower CSR column indices.
     * @param u_iptr Output upper CSR row pointers.
     * @param u_jind Output upper CSR column indices.
     *
     * @note the input lower CSR is expected to have diagonal entries and no duplicates.
     */
    void lower_csr_to_petsc_upper(int n,
                                  const std::vector<int> &l_row_ptr,
                                  const std::vector<int> &l_col_ind,
                                  std::vector<PetscInt> &u_iptr,
                                  std::vector<PetscInt> &u_jind)
    {
        std::vector<PetscInt> row_counts((std::size_t)n, 0);
        for (int i = 0; i < n; ++i)
        {
            for (int p = l_row_ptr[i]; p < l_row_ptr[i + 1]; ++p)
                row_counts[l_col_ind[p]] += 1;
        }

        u_iptr.assign(n + 1, 0);
        for (int i = 0; i < n; ++i)
            u_iptr[i + 1] = u_iptr[i] + row_counts[i];

        const PetscInt nnzU = u_iptr[n];
        u_jind.assign(nnzU, 0);
        std::vector<PetscInt> cursor = u_iptr;

        for (int i = 0; i < n; ++i)
        {
            for (int p = l_row_ptr[i]; p < l_row_ptr[i + 1]; ++p)
                u_jind[(std::size_t)cursor[l_col_ind[p]]++] = (PetscInt)i;
        }
    }

    Mat lower_csr_to_petsc_mat(int n,
                               const std::vector<int> &row_ptr,
                               const std::vector<int> &col_ind,
                               std::vector<PetscInt> &iptr_upper,
                               std::vector<PetscInt> &jind_upper,
                               std::vector<PetscScalar> &aval_upper)
    {
        lower_csr_to_petsc_upper(n, row_ptr, col_ind, iptr_upper, jind_upper);
        aval_upper.assign(jind_upper.size(), (PetscScalar)1.0); // dummy values; ordering uses graph
        Mat A = nullptr;
        PetscErrorCode ierr = MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,
                                                          (PetscInt)1, (PetscInt)n, (PetscInt)n,
                                                          iptr_upper.data(),
                                                          jind_upper.data(),
                                                          aval_upper.data(),
                                                          &A);

        ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
        ierr = MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE);

        return A;
    }

    Mat lower_csc_to_petsc_mat(
        int n,
        const std::vector<int> &col_ptr,
        const std::vector<int> &row_ind,
        std::vector<PetscInt> &iptr_upper,
        std::vector<PetscInt> &jind_upper,
        std::vector<PetscScalar> &aval_upper)
    {
        std::vector<int> row_ptr, col_ind, csr_to_csc_map;
        ichol::matrix::csc_to_csr_pattern_only(n, col_ptr, row_ind, row_ptr, col_ind, csr_to_csc_map);
        return lower_csr_to_petsc_mat(n, row_ptr, col_ind, iptr_upper, jind_upper, aval_upper);
    }

    /**
     * @brief Extract permutation from PETSc `MatGetOrdering`.
     *
     * @note The returned permutation implies perm[new]=old and inv_perm[old]=new.
     */
    inline ichol::symbolic::Permutation get_perm_from_petsc(Mat A, const char *ordering_type, int n)
    {
        init_petsc();
        IS row = nullptr, col = nullptr;
        PetscErrorCode ierr = MatGetOrdering(A, ordering_type, &row, &col);
        const PetscInt *idx = nullptr;
        ierr = ISGetIndices(row, &idx);

        ichol::symbolic::Permutation P;
        P.perm.assign((std::size_t)n, 0);
        P.inv_perm.assign((std::size_t)n, 0);

        for (int newi = 0; newi < n; ++newi)
        {
            const int oldi = idx[newi];
            P.perm[(std::size_t)newi] = oldi;
            P.inv_perm[(std::size_t)oldi] = newi;
        }

        ierr = ISRestoreIndices(row, &idx);
        ISDestroy(&row);
        ISDestroy(&col);
        return P;
    }
}

namespace ichol::symbolic
{
    Permutation amd_from_csr(int n,
                             const std::vector<int> &row_ptr,
                             const std::vector<int> &col_ind)
    {
        std::vector<int> Ap, Ai, col_csr_pos;
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

        for (int k = 0; k < n; ++k)
            P.inv_perm[P.perm[k]] = k;

        return P;
    }

    Permutation amd_from_csc(int n,
                             const std::vector<int> &col_ptr,
                             const std::vector<int> &row_ind)
    {
        ichol::symbolic::Permutation P;
        P.perm.assign(n, 0);
        P.inv_perm.assign(n, 0);

        double Info[AMD_INFO];
        const int status = amd_order(n, col_ptr.data(), row_ind.data(), P.perm.data(), nullptr, Info);

        for (int k = 0; k < n; ++k)
            P.inv_perm[P.perm[k]] = k;

        return P;
    }

    Permutation rcm_from_csr(int n,
                             const std::vector<int> &row_ptr,
                             const std::vector<int> &col_ind)
    {
        std::vector<PetscInt> iptr_upper, jind_upper;
        std::vector<PetscScalar> aval_upper;

        Mat A = lower_csr_to_petsc_mat(n, row_ptr, col_ind,
                                       iptr_upper, jind_upper, aval_upper);

        Permutation P = get_perm_from_petsc(A, MATORDERINGRCM, n);

        MatDestroy(&A);
        return P;
    }

    Permutation rcm_from_csc(int n,
                             const std::vector<int> &col_ptr,
                             const std::vector<int> &row_ind)
    {
        std::vector<PetscInt> iptr_upper, jind_upper;
        std::vector<PetscScalar> aval_upper;

        Mat A = lower_csc_to_petsc_mat(n, col_ptr, row_ind,
                                       iptr_upper, jind_upper, aval_upper);

        Permutation P = get_perm_from_petsc(A, MATORDERINGRCM, n);
        MatDestroy(&A);
        return P;
    }

    Permutation nd_from_csr(int n,
                            const std::vector<int> &row_ptr,
                            const std::vector<int> &col_ind)
    {
        std::vector<PetscInt> iptr_upper, jind_upper;
        std::vector<PetscScalar> aval_upper;

        Mat A = lower_csr_to_petsc_mat(n, row_ptr, col_ind,
                                       iptr_upper, jind_upper, aval_upper);

        Permutation P = get_perm_from_petsc(A, MATORDERINGND, n);

        MatDestroy(&A);
        return P;
    }

    Permutation nd_from_csc(int n,
                            const std::vector<int> &col_ptr,
                            const std::vector<int> &row_ind)
    {
        // Step 1: construct temporary PETSc matrix from pattern.
        std::vector<PetscInt> iptr_upper, jind_upper;
        std::vector<PetscScalar> aval_upper;

        Mat A = lower_csc_to_petsc_mat(n, col_ptr, row_ind,
                                       iptr_upper, jind_upper, aval_upper);

        // Step 2: extract ND ordering and destroy temporary matrix.
        Permutation P = get_perm_from_petsc(A, MATORDERINGND, n);
        MatDestroy(&A);
        return P;
    }

    Permutation identity_permutation(int n)
    {
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

    /**
     * @brief Apply symmetric permutation to a CSR symmetric matrix, with sorted rows.
     *
     * @note
     * It is an in-place permutation.
     * It expects a lower triangular pattern symmetric matrix.
     * It outputs a lower triangular with sorted rows.
     * No deduplication check.
     */
    template <typename T>
    void apply_permutation_csr(ichol::matrix::CsrMatrix<T> &A, const Permutation &P)
    {
        const int n = A.row_ptr.size() - 1;
        const auto &p = P.inv_perm; // inv_perm maps old->new
        const int nnz = A.row_ptr[n];

        // count entries per destination row after permutation.
        std::vector<int> row_counts(n, 0);
        for (int i = 0; i < n; ++i)
        {
            const int pi = p[i]; // destination row
            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            {
                const int pj = p[A.col_ind[k]]; // destination column
                int r = pi, c = pj;
                if (r < c)
                    std::swap(r, c); // project to lower triangle
                row_counts[r]++;
            }
        }

        std::vector<int> new_row_ptr(n + 1, 0);
        std::vector<int> new_col_ind(nnz);
        std::vector<T> new_values(nnz);

        for (int r = 0; r < n; ++r)
            new_row_ptr[r + 1] = new_row_ptr[r] + row_counts[r];

        // Per-row write cursor.
        std::vector<int> write_ptr = new_row_ptr;

        for (int i = 0; i < n; ++i)
        {
            const int pi = p[i]; // destination row
            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            {
                const int pj = p[A.col_ind[k]]; // destination column
                int r = pi, c = pj;
                if (r < c)
                {
                    std::swap(r, c);
                }

                const int dst = write_ptr[r]++;
                new_col_ind[dst] = c;
                new_values[dst] = A.values[k];
            }
        }

        struct Entry
        {
            int col;
            T val;
        };
        std::vector<Entry> tmp;

        for (int r = 0; r < n; ++r)
        {
            const int begin = new_row_ptr[r];
            const int end = new_row_ptr[r + 1];
            const int len = end - begin;
            if (len <= 1)
                continue;

            tmp.resize(len);
            for (int t = 0; t < len; ++t)
            {
                const int k = begin + t;
                tmp[static_cast<size_t>(t)] = {
                    new_col_ind[static_cast<size_t>(k)],
                    new_values[static_cast<size_t>(k)]};
            }

            std::sort(tmp.begin(), tmp.end(),
                      [](const Entry &a, const Entry &b)
                      { return a.col < b.col; });

            for (int t = 0; t < len; ++t)
            {
                const int k = begin + t;
                new_col_ind[static_cast<size_t>(k)] = tmp[static_cast<size_t>(t)].col;
                new_values[static_cast<size_t>(k)] = tmp[static_cast<size_t>(t)].val;
            }
        }

        A.row_ptr = std::move(new_row_ptr);
        A.col_ind = std::move(new_col_ind);
        A.values = std::move(new_values);
    }

    /**
     * @brief Apply symmetric permutation to a CSC square matrix, with sorted columns.
     *
     * @note See `apply_permutation_csr` for assumptions and output format.
     */
    template <typename T>
    void apply_permutation_csc(ichol::matrix::CscMatrix<T> &A, const Permutation &P)
    {
        const int n = A.col_ptr.size() - 1;
        const auto &p = P.inv_perm;
        const int nnz = A.col_ptr[n];

        // count entries per destination column after permutation.
        std::vector<int> col_counts(n, 0);
        for (int j = 0; j < n; ++j)
        {
            const int pj = p[j]; // destination column
            for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k)
            {
                const int pi = p[A.row_ind[k]]; // destination row
                int r = pi, c = pj;
                if (r < c)
                    std::swap(r, c); // project to lower triangle
                col_counts[c]++;
            }
        }

        std::vector<int> new_col_ptr(n + 1, 0);
        std::vector<int> new_row_ind(nnz);
        std::vector<T> new_values(nnz);

        for (int c = 0; c < n; ++c)
            new_col_ptr[c + 1] = new_col_ptr[c] + col_counts[c];

        // Per-column write cursor.
        std::vector<int> write_ptr = new_col_ptr;

        for (int j = 0; j < n; ++j)
        {
            const int pj = p[j]; // destination column
            for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k)
            {
                const int pi = p[A.row_ind[k]]; // destination row
                int r = pi, c = pj;
                if (r < c)
                {
                    std::swap(r, c);
                }

                const int dst = write_ptr[c]++;
                new_row_ind[dst] = r;
                new_values[dst] = A.values[k];
            }
        }

        struct Entry
        {
            int row;
            T val;
        };
        std::vector<Entry> tmp;

        for (int c = 0; c < n; ++c)
        {
            const int begin = new_col_ptr[c];
            const int end = new_col_ptr[c + 1];
            const int len = end - begin;
            if (len <= 1)
                continue;

            tmp.resize(len);
            for (int t = 0; t < len; ++t)
            {
                const int k = begin + t;
                tmp[static_cast<size_t>(t)] = {
                    new_row_ind[static_cast<size_t>(k)],
                    new_values[static_cast<size_t>(k)]};
            }

            std::sort(tmp.begin(), tmp.end(),
                      [](const Entry &a, const Entry &b)
                      { return a.row < b.row; });
            for (int t = 0; t < len; ++t)
            {
                const int k = begin + t;
                new_row_ind[static_cast<size_t>(k)] = tmp[static_cast<size_t>(t)].row;
                new_values[static_cast<size_t>(k)] = tmp[static_cast<size_t>(t)].val;
            }
        }
        A.col_ptr = std::move(new_col_ptr);
        A.row_ind = std::move(new_row_ind);
        A.values = std::move(new_values);
    }

    /**
     * @brief Apply permutation to a dense vector.
     *
     * @tparam T Element type.
     * @param v Input vector in old ordering.
     * @param P Permutation where `perm[new]=old`.
     * @return Vector in new ordering (`out[new] = v[old]`).
     */
    template <typename T>
    std::vector<T> apply_permutation_vec(const std::vector<T> &v,
                                         const ichol::symbolic::Permutation &P)
    {
        const int n = static_cast<int>(v.size());
        std::vector<T> out(n);
        for (int k = 0; k < n; ++k)
            out[static_cast<std::size_t>(k)] = v[static_cast<std::size_t>(P.perm[k])];
        return out;
    }

    template void apply_permutation_csr<double>(ichol::matrix::CsrMatrix<double> &, const Permutation &);
    template void apply_permutation_csr<float>(ichol::matrix::CsrMatrix<float> &, const Permutation &);
    template void apply_permutation_csr<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &, const Permutation &);

    template void apply_permutation_csc<double>(ichol::matrix::CscMatrix<double> &, const Permutation &);
    template void apply_permutation_csc<float>(ichol::matrix::CscMatrix<float> &, const Permutation &);
    template void apply_permutation_csc<half_float::half>(ichol::matrix::CscMatrix<half_float::half> &, const Permutation &);

    template std::vector<double> apply_permutation_vec<double>(const std::vector<double> &v,
                                                               const ichol::symbolic::Permutation &P);
    template std::vector<float> apply_permutation_vec<float>(const std::vector<float> &v,
                                                             const ichol::symbolic::Permutation &P);
    template std::vector<half_float::half> apply_permutation_vec<half_float::half>(
        const std::vector<half_float::half> &v,
        const ichol::symbolic::Permutation &P);
} // namespace ichol::symbolic
