#include "symbolic.hpp"

namespace
{
    template <typename T>
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::CsrMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree)
    {
        const int n = A.num_rows;
        ichol::symbolic::FactorPattern factor_pattern;

        return factor_pattern;
    }

    template <typename T>
    ichol::symbolic::FactorPattern compute_ic_factor_pattern(const ichol::CsrMatrix<T> &A,
                                                             int level_k)
    {
        const int n = A.num_rows;
        ichol::symbolic::FactorPattern factor_pattern;

        std::vector<std::vector<int>> L_row(n);   // columns present in row r
        std::vector<std::vector<int>> L_level(n); // the level-of-fill attached to each entry
        std::vector<int> marker(n, -1);           // marker[c] == i means column c is present in row i
        std::vector<int> work_col(n);             // temporary storage for columns in current row
        std::vector<int> work_level(n);           // temporary storage for levels in current row, keyed by column index

        for (int i = 0; i < n; ++i)
        {
            int used = 0;
            const int row_start = A.row_ptr[i];
            const int row_end = A.row_ptr[i + 1];

            // Initial pattern from A
            for (int p = row_start; p < row_end; ++p)
            {
                int j = A.col_ind[p];

                if (marker[j] != i)
                {
                    marker[j] = i;
                    work_col[used++] = j;
                    work_level[j] = 0;
                }
                else if (0 < work_level[j])
                {
                    work_level[j] = 0;
                }
            }

            for (int pos = 0; pos < used; ++pos)
            {
                int j = work_col[pos];
                if (j >= i)
                    continue;

                int level_ij = work_level[j];
                if (level_ij > level_k)
                    continue;

                // Propagate fill from row j
                const std::vector<int> &Lj_row = L_row[j];
                const std::vector<int> &Lj_level = L_level[j];
                const int Lj_size = static_cast<int>(Lj_row.size());

                for (int idx = 0; idx < Lj_size; ++idx)
                {
                    int col = Lj_row[idx];
                    int lev_jc = Lj_level[idx];
                    int new_level = level_ij + lev_jc + 1;
                    if (new_level > level_k)
                        continue;

                    if (marker[col] != i)
                    {
                        marker[col] = i;
                        work_col[used++] = col;
                        work_level[col] = new_level;
                    }
                    else if (new_level < work_level[col])
                    {
                        work_level[col] = new_level;
                    }
                }
            }

            std::sort(work_col.begin(), work_col.begin() + used);
            L_row[i].assign(work_col.begin(), work_col.begin() + used);
            L_level[i].resize(used);
            for (int idx = 0; idx < used; ++idx)
            {
                int col = work_col[idx];
                L_level[i][idx] = work_level[col];
            }
        }

        return factor_pattern;
    }

    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<double>(const ichol::CsrMatrix<double> &A,
                                                                                      const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<double>(const ichol::CsrMatrix<double> &A,
                                                                              int level_k);
    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<float>(const ichol::CsrMatrix<float> &A,
                                                                                     const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<float>(const ichol::CsrMatrix<float> &A,
                                                                             int level_k);
    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<half_float::half>(const ichol::CsrMatrix<half_float::half> &A,
                                                                                                const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<half_float::half>(const ichol::CsrMatrix<half_float::half> &A,
                                                                                        int level_k);
}