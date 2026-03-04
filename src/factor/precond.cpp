#include "ichol/matrix_formats.hpp"
#include "ichol/preconditioner.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/factorize.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ichol::precond
{
    namespace
    {
        inline int flatten_local_3d(int x, int y, int z, int w, int h)
        {
            return x + y * w + z * (w * h);
        }

        inline void unflatten_global_3d(int gi, int n, int &x, int &y, int &z)
        {
            const int plane = n * n;
            z = gi / plane;
            const int rem = gi - z * plane;
            y = rem / n;
            x = rem - y * n;
        }

        inline int local_from_global(int gj, int n, int x0, int x1, int y0, int y1, int z0, int z1, int lw, int lh)
        {
            int x = 0, y = 0, z = 0;
            unflatten_global_3d(gj, n, x, y, z);
            if (x < x0 || x >= x1 || y < y0 || y >= y1 || z < z0 || z >= z1)
                return -1;
            return flatten_local_3d(x - x0, y - y0, z - z0, lw, lh);
        }

        static ichol::matrix::CsrMatrix<double> extract_lower_subdomain_csr(
            const ichol::matrix::CsrMatrix<double> &A,
            int n,
            int x0,
            int x1,
            int y0,
            int y1,
            int z0,
            int z1)
        {
            const int lw = x1 - x0;
            const int lh = y1 - y0;
            const int ld = z1 - z0;
            const int nsub = lw * lh * ld;

            ichol::matrix::CsrMatrix<double> sub;
            sub.num_rows = nsub;
            sub.num_cols = nsub;
            sub.row_ptr.resize((size_t)nsub + 1, 0);

            std::vector<int> cols;
            std::vector<double> vals;

            for (int li = 0; li < nsub; ++li)
            {
                const int plane = lw * lh;
                const int lz = li / plane;
                const int rem = li - lz * plane;
                const int ly = rem / lw;
                const int lx = rem - ly * lw;
                const int gi = (x0 + lx) + (y0 + ly) * n + (z0 + lz) * (n * n);

                std::vector<std::pair<int, double>> row_entries;
                row_entries.reserve((size_t)(A.row_ptr[gi + 1] - A.row_ptr[gi]));

                for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
                {
                    const int lj = local_from_global(A.col_ind[kk], n, x0, x1, y0, y1, z0, z1, lw, lh);
                    if (lj < 0 || lj > li)
                        continue;
                    row_entries.push_back({lj, A.values[kk]});
                }

                std::sort(row_entries.begin(), row_entries.end(), [](const auto &u, const auto &v)
                          { return u.first < v.first; });

                int diag_pos = -1;
                for (int i = 0; i < (int)row_entries.size(); ++i)
                {
                    if (row_entries[(size_t)i].first == li)
                    {
                        diag_pos = i;
                        break;
                    }
                }
                if (diag_pos < 0)
                    throw std::runtime_error("extract_lower_subdomain_csr: missing diagonal");

                for (int i = 0; i < (int)row_entries.size(); ++i)
                {
                    if (i == diag_pos)
                        continue;
                    cols.push_back(row_entries[(size_t)i].first);
                    vals.push_back(row_entries[(size_t)i].second);
                }
                cols.push_back(li);
                vals.push_back(row_entries[(size_t)diag_pos].second);
                sub.row_ptr[(size_t)li + 1] = (int)cols.size();
            }

            sub.col_ind = std::move(cols);
            sub.values = std::move(vals);
            sub.nnz = (int)sub.values.size();
            return sub;
        }
    } // namespace

    /**
     * Generates Mx and My as defined in Section 4.1.1 of the paper.
     * A = Mx + My, where Mx is the x-derivatives and My is the y-derivatives.
     */
    template <typename T>
    void generateADIPreconditioners(int n, T epsilon, ichol::matrix::CsrMatrix<T> &Mx, ichol::matrix::CsrMatrix<T> &My)
    {
        int N = n * n;
        T h2 = (T)1.0 / ((n + 1) * (n + 1));

        auto initMat = [&](ichol::matrix::CsrMatrix<T> &M)
        {
            M.num_rows = N;
            M.num_cols = N;
            M.row_ptr.assign(N + 1, 0);
            M.col_ind.clear();
            M.values.clear();
        };

        initMat(Mx);
        initMat(My);

        for (int i = 0; i < n; ++i)
        { // y-index
            for (int j = 0; j < n; ++j)
            { // x-index
                int row = i * n + j;

                // --- Construct Mx (x-direction tridiagonal blocks) ---
                if (j > 0)
                {
                    Mx.col_ind.push_back(row - 1);
                    Mx.values.push_back(-1.0 / h2);
                }
                Mx.col_ind.push_back(row);
                Mx.values.push_back(2.0 / h2);
                if (j < n - 1)
                {
                    Mx.col_ind.push_back(row + 1);
                    Mx.values.push_back(-1.0 / h2);
                }
                Mx.row_ptr[row + 1] = Mx.col_ind.size();

                // --- Construct My (y-direction tridiagonal blocks) ---
                if (i > 0)
                {
                    My.col_ind.push_back(row - n);
                    My.values.push_back(-epsilon / h2);
                }
                My.col_ind.push_back(row);
                My.values.push_back(2.0 * epsilon / h2);
                if (i < n - 1)
                {
                    My.col_ind.push_back(row + n);
                    My.values.push_back(-epsilon / h2);
                }
                My.row_ptr[row + 1] = My.col_ind.size();
            }
        }
        Mx.nnz = Mx.values.size();
        My.nnz = My.values.size();
    }

    template void generateADIPreconditioners(int n, double epsilon, ichol::matrix::CsrMatrix<double> &Mx, ichol::matrix::CsrMatrix<double> &My);

    ichol::matrix::CsrMatrix<double> build_block_diagonal_exact_preconditioner_3d(
        const ichol::matrix::CsrMatrix<double> &A,
        int n,
        int sub_w,
        int sub_h,
        int sub_d)
    {
        if (n <= 0 || sub_w <= 0 || sub_h <= 0 || sub_d <= 0)
            throw std::runtime_error("build_block_diagonal_exact_preconditioner_3d: invalid dimensions");
        if (A.num_rows != n * n * n || A.num_cols != n * n * n)
            throw std::runtime_error("build_block_diagonal_exact_preconditioner_3d: matrix size does not match n^3");

        const int N = A.num_rows;
        std::vector<std::vector<std::pair<int, double>>> rows((size_t)N);

        for (int z0 = 0; z0 < n; z0 += sub_d)
        {
            const int z1 = std::min(z0 + sub_d, n);
            for (int y0 = 0; y0 < n; y0 += sub_h)
            {
                const int y1 = std::min(y0 + sub_h, n);
                for (int x0 = 0; x0 < n; x0 += sub_w)
                {
                    const int x1 = std::min(x0 + sub_w, n);
                    const int lw = x1 - x0;
                    const int lh = y1 - y0;
                    const int ld = z1 - z0;
                    const int nsub = lw * lh * ld;

                    auto A_sub = extract_lower_subdomain_csr(A, n, x0, x1, y0, y1, z0, z1);

                    ichol::SymbolicOptions sym_opts;
                    sym_opts.ordering = ichol::Ordering::Identity;
                    sym_opts.level_k = -1;
                    auto sym_plan = ichol::symbolic::ic_analyze(A_sub, sym_opts);

                    ichol::IncompleteCholeskyOptions ic_opts;
                    ic_opts.scaling = ichol::Scaling::None;
                    ic_opts.pivot_shift_strategy = ichol::PivotShiftStrategy::None;
                    ic_opts.algorithm = ichol::FactorizationAlgorithm::ICKDT;
                    ic_opts.max_restarts = 1;
                    ic_opts.verbose = false;
                    ic_opts.lfil = A_sub.num_rows;
                    ic_opts.drop_tol = 0.0;

                    ichol::numeric::NumericPlan num_plan;
                    auto L_sub = ichol::numeric::incomplete_cholesky_preconditioner<double>(A_sub, sym_plan, num_plan, ic_opts);

                    for (int li = 0; li < nsub; ++li)
                    {
                        const int plane = lw * lh;
                        const int lz = li / plane;
                        const int rem = li - lz * plane;
                        const int ly = rem / lw;
                        const int lx = rem - ly * lw;
                        const int gi = (x0 + lx) + (y0 + ly) * n + (z0 + lz) * (n * n);

                        rows[(size_t)gi].clear();
                        rows[(size_t)gi].reserve((size_t)(L_sub.row_ptr[li + 1] - L_sub.row_ptr[li]));

                        for (int p = L_sub.row_ptr[li]; p < L_sub.row_ptr[li + 1]; ++p)
                        {
                            const int lj = L_sub.col_ind[p];
                            const int llz = lj / plane;
                            const int lrem = lj - llz * plane;
                            const int lly = lrem / lw;
                            const int llx = lrem - lly * lw;
                            const int gj = (x0 + llx) + (y0 + lly) * n + (z0 + llz) * (n * n);
                            rows[(size_t)gi].push_back({gj, L_sub.values[p]});
                        }

                        std::sort(rows[(size_t)gi].begin(), rows[(size_t)gi].end(), [](const auto &a, const auto &b)
                                  { return a.first < b.first; });

                        int diag_pos = -1;
                        for (int t = 0; t < (int)rows[(size_t)gi].size(); ++t)
                        {
                            if (rows[(size_t)gi][(size_t)t].first == gi)
                            {
                                diag_pos = t;
                                break;
                            }
                        }
                        if (diag_pos < 0)
                            throw std::runtime_error("build_block_diagonal_exact_preconditioner_3d: missing diagonal in block factor");
                        if (diag_pos != (int)rows[(size_t)gi].size() - 1)
                        {
                            const auto diag = rows[(size_t)gi][(size_t)diag_pos];
                            rows[(size_t)gi].erase(rows[(size_t)gi].begin() + diag_pos);
                            rows[(size_t)gi].push_back(diag);
                        }
                    }
                }
            }
        }

        ichol::matrix::CsrMatrix<double> L;
        L.num_rows = N;
        L.num_cols = N;
        L.row_ptr.resize((size_t)N + 1, 0);
        for (int i = 0; i < N; ++i)
            L.row_ptr[(size_t)i + 1] = L.row_ptr[(size_t)i] + (int)rows[(size_t)i].size();
        L.nnz = L.row_ptr[(size_t)N];
        L.col_ind.resize((size_t)L.nnz);
        L.values.resize((size_t)L.nnz);

        int p = 0;
        for (int i = 0; i < N; ++i)
        {
            for (const auto &entry : rows[(size_t)i])
            {
                L.col_ind[(size_t)p] = entry.first;
                L.values[(size_t)p] = entry.second;
                ++p;
            }
        }

        return L;
    }

    ichol::matrix::CsrMatrix<double> build_block_diagonal_exact_preconditioner_2d(
        const ichol::matrix::CsrMatrix<double> &A,
        int n,
        int sub_w,
        int sub_h)
    {
        if (n <= 0 || sub_w <= 0 || sub_h <= 0)
            throw std::runtime_error("build_block_diagonal_exact_preconditioner_2d: invalid dimensions");
        if (A.num_rows != n * n || A.num_cols != n * n)
            throw std::runtime_error("build_block_diagonal_exact_preconditioner_2d: matrix size does not match n^2");

        const int N = A.num_rows;
        std::vector<std::vector<std::pair<int, double>>> rows((size_t)N);

        for (int y0 = 0; y0 < n; y0 += sub_h)
        {
            const int y1 = std::min(y0 + sub_h, n);
            for (int x0 = 0; x0 < n; x0 += sub_w)
            {
                const int x1 = std::min(x0 + sub_w, n);
                const int lw = x1 - x0;
                const int lh = y1 - y0;
                const int nsub = lw * lh;

                ichol::matrix::CsrMatrix<double> A_sub;
                A_sub.num_rows = nsub;
                A_sub.num_cols = nsub;
                A_sub.row_ptr.resize((size_t)nsub + 1, 0);

                std::vector<int> cols;
                std::vector<double> vals;
                for (int li = 0; li < nsub; ++li)
                {
                    const int ly = li / lw;
                    const int lx = li - ly * lw;
                    const int gi = (x0 + lx) + (y0 + ly) * n;

                    std::vector<std::pair<int, double>> row_entries;
                    row_entries.reserve((size_t)(A.row_ptr[gi + 1] - A.row_ptr[gi]));
                    for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
                    {
                        const int gj = A.col_ind[kk];
                        const int gx = gj % n;
                        const int gy = gj / n;
                        if (gx < x0 || gx >= x1 || gy < y0 || gy >= y1)
                            continue;
                        const int lj = (gx - x0) + (gy - y0) * lw;
                        if (lj > li)
                            continue;
                        row_entries.push_back({lj, A.values[kk]});
                    }

                    std::sort(row_entries.begin(), row_entries.end(), [](const auto &u, const auto &v)
                              { return u.first < v.first; });
                    int diag_pos = -1;
                    for (int t = 0; t < (int)row_entries.size(); ++t)
                    {
                        if (row_entries[(size_t)t].first == li)
                        {
                            diag_pos = t;
                            break;
                        }
                    }
                    if (diag_pos < 0)
                        throw std::runtime_error("build_block_diagonal_exact_preconditioner_2d: missing diagonal");

                    for (int t = 0; t < (int)row_entries.size(); ++t)
                    {
                        if (t == diag_pos)
                            continue;
                        cols.push_back(row_entries[(size_t)t].first);
                        vals.push_back(row_entries[(size_t)t].second);
                    }
                    cols.push_back(li);
                    vals.push_back(row_entries[(size_t)diag_pos].second);
                    A_sub.row_ptr[(size_t)li + 1] = (int)cols.size();
                }
                A_sub.col_ind = std::move(cols);
                A_sub.values = std::move(vals);
                A_sub.nnz = (int)A_sub.values.size();

                ichol::SymbolicOptions sym_opts;
                sym_opts.ordering = ichol::Ordering::Identity;
                sym_opts.level_k = -1;
                auto sym_plan = ichol::symbolic::ic_analyze(A_sub, sym_opts);

                ichol::IncompleteCholeskyOptions ic_opts;
                ic_opts.scaling = ichol::Scaling::None;
                ic_opts.pivot_shift_strategy = ichol::PivotShiftStrategy::None;
                ic_opts.algorithm = ichol::FactorizationAlgorithm::ICKDT;
                ic_opts.max_restarts = 1;
                ic_opts.verbose = false;
                ic_opts.lfil = A_sub.num_rows;
                ic_opts.drop_tol = 0.0;

                ichol::numeric::NumericPlan num_plan;
                auto L_sub = ichol::numeric::incomplete_cholesky_preconditioner<double>(A_sub, sym_plan, num_plan, ic_opts);

                for (int li = 0; li < nsub; ++li)
                {
                    const int ly = li / lw;
                    const int lx = li - ly * lw;
                    const int gi = (x0 + lx) + (y0 + ly) * n;

                    rows[(size_t)gi].clear();
                    rows[(size_t)gi].reserve((size_t)(L_sub.row_ptr[li + 1] - L_sub.row_ptr[li]));
                    for (int p = L_sub.row_ptr[li]; p < L_sub.row_ptr[li + 1]; ++p)
                    {
                        const int lj = L_sub.col_ind[p];
                        const int lly = lj / lw;
                        const int llx = lj - lly * lw;
                        const int gj = (x0 + llx) + (y0 + lly) * n;
                        rows[(size_t)gi].push_back({gj, L_sub.values[p]});
                    }

                    std::sort(rows[(size_t)gi].begin(), rows[(size_t)gi].end(), [](const auto &a, const auto &b)
                              { return a.first < b.first; });
                    int diag_pos = -1;
                    for (int t = 0; t < (int)rows[(size_t)gi].size(); ++t)
                    {
                        if (rows[(size_t)gi][(size_t)t].first == gi)
                        {
                            diag_pos = t;
                            break;
                        }
                    }
                    if (diag_pos < 0)
                        throw std::runtime_error("build_block_diagonal_exact_preconditioner_2d: missing factor diagonal");
                    if (diag_pos != (int)rows[(size_t)gi].size() - 1)
                    {
                        const auto diag = rows[(size_t)gi][(size_t)diag_pos];
                        rows[(size_t)gi].erase(rows[(size_t)gi].begin() + diag_pos);
                        rows[(size_t)gi].push_back(diag);
                    }
                }
            }
        }

        ichol::matrix::CsrMatrix<double> L;
        L.num_rows = N;
        L.num_cols = N;
        L.row_ptr.resize((size_t)N + 1, 0);
        for (int i = 0; i < N; ++i)
            L.row_ptr[(size_t)i + 1] = L.row_ptr[(size_t)i] + (int)rows[(size_t)i].size();
        L.nnz = L.row_ptr[(size_t)N];
        L.col_ind.resize((size_t)L.nnz);
        L.values.resize((size_t)L.nnz);

        int p = 0;
        for (int i = 0; i < N; ++i)
        {
            for (const auto &entry : rows[(size_t)i])
            {
                L.col_ind[(size_t)p] = entry.first;
                L.values[(size_t)p] = entry.second;
                ++p;
            }
        }
        return L;
    }
} // namespace ichol::precond
