// src/apps/ict_pcg_main.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <petscsys.h>

#include "ichol/half.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "ichol/pcg.hpp"
#include "factor/numerical/factorize.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "backends/cpu/util/cast.hpp"

namespace
{
    struct AppOptions
    {
        std::string matrix_path = "lap3d";
        int laplacian_dim = 100;
        std::string precision_fact = "float";
        std::string precision_pcg = "float";

        ichol::SymbolicOptions sym;
        ichol::IncompleteCholeskyOptions ic;
    };

    // -------------------- string helpers

    std::string trim_copy(std::string s)
    {
        auto not_space = [](unsigned char c)
        { return !std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
    }

    std::string to_lower_copy(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    bool ends_with(const std::string &s, const std::string &suffix)
    {
        if (s.size() < suffix.size())
            return false;
        return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
    }

    bool is_list_value(const std::string &raw)
    {
        auto s = trim_copy(raw);
        return s.size() >= 2 && s.front() == '[' && s.back() == ']';
    }

    std::vector<std::string> split_list_tokens(const std::string &raw)
    {
        // raw is something like "[0, 2, 4]" or "0"
        auto s = trim_copy(raw);
        if (!is_list_value(s))
            return {trim_copy(s)};

        std::string inside = trim_copy(s.substr(1, s.size() - 2));
        std::vector<std::string> out;

        std::string tok;
        std::stringstream ss(inside);
        while (std::getline(ss, tok, ','))
        {
            tok = trim_copy(tok);
            if (!tok.empty())
                out.push_back(tok);
        }
        if (out.empty())
            out.push_back(std::string{}); // allow "[]", though likely unintended
        return out;
    }

    // -------------------- kv parsing

    std::unordered_map<std::string, std::string> read_kv_file(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Failed to open options file: " + path);

        std::unordered_map<std::string, std::string> kv;
        std::string line;
        while (std::getline(in, line))
        {
            auto trimmed = trim_copy(line);
            if (trimmed.empty() || trimmed[0] == '#')
                continue;

            auto pos = trimmed.find('=');
            if (pos == std::string::npos)
                continue;

            std::string key = trim_copy(trimmed.substr(0, pos));
            std::string value = trim_copy(trimmed.substr(pos + 1));
            if (!key.empty())
                kv[key] = value;
        }
        return kv;
    }

    std::string get_raw(const std::unordered_map<std::string, std::string> &kv,
                        const std::string &key,
                        const std::string &def)
    {
        auto it = kv.find(key);
        return (it == kv.end()) ? def : it->second;
    }

    std::vector<std::string> get_tokens(const std::unordered_map<std::string, std::string> &kv,
                                        const std::string &key,
                                        const std::string &def_raw)
    {
        return split_list_tokens(get_raw(kv, key, def_raw));
    }

    std::vector<int> parse_int_list(const std::vector<std::string> &tokens, const std::string &key)
    {
        std::vector<int> out;
        out.reserve(tokens.size());
        try
        {
            for (const auto &t : tokens)
                out.push_back(std::stoi(t));
        }
        catch (...)
        {
            throw std::runtime_error("Failed to parse int list for key '" + key + "'");
        }
        return out;
    }

    std::vector<double> parse_double_list(const std::vector<std::string> &tokens, const std::string &key)
    {
        std::vector<double> out;
        out.reserve(tokens.size());
        try
        {
            for (const auto &t : tokens)
                out.push_back(std::stod(t));
        }
        catch (...)
        {
            throw std::runtime_error("Failed to parse double list for key '" + key + "'");
        }
        return out;
    }

    bool parse_ordering_token(const std::string &s, ichol::Ordering &out)
    {
        auto v = to_lower_copy(s);
        if (v == "identity")
            out = ichol::Ordering::Identity;
        else if (v == "amd")
            out = ichol::Ordering::AMD;
        else if (v == "nesteddissection")
            out = ichol::Ordering::NestedDissection;
        else if (v == "rcm")
            out = ichol::Ordering::RCM;
        else
            return false;
        return true;
    }

    bool parse_scaling_token(const std::string &s, ichol::Scaling &out)
    {
        auto v = to_lower_copy(s);
        if (v == "none")
            out = ichol::Scaling::None;
        else if (v == "unitsqrtdiag")
            out = ichol::Scaling::UnitSqrtDiag;
        else if (v == "unitcolnorm")
            out = ichol::Scaling::UnitColNorm;
        else
            return false;
        return true;
    }

    bool parse_pivot_strategy_token(const std::string &s, ichol::PivotShiftStrategy &out)
    {
        auto v = to_lower_copy(s);
        if (v == "none")
            out = ichol::PivotShiftStrategy::None;
        else if (v == "machineepsilon")
            out = ichol::PivotShiftStrategy::MachineEpsilon;
        else if (v == "static")
            out = ichol::PivotShiftStrategy::Static;
        else if (v == "dynamic")
            out = ichol::PivotShiftStrategy::Dynamic;
        else
            return false;
        return true;
    }

    std::vector<ichol::Ordering> parse_ordering_list(const std::vector<std::string> &tokens, const std::string &key)
    {
        std::vector<ichol::Ordering> out;
        out.reserve(tokens.size());
        for (const auto &t : tokens)
        {
            ichol::Ordering v;
            if (!parse_ordering_token(t, v))
                throw std::runtime_error("Failed to parse ordering token '" + t + "' for key '" + key + "'");
            out.push_back(v);
        }
        return out;
    }

    std::vector<ichol::Scaling> parse_scaling_list(const std::vector<std::string> &tokens, const std::string &key)
    {
        std::vector<ichol::Scaling> out;
        out.reserve(tokens.size());
        for (const auto &t : tokens)
        {
            ichol::Scaling v;
            if (!parse_scaling_token(t, v))
                throw std::runtime_error("Failed to parse scaling token '" + t + "' for key '" + key + "'");
            out.push_back(v);
        }
        return out;
    }

    std::vector<ichol::PivotShiftStrategy> parse_pivot_strategy_list(const std::vector<std::string> &tokens, const std::string &key)
    {
        std::vector<ichol::PivotShiftStrategy> out;
        out.reserve(tokens.size());
        for (const auto &t : tokens)
        {
            ichol::PivotShiftStrategy v;
            if (!parse_pivot_strategy_token(t, v))
                throw std::runtime_error("Failed to parse pivot_shift_strategy token '" + t + "' for key '" + key + "'");
            out.push_back(v);
        }
        return out;
    }

    // -------------------- matrix loader

    ichol::matrix::CsrMatrix<double> load_matrix(const AppOptions &opts)
    {
        if (ends_with(opts.matrix_path, ".mtx"))
            return ichol::io::mtx_to_csr<double>(opts.matrix_path, false);
        else if (opts.matrix_path == "lap2d")
            return ichol::io::gen_2dlap_csr<double>(opts.laplacian_dim);
        else if (opts.matrix_path == "lap3d")
            return ichol::io::gen_3dlap_csr<double>(opts.laplacian_dim);
        else
            throw std::runtime_error("Unknown matrix path: " + opts.matrix_path);
    }

    // -------------------- core run

    template <typename FactT, typename PcgT>
    int run_ic_pcg(const AppOptions &opts)
    {
        auto A = load_matrix(opts);
        const int n = A.num_rows;

        auto sym_plan = ichol::symbolic::ic_analyze<double>(A, opts.sym);
        ichol::numeric::NumericPlan num_plan;

        auto fact_start = std::chrono::high_resolution_clock::now();
        auto ic_opts = opts.ic;
        auto L = ichol::numeric::incomplete_cholesky_preconditioner<FactT>(A, sym_plan, num_plan, ic_opts);
        auto fact_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> fact_duration = fact_end - fact_start;
        std::cout << "IC factorization time: " << fact_duration.count() << " seconds.\n";

        const auto &D = num_plan.prescaling.D;

        std::vector<double> b(n, 1.0);
        std::vector<double> b_perm = (opts.sym.ordering == ichol::Ordering::Identity)
                                         ? b
                                         : ichol::symbolic::apply_permutation_vec(b, sym_plan.perm);
        std::vector<double> b_tilde(n);
        for (int i = 0; i < n; ++i)
            b_tilde[i] = b_perm[i] / D[i];

        std::vector<PcgT> L_values_pcg;
        L_values_pcg.reserve(L.values.size());
        for (const auto &v : L.values)
            L_values_pcg.push_back(ichol::util::cast_fp_type<PcgT>(v));

        std::vector<double> y;
        int iters = 0;
        double finalRes = 0.0;

        auto pcg_start = std::chrono::high_resolution_clock::now();
        ichol::solver::pcg<PcgT>(
            A.row_ptr,
            A.col_ind,
            A.values,
            L.row_ptr,
            L.col_ind,
            L_values_pcg,
            b_tilde,
            y,
            D,
            iters,
            finalRes);
        auto pcg_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> pcg_duration = pcg_end - pcg_start;
        std::cout << "PCG time: " << pcg_duration.count() << " seconds.\n";

        auto vec_norm = [](const std::vector<double> &v)
        {
            double s = 0.0;
            for (double a : v)
                s += a * a;
            return std::sqrt(s);
        };

        auto symm_lower_csr_matvec = [&](const ichol::matrix::CsrMatrix<double> &M,
                                         const std::vector<double> &x,
                                         std::vector<double> &out)
        {
            out.assign(n, 0.0);
            for (int i = 0; i < n; ++i)
            {
                for (int p = M.row_ptr[i]; p < M.row_ptr[i + 1]; ++p)
                {
                    int j = M.col_ind[p];
                    double aij = M.values[p];
                    out[i] += aij * x[j];
                    if (j != i)
                        out[j] += aij * x[i];
                }
            }
        };

        std::vector<double> By(n), rB(n);
        symm_lower_csr_matvec(A, y, By);
        for (int i = 0; i < n; ++i)
            rB[i] = By[i] - b_tilde[i];

        double rBnorm = vec_norm(rB);
        double bTildenorm = vec_norm(b_tilde);
        double relresB = (bTildenorm == 0.0) ? rBnorm : rBnorm / bTildenorm;

        std::cout << "nnz of L : " << L.values.size() << "\n";
        std::cout << "Scaled-system relative residual (B y = b_tilde): " << relresB << "\n";
        std::cout << "Iterations taken by PCG: " << iters << "\n";

        return 0;
    }

    enum class Precision
    {
        Double,
        Float,
        Half
    };

    Precision parse_precision(const std::string &s)
    {
        auto v = to_lower_copy(s);
        if (v == "double")
            return Precision::Double;
        if (v == "float")
            return Precision::Float;
        return Precision::Half;
    }

    int dispatch_run(const AppOptions &opts)
    {
        Precision fact = parse_precision(opts.precision_fact);
        Precision pcg = parse_precision(opts.precision_pcg);

        if (fact == Precision::Double && pcg == Precision::Double)
            return run_ic_pcg<double, double>(opts);
        if (fact == Precision::Double && pcg == Precision::Float)
            return run_ic_pcg<double, float>(opts);
        if (fact == Precision::Double && pcg == Precision::Half)
            return run_ic_pcg<double, half_float::half>(opts);
        if (fact == Precision::Float && pcg == Precision::Double)
            return run_ic_pcg<float, double>(opts);
        if (fact == Precision::Float && pcg == Precision::Float)
            return run_ic_pcg<float, float>(opts);
        if (fact == Precision::Float && pcg == Precision::Half)
            return run_ic_pcg<float, half_float::half>(opts);
        if (fact == Precision::Half && pcg == Precision::Double)
            return run_ic_pcg<half_float::half, double>(opts);
        if (fact == Precision::Half && pcg == Precision::Float)
            return run_ic_pcg<half_float::half, float>(opts);
        return run_ic_pcg<half_float::half, half_float::half>(opts);
    }

    // -------------------- grid search driver

    struct GridField
    {
        std::string key;
        std::size_t count = 1;
        std::function<void(AppOptions &, std::size_t)> apply;
        std::function<void(const AppOptions &, std::ostream &)> print_one; // prints selected value(s) from opts
    };

    std::uint64_t safe_mul_u64(std::uint64_t a, std::uint64_t b)
    {
        if (a == 0 || b == 0)
            return 0;
        if (a > (std::numeric_limits<std::uint64_t>::max() / b))
            return std::numeric_limits<std::uint64_t>::max();
        return a * b;
    }

    void enumerate_grid(const std::vector<GridField> &fields,
                        const AppOptions &base,
                        std::size_t idx,
                        AppOptions &work,
                        std::vector<std::size_t> &choice,
                        const std::function<void(const AppOptions &, const std::vector<std::size_t> &)> &fn)
    {
        if (idx == fields.size())
        {
            fn(work, choice);
            return;
        }

        const auto &f = fields[idx];
        for (std::size_t i = 0; i < f.count; ++i)
        {
            f.apply(work, i);
            choice[idx] = i;
            enumerate_grid(fields, base, idx + 1, work, choice, fn);
        }
    }

    std::vector<GridField> make_grid_fields(const std::unordered_map<std::string, std::string> &kv)
    {
        AppOptions def;

        std::vector<GridField> fields;
        fields.reserve(32);

        // matrix_path
        {
            auto toks = get_tokens(kv, "matrix_path", def.matrix_path);
            std::vector<std::string> vals = toks;
            fields.push_back(GridField{
                "matrix_path",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.matrix_path = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "matrix_path=" << o.matrix_path; }});
        }

        // laplacian_dim
        {
            auto toks = get_tokens(kv, "laplacian_dim", std::to_string(def.laplacian_dim));
            auto vals = parse_int_list(toks, "laplacian_dim");
            fields.push_back(GridField{
                "laplacian_dim",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.laplacian_dim = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "laplacian_dim=" << o.laplacian_dim; }});
        }

        // precision_fact
        {
            auto toks = get_tokens(kv, "precision_fact", def.precision_fact);
            std::vector<std::string> vals = toks;
            fields.push_back(GridField{
                "precision_fact",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.precision_fact = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "precision_fact=" << o.precision_fact; }});
        }

        // precision_pcg
        {
            auto toks = get_tokens(kv, "precision_pcg", def.precision_pcg);
            std::vector<std::string> vals = toks;
            fields.push_back(GridField{
                "precision_pcg",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.precision_pcg = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "precision_pcg=" << o.precision_pcg; }});
        }

        // ordering
        {
            auto toks = get_tokens(kv, "ordering", "Identity");
            auto vals = parse_ordering_list(toks, "ordering");
            fields.push_back(GridField{
                "ordering",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.sym.ordering = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                {
                    os << "ordering=";
                    switch (o.sym.ordering)
                    {
                    case ichol::Ordering::Identity:
                        os << "Identity";
                        break;
                    case ichol::Ordering::AMD:
                        os << "AMD";
                        break;
                    case ichol::Ordering::NestedDissection:
                        os << "NestedDissection";
                        break;
                    case ichol::Ordering::RCM:
                        os << "RCM";
                        break;
                    default:
                        os << "Unknown";
                        break;
                    }
                }});
        }

        // level_k
        {
            auto toks = get_tokens(kv, "level_k", std::to_string(def.sym.level_k));
            auto vals = parse_int_list(toks, "level_k");
            fields.push_back(GridField{
                "level_k",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.sym.level_k = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "level_k=" << o.sym.level_k; }});
        }

        // scaling
        {
            auto toks = get_tokens(kv, "scaling", "UnitSqrtDiag");
            auto vals = parse_scaling_list(toks, "scaling");
            fields.push_back(GridField{
                "scaling",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.ic.scaling = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                {
                    os << "scaling=";
                    switch (o.ic.scaling)
                    {
                    case ichol::Scaling::None:
                        os << "None";
                        break;
                    case ichol::Scaling::UnitSqrtDiag:
                        os << "UnitSqrtDiag";
                        break;
                    case ichol::Scaling::UnitColNorm:
                        os << "UnitColNorm";
                        break;
                    default:
                        os << "Unknown";
                        break;
                    }
                }});
        }

        // pivot_shift_strategy
        {
            auto toks = get_tokens(kv, "pivot_shift_strategy", "Static");
            auto vals = parse_pivot_strategy_list(toks, "pivot_shift_strategy");
            fields.push_back(GridField{
                "pivot_shift_strategy",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.ic.pivot_shift_strategy = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                {
                    os << "pivot_shift_strategy=";
                    switch (o.ic.pivot_shift_strategy)
                    {
                    case ichol::PivotShiftStrategy::None:
                        os << "None";
                        break;
                    case ichol::PivotShiftStrategy::MachineEpsilon:
                        os << "MachineEpsilon";
                        break;
                    case ichol::PivotShiftStrategy::Static:
                        os << "Static";
                        break;
                    case ichol::PivotShiftStrategy::Dynamic:
                        os << "Dynamic";
                        break;
                    default:
                        os << "Unknown";
                        break;
                    }
                }});
        }

        // static_shift
        {
            auto toks = get_tokens(kv, "static_shift", std::to_string(def.ic.static_shift));
            auto vals = parse_double_list(toks, "static_shift");
            fields.push_back(GridField{
                "static_shift",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.ic.static_shift = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "static_shift=" << o.ic.static_shift; }});
        }

        // lfil
        {
            auto toks = get_tokens(kv, "lfil", std::to_string(def.ic.lfil));
            auto vals = parse_int_list(toks, "lfil");
            fields.push_back(GridField{
                "lfil",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.ic.lfil = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "lfil=" << o.ic.lfil; }});
        }

        // drop_tol
        {
            auto toks = get_tokens(kv, "drop_tol", std::to_string(def.ic.drop_tol));
            auto vals = parse_double_list(toks, "drop_tol");
            fields.push_back(GridField{
                "drop_tol",
                vals.size(),
                [vals](AppOptions &o, std::size_t i)
                { o.ic.drop_tol = vals[i]; },
                [](const AppOptions &o, std::ostream &os)
                { os << "drop_tol=" << o.ic.drop_tol; }});
        }

        return fields;
    }

    void print_run_config_line(const std::vector<GridField> &fields, const AppOptions &opts)
    {
        for (std::size_t i = 0; i < fields.size(); ++i)
        {
            fields[i].print_one(opts, std::cout);
            if (i + 1 < fields.size())
                std::cout << "  ";
        }
        std::cout << "\n";
    }

} // namespace

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr)
        return ierr;
    const std::string options_path = (argc > 1) ? argv[1] : "config/ict_pcg_options.txt";

    std::unordered_map<std::string, std::string> kv;
    try
    {
        kv = read_kv_file(options_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::vector<GridField> fields;
    try
    {
        fields = make_grid_fields(kv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::uint64_t total = 1;
    for (const auto &f : fields)
        total = safe_mul_u64(total, static_cast<std::uint64_t>(f.count));

    std::cout << "Grid search total runs: " << total << "\n";

    AppOptions base;
    AppOptions work = base;
    std::vector<std::size_t> choice(fields.size(), 0);

    std::uint64_t run_id = 0;
    enumerate_grid(fields, base, 0, work, choice,
                   [&](const AppOptions &opts, const std::vector<std::size_t> &)
                   {
                       ++run_id;
                       std::cout << "-------------------- run " << run_id << " / " << total << "\n";
                       print_run_config_line(fields, opts);
                       (void)dispatch_run(opts);
                   });

    ierr = PetscFinalize();
    if (ierr)
        return ierr;
    return 0;
}
