#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ichol/matrix_formats.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"
#include "ichol/subdomain_preconditioner_gpu.hpp"
#include "factor/numerical/factorize.hpp"

namespace
{
struct Int3
{
    int x = 0;
    int y = 0;
    int z = 0;
};

struct AppOptions
{
    int n = 32;
    Int3 subdomain{16, 16, 16};
    unsigned int seed = 20260303u;

    ichol::precond::SubdomainPreconditionerKind precond_kind =
        ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    int ic_level_k = 0;
    int fsai_level_k = 0;
    int spai_radius = 1;

    ichol::solver::PCGParams params;
    bool show_help = false;
};

struct SubdomainContextDeleter
{
    void operator()(ichol::precond::SubdomainSpSVContext *ctx) const
    {
        if (ctx != nullptr)
            ichol::precond::destroy_subdomain_spsv_context(ctx);
    }
};

using SubdomainContextPtr = std::unique_ptr<ichol::precond::SubdomainSpSVContext, SubdomainContextDeleter>;

struct SubdomainBundle
{
    std::vector<ichol::precond::SubdomainRegion> regions;
    std::vector<SubdomainContextPtr> contexts;
    std::vector<ichol::precond::PrecondApply> preconds;
    double build_secs = 0.0;
};

struct SolverRun
{
    ichol::solver::PCGResult result;
    double precond_secs = 0.0;
    double solve_secs = 0.0;
};

constexpr int kTruncatedHistory = 20;
constexpr int kMixedHistory64 = 10;
constexpr int kMixedHistory32 = 10;
constexpr int kMixedHistory16 = 10;

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

std::string precision_to_string(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (prec)
    {
    case Prec::FP64:
        return "fp64";
    case Prec::FP32:
        return "fp32";
    case Prec::TF32:
        return "tf32";
    case Prec::FP16:
        return "fp16";
    case Prec::BF16:
        return "bf16";
    case Prec::FP8_E4M3:
        return "fp8_e4m3";
    case Prec::FP8_E5M2:
        return "fp8_e5m2";
    }
    return "unknown";
}

ichol::solver::ComputePrecision parse_compute_precision(const std::string &raw)
{
    const std::string v = to_lower_copy(trim_copy(raw));
    using Prec = ichol::solver::ComputePrecision;
    if (v == "fp64" || v == "double")
        return Prec::FP64;
    if (v == "fp32" || v == "float")
        return Prec::FP32;
    if (v == "tf32")
        return Prec::TF32;
    if (v == "fp16" || v == "half")
        return Prec::FP16;
    if (v == "bf16" || v == "bfloat16")
        return Prec::BF16;
    if (v == "fp8_e4m3" || v == "fp8-e4m3")
        return Prec::FP8_E4M3;
    if (v == "fp8_e5m2" || v == "fp8-e5m2")
        return Prec::FP8_E5M2;
    throw std::runtime_error("Unknown precision: " + raw);
}

Int3 parse_triplet(const std::string &raw)
{
    std::string s = to_lower_copy(trim_copy(raw));
    for (char &c : s)
    {
        if (c == 'x' || c == ',' || c == ':')
            c = ' ';
    }
    std::stringstream ss(s);
    Int3 out{};
    if (!(ss >> out.x >> out.y >> out.z))
        throw std::runtime_error("Expected triplet like 16x16x16, got: " + raw);
    return out;
}

ichol::precond::SubdomainPreconditionerKind parse_precond_kind(const std::string &raw)
{
    const std::string v = to_lower_copy(trim_copy(raw));
    if (v == "exact" || v == "exactcholesky")
        return ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    if (v == "ic" || v == "ichol" || v == "incompletecholesky")
        return ichol::precond::SubdomainPreconditionerKind::IncompleteCholesky;
    if (v == "fsai")
        return ichol::precond::SubdomainPreconditionerKind::FSAI;
    if (v == "spai")
        return ichol::precond::SubdomainPreconditionerKind::SPAI;
    throw std::runtime_error("Unknown preconditioner kind: " + raw);
}

std::string precond_kind_to_string(ichol::precond::SubdomainPreconditionerKind kind)
{
    switch (kind)
    {
    case ichol::precond::SubdomainPreconditionerKind::ExactCholesky:
        return "exact";
    case ichol::precond::SubdomainPreconditionerKind::IncompleteCholesky:
        return "ic";
    case ichol::precond::SubdomainPreconditionerKind::FSAI:
        return "fsai";
    case ichol::precond::SubdomainPreconditionerKind::SPAI:
        return "spai";
    }
    return "unknown";
}

int parse_int(const std::string &raw, const char *name)
{
    try
    {
        return std::stoi(raw);
    }
    catch (...)
    {
        throw std::runtime_error(std::string("Failed to parse integer for ") + name + ": " + raw);
    }
}

double parse_double(const std::string &raw, const char *name)
{
    try
    {
        return std::stod(raw);
    }
    catch (...)
    {
        throw std::runtime_error(std::string("Failed to parse floating-point value for ") + name + ": " + raw);
    }
}

unsigned int parse_uint(const std::string &raw, const char *name)
{
    const int v = parse_int(raw, name);
    if (v < 0)
        throw std::runtime_error(std::string(name) + " must be non-negative");
    return static_cast<unsigned int>(v);
}

void apply_unit_col_prescaling_system(ichol::matrix::CsrMatrix<double> &A,
                                      std::vector<double> &b)
{
    const auto D = ichol::numeric::scale_diag_sqrt(A);
    ichol::numeric::apply_prescaling(A, D);
    ichol::numeric::apply_rhs_prescaling(b, D);
}

void set_default_params(AppOptions &opts)
{
    opts.params.maxits = 60;
    opts.params.tol = 1e-10;
    opts.params.restart = 0;
    opts.params.m_64 = kMixedHistory64;
    opts.params.m_32 = kMixedHistory32;
    opts.params.m_16 = kMixedHistory16;
    opts.params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    opts.params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    opts.params.prec_precond = ichol::solver::ComputePrecision::FP64;
    opts.params.prec_acc = ichol::solver::ComputePrecision::FP64;
    opts.params.acc_prec = ichol::solver::ComputePrecision::FP64;
    opts.params.store_Znew = ichol::solver::ComputePrecision::FP64;
    opts.params.store_Pnew = ichol::solver::ComputePrecision::FP64;
    opts.params.store_Wnew = ichol::solver::ComputePrecision::FP64;
    opts.params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    opts.params.store_W_hist = ichol::solver::ComputePrecision::FP64;
    opts.params.use_svd = false;
}

void print_usage(const char *argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --n INT                       3D Poisson grid size per dimension\n"
        << "  --subdomain WxHxD             subdomain size, e.g. 16x16x16\n"
        << "  --precond exact|ic|fsai|spai  subdomain preconditioner family\n"
        << "  --ic-level-k INT              level-k for incomplete Cholesky subdomains\n"
        << "  --fsai-level-k INT            level-k symbolic IC pattern used by FSAI subdomains\n"
        << "  --spai-radius INT             radius hint for SPAI subdomains\n"
        << "  --seed INT                    RHS RNG seed\n"
        << "  --tol FLOAT                   solver tolerance\n"
        << "  --mpcg-maxits INT             MPCG max iterations\n"
        << "  --mixed-m64 INT               FP64 mixed-history length\n"
        << "  --mixed-m32 INT               FP32 mixed-history length\n"
        << "  --mixed-m16 INT               FP16 mixed-history length\n"
        << "  --prec-gemm PREC              fp64|fp32|tf32|fp16|bf16\n"
        << "  --prec-spmm PREC              fp64|fp32|tf32|fp16|bf16\n"
        << "  --prec-precond PREC           fp64|fp32|tf32\n"
        << "  --prec-acc PREC               fp64|fp32|tf32|fp16|bf16\n"
        << "  --acc-prec PREC               fp64|fp32|fp16 (mixed-history projection accumulation)\n"
        << "  --store-znew PREC             fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-pnew PREC             fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-wnew PREC             fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-p-hist PREC           fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-w-hist PREC           fp64|fp32|tf32|fp16|bf16\n"
        << "  --use-svd                     use SVD (pinv) for alpha solve; default is Cholesky\n"
        << "  Fixed runs: full mpcg, truncated mpcg(restart=20), mixed mpcg(m64/m32/m16 from params)\n"
        << "  Each method is executed twice: warmup first, then timed run.\n"
        << "  --help                        show this message\n";
}

AppOptions parse_args(int argc, char **argv)
{
    AppOptions opts;
    set_default_params(opts);

    auto require_value = [&](int &i, const char *flag) -> std::string
    {
        if (i + 1 >= argc)
            throw std::runtime_error(std::string("Missing value for ") + flag);
        ++i;
        return argv[i];
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help")
        {
            opts.show_help = true;
        }
        else if (arg == "--n")
        {
            opts.n = parse_int(require_value(i, "--n"), "--n");
        }
        else if (arg == "--subdomain")
        {
            opts.subdomain = parse_triplet(require_value(i, "--subdomain"));
        }
        else if (arg == "--precond")
        {
            opts.precond_kind = parse_precond_kind(require_value(i, "--precond"));
        }
        else if (arg == "--ic-level-k")
        {
            opts.ic_level_k = parse_int(require_value(i, "--ic-level-k"), "--ic-level-k");
        }
        else if (arg == "--fsai-level-k")
        {
            opts.fsai_level_k = parse_int(require_value(i, "--fsai-level-k"), "--fsai-level-k");
        }
        else if (arg == "--spai-radius")
        {
            opts.spai_radius = parse_int(require_value(i, "--spai-radius"), "--spai-radius");
        }
        else if (arg == "--seed")
        {
            opts.seed = parse_uint(require_value(i, "--seed"), "--seed");
        }
        else if (arg == "--tol")
        {
            opts.params.tol = parse_double(require_value(i, "--tol"), "--tol");
        }
        else if (arg == "--mpcg-maxits")
        {
            opts.params.maxits = parse_int(require_value(i, "--mpcg-maxits"), "--mpcg-maxits");
        }
        else if (arg == "--mixed-m64")
        {
            opts.params.m_64 = parse_int(require_value(i, "--mixed-m64"), "--mixed-m64");
        }
        else if (arg == "--mixed-m32")
        {
            opts.params.m_32 = parse_int(require_value(i, "--mixed-m32"), "--mixed-m32");
        }
        else if (arg == "--mixed-m16")
        {
            opts.params.m_16 = parse_int(require_value(i, "--mixed-m16"), "--mixed-m16");
        }
        else if (arg == "--prec-gemm")
        {
            opts.params.prec_gemm = parse_compute_precision(require_value(i, "--prec-gemm"));
        }
        else if (arg == "--prec-spmm")
        {
            opts.params.prec_spmm = parse_compute_precision(require_value(i, "--prec-spmm"));
        }
        else if (arg == "--prec-precond")
        {
            opts.params.prec_precond = parse_compute_precision(require_value(i, "--prec-precond"));
        }
        else if (arg == "--prec-acc")
        {
            opts.params.prec_acc = parse_compute_precision(require_value(i, "--prec-acc"));
        }
        else if (arg == "--acc-prec")
        {
            opts.params.acc_prec = parse_compute_precision(require_value(i, "--acc-prec"));
        }
        else if (arg == "--store-znew")
        {
            opts.params.store_Znew = parse_compute_precision(require_value(i, "--store-znew"));
        }
        else if (arg == "--store-pnew")
        {
            opts.params.store_Pnew = parse_compute_precision(require_value(i, "--store-pnew"));
        }
        else if (arg == "--store-wnew")
        {
            opts.params.store_Wnew = parse_compute_precision(require_value(i, "--store-wnew"));
        }
        else if (arg == "--store-p-hist")
        {
            opts.params.store_P_hist = parse_compute_precision(require_value(i, "--store-p-hist"));
        }
        else if (arg == "--store-w-hist")
        {
            opts.params.store_W_hist = parse_compute_precision(require_value(i, "--store-w-hist"));
        }
        else if (arg == "--use-svd")
        {
            opts.params.use_svd = true;
        }
        else
        {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (opts.n <= 0)
        throw std::runtime_error("--n must be positive");
    if (opts.subdomain.x <= 0 || opts.subdomain.y <= 0 || opts.subdomain.z <= 0)
        throw std::runtime_error("--subdomain entries must be positive");
    if (opts.params.maxits <= 0)
        throw std::runtime_error("--mpcg-maxits must be positive");
    if (opts.params.m_64 <= 0)
        throw std::runtime_error("--mixed-m64 must be positive");
    if (opts.params.m_32 < 0)
        throw std::runtime_error("--mixed-m32 must be non-negative");
    if (opts.params.m_16 < 0)
        throw std::runtime_error("--mixed-m16 must be non-negative");
    if (opts.params.tol <= 0.0)
        throw std::runtime_error("--tol must be positive");
    if (opts.ic_level_k < 0)
        throw std::runtime_error("--ic-level-k must be non-negative");
    if (opts.fsai_level_k < 0)
        throw std::runtime_error("--fsai-level-k must be non-negative");
    if (opts.spai_radius < 0)
        throw std::runtime_error("--spai-radius must be non-negative");

    return opts;
}

SubdomainBundle build_subdomain_bundle(const ichol::matrix::CsrMatrix<double> &A,
                                       const AppOptions &opts)
{
    const ichol::precond::GridShape global_shape{opts.n, opts.n, opts.n};
    const ichol::precond::SubdomainSize subdomain_size{opts.subdomain.x, opts.subdomain.y, opts.subdomain.z};

    SubdomainBundle bundle;
    bundle.regions = ichol::precond::partition_subdomains(global_shape, subdomain_size);
    if (bundle.regions.empty())
        throw std::runtime_error("partition_subdomains returned no regions");

    ichol::precond::SubdomainPreconditionerOptions precond_opts;
    precond_opts.kind = opts.precond_kind;
    precond_opts.ic_level_k = opts.ic_level_k;
    precond_opts.fsai_level_k = opts.fsai_level_k;
    precond_opts.spai_radius = opts.spai_radius;
    precond_opts.precision = opts.params.prec_precond;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto raw_contexts = ichol::precond::create_subdomain_preconditioner_contexts_parallel(
        A, global_shape, bundle.regions, precond_opts);

    bundle.contexts.reserve(raw_contexts.size());
    for (auto *ctx : raw_contexts)
        bundle.contexts.emplace_back(ctx);

    bundle.preconds.reserve(bundle.contexts.size());
    for (const auto &ctx : bundle.contexts)
        bundle.preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, ctx.get()});

    auto t1 = std::chrono::high_resolution_clock::now();
    bundle.build_secs = std::chrono::duration<double>(t1 - t0).count();
    return bundle;
}

SolverRun run_mpcg_with_params(const ichol::matrix::CsrMatrix<double> &A,
                               const std::vector<double> &b,
                               const SubdomainBundle &bundle,
                               const ichol::solver::PCGParams &params)
{
    SolverRun run;
    run.precond_secs = bundle.build_secs;

    std::vector<double> x(A.num_rows, 0.0);
    auto t0 = std::chrono::high_resolution_clock::now();
    run.result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        bundle.preconds,
        b,
        x,
        params);
    auto t1 = std::chrono::high_resolution_clock::now();
    run.solve_secs = std::chrono::duration<double>(t1 - t0).count();
    return run;
}

SolverRun run_mpcg_mixed_with_params(const ichol::matrix::CsrMatrix<double> &A,
                                     const std::vector<double> &b,
                                     const SubdomainBundle &bundle,
                                     const ichol::solver::PCGParams &params)
{
    SolverRun run;
    run.precond_secs = bundle.build_secs;

    std::vector<double> x(A.num_rows, 0.0);
    auto t0 = std::chrono::high_resolution_clock::now();
    run.result = ichol::solver::mpcg_mixed<double>(
        A.row_ptr, A.col_ind, A.values,
        bundle.preconds,
        b,
        x,
        params);
    auto t1 = std::chrono::high_resolution_clock::now();
    run.solve_secs = std::chrono::duration<double>(t1 - t0).count();
    return run;
}

std::string trunc_label(const ichol::solver::PCGParams &params)
{
    return "[MPCG-TRUNC" + std::to_string(params.restart) + "]";
}

std::string mixed_label(const ichol::solver::PCGParams &params)
{
    return "[MPCG-MIXED-" +
           std::to_string(params.m_64) + "-" +
           std::to_string(params.m_32) + "-" +
           std::to_string(params.m_16) + "]";
}

void print_config(const AppOptions &opts,
                  const SubdomainBundle &bundle)
{
    std::cout
        << "[Config] n=" << opts.n
        << " N=" << opts.n * opts.n * opts.n
        << " subdomain=(" << opts.subdomain.x << "," << opts.subdomain.y << "," << opts.subdomain.z << ")"
        << " subdomains=" << bundle.regions.size()
        << " precond_kind=" << precond_kind_to_string(opts.precond_kind)
        << " fsai_level_k=" << opts.fsai_level_k
        << " mpcg_restart=" << opts.params.restart
        << " mpcg_maxits=" << opts.params.maxits
        << " mixed_m64=" << opts.params.m_64
        << " mixed_m32=" << opts.params.m_32
        << " mixed_m16=" << opts.params.m_16
        << " truncated_restart=" << kTruncatedHistory
        << " prec_gemm=" << precision_to_string(opts.params.prec_gemm)
        << " prec_spmm=" << precision_to_string(opts.params.prec_spmm)
        << " prec_precond=" << precision_to_string(opts.params.prec_precond)
        << " prec_acc=" << precision_to_string(opts.params.prec_acc)
        << " acc_prec=" << precision_to_string(opts.params.acc_prec)
        << " store_znew=" << precision_to_string(opts.params.store_Znew)
        << " store_pnew=" << precision_to_string(opts.params.store_Pnew)
        << " store_wnew=" << precision_to_string(opts.params.store_Wnew)
        << " store_p_hist=" << precision_to_string(opts.params.store_P_hist)
        << " store_w_hist=" << precision_to_string(opts.params.store_W_hist)
        << " tol=" << opts.params.tol
        << " seed=" << opts.seed
        << "\n";
}

void print_run(const char *label,
               const SolverRun &run)
{
    std::cout << label
              << " iters=" << run.result.iterations
              << " finalRes=" << run.result.finalRes
              << " end_to_end=" << (run.precond_secs + run.solve_secs) << "s"
              << " solve_time=" << run.solve_secs << "s"
              << " precond_gen_time=" << run.precond_secs << "s"
              << " solver_total_ms=" << run.result.timing.total_ms
              << " solver_setup_ms=" << run.result.timing.setup_ms
              << " solver_iter_ms=" << run.result.timing.iter_ms
              << " solver_finalize_ms=" << run.result.timing.finalize_ms
              << " precond_apply_ms=" << run.result.timing.preconditioner_apply_ms
              << " ortho_ms=" << run.result.timing.orthogonalization_ms
              << " spmm_ms=" << run.result.timing.spmm_ms
              << " dense_ms=" << run.result.timing.dense_ms
              << " reset_ms=" << run.result.timing.residual_reset_ms
              << " other_iter_ms=" << run.result.timing.other_iter_ms
              << "\n";
}

void print_rel_residuals(const std::string &label,
                         const ichol::solver::PCGResult &result)
{
    std::cout << label << " relres:";
    if (result.relResiduals.empty())
    {
        std::cout << " <empty>\n";
        return;
    }

    for (std::size_t i = 0; i < result.relResiduals.size(); ++i)
        std::cout << " (" << i << "," << result.relResiduals[i] << ")";
    std::cout << "\n";
}
} // namespace

int main(int argc, char **argv)
{
    AppOptions opts;
    try
    {
        opts = parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (opts.show_help)
    {
        print_usage(argv[0]);
        return 0;
    }

    int exit_code = 0;
    try
    {
        auto A = ichol::io::gen_3dpoi<double>(opts.n);
        std::vector<double> b(A.num_rows);
        {
            std::mt19937 rng(opts.seed);
            std::normal_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < A.num_rows; ++i)
                b[static_cast<std::size_t>(i)] = dist(rng);
        }
        apply_unit_col_prescaling_system(A, b);

        const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        const double stopping_scale = (bnorm > 0.0) ? bnorm : 1.0;

        SubdomainBundle bundle = build_subdomain_bundle(A, opts);
        print_config(opts, bundle);

        const ichol::solver::PCGParams full_params = opts.params;

        ichol::solver::PCGParams truncated_params = opts.params;
        truncated_params.restart = kTruncatedHistory;

        const ichol::solver::PCGParams mixed_params = opts.params;

        // Warmup
        SolverRun full_run = run_mpcg_with_params(A, b, bundle, full_params);
        SolverRun trunc_run = run_mpcg_with_params(A, b, bundle, truncated_params);
        SolverRun mixed_run = run_mpcg_mixed_with_params(A, b, bundle, mixed_params);

        full_run = run_mpcg_with_params(A, b, bundle, full_params);
        const std::string trunc_run_label = trunc_label(truncated_params);
        const std::string mixed_run_label = mixed_label(mixed_params);

        print_run("[MPCG]", full_run);
        print_rel_residuals("[MPCG]", full_run.result);

        trunc_run = run_mpcg_with_params(A, b, bundle, truncated_params);
        print_run(trunc_run_label.c_str(), trunc_run);
        print_rel_residuals(trunc_run_label, trunc_run.result);

        mixed_run = run_mpcg_mixed_with_params(A, b, bundle, mixed_params);
        print_run(mixed_run_label.c_str(), mixed_run);
        print_rel_residuals(mixed_run_label, mixed_run.result);

        if (full_run.result.finalRes > opts.params.tol * stopping_scale)
            std::cerr << "[Warn] MPCG did not reach the requested tolerance.\n";
        if (trunc_run.result.finalRes > opts.params.tol * stopping_scale)
            std::cerr << "[Warn] " << trunc_run_label << " did not reach the requested tolerance.\n";
        if (mixed_run.result.finalRes > opts.params.tol * stopping_scale)
            std::cerr << "[Warn] " << mixed_run_label << " did not reach the requested tolerance.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        exit_code = 1;
    }

    return exit_code;
}
