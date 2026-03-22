#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ichol/half.hpp"
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

enum class PcgMode
{
    Auto,
    BlockDiagonal,
    Additive
};

struct AppOptions
{
    int n = 32;
    Int3 subdomain{16, 16, 16};
    unsigned int seed = 20260303u;

    ichol::precond::SubdomainPreconditionerKind precond_kind =
        ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    int ic_level_k = 0;
    int spai_radius = 1;

    ichol::solver::PCGParams params;
    int pcg_maxits = 100;
    PcgMode pcg_mode = PcgMode::Auto;
    ichol::solver::ComputePrecision pcg_factor_precision = ichol::solver::ComputePrecision::FP64;
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

void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void cublas_check(cublasStatus_t st, const char *what)
{
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed with cuBLAS status " + std::to_string(static_cast<int>(st)));
}

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

ichol::solver::ComputePrecision normalize_custom_precond_precision(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (prec)
    {
    case Prec::FP64:
        return Prec::FP64;
    case Prec::FP32:
    case Prec::TF32:
        return Prec::FP32;
    default:
        throw std::runtime_error("custom preconditioner apply path supports FP64 and FP32 only");
    }
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
    case ichol::precond::SubdomainPreconditionerKind::SPAI:
        return "spai";
    }
    return "unknown";
}

PcgMode parse_pcg_mode(const std::string &raw)
{
    const std::string v = to_lower_copy(trim_copy(raw));
    if (v == "auto")
        return PcgMode::Auto;
    if (v == "blockdiag" || v == "block_diagonal")
        return PcgMode::BlockDiagonal;
    if (v == "additive")
        return PcgMode::Additive;
    throw std::runtime_error("Unknown pcg mode: " + raw);
}

std::string pcg_mode_to_string(PcgMode mode)
{
    switch (mode)
    {
    case PcgMode::Auto:
        return "auto";
    case PcgMode::BlockDiagonal:
        return "blockdiag";
    case PcgMode::Additive:
        return "additive";
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
    opts.params.maxits = 40;
    opts.params.tol = 1e-10;
    opts.params.restart = 0;
    opts.params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    opts.params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    opts.params.prec_precond = ichol::solver::ComputePrecision::FP64;
    opts.params.prec_acc = ichol::solver::ComputePrecision::FP64;
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
        << "  --precond-kind exact|ic|spai  subdomain preconditioner family\n"
        << "  --ic-level-k INT              level-k for incomplete Cholesky subdomains\n"
        << "  --spai-radius INT             radius hint for SPAI subdomains\n"
        << "  --seed INT                    RHS RNG seed\n"
        << "  --tol FLOAT                   solver tolerance\n"
        << "  --mpcg-maxits INT             MPCG max iterations\n"
        << "  --pcg-maxits INT              PCG max iterations\n"
        << "  --mpcg-restart INT            MPCG restart/window size; 0 => full history\n"
        << "  --prec-gemm PREC              fp64|fp32|tf32|fp16|bf16\n"
        << "  --prec-spmm PREC              fp64|fp32|tf32|fp16|bf16\n"
        << "  --prec-precond PREC           fp64|fp32|tf32\n"
        << "  --prec-acc PREC               fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-znew PREC             fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-pnew PREC             fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-wnew PREC             fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-p-hist PREC           fp64|fp32|tf32|fp16|bf16\n"
        << "  --store-w-hist PREC           fp64|fp32|tf32|fp16|bf16\n"
        << "  --pcg-mode auto|blockdiag|additive\n"
        << "  --pcg-factor-precision PREC   fp64|fp32|fp16 for blockdiag pcg_cusparse_spsv\n"
        << "  --use-svd                     use SVD (pinv) for alpha solve; default is Cholesky\n"
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
        else if (arg == "--precond-kind")
        {
            opts.precond_kind = parse_precond_kind(require_value(i, "--precond-kind"));
        }
        else if (arg == "--ic-level-k")
        {
            opts.ic_level_k = parse_int(require_value(i, "--ic-level-k"), "--ic-level-k");
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
        else if (arg == "--pcg-maxits")
        {
            opts.pcg_maxits = parse_int(require_value(i, "--pcg-maxits"), "--pcg-maxits");
        }
        else if (arg == "--mpcg-restart")
        {
            opts.params.restart = parse_int(require_value(i, "--mpcg-restart"), "--mpcg-restart");
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
        else if (arg == "--pcg-mode")
        {
            opts.pcg_mode = parse_pcg_mode(require_value(i, "--pcg-mode"));
        }
        else if (arg == "--pcg-factor-precision")
        {
            opts.pcg_factor_precision = parse_compute_precision(require_value(i, "--pcg-factor-precision"));
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
    if (opts.pcg_maxits <= 0)
        throw std::runtime_error("--pcg-maxits must be positive");
    if (opts.params.restart < 0)
        throw std::runtime_error("--mpcg-restart must be non-negative");
    if (opts.params.tol <= 0.0)
        throw std::runtime_error("--tol must be positive");
    if (opts.ic_level_k < 0)
        throw std::runtime_error("--ic-level-k must be non-negative");
    if (opts.spai_radius < 0)
        throw std::runtime_error("--spai-radius must be non-negative");

    using Prec = ichol::solver::ComputePrecision;
    if (opts.pcg_factor_precision != Prec::FP64 &&
        opts.pcg_factor_precision != Prec::FP32 &&
        opts.pcg_factor_precision != Prec::FP16)
    {
        throw std::runtime_error("--pcg-factor-precision supports fp64, fp32, or fp16 only");
    }

    return opts;
}

class AdditivePrecondContext
{
public:
    explicit AdditivePrecondContext(const std::vector<ichol::precond::PrecondApply> &preconds)
        : preconds_(preconds)
    {
        cublas_check(cublasCreate(&cublas_), "cublasCreate");
    }

    ~AdditivePrecondContext()
    {
        if (tmp64_ != nullptr)
            cudaFree(tmp64_);
        if (tmp32_ != nullptr)
            cudaFree(tmp32_);
        if (cublas_ != nullptr)
            cublasDestroy(cublas_);
    }

    static void apply(void *ctx,
                      const void *d_r,
                      void *d_z,
                      int n,
                      ichol::solver::ComputePrecision prec,
                      cudaStream_t stream)
    {
        static_cast<AdditivePrecondContext *>(ctx)->apply_impl(d_r, d_z, n, prec, stream);
    }

private:
    void ensure_capacity(int n)
    {
        if (n <= capacity_)
            return;
        if (tmp64_ != nullptr)
            cuda_check(cudaFree(tmp64_), "cudaFree(tmp64)");
        if (tmp32_ != nullptr)
            cuda_check(cudaFree(tmp32_), "cudaFree(tmp32)");
        cuda_check(cudaMalloc(&tmp64_, static_cast<std::size_t>(n) * sizeof(double)), "cudaMalloc(tmp64)");
        cuda_check(cudaMalloc(&tmp32_, static_cast<std::size_t>(n) * sizeof(float)), "cudaMalloc(tmp32)");
        capacity_ = n;
    }

    void apply_impl(const void *d_r,
                    void *d_z,
                    int n,
                    ichol::solver::ComputePrecision prec,
                    cudaStream_t stream)
    {
        const auto norm_prec = normalize_custom_precond_precision(prec);
        ensure_capacity(n);
        cublas_check(cublasSetStream(cublas_, stream), "cublasSetStream");

        if (norm_prec == ichol::solver::ComputePrecision::FP64)
        {
            auto *out = static_cast<double *>(d_z);
            cuda_check(cudaMemsetAsync(out, 0, static_cast<std::size_t>(n) * sizeof(double), stream), "cudaMemsetAsync(additive out fp64)");
            const double alpha = 1.0;
            for (const auto &precond : preconds_)
            {
                cuda_check(cudaMemsetAsync(tmp64_, 0, static_cast<std::size_t>(n) * sizeof(double), stream), "cudaMemsetAsync(additive tmp fp64)");
                precond.apply(precond.ctx, d_r, tmp64_, n, norm_prec, stream);
                cublas_check(cublasDaxpy(cublas_, n, &alpha, tmp64_, 1, out, 1), "cublasDaxpy");
            }
        }
        else
        {
            auto *out = static_cast<float *>(d_z);
            cuda_check(cudaMemsetAsync(out, 0, static_cast<std::size_t>(n) * sizeof(float), stream), "cudaMemsetAsync(additive out fp32)");
            const float alpha = 1.0f;
            for (const auto &precond : preconds_)
            {
                cuda_check(cudaMemsetAsync(tmp32_, 0, static_cast<std::size_t>(n) * sizeof(float), stream), "cudaMemsetAsync(additive tmp fp32)");
                precond.apply(precond.ctx, d_r, tmp32_, n, norm_prec, stream);
                cublas_check(cublasSaxpy(cublas_, n, &alpha, tmp32_, 1, out, 1), "cublasSaxpy");
            }
        }
    }

    std::vector<ichol::precond::PrecondApply> preconds_;
    cublasHandle_t cublas_ = nullptr;
    double *tmp64_ = nullptr;
    float *tmp32_ = nullptr;
    int capacity_ = 0;
};

PcgMode resolve_pcg_mode(const AppOptions &opts)
{
    if (opts.pcg_mode != PcgMode::Auto)
        return opts.pcg_mode;
    if (opts.precond_kind == ichol::precond::SubdomainPreconditionerKind::ExactCholesky)
        return PcgMode::BlockDiagonal;
    return PcgMode::Additive;
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

SolverRun run_mpcg(const ichol::matrix::CsrMatrix<double> &A,
                   const std::vector<double> &b,
                   const AppOptions &opts,
                   const SubdomainBundle &bundle)
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
        opts.params);
    auto t1 = std::chrono::high_resolution_clock::now();
    run.solve_secs = std::chrono::duration<double>(t1 - t0).count();
    return run;
}

template <typename T>
ichol::solver::PCGResult run_blockdiag_pcg_impl(const ichol::matrix::CsrMatrix<double> &A,
                                                const ichol::matrix::CsrMatrix<double> &L_block,
                                                const std::vector<double> &b,
                                                const AppOptions &opts)
{
    auto L_t = ichol::matrix::convert_csr_precision<double, T>(L_block);
    std::vector<double> x(A.num_rows, 0.0);
    std::vector<double> h_D(A.num_rows, 1.0);
    ichol::solver::PCGParams pcg_params = opts.params;
    pcg_params.custom_precond = nullptr;
    pcg_params.maxits = opts.pcg_maxits;
    return ichol::solver::pcg_cusparse_spsv<T>(
        A.row_ptr, A.col_ind, A.values,
        L_t.row_ptr, L_t.col_ind, L_t.values,
        b,
        x,
        h_D,
        pcg_params);
}

SolverRun run_blockdiag_pcg(const ichol::matrix::CsrMatrix<double> &A,
                            const std::vector<double> &b,
                            const AppOptions &opts)
{
    if (opts.precond_kind != ichol::precond::SubdomainPreconditionerKind::ExactCholesky)
        throw std::runtime_error("blockdiag PCG mode requires --precond-kind exact");

    SolverRun run;
    auto t_build0 = std::chrono::high_resolution_clock::now();
    auto L_block = ichol::precond::build_block_diagonal_exact_preconditioner_3d(
        A, opts.n, opts.subdomain.x, opts.subdomain.y, opts.subdomain.z);
    auto t_build1 = std::chrono::high_resolution_clock::now();
    run.precond_secs = std::chrono::duration<double>(t_build1 - t_build0).count();

    auto t_solve0 = std::chrono::high_resolution_clock::now();
    switch (opts.pcg_factor_precision)
    {
    case ichol::solver::ComputePrecision::FP64:
        run.result = run_blockdiag_pcg_impl<double>(A, L_block, b, opts);
        break;
    case ichol::solver::ComputePrecision::FP32:
        run.result = run_blockdiag_pcg_impl<float>(A, L_block, b, opts);
        break;
    case ichol::solver::ComputePrecision::FP16:
        run.result = run_blockdiag_pcg_impl<half_float::half>(A, L_block, b, opts);
        break;
    default:
        throw std::runtime_error("unsupported pcg factor precision");
    }
    auto t_solve1 = std::chrono::high_resolution_clock::now();
    run.solve_secs = std::chrono::duration<double>(t_solve1 - t_solve0).count();
    return run;
}

SolverRun run_additive_pcg(const ichol::matrix::CsrMatrix<double> &A,
                           const std::vector<double> &b,
                           const AppOptions &opts,
                           const SubdomainBundle &bundle)
{
    normalize_custom_precond_precision(opts.params.prec_precond);

    SolverRun run;
    run.precond_secs = bundle.build_secs;

    AdditivePrecondContext additive_ctx(bundle.preconds);
    ichol::precond::PrecondApply additive_precond{&AdditivePrecondContext::apply, &additive_ctx};

    std::vector<double> x(A.num_rows, 0.0);
    std::vector<double> h_D(A.num_rows, 1.0);
    ichol::solver::PCGParams pcg_params = opts.params;
    pcg_params.custom_precond = &additive_precond;
    pcg_params.maxits = opts.pcg_maxits;

    const std::vector<int> empty_row_ptr;
    const std::vector<int> empty_col_ind;
    const std::vector<double> empty_vals;

    auto t0 = std::chrono::high_resolution_clock::now();
    run.result = ichol::solver::pcg<double>(
        A.row_ptr, A.col_ind, A.values,
        empty_row_ptr, empty_col_ind, empty_vals,
        b,
        x,
        h_D,
        pcg_params);
    auto t1 = std::chrono::high_resolution_clock::now();
    run.solve_secs = std::chrono::duration<double>(t1 - t0).count();
    return run;
}

void print_config(const AppOptions &opts,
                  const SubdomainBundle &bundle,
                  PcgMode resolved_pcg_mode)
{
    std::cout
        << "[Config] n=" << opts.n
        << " N=" << opts.n * opts.n * opts.n
        << " subdomain=(" << opts.subdomain.x << "," << opts.subdomain.y << "," << opts.subdomain.z << ")"
        << " subdomains=" << bundle.regions.size()
        << " precond_kind=" << precond_kind_to_string(opts.precond_kind)
        << " mpcg_restart=" << opts.params.restart
        << " mpcg_maxits=" << opts.params.maxits
        << " pcg_maxits=" << opts.pcg_maxits
        << " pcg_mode=" << pcg_mode_to_string(resolved_pcg_mode)
        << " prec_gemm=" << precision_to_string(opts.params.prec_gemm)
        << " prec_spmm=" << precision_to_string(opts.params.prec_spmm)
        << " prec_precond=" << precision_to_string(opts.params.prec_precond)
        << " prec_acc=" << precision_to_string(opts.params.prec_acc)
        << " store_znew=" << precision_to_string(opts.params.store_Znew)
        << " store_pnew=" << precision_to_string(opts.params.store_Pnew)
        << " store_wnew=" << precision_to_string(opts.params.store_Wnew)
        << " store_p_hist=" << precision_to_string(opts.params.store_P_hist)
        << " store_w_hist=" << precision_to_string(opts.params.store_W_hist)
        << " pcg_factor_precision=" << precision_to_string(opts.pcg_factor_precision)
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
              << " precond_gen_time=" << run.precond_secs << "s\n";
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
        const PcgMode resolved_pcg_mode = resolve_pcg_mode(opts);
        if (resolved_pcg_mode == PcgMode::BlockDiagonal &&
            opts.precond_kind != ichol::precond::SubdomainPreconditionerKind::ExactCholesky)
        {
            throw std::runtime_error("pcg_mode=blockdiag requires --precond-kind exact");
        }

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
        print_config(opts, bundle, resolved_pcg_mode);

        const SolverRun mpcg_run = run_mpcg(A, b, opts, bundle);
        print_run("[MPCG]", mpcg_run);

        SolverRun pcg_run;
        if (resolved_pcg_mode == PcgMode::BlockDiagonal)
        {
            pcg_run = run_blockdiag_pcg(A, b, opts);
            print_run("[PCG blockdiag/cuSPARSE-SpSV]", pcg_run);
            std::cout << "[Note] blockdiag PCG ignores prec_precond; its preconditioner solve precision is set by --pcg-factor-precision.\n";
        }
        else
        {
            pcg_run = run_additive_pcg(A, b, opts, bundle);
            print_run("[PCG additive/custom-precond]", pcg_run);
            std::cout << "[Note] additive PCG reuses the subdomain PrecondApply objects, so prec_precond affects this PCG path.\n";
        }

        if (mpcg_run.result.finalRes > opts.params.tol * stopping_scale)
            std::cerr << "[Warn] MPCG did not reach the requested tolerance.\n";
        if (pcg_run.result.finalRes > opts.params.tol * stopping_scale)
            std::cerr << "[Warn] PCG did not reach the requested tolerance.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        exit_code = 1;
    }

    return exit_code;
}
