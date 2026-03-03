#include "backends/CUDA/subdomain_precond_impl.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    inline void cuda_check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
    }

    inline void cuda_check_last_kernel()
    {
        cuda_check(cudaGetLastError());
    }

    __host__ __device__ __forceinline__ int flatten_local_3d(int x, int y, int z, int w, int h)
    {
        return x + y * w + z * (w * h);
    }

    __host__ __device__ __forceinline__ void unflatten_global_3d(int gi, int gw, int gh, int &x, int &y, int &z)
    {
        const int plane = gw * gh;
        z = gi / plane;
        const int rem = gi - z * plane;
        y = rem / gw;
        x = rem - y * gw;
    }

    __device__ __forceinline__ int local_from_global(
        int gj, int gw, int gh,
        int x0, int x1, int y0, int y1, int z0, int z1,
        int lw, int lh)
    {
        int x, y, z;
        unflatten_global_3d(gj, gw, gh, x, y, z);
        if (x < x0 || x >= x1 || y < y0 || y >= y1 || z < z0 || z >= z1)
            return -1;
        return flatten_local_3d(x - x0, y - y0, z - z0, lw, lh);
    }

    struct DeviceCsrCache
    {
        int n = 0;
        int nnz = 0;
        int *d_row = nullptr;
        int *d_col = nullptr;
        double *d_val = nullptr;

        void release()
        {
            if (d_row)
                cudaFree(d_row);
            if (d_col)
                cudaFree(d_col);
            if (d_val)
                cudaFree(d_val);
            d_row = nullptr;
            d_col = nullptr;
            d_val = nullptr;
            n = 0;
            nnz = 0;
        }

        void ensure(const ichol::matrix::CsrMatrix<double> &A)
        {
            if (d_row && n == A.num_rows && nnz == A.nnz)
                return;

            release();
            n = A.num_rows;
            nnz = A.nnz;

            cuda_check(cudaMalloc(&d_row, (n + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&d_col, nnz * sizeof(int)));
            cuda_check(cudaMalloc(&d_val, nnz * sizeof(double)));

            cuda_check(cudaMemcpy(d_row, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_col, A.col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_val, A.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        }

        ~DeviceCsrCache() { release(); }
    };

    static DeviceCsrCache gA;

    __global__ void k_build_psai(
        const int *row_ptr, const int *col_ind, const double *val,
        int *Mcol, double *Mval, int *status,
        int lw, int lh, int ld, int nsub,
        int gw, int gh,
        int x0, int x1, int y0, int y1, int z0, int z1,
        int r)
    {
        const int li = (int)blockIdx.x;
        if (li >= nsub)
            return;

        const int diam = 2 * r + 1;
        const int pmax = diam * diam * diam;

        extern __shared__ unsigned char smem_raw[];
        int *p_loc = reinterpret_cast<int *>(smem_raw);
        int *p_g = reinterpret_cast<int *>(p_loc + pmax);
        double *S = reinterpret_cast<double *>(p_g + pmax);
        double *rhs = S + (size_t)pmax * (size_t)pmax;
        double *sol = rhs + pmax;

        const int plane = lw * lh;
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;

        int p = 0;
        if (threadIdx.x == 0)
        {
            for (int dz = -r; dz <= r; ++dz)
            {
                const int zz = lz + dz;
                if (zz < 0 || zz >= ld)
                    continue;
                for (int dy = -r; dy <= r; ++dy)
                {
                    const int yy = ly + dy;
                    if (yy < 0 || yy >= lh)
                        continue;
                    for (int dx = -r; dx <= r; ++dx)
                    {
                        const int xx = lx + dx;
                        if (xx < 0 || xx >= lw)
                            continue;

                        const int lj = flatten_local_3d(xx, yy, zz, lw, lh);
                        const int gj = (x0 + xx) + (y0 + yy) * gw + (z0 + zz) * (gw * gh);

                        p_loc[p] = lj;
                        p_g[p] = gj;
                        ++p;
                    }
                }
            }

            for (int i = 0; i < pmax * pmax; ++i)
                S[i] = 0.0;
            for (int i = 0; i < pmax; ++i)
            {
                rhs[i] = 0.0;
                sol[i] = 0.0;
            }

            for (int t = 0; t < p; ++t)
                rhs[t] = (p_loc[t] == li) ? 1.0 : 0.0;

            for (int u = 0; u < p; ++u)
            {
                const int gu = p_g[u];
                for (int kk = row_ptr[gu]; kk < row_ptr[gu + 1]; ++kk)
                {
                    const int gv = col_ind[kk];
                    const int lv = local_from_global(gv, gw, gh, x0, x1, y0, y1, z0, z1, lw, lh);
                    if (lv < 0)
                        continue;

                    int v = -1;
                    for (int t = 0; t < p; ++t)
                    {
                        if (p_loc[t] == lv)
                        {
                            v = t;
                            break;
                        }
                    }
                    if (v >= 0)
                        S[u * pmax + v] = val[kk];
                }
            }

            bool ok = true;
            for (int k = 0; k < p; ++k)
            {
                double s = S[k * pmax + k];
                for (int t = 0; t < k; ++t)
                {
                    const double lkt = S[k * pmax + t];
                    s -= lkt * lkt;
                }
                if (s <= 0.0)
                {
                    ok = false;
                    break;
                }
                const double lkk = sqrt(s);
                S[k * pmax + k] = lkk;

                for (int i = k + 1; i < p; ++i)
                {
                    double aik = S[i * pmax + k];
                    for (int t = 0; t < k; ++t)
                        aik -= S[i * pmax + t] * S[k * pmax + t];
                    S[i * pmax + k] = aik / lkk;
                }
            }

            if (!ok)
            {
                atomicExch(status, 2);

                double aii = 0.0;
                const int gi = (x0 + lx) + (y0 + ly) * gw + (z0 + lz) * (gw * gh);
                for (int kk = row_ptr[gi]; kk < row_ptr[gi + 1]; ++kk)
                {
                    if (col_ind[kk] == gi)
                    {
                        aii = val[kk];
                        break;
                    }
                }
                const double inv = (aii != 0.0) ? (1.0 / aii) : 1.0;

                for (int t = 0; t < pmax; ++t)
                {
                    const int out = li * pmax + t;
                    Mcol[out] = -1;
                    Mval[out] = 0.0;
                }
                int self_pos = -1;
                for (int t = 0; t < p; ++t)
                    if (p_loc[t] == li)
                    {
                        self_pos = t;
                        break;
                    }

                if (self_pos >= 0)
                {
                    const int out = li * pmax + 0;
                    Mcol[out] = p_g[self_pos];
                    Mval[out] = inv;
                }
                return;
            }

            for (int i = 0; i < p; ++i)
            {
                double s = rhs[i];
                for (int t = 0; t < i; ++t)
                    s -= S[i * pmax + t] * sol[t];
                sol[i] = s / S[i * pmax + i];
            }

            for (int i = p - 1; i >= 0; --i)
            {
                double s = sol[i];
                for (int t = i + 1; t < p; ++t)
                    s -= S[t * pmax + i] * rhs[t];
                rhs[i] = s / S[i * pmax + i];
            }

            for (int t = 0; t < pmax; ++t)
            {
                const int out = li * pmax + t;
                if (t < p)
                {
                    Mcol[out] = p_g[t];
                    Mval[out] = rhs[t];
                }
                else
                {
                    Mcol[out] = -1;
                    Mval[out] = 0.0;
                }
            }
        }
    }

    __global__ void k_apply_psai(
        const double *d_r, double *d_z,
        const int *Mcol, const double *Mval,
        int lw, int lh, int ld, int nsub,
        int gw, int gh,
        int x0, int y0, int z0,
        int r)
    {
        const int li = blockIdx.x * blockDim.x + threadIdx.x;
        if (li >= nsub)
            return;

        const int diam = 2 * r + 1;
        const int pmax = diam * diam * diam;

        const int plane = lw * lh;
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;

        const int gi = (x0 + lx) + (y0 + ly) * gw + (z0 + lz) * (gw * gh);

        double sum = 0.0;
        const int base = li * pmax;
#pragma unroll
        for (int k = 0; k < 125; ++k)
        {
            if (k >= pmax)
                break;
            const int gj = Mcol[base + k];
            if (gj >= 0)
                sum += Mval[base + k] * d_r[gj];
        }

        d_z[gi] = sum;
    }
} // namespace

namespace ichol::precond::detail
{
    struct SubdomainPsaiContext
    {
        int lw, lh, ld, nsub;
        int gw, gh;
        int x0, x1, y0, y1, z0, z1;

        int r;
        int pmax;

        int *d_Mcol = nullptr;
        double *d_Mval = nullptr;
        int *d_status = nullptr;
    };

    void *create_subdomain_psai_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &reg,
        const SubdomainPreconditionerOptions &options)
    {
        gA.ensure(A);

        auto *ctx = new SubdomainPsaiContext();
        ctx->lw = reg.x1 - reg.x0;
        ctx->lh = reg.y1 - reg.y0;
        ctx->ld = reg.z1 - reg.z0;
        ctx->nsub = ctx->lw * ctx->lh * ctx->ld;

        ctx->gw = global.w;
        ctx->gh = global.h;

        ctx->x0 = reg.x0;
        ctx->x1 = reg.x1;
        ctx->y0 = reg.y0;
        ctx->y1 = reg.y1;
        ctx->z0 = reg.z0;
        ctx->z1 = reg.z1;

        ctx->r = options.psai_radius;
        if (ctx->r < 0)
            throw std::runtime_error("create_subdomain_psai_context: psai_radius must be non-negative");

        ctx->pmax = (2 * ctx->r + 1);
        ctx->pmax = ctx->pmax * ctx->pmax * ctx->pmax;

        cuda_check(cudaMalloc(&ctx->d_Mcol, (size_t)ctx->nsub * (size_t)ctx->pmax * sizeof(int)));
        cuda_check(cudaMalloc(&ctx->d_Mval, (size_t)ctx->nsub * (size_t)ctx->pmax * sizeof(double)));
        cuda_check(cudaMalloc(&ctx->d_status, sizeof(int)));
        cuda_check(cudaMemset(ctx->d_status, 0, sizeof(int)));

        const int pmax = ctx->pmax;
        const size_t shmem =
            (size_t)pmax * sizeof(int) +
            (size_t)pmax * sizeof(int) +
            (size_t)pmax * (size_t)pmax * sizeof(double) +
            (size_t)pmax * sizeof(double) +
            (size_t)pmax * sizeof(double);

        k_build_psai<<<ctx->nsub, 1, shmem>>>(
            gA.d_row, gA.d_col, gA.d_val,
            ctx->d_Mcol, ctx->d_Mval, ctx->d_status,
            ctx->lw, ctx->lh, ctx->ld, ctx->nsub,
            global.w, global.h,
            reg.x0, reg.x1, reg.y0, reg.y1, reg.z0, reg.z1,
            ctx->r);
        cuda_check_last_kernel();

        cuda_check(cudaDeviceSynchronize());
        return ctx;
    }

    void apply_subdomain_psai(void *vctx, const double *d_r, double *d_z, int /*N*/, cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainPsaiContext *>(vctx);

        const int tpb = 256;
        const int bpg = (ctx->nsub + tpb - 1) / tpb;

        k_apply_psai<<<bpg, tpb, 0, stream>>>(
            d_r, d_z,
            ctx->d_Mcol, ctx->d_Mval,
            ctx->lw, ctx->lh, ctx->ld, ctx->nsub,
            ctx->gw, ctx->gh,
            ctx->x0, ctx->y0, ctx->z0,
            ctx->r);
        cuda_check_last_kernel();
    }

    void destroy_subdomain_psai_context(void *vctx)
    {
        auto *ctx = reinterpret_cast<SubdomainPsaiContext *>(vctx);
        if (!ctx)
            return;
        cudaFree(ctx->d_Mcol);
        cudaFree(ctx->d_Mval);
        cudaFree(ctx->d_status);
        delete ctx;
    }
} // namespace ichol::precond::detail
