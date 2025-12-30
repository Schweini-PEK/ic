#pragma once

namespace ichol::cuda
{
    template <class G>
    __device__ __forceinline__ void bitonic_sort_desc_abs(G *absv, int *idx, int N)
    {
        for (int k = 2; k <= N; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                for (int i = threadIdx.x; i < N; i += blockDim.x)
                {
                    int ixj = i ^ j;
                    if (ixj > i)
                    {
                        bool up = ((i & k) == 0);
                        bool swap_needed = up ? cuda::GMath<G>::lt(absv[i], absv[ixj])
                                              : cuda::GMath<G>::gt(absv[i], absv[ixj]);
                        if (swap_needed)
                        {
                            G ta = absv[i];
                            absv[i] = absv[ixj];
                            absv[ixj] = ta;
                            int ti = idx[i];
                            idx[i] = idx[ixj];
                            idx[ixj] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
} // namespace ichol::cuda