#pragma once

namespace ichol::cuda
{
    __device__ __forceinline__ unsigned hash_u32(unsigned x) { return x * 2654435761u; }

    __device__ __forceinline__ void hash_init(int *keys, int *vals, int H)
    {
        for (int t = threadIdx.x; t < H; t += blockDim.x)
        {
            keys[t] = -1;
            vals[t] = -1;
        }
    }

    __device__ __forceinline__ void hash_insert(int *keys, int *vals, int H, int key, int val)
    {
        unsigned h = hash_u32((unsigned)key) & (unsigned)(H - 1);
        for (int it = 0; it < H; ++it)
        {
            int slot = (int)((h + (unsigned)it) & (unsigned)(H - 1));
            int prev = atomicCAS(&keys[slot], -1, key);
            if (prev == -1 || prev == key)
            {
                vals[slot] = val;
                return;
            }
        }
    }

    __device__ __forceinline__ int hash_find(const int *keys, const int *vals, int H, int key)
    {
        unsigned h = hash_u32((unsigned)key) & (unsigned)(H - 1);
        for (int it = 0; it < H; ++it)
        {
            int slot = (int)((h + (unsigned)it) & (unsigned)(H - 1));
            int k = keys[slot];
            if (k == key)
                return vals[slot];
            if (k == -1)
                return -1;
        }
        return -1;
    }
} // namespace ichol::cuda