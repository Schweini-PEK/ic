#pragma once

namespace ichol::cuda
{
    template <class G>
    static inline size_t shmem_bytes_for_level(int max_off_level, int H_level, int N_level)
    {
        size_t off = 0;

        // hash_keys + hash_vals
        off += 2ull * (size_t)H_level * sizeof(int);

        // w_val + lik_val
        off = align_up(off, alignof(G));
        off += 2ull * (size_t)max_off_level * sizeof(G);

        // keep
        off = align_up(off, alignof(int));
        off += 1ull * (size_t)max_off_level * sizeof(int);

        // absbuf
        off = align_up(off, alignof(G));
        off += 1ull * (size_t)N_level * sizeof(G);

        // idxbuf
        off = align_up(off, alignof(int));
        off += 1ull * (size_t)N_level * sizeof(int);

        return off;
    }
} // namespace ichol::cuda