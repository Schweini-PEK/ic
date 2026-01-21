#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

namespace ichol::numeric {

// Thread-local workspace for supernodal LL numeric factorization.
// Owns temporary buffers used per-supernode to avoid repeated allocations.
struct SupernodalWorkspace {
    int n = 0;
    int max_front = 0;

    // Dense front buffer (max_front x max_front), column-major.
    std::vector<double> F;

    // Global row -> local index map for the current front.
    // Size = n, initialized to -1 once.
    std::vector<int> g2l;

    // The global row indices that are active in the current front.
    // Also used as the "touched set" so we can reset g2l in O(front).
    std::vector<int> front_rows;

    // Scratch for child update scatter positions.
    std::vector<int> tmp_pos;

    void init(int n_, int max_front_)
    {
        n = n_;
        max_front = std::max(0, max_front_);

        g2l.assign((size_t)n, -1);
        F.assign((size_t)max_front * (size_t)max_front, 0.0);

        front_rows.clear();
        front_rows.reserve((size_t)max_front);

        tmp_pos.clear();
        tmp_pos.reserve((size_t)max_front);
    }

    // Reset only the portion of the front that will be used (nsrow x nsrow).
    void reset_front(int nsrow)
    {
        const size_t nn = (size_t)nsrow * (size_t)nsrow;
        std::fill(F.begin(), F.begin() + nn, 0.0);
        front_rows.clear();
    }

    // Build g2l map from a contiguous slice of sym.s (pi0..pi0+nsrow).
    void build_mapping_from_s(const std::vector<int>& s, int pi0, int nsrow)
    {
        front_rows.reserve((size_t)nsrow);
        for (int t = 0; t < nsrow; ++t) {
            const int r = s[(size_t)(pi0 + t)];
            g2l[(size_t)r] = t;
            front_rows.push_back(r);
        }
    }

    // Clear g2l entries touched by the current front.
    void clear_mapping()
    {
        for (int r : front_rows) g2l[(size_t)r] = -1;
    }
};

} // namespace ichol::numeric
