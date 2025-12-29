// src/factor/symbolic/supernodes.cpp
#include <vector>
#include <utility>
#include <cassert>

#include "symbolic.hpp"


namespace ichol::symbolic {

        // Conservative supernode detection: merge adjacent columns only when their row lists are identical.
        // Returns vector of pairs [start_col, end_col) where end_col is exclusive.
        std::vector<std::pair<int,int>> detect_supernodes(const FactorPattern &pattern, const ETree & /*etree*/)
        {
            int n = static_cast<int>(pattern.row_ptr_L.size()) - 1;
            std::vector<std::pair<int,int>> sn;
            if (n <= 0) return sn;

            int cur_start = 0;
            for (int c = 1; c < n; ++c)
            {
                int s1 = pattern.row_ptr_L[c - 1];
                int e1 = pattern.row_ptr_L[c - 1 + 1];
                int s2 = pattern.row_ptr_L[c];
                int e2 = pattern.row_ptr_L[c + 1];

                bool equal = (e1 - s1 == e2 - s2);
                if (equal) {
                    for (int t = 0; t < e1 - s1; ++t) {
                        if (pattern.col_ind_L[s1 + t] != pattern.col_ind_L[s2 + t]) { equal = false; break; }
                    }
                }

                if (!equal) {
                    sn.emplace_back(cur_start, c);
                    cur_start = c;
                }
            }
            sn.emplace_back(cur_start, n);
            return sn;
        }

        // Approximate supernode detection based on overlap ratio.
        // Merge adjacent columns if their row-list intersection size >= overlap_threshold * average_size.
        // overlap_threshold in (0,1], e.g. 1.0 -> exact (same as conservative), 0.8 -> allow 80% overlap.
        std::vector<std::pair<int,int>> detect_supernodes_approx(const FactorPattern &pattern, const ETree & /*etree*/, double overlap_threshold)
        {
            assert(overlap_threshold > 0.0 && overlap_threshold <= 1.0);
            int n = static_cast<int>(pattern.row_ptr_L.size()) - 1;
            std::vector<std::pair<int,int>> sn;
            if (n <= 0) return sn;

            int cur_start = 0;
            for (int c = 1; c < n; ++c)
            {
                int s1 = pattern.row_ptr_L[c - 1], e1 = pattern.row_ptr_L[c - 1 + 1];
                int s2 = pattern.row_ptr_L[c], e2 = pattern.row_ptr_L[c + 1];
                int len1 = e1 - s1;
                int len2 = e2 - s2;
                if (len1 == 0 || len2 == 0) {
                    // if either empty, require exact equality (both empty) to merge
                    if (!(len1 == 0 && len2 == 0)) {
                        sn.emplace_back(cur_start, c);
                        cur_start = c;
                    }
                    continue;
                }

                // compute intersection size of two sorted lists
                int i = s1, j = s2, inter = 0;
                while (i < e1 && j < e2) {
                    int v1 = pattern.col_ind_L[i];
                    int v2 = pattern.col_ind_L[j];
                    if (v1 == v2) { ++inter; ++i; ++j; }
                    else if (v1 < v2) ++i;
                    else ++j;
                }

                double avg_len = 0.5 * (len1 + len2);
                double ratio = static_cast<double>(inter) / avg_len;
                if (ratio >= overlap_threshold) {
                    // merge: continue current supernode
                    continue;
                } else {
                    sn.emplace_back(cur_start, c);
                    cur_start = c;
                }
            }
            sn.emplace_back(cur_start, n);
            return sn;
        }

        // Build a mapping from column -> supernode id
        std::vector<int> build_col2snode(const std::vector<std::pair<int,int>>& snodes, int ncols)
        {
            std::vector<int> col2s(ncols, -1);
            for (size_t id = 0; id < snodes.size(); ++id) {
                int s = snodes[id].first;
                int e = snodes[id].second;
                for (int c = s; c < e; ++c) col2s[c] = static_cast<int>(id);
            }
            return col2s;
        }

        // For each supernode [s,e), collect union of rows across its columns (rows kept sorted & unique)
        std::vector<std::vector<int>> compute_snode_rows(const FactorPattern& pat,
                                                         const std::vector<std::pair<int,int>>& snodes)
        {
            std::vector<std::vector<int>> out;
            out.reserve(snodes.size());
            for (const auto &pr : snodes) {
                int s = pr.first;
                int e = pr.second;
                std::vector<int> rows;
                for (int c = s; c < e; ++c) {
                    int a = pat.row_ptr_L[c];
                    int b = pat.row_ptr_L[c + 1];
                    rows.insert(rows.end(), pat.col_ind_L.begin() + a, pat.col_ind_L.begin() + b);
                }
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                out.push_back(std::move(rows));
            }
            return out;
        }
} // namespace ichol::symbolic

