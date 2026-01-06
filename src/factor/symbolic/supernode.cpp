#include <vector>
#include <utility>
#include <cassert>
#include <algorithm>
#include <climits>
#include <cstdint>

#include "symbolic.hpp"

namespace ichol::symbolic {

/*
  Full CHOLMOD-style ported implementation (best-effort automated port):

  - detect_supernodes_cholmod: a near-line-by-line logical port of CHOLMOD's
    supernodal symbolic analysis (super_symbolic2), adapted to the project's
    FactorPattern/ETree types and with names changed to fit the project.

  - detect_supernodes: legacy API kept for compatibility; it calls the more
    complete port and returns only the supernode ranges (vector<pair<int,int>>).

  Notes:
  - This port focuses on the symmetric case (pattern for L available).
  - GPU-related code paths are omitted (you can reintroduce them later if
    needed). The core logic (fundamental supernodes, relaxed amalgamation,
    subtree traversal to build Lpi/Ls) is ported as-is.
  - The port returns optional outputs Lpi/Ls when requested (so you can
    write them back into your Factor structure).
*/

struct SupernodalPattern {
    std::vector<std::pair<int,int>> snodes; // [s,e) ranges
    std::vector<int> Lpi; // pointers into Ls for each relaxed supernode
    std::vector<int> Ls;  // concatenated row indices for relaxed supernodes
};

// Ported CHOLMOD-style subtree used by the main routine.  Symmetric case only.
static void subtree_cholmod(
    int j, int k,                          // j: column to scan, k: current column in supernode
    const FactorPattern& pat,              // pattern (Ap/Ai equivalent)
    const std::vector<int>& Anz,           // optional column nz counts (can be empty)
    const std::vector<int>& SuperMap,      // mapping column -> relaxed supernode id
    const std::vector<int>& Sparent,       // parent per relaxed supernode
    int mark,
    int sorted,                             // not used, kept for similarity
    int k1,                                 // only consider rows < k1
    std::vector<int>& Flag,                 // size = nsuper
    std::vector<int>& Ls,                   // result storage
    std::vector<int>& Lpi2)                 // write pointer per supernode
{
    int p = pat.row_ptr_L[j];
    int pend = pat.row_ptr_L[j+1];
    // if there is an Anz vector, respect it (simulate Ap+Anz behavior)
    if (!Anz.empty()) pend = p + Anz[j];
    for (; p < pend; ++p) {
        int i = pat.col_ind_L[p];
        if (i < k1) {
            int si = SuperMap[i];
            for (; si >= 0 && Flag[si] < mark; si = Sparent[si]) {
                Ls[Lpi2[si]++] = k;
                Flag[si] = mark;
            }
        } else {
            // If columns are sorted, we could break here. We do not assume that.
        }
    }
}

// A near-complete CHOLMOD-style supernodal symbolic analysis port.
// - pattern: contains row_ptr_L and col_ind_L (L's column structure)
// - etree: elimination tree, etree.parent length == n
// - out (optional): fill Lpi / Ls if non-null
static SupernodalPattern detect_supernodes_cholmod(const FactorPattern& pattern,
                                                   const ETree& etree,
                                                   std::vector<int>* out_Lpi = nullptr,
                                                   std::vector<int>* out_Ls = nullptr)
{
    SupernodalPattern result;

    int n = static_cast<int>(pattern.row_ptr_L.size()) - 1;
    if (n <= 0) return result;

    const std::vector<int>& Parent = etree.parent;

    // 1) ColCount
    std::vector<int> ColCount(n);
    for (int j = 0; j < n; ++j) ColCount[j] = pattern.row_ptr_L[j+1] - pattern.row_ptr_L[j];

    // 2) Wi: child counts
    std::vector<int> Wi(n, 0);
    for (int j = 0; j < n; ++j) {
        int p = Parent[j];
        if (p >= 0 && p < n) Wi[p]++;
    }

    // 3) Fundamental supernodes (Super)
    std::vector<int> Super; Super.reserve(n + 1);
    Super.push_back(0);
    for (int j = 1; j < n; ++j) {
        if (Parent[j-1] != j || ColCount[j-1] != ColCount[j] + 1 || Wi[j] > 1) {
            Super.push_back(j);
        }
    }
    Super.push_back(n);
    int nfsuper = static_cast<int>(Super.size()) - 1;
    if (nfsuper <= 0) return result;

    // 4) SuperMap (fundamental)
    std::vector<int> SuperMap(n, -1);
    for (int s = 0; s < nfsuper; ++s) for (int k = Super[s]; k < Super[s+1]; ++k) SuperMap[k] = s;

    // 5) Sparent (fundamental)
    std::vector<int> Sparent(nfsuper, -1);
    for (int s = 0; s < nfsuper; ++s) {
        int lastcol = Super[s+1] - 1;
        int parent = (lastcol >= 0 && lastcol < n) ? Parent[lastcol] : -1;
        Sparent[s] = (parent == -1) ? -1 : SuperMap[parent];
    }

    // 6) initialize for relaxation
    const int EMPTY = -1;
    std::vector<int> Merged(nfsuper, EMPTY);
    std::vector<int> Nscol(nfsuper, 0);
    std::vector<int> Snz(nfsuper, 0);
    std::vector<int> Zeros(nfsuper, 0);
    for (int s = 0; s < nfsuper; ++s) {
        Nscol[s] = Super[s+1] - Super[s];
        Snz[s] = ColCount[Super[s]];
        Zeros[s] = 0;
    }

    // 7) relaxation parameters (CHOLMOD defaults)
    int nrelax0 = 1, nrelax1 = 6, nrelax2 = 20;
    double zrelax0 = 0.1, zrelax1 = 0.5, zrelax2 = 0.9;

    // 8) relaxed amalgamation (reverse order)
    for (int s = nfsuper - 2; s >= 0; --s) {
        int ss = Sparent[s];
        if (ss == EMPTY) continue;
        int sparent = ss;
        while (sparent != EMPTY && Merged[sparent] != EMPTY) sparent = Merged[sparent];
        for (int ss2 = ss; ss2 != EMPTY && Merged[ss2] != EMPTY; ) {
            int snext = Merged[ss2];
            Merged[ss2] = sparent;
            ss2 = snext;
        }
        if (sparent != s + 1) continue;
        int nscol0 = Nscol[s];
        int nscol1 = Nscol[s+1];
        int ns = nscol0 + nscol1;
        int totzeros = Zeros[s+1];
        double lnz1 = static_cast<double>(Snz[s+1]);
        bool merge = false;
        if (ns <= nrelax0) merge = true;
        else {
            double lnz0 = static_cast<double>(Snz[s]);
            double xnewzeros = static_cast<double>(nscol0) * (lnz1 + nscol0 - lnz0);
            int newzeros = nscol0 * (Snz[s+1] + nscol0 - Snz[s]);
            if (xnewzeros == 0.0) merge = true;
            else {
                double xtotzeros = static_cast<double>(totzeros) + xnewzeros;
                double xns = static_cast<double>(ns);
                double xtotsize = (xns * (xns + 1.0) / 2.0) + xns * (lnz1 - nscol1);
                double z = xtotzeros / xtotsize;
                double max_xtotsize = static_cast<double>(INT_MAX) / static_cast<double>(sizeof(double));
                bool size_ok = (xtotsize < max_xtotsize);
                if (((ns <= nrelax1 && z < zrelax0) || (ns <= nrelax2 && z < zrelax1) || (z < zrelax2)) && size_ok) merge = true;
            }
        }
        if (merge) {
            // compute newzeros using original Snz[s] before overwriting it
            int newzeros = nscol0 * (Snz[s+1] + nscol0 - Snz[s]); // Snz[s] is original here
            int totzeros = Zeros[s+1] + newzeros;
            Zeros[s] = totzeros;
            Snz[s] = nscol0 + Snz[s+1];
            Merged[s+1] = s;
            Nscol[s] += Nscol[s+1];
        }
    }

    // 9) build final relaxed supernode list (Super2)
    std::vector<int> Super2; Super2.reserve(nfsuper + 1);
    for (int s = 0; s < nfsuper; ++s) if (Merged[s] == EMPTY) Super2.push_back(Super[s]);
    Super2.push_back(n);
    int nsuper = static_cast<int>(Super2.size()) - 1;

    // SuperMap2, Sparent2 for relaxed nodes
    std::vector<int> SuperMap2(n, -1);
    for (int s = 0; s < nsuper; ++s) for (int k = Super2[s]; k < Super2[s+1]; ++k) SuperMap2[k] = s;
    std::vector<int> Sparent2(nsuper, -1);
    for (int s = 0; s < nsuper; ++s) {
        int lastcol = Super2[s+1] - 1;
        int parent = (lastcol >= 0 && lastcol < n) ? Parent[lastcol] : -1;
        Sparent2[s] = (parent == -1) ? -1 : SuperMap2[parent];
    }

    // extract Snz_live and Nscol_live
    std::vector<int> Snz_live; Snz_live.reserve(nsuper);
    std::vector<int> Nscol_live; Nscol_live.reserve(nsuper);
    for (int s = 0; s < nfsuper; ++s) if (Merged[s] == EMPTY) { Snz_live.push_back(Snz[s]); Nscol_live.push_back(Nscol[s]); }

    // compute ssize / xsize
    uint64_t ssize64 = 0, xsize64 = 0;
    for (int s = 0; s < nsuper; ++s) {
        uint64_t nscol = static_cast<uint64_t>(Nscol_live[s]);
        uint64_t nsrow = static_cast<uint64_t>(Snz_live[s]);
        ssize64 += nsrow;
        xsize64 += nscol * nsrow;
    }

    bool ok = (ssize64 < static_cast<uint64_t>(INT_MAX)) && (xsize64 < static_cast<uint64_t>(INT_MAX));
    // ssize/xsize values available if needed

    // Build Lpi and Ls using subtree traversal
    std::vector<int> Lpi(nsuper + 1, 0);
    int p = 0;
    for (int s = 0; s < nsuper; ++s) {
        Lpi[s] = p;
        p += Snz_live[s];
    }
    Lpi[nsuper] = p;

    std::vector<int> Ls; Ls.resize(p);
    std::vector<int> Lpi2 = Lpi; // working pointers

    // Flag workspace per relaxed supernode
    std::vector<int> Flag(nsuper, 0);
    int mark = 1;

    // For symmetric case: fill leading columns and traverse subtree
    for (int s = 0; s < nsuper; ++s) {
        int k1 = Super2[s];
        int k2 = Super2[s+1];
        // place diagonal/block columns in leading column
        for (int kcol = k1; kcol < k2; ++kcol) Ls[Lpi2[s]++] = kcol;

        for (int kcol = k1; kcol < k2; ++kcol) {
            if (++mark == 0) { std::fill(Flag.begin(), Flag.end(), 0); mark = 1; }
            Flag[s] = mark;
            subtree_cholmod(kcol, kcol, pattern, std::vector<int>(), SuperMap2, Sparent2, mark, 1, k1, Flag, Ls, Lpi2);
        }
    }

    // validate
    // for (int s = 0; s < nsuper; ++s) {
    //     assert(Lpi2[s] == Lpi[s+1]);
    // }

    // fill result
    result.snodes.reserve(nsuper);
    for (int s = 0; s < nsuper; ++s) result.snodes.emplace_back(Super2[s], Super2[s+1]);
    result.Lpi.swap(Lpi);
    result.Ls.swap(Ls);

    if (out_Lpi) *out_Lpi = result.Lpi;
    if (out_Ls) *out_Ls = result.Ls;

    return result;
}

// Legacy API: keep original signature, but call the ported CHOLMOD function.
std::vector<std::pair<int, int>> detect_supernodes(const FactorPattern& pattern, const ETree& etree) {
    SupernodalPattern p = detect_supernodes_cholmod(pattern, etree, nullptr, nullptr);
    return p.snodes;
}

// keep existing helpers
std::vector<std::pair<int, int>> detect_supernodes_approx(const FactorPattern& pattern, const ETree& /*etree*/, double overlap_threshold)
{
    assert(overlap_threshold > 0.0 && overlap_threshold <= 1.0);
    int n = static_cast<int>(pattern.row_ptr_L.size()) - 1;
    std::vector<std::pair<int, int>> sn;
    if (n <= 0) return sn;

    int cur_start = 0;
    for (int c = 1; c < n; ++c)
    {
        int s1 = pattern.row_ptr_L[c - 1], e1 = pattern.row_ptr_L[c - 1 + 1];
        int s2 = pattern.row_ptr_L[c], e2 = pattern.row_ptr_L[c + 1];
        int len1 = e1 - s1;
        int len2 = e2 - s2;
        if (len1 == 0 || len2 == 0) {
            if (!(len1 == 0 && len2 == 0)) {
                sn.emplace_back(cur_start, c);
                cur_start = c;
            }
            continue;
        }

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
            continue;
        } else {
            sn.emplace_back(cur_start, c);
            cur_start = c;
        }
    }
    sn.emplace_back(cur_start, n);
    return sn;
}

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
