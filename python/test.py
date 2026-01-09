#!/usr/bin/env python3
"""
test.py

Python 版本的 supernodal 符号分析对比测试。

用法:
    python test.py path/to/matrix.mtx

如果未提供路径，脚本使用内置示例矩阵。
"""
import sys
import os
import numpy as np
import scipy.io
import scipy.sparse as sp

try:
    from sksparse.cholmod import cholesky as sk_cholesky
    SKSPARSE = True
except Exception:
    SKSPARSE = False

# ---------------------
# Utilities / IO
# ---------------------
def read_matrix_market(path):
    M = scipy.io.mmread(path)
    # ensure symmetric pattern (we'll symmetrize to be safe)
    if sp.issparse(M):
        M = M.tocsc()
    else:
        M = sp.csc_matrix(M)
    # If not symmetric, symmetrize pattern (A + A.T)/2 for SPD pattern
    if (M - M.T).nnz != 0:
        M = (M + M.T) * 0.5
        M = sp.csc_matrix(M)
    return M

def build_example_matrix(n=12):
    A = sp.lil_matrix((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 4.0
        if i + 1 < n:
            A[i, i+1] = -1.0
            A[i+1, i] = -1.0
        if i + 3 < n:
            A[i, i+3] = -0.5
            A[i+3, i] = -0.5
    return A.tocsc()

# ---------------------
# elimination tree (simple pattern-only algorithm)
# ---------------------
def elimination_tree_from_pattern(A_csc):
    """
    Build elimination tree parent array using the standard ancestor trick.
    A_csc: scipy.sparse.csc_matrix (assumed symmetric pattern)
    returns parent: numpy array length n, parent[i] is parent of node i or -1
    """
    n = A_csc.shape[0]
    indptr = A_csc.indptr
    indices = A_csc.indices
    parent = -1 * np.ones(n, dtype=int)
    ancestor = -1 * np.ones(n, dtype=int)
    for k in range(n):
        # iterate rows i where A[i,k] != 0 and i < k
        for idx in range(indptr[k], indptr[k+1]):
            i = indices[idx]
            if i >= k:
                continue
            j = i
            while ancestor[j] != -1 and ancestor[j] != k:
                t = ancestor[j]
                ancestor[j] = k
                j = t
            if ancestor[j] == -1:
                ancestor[j] = k
                parent[j] = k
    return parent

# ---------------------
# compute complete cholesky pattern (fp) — fallback + CHOLMOD path
# ---------------------
class FactorPattern:
    """Mimic the minimal fields used in your C++ test: col_ind_L (vector of rows),
       row_ptr_L (pointer into col_ind_L per column)."""
    def __init__(self, n):
        self.n = n
        self.col_ind_L = []  # flattened row indices of L columns
        self.row_ptr_L = [0] * (n + 1)  # length n+1

def compute_complete_cholesky_pattern(A_csc, etree):
    """
    If sksparse available, use CHOLMOD to get L and derive pattern.
    Else, use a naive pattern: each column's lower pattern = rows >= col where A has nonzero.
    Note: fallback is a heuristic (may under/over-approximate fill-in).
    """
    n = A_csc.shape[0]
    fp = FactorPattern(n)
    if SKSPARSE:
        # numeric factorization but we only use sparsity pattern of L
        F = sk_cholesky(A_csc)
        L = F.L()  # scipy.sparse.csc_matrix
        # ensure CSC
        L = L.tocsc()
        for j in range(n):
            start, end = L.indptr[j], L.indptr[j+1]
            col_rows = L.indices[start:end]
            fp.col_ind_L.extend(int(x) for x in col_rows)
            fp.row_ptr_L[j+1] = len(fp.col_ind_L)
        return fp
    else:
        # fallback: use A's lower structure as L pattern (no fill-in)
        indptr = A_csc.indptr
        indices = A_csc.indices
        for j in range(n):
            # collect rows i >= j where A[i,j] != 0
            rows = [int(i) for i in indices[indptr[j]:indptr[j+1]] if i >= j]
            fp.col_ind_L.extend(rows)
            fp.row_ptr_L[j+1] = len(fp.col_ind_L)
        return fp

# ---------------------
# helper: build column pattern sets from fp
# ---------------------
def fp_column_patterns(fp):
    patterns = []
    for j in range(fp.n):
        s = set(fp.col_ind_L[fp.row_ptr_L[j]: fp.row_ptr_L[j+1]])
        patterns.append(s)
    return patterns

# ---------------------
# supernode detection (conservative + approx)
# conservative: exact pattern equality + parent[k] == k+1
# approx: allow similarity threshold on pattern intersection/union (Jaccard)
# ---------------------
def detect_supernodes(fp, etree):
    patterns = fp_column_patterns(fp)
    n = fp.n
    snodes = []
    j = 0
    while j < n:
        start = j
        j_next = j + 1
        while j_next < n:
            cond_parent = (etree[j] == j_next)
            pat_j = set(x for x in patterns[j] if x > j)
            pat_jn = set(x for x in patterns[j_next] if x > j_next)
            pat_j_minus = set(x for x in pat_j if x != j_next)
            if cond_parent and (pat_j_minus == pat_jn):
                j = j_next
                j_next += 1
                continue
            else:
                break
        snodes.append((start, j+1))  # half-open [start, end)
        j = j + 1
    return snodes

def jaccard(a,b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    if uni == 0:
        return 1.0
    return inter / uni

def detect_supernodes_approx(fp, etree, thr=1.0):
    patterns = fp_column_patterns(fp)
    n = fp.n
    snodes = []
    j = 0
    while j < n:
        start = j
        current_pat = patterns[j]
        j += 1
        while j < n:
            cond_parent = (etree[j-1] == j)  # parent of previous column equals next column
            # compare pattern of previous column and this column (restrict to strict lower)
            pat_prev = set(x for x in current_pat if x > (j-1))
            pat_cur = set(x for x in patterns[j] if x > j)
            # compute similarity (Jaccard)
            sim = jaccard(pat_prev, pat_cur)
            if cond_parent and sim >= thr:
                # merge: update current_pat to intersection/union? we keep union to be conservative
                current_pat = current_pat | patterns[j]
                j += 1
            else:
                break
        snodes.append((start, j))  # [start, j)
    return snodes

# ---------------------
# build col->snode map
# ---------------------
def build_col2snode(snodes, ncols):
    col2s = [-1] * ncols
    for sid, (s,e) in enumerate(snodes):
        for c in range(s, e):
            col2s[c] = sid
    return col2s

# ---------------------
# compute snode rows (union of rows of columns in snode)
# ---------------------
def compute_snode_rows(fp, snodes):
    rows_per_snode = []
    for (s,e) in snodes:
        srows = set()
        for j in range(s, e):
            srows.update(fp.col_ind_L[fp.row_ptr_L[j]:fp.row_ptr_L[j+1]])
        rows_per_snode.append(sorted(srows))
    return rows_per_snode

# ---------------------
# level sets (simplified): compute column-level as distance to root in etree
# and build simple bucket structure used later
# ---------------------
class LevelSets:
    def __init__(self, level_ptr, level_idx):
        self.level_ptr = level_ptr
        self.level_idx = level_idx

class SnodeLevelSets:
    def __init__(self, snode_level, level_sets):
        self.snode_level = snode_level
        self.level_sets = level_sets

def build_level_sets(fp, symopts=None):
    # simplified: column level = depth in etree
    # we build shallow fake structure: all zeros — placeholder, not used heavily
    n = fp.n
    levels = [0]*n
    # compute parent using elimination_tree from fp (recompute)
    # For simplicity, compute a trivial level based on number of rows in column
    for j in range(n):
        levels[j] = fp.row_ptr_L[j+1] - fp.row_ptr_L[j]
    # bucket columns by these levels
    unique = sorted(set(levels))
    level_map = {v:i for i,v in enumerate(unique)}
    buckets = {i:[] for i in range(len(unique))}
    for j,v in enumerate(levels):
        buckets[level_map[v]].append(j)
    # flatten
    level_ptr = [0]
    level_idx = []
    for i in range(len(unique)):
        level_idx.extend(buckets[i])
        level_ptr.append(len(level_idx))
    return LevelSets(level_ptr, level_idx)

def build_snode_level_sets(col_ls, snodes):
    # assign each snode a level equal to min column level index in col_ls (simplified)
    n_sn = len(snodes)
    # We approximate: snode_level = number of columns in snode (placeholder)
    snode_level = [e-s for (s,e) in snodes]
    # level_sets: bucket by snode_level values
    unique = sorted(set(snode_level))
    buckets = {i:[] for i in range(len(unique))}
    for sid, val in enumerate(snode_level):
        idx = unique.index(val)
        buckets[idx].append(sid)
    level_ptr = [0]
    level_idx = []
    for i in range(len(unique)):
        level_idx.extend(buckets[i])
        level_ptr.append(len(level_idx))
    return SnodeLevelSets(snode_level, LevelSets(level_ptr, level_idx))

# ---------------------
# histogram printing
# ---------------------
def snode_size_histogram(snodes, max_bucket=20):
    hist = [0] * (max_bucket + 1)
    for s,e in snodes:
        sz = e - s
        if sz >= max_bucket:
            hist[max_bucket] += 1
        else:
            hist[sz] += 1
    return hist

def print_sn_range_list(snodes, limit=20):
    m = len(snodes)
    print("  total supernodes = {}".format(m))
    shown = min(m, limit)
    print("  first {} supernodes [start,end):".format(shown))
    for i in range(shown):
        s,e = snodes[i]
        print("    [{}, {}) size={}".format(s,e,e-s))
    if m > shown:
        print("    ... (+{} more)".format(m-shown))

def print_histogram(hist):
    for i in range(len(hist)-1):
        print("    size={:2d} -> {}".format(i, hist[i]))
    print("    size>={} -> {}".format(len(hist)-1, hist[-1]))

# ---------------------
# Main test logic (mirrors your C++ test)
# ---------------------
def main(path=None):
    if path is None:
        print("No path provided -> using built-in example matrix.")
        A = build_example_matrix(12)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        print("Reading matrix:", path)
        A = read_matrix_market(path)

    ncols = A.shape[1]
    nnz = A.nnz
    print(f"Matrix: {path if path else '<example>'}  ncols={ncols} nnz={nnz}")

    # 1) build etree and factor pattern
    etree = elimination_tree_from_pattern(A)
    fp = compute_complete_cholesky_pattern(A, etree)

    assert fp.row_ptr_L[-1] == len(fp.col_ind_L)

    # 2) column-level level sets (simplified)
    col_ls = build_level_sets(fp)

    # 3) conservative detection
    sn_cons = detect_supernodes(fp, etree)
    print("\nConservative detect:")
    print_sn_range_list(sn_cons)
    hist_cons = snode_size_histogram(sn_cons, 16)
    print("Conservative size histogram:")
    print_histogram(hist_cons)

    # 4) approximate detect thr=1.0 (should match conservative)
    sn_appx1 = detect_supernodes_approx(fp, etree, 1.0)
    print("\nApproximate detect (threshold=1.0):")
    print_sn_range_list(sn_appx1)
    hist_appx1 = snode_size_histogram(sn_appx1, 16)
    print("Approx(1.0) size histogram:")
    print_histogram(hist_appx1)

    # 5) approximate detect thr=0.8
    thr = 0.8
    sn_appx08 = detect_supernodes_approx(fp, etree, thr)
    print("\nApproximate detect (threshold={}):".format(thr))
    print_sn_range_list(sn_appx08)
    hist_appx08 = snode_size_histogram(sn_appx08, 16)
    print("Approx(0.8) size histogram:")
    print_histogram(hist_appx08)

    # 6) relationship assertion-like info
    if len(sn_appx08) <= len(sn_cons):
        print("\nOK: approx(0.8) supernodes <= conservative supernodes (expected).")
    else:
        print("\nWarning: approx produced MORE supernodes than conservative (unexpected).")

    # 7) col->snode maps
    col2s_cons = build_col2snode(sn_cons, ncols)
    col2s_appx08 = build_col2snode(sn_appx08, ncols)
    assert len(col2s_cons) == ncols
    assert len(col2s_appx08) == ncols

    for c in range(ncols):
        if not (0 <= col2s_cons[c] < len(sn_cons)):
            print("col", c, "bad mapping in conservative")
        if not (0 <= col2s_appx08[c] < len(sn_appx08)):
            print("col", c, "bad mapping in approx(0.8)")

    # 8) compute snode rows
    snrows_cons = compute_snode_rows(fp, sn_cons)
    snrows_appx08 = compute_snode_rows(fp, sn_appx08)
    print("\nSnode rows: conservative has {} blocks, approx(0.8) has {}".format(len(snrows_cons), len(snrows_appx08)))

    # 9) build snode-level-sets (simplified)
    snode_level_cons = build_snode_level_sets(col_ls, sn_cons)
    snode_level_appx08 = build_snode_level_sets(col_ls, sn_appx08)
    print("\nSnode-level sets: conservative levels = {}; level buckets = {}".format(
        len(snode_level_cons.snode_level),
        len(s
