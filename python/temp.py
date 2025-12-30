from petsc4py import PETSc
import scipy.io
import numpy as np

mtx_file = "../test/data/Kuu.mtx"
k = 3

# ---- load MatrixMarket into CSR ----
A = scipy.io.mmread(mtx_file).tocsr()
A = (A + A.T) * 0.5  # enforce symmetry
n = A.shape[0]

# ---- build symmetric lower pattern (level 0) ----
# S_row[i] = sorted list of j <= i where A(i,j) or A(j,i) is nonzero
S_row = [[] for _ in range(n)]

for i in range(n):
    row_cols = set(A.indices[A.indptr[i]:A.indptr[i+1]])
    for j in row_cols:
        if j <= i:
            S_row[i].append(j)

# ---- symbolic IC(k) ----
L_row = [[] for _ in range(n)]
L_lev = [[] for _ in range(n)]

marker = np.full(n, -1, dtype=int)
work_col = np.empty(n, dtype=int)
work_lev = np.empty(n, dtype=int)

for i in range(n):
    used = 0

    # level-0 structure
    for j in S_row[i]:
        if marker[j] != i:
            marker[j] = i
            work_col[used] = j
            work_lev[j] = 0
            used += 1

    # ensure diagonal
    if marker[i] != i:
        marker[i] = i
        work_col[used] = i
        work_lev[i] = 0
        used += 1

    # propagate fill
    pos = 0
    while pos < used:
        j = work_col[pos]
        pos += 1

        if j >= i:
            continue

        lev_ij = work_lev[j]
        if lev_ij > k:
            continue

        Lj = L_row[j]
        Lj_lev = L_lev[j]

        for m, lev_jm in zip(Lj, Lj_lev):
            if m >= j:
                continue  # strict lower only

            lev_new = lev_ij + lev_jm + 1
            if lev_new > k:
                continue

            if marker[m] != i:
                marker[m] = i
                work_col[used] = m
                work_lev[m] = lev_new
                used += 1
            else:
                work_lev[m] = min(work_lev[m], lev_new)

    cols = np.sort(work_col[:used])
    L_row[i] = cols.tolist()
    L_lev[i] = [work_lev[c] for c in cols]

# ---- count nnz in L (including diagonal) ----
nnz_L = sum(len(r) for r in L_row)

print("Predicted nnz in L (including diagonal):", nnz_L)
