from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = PROJECT_ROOT / "demo_web"
FRONTEND_DIR = DEMO_ROOT / "frontend"
RUNS_DIR = DEMO_ROOT / "runs"


@dataclass(frozen=True)
class MatrixInfo:
    index: int
    name: str
    order: int
    nnz: int
    matrix_path: str


MATRICES: list[MatrixInfo] = [
    MatrixInfo(1, "bcsstk21", 3600, 26600, "./data/matrices/bcsstk21/bcsstk21.mtx"),
    MatrixInfo(2, "Dubcova1", 16129, 253009, "./data/matrices/Dubcova1/Dubcova1.mtx"),
    MatrixInfo(3, "Dubcova2", 65025, 1030225, "./data/matrices/Dubcova2/Dubcova2.mtx"),
    MatrixInfo(4, "fv1", 9604, 85264, "./data/matrices/fv1/fv1.mtx"),
    MatrixInfo(5, "Kuu", 7102, 340200, "./data/matrices/kuu/kuu.mtx"),
    MatrixInfo(6, "nasa2146", 2146, 72250, "./data/matrices/nasa2146/nasa2146.mtx"),
    MatrixInfo(7, "Pres_Poisson", 14822, 715804, "./data/matrices/Pres_Poisson/Pres_Poisson.mtx"),
]

ROUTES = [
    {"value": "level", "label": "Level"},
    {"value": "supernode", "label": "Supernode"},
]

PRECISIONS = [
    {"value": "double", "label": "FP64"},
    {"value": "float", "label": "FP32"},
]

FIXED_OPTIONS = {
    "ordering": "RCM",
    "scaling": "UnitSqrtDiag",
    "precision_fact": "double",
    "level_k": "-1",
    "pivot_shift_strategy": "Static",
    "static_shift": "1e-06",
    "lfil": "20",
    "drop_tol": "0.0001",
    "matrix_scope": "all",
}

EXECUTABLE = PROJECT_ROOT / "build" / "src" / "ict_pcg"
