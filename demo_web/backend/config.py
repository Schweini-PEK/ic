from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = PROJECT_ROOT / "demo_web"
FRONTEND_DIR = DEMO_ROOT / "frontend"
RUNS_DIR = DEMO_ROOT / "runs"
DEMO_DATA_DIR = DEMO_ROOT / "data"
VERSION2_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "version2"
VERSION2_COMPARE_SOURCE_DIR = VERSION2_RESULTS_DIR / "super"
VERSION2_COMPARE_DATA_PATH = DEMO_DATA_DIR / "version2_compare_results.json"


@dataclass(frozen=True)
class MatrixInfo:
    index: int
    name: str
    order: int
    nnz: int
    domain: str
    matrix_path: str


MATRICES: list[MatrixInfo] = [
    MatrixInfo(1, "nasa2146", 2146, 72250, "结构力学", "./data/matrices/nasa2146/nasa2146.mtx"),
    MatrixInfo(2, "bcsstk21", 3600, 26600, "结构力学", "./data/matrices/bcsstk21/bcsstk21.mtx"),
    MatrixInfo(3, "Kuu", 7102, 340200, "结构力学", "./data/matrices/kuu/kuu.mtx"),
    MatrixInfo(4, "fv1", 9604, 85264, "生物工程", "./data/matrices/fv1/fv1.mtx"),
    MatrixInfo(5, "Pres_Poisson", 14822, 715804, "流体力学", "./data/matrices/Pres_Poisson/Pres_Poisson.mtx"),
    MatrixInfo(6, "Dubcova1", 16129, 253009, "偏微分方程", "./data/matrices/Dubcova1/Dubcova1.mtx"),
    MatrixInfo(7, "Dubcova2", 65025, 1030225, "偏微分方程", "./data/matrices/Dubcova2/Dubcova2.mtx"),
]

ROUTES = [
    {"value": "level", "label": "Level"},
    {"value": "supernode", "label": "Supernode"},
]

PRECISIONS = [
    {"value": "double", "label": "FP64"},
    {"value": "float", "label": "FP32"},
]

RUN_GROUPS = [
    {"value": "level_double", "label": "Level + FP64", "route": "level", "precision": "double"},
    {"value": "level_float", "label": "Level + FP32", "route": "level", "precision": "float"},
    {"value": "super_double", "label": "Supernode + FP64", "route": "supernode", "precision": "double"},
    {"value": "super_float", "label": "Supernode + FP32", "route": "supernode", "precision": "float"},
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
