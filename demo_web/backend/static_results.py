from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    MATRICES,
    PROJECT_ROOT,
    RUN_GROUPS,
    VERSION2_COMPARE_DATA_PATH,
    VERSION2_COMPARE_SOURCE_DIR,
    MatrixInfo,
)


RUN_HEADER_RE = re.compile(r"-{20,}\s*run\s+\d+\s*/\s*\d+", re.IGNORECASE)

RUN_GROUP_SELECTOR = {
    "level_double": {"precision": "double", "policy_prefix": "level"},
    "level_float": {"precision": "float", "policy_prefix": "level"},
    "super_double": {"precision": "double", "policy_prefix": "supernode"},
    "super_float": {"precision": "float", "policy_prefix": "supernode"},
}

RUN_GROUP_LABELS = {
    str(item["value"]): str(item["label"])
    for item in RUN_GROUPS
}


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _parse_run_chunk(chunk: str) -> dict[str, Any] | None:
    precision = re.search(r"precision_pcg=([A-Za-z0-9_-]+)", chunk)
    policy = re.search(r"factorized_precond_policy=([A-Za-z0-9_-]+)", chunk)
    ordering = re.search(r"ordering=([A-Za-z0-9_-]+)", chunk)
    scaling = re.search(r"scaling=([A-Za-z0-9_-]+)", chunk)
    if not (precision and policy):
        return None

    pcg_time_s = _last_float(r"PCG time:\s*([0-9.eE+-]+)\s*seconds", chunk)
    return {
        "precision": precision.group(1),
        "policy": policy.group(1),
        "ordering": ordering.group(1) if ordering else None,
        "scaling": scaling.group(1) if scaling else None,
        "pcg_iterations": _last_int(r"PCG iterations=([0-9]+)", chunk),
        "final_residual": _last_float(r"finalRes=([0-9.eE+-]+)", chunk),
        "sptrsv_avg_time_ms": _last_float(r"avg_sptrsv_ms=([0-9.eE+-]+)", chunk),
        "pcg_time_ms": pcg_time_s * 1000.0 if pcg_time_s is not None else None,
        "relative_residual": _last_float(
            r"relative residual of the scaled linear system .*?:\s*([0-9.eE+-]+)",
            chunk,
        ),
        "nnz_l": _last_int(r"nnz of L\s*:\s*([0-9]+)", chunk),
        "exception": _parse_exception(chunk),
    }


def _parse_exception(text: str) -> str | None:
    match = re.search(r"Exception:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None


def _parse_result_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = RUN_HEADER_RE.split(text)
    return [
        parsed
        for chunk in chunks[1:]
        if (parsed := _parse_run_chunk(chunk)) is not None
    ]


def _matrix_result_path(matrix: MatrixInfo) -> Path:
    matrix_stem = Path(matrix.matrix_path).stem
    return VERSION2_COMPARE_SOURCE_DIR / f"results_{matrix_stem}.txt"


def _find_group_run(runs: list[dict[str, Any]], group_id: str) -> dict[str, Any] | None:
    selector = RUN_GROUP_SELECTOR[group_id]
    for item in runs:
        if item["precision"] != selector["precision"]:
            continue
        if not str(item["policy"]).startswith(selector["policy_prefix"]):
            continue
        if item.get("ordering") != "RCM" or item.get("scaling") != "UnitSqrtDiag":
            continue
        return item
    return None


def _empty_matrix_row(matrix: MatrixInfo, source_path: Path | None, message: str) -> dict[str, Any]:
    return {
        "index": matrix.index,
        "name": matrix.name,
        "order": matrix.order,
        "nnz": matrix.nnz,
        "status": "failed",
        "source_path": _display_path(source_path),
        "pcg_iterations": None,
        "sptrsv_avg_time_ms": None,
        "pcg_time_ms": None,
        "relative_residual": None,
        "final_residual": None,
        "nnz_l": None,
        "exception": message,
    }


def _matrix_row(matrix: MatrixInfo, source_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    status = "success" if run.get("exception") is None and run.get("pcg_iterations") is not None else "failed"
    return {
        "index": matrix.index,
        "name": matrix.name,
        "order": matrix.order,
        "nnz": matrix.nnz,
        "status": status,
        "source_path": _display_path(source_path),
        "pcg_iterations": run.get("pcg_iterations"),
        "sptrsv_avg_time_ms": run.get("sptrsv_avg_time_ms"),
        "pcg_time_ms": run.get("pcg_time_ms"),
        "relative_residual": run.get("relative_residual"),
        "final_residual": run.get("final_residual"),
        "nnz_l": run.get("nnz_l"),
        "exception": run.get("exception"),
    }


def build_version2_compare_data() -> dict[str, Any]:
    """Parse immutable version2 result files into a demo-local JSON cache."""

    VERSION2_COMPARE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, Any]] = {}
    parsed_by_matrix: dict[str, tuple[Path, list[dict[str, Any]]] | None] = {}

    for matrix in MATRICES:
        path = _matrix_result_path(matrix)
        if path.exists():
            parsed_by_matrix[matrix.name] = (path, _parse_result_file(path))
        else:
            parsed_by_matrix[matrix.name] = None

    for group in RUN_GROUPS:
        group_id = str(group["value"])
        matrices: list[dict[str, Any]] = []
        for matrix in MATRICES:
            parsed = parsed_by_matrix[matrix.name]
            if parsed is None:
                matrices.append(_empty_matrix_row(matrix, None, "version2结果文件不存在"))
                continue
            source_path, runs = parsed
            run = _find_group_run(runs, group_id)
            if run is None:
                matrices.append(_empty_matrix_row(matrix, source_path, "未找到匹配的version2实验结果"))
                continue
            matrices.append(_matrix_row(matrix, source_path, run))

        success_count = sum(1 for item in matrices if item["status"] == "success")
        groups[group_id] = {
            "run_id": group_id,
            "group_id": group_id,
            "group_label": RUN_GROUP_LABELS.get(group_id, group_id),
            "status": "success" if success_count == len(MATRICES) else "partial_failed",
            "message": f"version2结果数据：{success_count}/{len(MATRICES)} 个矩阵可用。",
            "source": _display_path(VERSION2_COMPARE_SOURCE_DIR),
            "matrices": matrices,
        }

    payload = {
        "source": _display_path(VERSION2_COMPARE_SOURCE_DIR),
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "groups": groups,
    }
    VERSION2_COMPARE_DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_version2_compare_data() -> dict[str, Any]:
    return build_version2_compare_data()


def list_version2_compare_runs() -> list[dict[str, Any]]:
    payload = _load_version2_compare_data()
    groups = payload.get("groups", {})
    return [groups[str(item["value"])] for item in RUN_GROUPS if str(item["value"]) in groups]


def read_version2_compare_run(group_id: str) -> dict[str, Any] | None:
    payload = _load_version2_compare_data()
    group = payload.get("groups", {}).get(group_id)
    return group if isinstance(group, dict) else None


def compare_version2_runs(base_group: str, target_group: str) -> dict[str, Any]:
    base = read_version2_compare_run(base_group)
    target = read_version2_compare_run(target_group)
    if base is None or target is None:
        raise FileNotFoundError("请选择可用的version2实验结果。")

    base_rows = {str(item.get("name")): item for item in base.get("matrices", [])}
    target_rows = {str(item.get("name")): item for item in target.get("matrices", [])}
    comparisons: list[dict[str, Any]] = []
    for matrix in MATRICES:
        base_row = base_rows.get(matrix.name, {})
        target_row = target_rows.get(matrix.name, {})
        base_sptrsv = base_row.get("sptrsv_avg_time_ms")
        target_sptrsv = target_row.get("sptrsv_avg_time_ms")
        base_pcg = base_row.get("pcg_time_ms")
        target_pcg = target_row.get("pcg_time_ms")
        comparisons.append(
            {
                "index": matrix.index,
                "name": matrix.name,
                "base_sptrsv_avg_time_ms": base_sptrsv,
                "target_sptrsv_avg_time_ms": target_sptrsv,
                "sptrsv_speedup": _speedup(base_sptrsv, target_sptrsv),
                "base_pcg_time_ms": base_pcg,
                "target_pcg_time_ms": target_pcg,
                "pcg_speedup": _speedup(base_pcg, target_pcg),
            }
        )

    return {
        "base_group": base_group,
        "base_label": base.get("group_label", base_group),
        "target_group": target_group,
        "target_label": target.get("group_label", target_group),
        "definition": "Speedup = T_A / T_B",
        "source": _display_path(VERSION2_COMPARE_SOURCE_DIR),
        "matrices": comparisons,
    }


def _speedup(base_value: object, target_value: object) -> float | None:
    try:
        base_float = float(base_value)
        target_float = float(target_value)
    except (TypeError, ValueError):
        return None
    if target_float <= 0:
        return None
    return base_float / target_float
