from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from config import EXECUTABLE, FIXED_OPTIONS, MATRICES, PROJECT_ROOT, RUNS_DIR, RUN_GROUPS, MatrixInfo
from parser import parse_result_file


RUN_TIMEOUT_SECONDS = 30 * 60
MAX_PARALLEL_JOBS = len(MATRICES)
RUN_GROUP_BY_CONFIG = {
    (str(item["route"]), str(item["precision"])): str(item["value"])
    for item in RUN_GROUPS
}
RUN_GROUP_LABELS = {
    str(item["value"]): str(item["label"])
    for item in RUN_GROUPS
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_policy(route: str) -> str:
    if route == "level":
        return "level"
    if route == "supernode":
        return "supernode"
    raise ValueError(f"unsupported route: {route}")


def _create_matrix_config(config_dir: Path, matrix: MatrixInfo, route: str, precision: str) -> Path:
    config_path = config_dir / f"ict_pcg_options_{matrix.name}.txt"
    lines = [
        f"matrix_path={matrix.matrix_path}",
        f"ordering={FIXED_OPTIONS['ordering']}",
        f"scaling={FIXED_OPTIONS['scaling']}",
        f"precision_fact={FIXED_OPTIONS['precision_fact']}",
        f"precision_pcg={precision}",
        f"factorized_precond_policy={_format_policy(route)}",
        f"level_k={FIXED_OPTIONS['level_k']}",
        f"pivot_shift_strategy={FIXED_OPTIONS['pivot_shift_strategy']}",
        f"static_shift={FIXED_OPTIONS['static_shift']}",
        f"lfil={FIXED_OPTIONS['lfil']}",
        f"drop_tol={FIXED_OPTIONS['drop_tol']}",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _matrix_output_name(matrix: MatrixInfo) -> str:
    return f"{Path(matrix.matrix_path).stem}.txt"


def _initial_matrix_row(matrix: MatrixInfo) -> dict[str, Any]:
    return {
        "index": matrix.index,
        "name": matrix.name,
        "order": matrix.order,
        "nnz": matrix.nnz,
        "status": "pending",
        "exit_code": None,
        "pcg_iterations": None,
        "sptrsv_avg_time_ms": None,
        "pcg_time_ms": None,
        "relative_residual": None,
        "final_residual": None,
        "nnz_l": None,
        "exception": None,
        "raw_path": None,
        "config_path": None,
    }


def _failed_matrix_row(matrix: MatrixInfo, message: str) -> dict[str, Any]:
    row = _initial_matrix_row(matrix)
    row["status"] = "failed"
    row["exception"] = message
    return row


def _run_one_matrix(
    matrix: MatrixInfo,
    route: str,
    precision: str,
    output_dir: Path,
    config_dir: Path,
) -> dict[str, Any]:
    row = _initial_matrix_row(matrix)
    config_path = _create_matrix_config(config_dir, matrix, route, precision)
    raw_path = output_dir / _matrix_output_name(matrix)
    row["config_path"] = str(config_path)
    row["raw_path"] = str(raw_path)

    matrix_file = PROJECT_ROOT / matrix.matrix_path
    if not matrix_file.exists():
        message = f"矩阵文件不存在：{matrix.matrix_path}"
        raw_path.write_text(f"Exception: {message}\n", encoding="utf-8")
        row.update(parse_result_file(raw_path))
        row["status"] = "failed"
        return row

    command = [str(EXECUTABLE), str(config_path)]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
        raw_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        row.update(parse_result_file(raw_path))
        row["exit_code"] = completed.returncode
        if completed.returncode == 0 and row.get("exception") is None and row.get("pcg_iterations") is not None:
            row["status"] = "success"
        else:
            row["status"] = "failed"
            if row.get("exception") is None:
                row["exception"] = f"ict_pcg exited with code {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        raw_path.write_text(f"{output}\nException: run timed out after {RUN_TIMEOUT_SECONDS}s\n", encoding="utf-8")
        row.update(parse_result_file(raw_path))
        row["status"] = "failed"
        row["exception"] = f"运行超时：超过 {RUN_TIMEOUT_SECONDS} 秒"
    except OSError as exc:
        raw_path.write_text(f"Exception: {exc}\n", encoding="utf-8")
        row.update(parse_result_file(raw_path))
        row["status"] = "failed"
        row["exception"] = str(exc)
    return row


def _group_id(route: str, precision: str) -> str:
    group_id = RUN_GROUP_BY_CONFIG.get((route, precision))
    if group_id is None:
        raise ValueError(f"unsupported run group: route={route}, precision={precision}")
    return group_id


def _summary_path(group_id: str) -> Path:
    return RUNS_DIR / group_id / "summary.json"


def create_run(route: str, precision: str) -> dict[str, Any]:
    """Run ict_pcg for all demo matrices and write outputs under a fixed group directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_id = _group_id(route, precision)
    run_dir = RUNS_DIR / group_id
    config_dir = run_dir / "configs"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": group_id,
        "group_id": group_id,
        "group_label": RUN_GROUP_LABELS[group_id],
        "status": "running",
        "message": "实验运行中。",
        "config": {
            "route": route,
            "precision": precision,
            **FIXED_OPTIONS,
        },
        "matrices": [_initial_matrix_row(item) for item in MATRICES],
        "created_at": timestamp,
        "updated_at": timestamp,
        "finished_at": None,
        "log": [
            f"group_id: {group_id}",
            f"route: {route}",
            f"precision_pcg: {precision}",
            "ordering: RCM",
            "scaling: UnitSqrtDiag",
        ],
    }

    summary_path = _summary_path(group_id)
    _write_json(summary_path, summary)

    if not EXECUTABLE.exists():
        message = f"未找到可执行程序：{EXECUTABLE}。请先在项目根目录执行 cmake 构建。"
        summary["status"] = "failed"
        summary["message"] = message
        summary["matrices"] = [_failed_matrix_row(item, message) for item in MATRICES]
        summary["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary["log"].append(message)
        _write_json(summary_path, summary)
        return summary

    results: list[dict[str, Any] | None] = [None] * len(MATRICES)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_matrix = {}
        for matrix in MATRICES:
            summary["log"].append(f"提交运行 {matrix.index}. {matrix.name}")
            future = executor.submit(_run_one_matrix, matrix, route, precision, run_dir, config_dir)
            future_to_matrix[future] = matrix
        _write_json(summary_path, summary)

        for future in as_completed(future_to_matrix):
            matrix = future_to_matrix[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep demo run alive if one matrix crashes
                result = _failed_matrix_row(matrix, str(exc))
            results[matrix.index - 1] = result
            summary["matrices"] = [
                item if item is not None else _initial_matrix_row(MATRICES[idx])
                for idx, item in enumerate(results)
            ]
            if result["status"] == "success":
                summary["log"].append(
                    f"完成 {matrix.name}: iter={result.get('pcg_iterations')}, "
                    f"SpTRSV={result.get('sptrsv_avg_time_ms')} ms, PCG={result.get('pcg_time_ms')} ms"
                )
            else:
                summary["log"].append(f"{matrix.name} 运行失败：{result.get('exception')}")
            _write_json(summary_path, summary)

    final_results = [
        item if item is not None else _failed_matrix_row(MATRICES[idx], "任务未返回结果")
        for idx, item in enumerate(results)
    ]
    success_count = sum(1 for item in final_results if item["status"] == "success")
    summary["status"] = "success" if success_count == len(MATRICES) else "partial_failed"
    summary["message"] = f"实验完成：{success_count}/{len(MATRICES)} 个矩阵运行成功。"
    summary["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary["updated_at"] = summary["finished_at"]
    summary["matrices"] = final_results
    summary["log"].append(summary["message"])
    _write_json(summary_path, summary)
    return summary


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for group in RUN_GROUPS:
        group_id = str(group["value"])
        summary_path = _summary_path(group_id)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            summary = {
                "run_id": group_id,
                "group_id": group_id,
                "group_label": group["label"],
                "status": "not_run",
                "message": "尚未运行。",
                "config": {
                    "route": group["route"],
                    "precision": group["precision"],
                    **FIXED_OPTIONS,
                },
                "matrices": [_initial_matrix_row(item) for item in MATRICES],
                "created_at": None,
                "updated_at": None,
                "finished_at": None,
                "log": [f"{group['label']} 尚未运行。"],
            }
        runs.append(summary)
    return runs


def read_run(run_id: str) -> dict[str, Any] | None:
    summary_path = _summary_path(run_id)
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compare_runs(base_group: str, target_group: str) -> dict[str, Any]:
    base = read_run(base_group)
    target = read_run(target_group)
    if base is None or target is None:
        raise FileNotFoundError("请选择已经运行过的两组实验结果。")

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
