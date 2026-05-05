from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from config import EXECUTABLE, FIXED_OPTIONS, MATRICES, PROJECT_ROOT, RUNS_DIR, MatrixInfo
from parser import parse_result_file


RUN_TIMEOUT_SECONDS = 30 * 60


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
    raw_dir: Path,
    config_dir: Path,
) -> dict[str, Any]:
    row = _initial_matrix_row(matrix)
    config_path = _create_matrix_config(config_dir, matrix, route, precision)
    raw_path = raw_dir / f"results_{matrix.name}.txt"
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


def create_run(route: str, precision: str) -> dict[str, Any]:
    """Run ict_pcg for all demo matrices and write outputs under demo_web/runs."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    config_dir = run_dir / "configs"
    charts_dir = run_dir / "charts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "status": "running",
        "message": "实验运行中。",
        "config": {
            "route": route,
            "precision": precision,
            **FIXED_OPTIONS,
        },
        "matrices": [_initial_matrix_row(item) for item in MATRICES],
        "created_at": timestamp,
        "finished_at": None,
        "log": [
            f"run_id: {run_id}",
            f"route: {route}",
            f"precision_pcg: {precision}",
            "ordering: RCM",
            "scaling: UnitSqrtDiag",
        ],
    }

    summary_path = run_dir / "summary.json"
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

    results: list[dict[str, Any]] = []
    for matrix in MATRICES:
        summary["log"].append(f"开始运行 {matrix.index}. {matrix.name}")
        _write_json(summary_path, summary)
        result = _run_one_matrix(matrix, route, precision, raw_dir, config_dir)
        results.append(result)
        summary["matrices"] = results + [_initial_matrix_row(item) for item in MATRICES[len(results) :]]
        if result["status"] == "success":
            summary["log"].append(
                f"完成 {matrix.name}: iter={result.get('pcg_iterations')}, "
                f"SpTRSV={result.get('sptrsv_avg_time_ms')} ms, PCG={result.get('pcg_time_ms')} ms"
            )
        else:
            summary["log"].append(f"{matrix.name} 运行失败：{result.get('exception')}")
        _write_json(summary_path, summary)

    success_count = sum(1 for item in results if item["status"] == "success")
    summary["status"] = "success" if success_count == len(MATRICES) else "partial_failed"
    summary["message"] = f"实验完成：{success_count}/{len(MATRICES)} 个矩阵运行成功。"
    summary["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary["matrices"] = results
    summary["log"].append(summary["message"])
    _write_json(summary_path, summary)
    return summary


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return runs
    for summary_path in sorted(RUNS_DIR.glob("run_*/summary.json"), reverse=True):
        try:
            runs.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return runs


def read_run(run_id: str) -> dict[str, Any] | None:
    summary_path = RUNS_DIR / run_id / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))
