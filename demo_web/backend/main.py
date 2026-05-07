from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import FIXED_OPTIONS, FRONTEND_DIR, MATRICES, PRECISIONS, ROUTES, RUN_GROUPS
from runner import create_run, list_runs, read_run
from static_results import compare_version2_runs, list_version2_compare_runs


class RunRequest(BaseModel):
    route: str
    precision: str


class CompareRequest(BaseModel):
    base_group: str
    target_group: str


app = FastAPI(title="IC-PCG SpTRSV Demo System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict[str, object]:
    return {
        "routes": ROUTES,
        "precisions": PRECISIONS,
        "run_groups": RUN_GROUPS,
        "fixed": FIXED_OPTIONS,
        "matrices": [item.__dict__ for item in MATRICES],
    }


@app.get("/api/runs")
def runs() -> dict[str, object]:
    return {"runs": list_runs()}


@app.get("/api/compare-runs")
def compare_runs() -> dict[str, object]:
    return {"runs": list_version2_compare_runs()}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, object]:
    summary = read_run(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return summary


@app.post("/api/runs")
def start_run(request: RunRequest) -> dict[str, object]:
    valid_routes = {item["value"] for item in ROUTES}
    valid_precisions = {item["value"] for item in PRECISIONS}
    if request.route not in valid_routes:
        raise HTTPException(status_code=400, detail="Invalid route")
    if request.precision not in valid_precisions:
        raise HTTPException(status_code=400, detail="Invalid precision")
    return create_run(request.route, request.precision)


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict[str, object]:
    valid_groups = {item["value"] for item in RUN_GROUPS}
    if request.base_group not in valid_groups:
        raise HTTPException(status_code=400, detail="Invalid base group")
    if request.target_group not in valid_groups:
        raise HTTPException(status_code=400, detail="Invalid target group")
    try:
        return compare_version2_runs(request.base_group, request.target_group)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
