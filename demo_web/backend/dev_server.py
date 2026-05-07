from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config import FIXED_OPTIONS, FRONTEND_DIR, MATRICES, PRECISIONS, ROUTES, RUN_GROUPS
from runner import create_run, list_runs, read_run
from static_results import compare_version2_runs, list_version2_compare_runs


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if path == "/api/options":
            self._send_json(
                {
                    "routes": ROUTES,
                    "precisions": PRECISIONS,
                    "run_groups": RUN_GROUPS,
                    "fixed": FIXED_OPTIONS,
                    "matrices": [item.__dict__ for item in MATRICES],
                }
            )
            return
        if path == "/api/runs":
            self._send_json({"runs": list_runs()})
            return
        if path == "/api/compare-runs":
            self._send_json({"runs": list_version2_compare_runs()})
            return
        if path.startswith("/api/runs/"):
            run_id = Path(path).name
            summary = read_run(run_id)
            if summary is None:
                self._send_json({"detail": "Run not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(summary)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/runs", "/api/compare"}:
            self._send_json({"detail": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            request = self._read_json()
            if path == "/api/compare":
                base_group = str(request.get("base_group", ""))
                target_group = str(request.get("target_group", ""))
                valid_groups = {item["value"] for item in RUN_GROUPS}
                if base_group not in valid_groups or target_group not in valid_groups:
                    self._send_json({"detail": "Invalid request"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(compare_version2_runs(base_group, target_group))
                return

            route = str(request.get("route", ""))
            precision = str(request.get("precision", ""))
            valid_routes = {item["value"] for item in ROUTES}
            valid_precisions = {item["value"] for item in PRECISIONS}
            if route not in valid_routes or precision not in valid_precisions:
                self._send_json({"detail": "Invalid request"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(create_run(route, precision))
        except Exception as exc:  # noqa: BLE001 - dev server should surface errors as JSON
            self._send_json({"detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("Demo system running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
