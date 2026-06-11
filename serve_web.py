from __future__ import annotations

import argparse
import json
import socket
import time
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from DQN.src.utils.live_feed import current_state, wait_for_update
from DQN.src.utils.paths import workspace_root


SNAKE_GRID_OPTIONS = (16, 32, 64, 128)
MIN_SIMULATION_FPS = 1
MAX_SIMULATION_FPS = 120
DEFAULT_SIMULATION_FPS = 12

_job_lock = threading.Lock()
_job_state: dict[str, Any] = {
    "running": False,
    "kind": None,
    "gridSize": None,
    "startedAt": None,
    "finishedAt": None,
    "status": "idle",
    "message": "",
    "fps": DEFAULT_SIMULATION_FPS,
}


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}
    raw_body = handler.rfile.read(content_length)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _snake_model_status() -> list[dict[str, Any]]:
    dqn_root = workspace_root() / "DQN"
    models = []
    for grid_size in SNAKE_GRID_OPTIONS:
        run_name = f"snake_{grid_size}x{grid_size}"
        checkpoint_dir = dqn_root / "checkpoints" / run_name
        best_eval = checkpoint_dir / "best_eval.pth"
        latest = checkpoint_dir / "latest.pth"
        metadata = _checkpoint_metadata(best_eval if best_eval.exists() else latest)
        models.append(
            {
                "gridSize": grid_size,
                "runName": run_name,
                "bestEvalExists": best_eval.exists(),
                "latestExists": latest.exists(),
                "bestEvalPath": str(best_eval),
                "latestPath": str(latest),
                "episode": metadata.get("episode"),
                "evalAvgScore": metadata.get("eval_avg_score"),
                "evalAvgReward": metadata.get("eval_avg_reward"),
                "evalAvgSteps": metadata.get("eval_avg_steps"),
                "evalMetric": metadata.get("eval_metric"),
                "evalMetricValue": metadata.get("eval_metric_value"),
            }
        )
    return models


def _public_job_state() -> dict[str, Any]:
    with _job_lock:
        return dict(_job_state)


def _set_job_state(**updates: Any) -> None:
    with _job_lock:
        _job_state.update(updates)


def _validate_fps(raw_value: Any) -> int:
    try:
        fps = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a number") from exc
    return max(MIN_SIMULATION_FPS, min(MAX_SIMULATION_FPS, fps))


def _current_job_fps() -> int:
    with _job_lock:
        return _validate_fps(_job_state.get("fps", DEFAULT_SIMULATION_FPS))


def _validate_grid_size(raw_value: Any) -> int:
    try:
        grid_size = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gridSize must be a number") from exc
    if grid_size not in SNAKE_GRID_OPTIONS:
        raise ValueError(f"gridSize must be one of: {', '.join(str(value) for value in SNAKE_GRID_OPTIONS)}")
    return grid_size


def _start_background_job(kind: str, grid_size: int, target, *, args: tuple[Any, ...]) -> bool:
    with _job_lock:
        if _job_state["running"]:
            return False
        _job_state.update(
            {
                "running": True,
                "kind": kind,
                "gridSize": grid_size,
                "startedAt": time.time(),
                "finishedAt": None,
                "status": "running",
                "message": "",
                "fps": _validate_fps(_job_state.get("fps", DEFAULT_SIMULATION_FPS)),
            }
        )

    def _run() -> None:
        try:
            target(*args)
            _set_job_state(running=False, finishedAt=time.time(), status="done", message=f"{kind} klaar")
        except Exception as exc:
            _set_job_state(
                running=False,
                finishedAt=time.time(),
                status="error",
                message=f"{exc.__class__.__name__}: {exc}",
            )

    thread = threading.Thread(target=_run, name=f"{kind}-{grid_size}x{grid_size}", daemon=True)
    thread.start()
    return True


def _run_simulation_job(grid_size: int, episodes: int, max_steps: int, fps: int) -> None:
    from DQN.simulate import run_simulation

    run_simulation(
        game="snake",
        checkpoint="best_eval.pth",
        episodes=episodes,
        loop=False,
        grid_size=grid_size,
        max_steps=max_steps,
        render=False,
        fps=fps,
        fps_provider=_current_job_fps,
        live_feed=True,
        live_every_n_steps=1,
        serve_live=False,
        open_browser=False,
        solver="dqn",
    )


def _run_training_job(grid_size: int, episodes: int, profile: str, device: str) -> None:
    from DQN.train import run_training

    run_training(
        game="snake",
        episodes=episodes,
        resume=True,
        grid_size=grid_size,
        enable_live_feed=True,
        profile=profile,
        device=device,
    )


class LiveFeedHandler(SimpleHTTPRequestHandler):
    project_root = Path(__file__).resolve().parent
    sse_keepalive_seconds = 10.0

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        clean_path = self.path.split("?", 1)[0]
        if clean_path in ("/events", "/web/events"):
            self._stream_events()
            return
        if clean_path in ("/api/status", "/web/api/status"):
            _json_response(self, 200, {"models": _snake_model_status(), "job": _public_job_state()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        clean_path = self.path.split("?", 1)[0]
        if clean_path in ("/api/simulate", "/web/api/simulate"):
            self._start_simulation()
            return
        if clean_path in ("/api/train", "/web/api/train"):
            self._start_training()
            return
        if clean_path in ("/api/speed", "/web/api/speed"):
            self._set_speed()
            return
        _json_response(self, 404, {"error": "Unknown endpoint"})

    def _start_simulation(self) -> None:
        try:
            payload = _read_json_body(self)
            grid_size = _validate_grid_size(payload.get("gridSize"))
            episodes = max(1, min(100, int(payload.get("episodes", 3))))
            max_steps = max(0, min(2_000_000, int(payload.get("maxSteps", 0))))
            fps = _validate_fps(payload.get("fps", DEFAULT_SIMULATION_FPS))
        except (TypeError, ValueError) as exc:
            _json_response(self, 400, {"error": str(exc)})
            return

        _set_job_state(fps=fps)

        best_eval = workspace_root() / "DQN" / "checkpoints" / f"snake_{grid_size}x{grid_size}" / "best_eval.pth"
        if not best_eval.exists():
            _json_response(
                self,
                409,
                {
                    "error": f"best_eval.pth ontbreekt voor {grid_size}x{grid_size}. Train deze grid eerst.",
                    "checkpoint": str(best_eval),
                },
            )
            return

        started = _start_background_job(
            "simulate",
            grid_size,
            _run_simulation_job,
            args=(grid_size, episodes, max_steps, fps),
        )
        if not started:
            _json_response(self, 409, {"error": "Er draait al een training of simulatie.", "job": _public_job_state()})
            return
        _json_response(self, 202, {"ok": True, "job": _public_job_state()})

    def _set_speed(self) -> None:
        try:
            payload = _read_json_body(self)
            fps = _validate_fps(payload.get("fps", DEFAULT_SIMULATION_FPS))
        except (TypeError, ValueError) as exc:
            _json_response(self, 400, {"error": str(exc)})
            return

        _set_job_state(fps=fps)
        _json_response(self, 200, {"ok": True, "fps": fps, "job": _public_job_state()})

    def _start_training(self) -> None:
        try:
            payload = _read_json_body(self)
            grid_size = _validate_grid_size(payload.get("gridSize"))
            episodes = max(1, min(1_000_000, int(payload.get("episodes", 10_000))))
            profile = str(payload.get("profile", "balanced")).lower().strip()
            if profile not in ("fast", "balanced", "quality"):
                raise ValueError("profile must be fast, balanced or quality")
            device = str(payload.get("device", "auto")).lower().strip()
            if device not in ("auto", "cpu", "cuda"):
                raise ValueError("device must be auto, cpu or cuda")
        except (TypeError, ValueError) as exc:
            _json_response(self, 400, {"error": str(exc)})
            return

        started = _start_background_job(
            "train",
            grid_size,
            _run_training_job,
            args=(grid_size, episodes, profile, device),
        )
        if not started:
            _json_response(self, 409, {"error": "Er draait al een training of simulatie.", "job": _public_job_state()})
            return
        _json_response(self, 202, {"ok": True, "job": _public_job_state()})

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_version, payload = current_state()
        if payload is not None:
            data = json.dumps(payload)
            self.wfile.write(f"event: state\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        while True:
            try:
                next_version, payload = wait_for_update(last_version, timeout=self.sse_keepalive_seconds)
                if next_version == last_version:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                last_version = next_version
                if payload is not None:
                    data = json.dumps(payload)
                    self.wfile.write(f"event: state\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError, socket.error):
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve web UI with SSE live feed")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--keepalive-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait before sending an SSE keepalive comment",
    )
    return parser.parse_args()


def create_server(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    handler = lambda *a, **kw: LiveFeedHandler(*a, directory=str(LiveFeedHandler.project_root), **kw)
    return ThreadingHTTPServer((host, port), handler)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = create_server(host=host, port=port)
    print(f"Serving project at http://{host}:{port}/web/")
    print("Open /web/ in your browser")
    print("SSE endpoint available at /events and /web/events")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def start_server_in_thread(host: str = "0.0.0.0", port: int = 8000) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = create_server(host=host, port=port)

    def _serve() -> None:
        print(f"Serving project at http://{host}:{port}/web/")
        print("Open /web/ in your browser")
        print("SSE endpoint available at /events and /web/events")
        server.serve_forever()

    thread = threading.Thread(target=_serve, name="live-feed-server", daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    args = parse_args()
    LiveFeedHandler.sse_keepalive_seconds = max(1.0, float(args.keepalive_seconds))
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
