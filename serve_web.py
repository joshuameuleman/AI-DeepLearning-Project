from __future__ import annotations

import argparse
import json
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from DQN.src.utils.live_feed import current_state, wait_for_update


class LiveFeedHandler(SimpleHTTPRequestHandler):
    project_root = Path(__file__).resolve().parent
    sse_keepalive_seconds = 10.0

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/events", "/web/events"):
            self._stream_events()
            return
        super().do_GET()

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
    print(f"Serving project at http://{host}:{port}")
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
        print(f"Serving project at http://{host}:{port}")
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
