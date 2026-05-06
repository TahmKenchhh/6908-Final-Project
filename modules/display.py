"""
Display module — sci-fi avatar in Chromium via WebSocket.

Architecture:
  Python (asyncio WebSocket server on :8765)
      ↕  JSON {"state":…} / {"mouth":…}   server→browser
      ↕  JSON {"cmd":"interrupt"|"vision"}  browser→server
  Chromium (fullscreen, serves web/index.html via local HTTP on :8766)

Extra HTTP endpoints (served on :8766):
  GET /snapshot.jpg  → current camera frame as JPEG
"""

import asyncio
import http.server
import io
import json
import logging
import os
import subprocess
import threading
import time
from enum import Enum


class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"


WS_PORT   = 8765
HTTP_PORT = 8766
WEB_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

logging.getLogger("websockets").setLevel(logging.WARNING)


class Display:
    def __init__(self):
        self._state   = State.IDLE
        self._lock    = threading.Lock()
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Events set by browser button presses
        self.interrupt_event    = threading.Event()
        self.vision_trigger     = threading.Event()
        self.camera_open        = threading.Event()  # set while camera overlay is open

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        t = threading.Thread(target=self._run_servers, daemon=True)
        t.start()
        time.sleep(1.5)
        self._launch_chromium()

    def set_state(self, state: State) -> None:
        with self._lock:
            self._state = state
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(state.value), self._loop)

    def force_camera(self, open_: bool) -> None:
        """Server-side trigger to open/close the camera overlay (for testing)."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_raw(json.dumps({"force_camera": open_})),
                self._loop,
            )

    def send_transcript(self, role: str, text: str) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_raw(json.dumps({"transcript": {"role": role, "text": text}})),
                self._loop,
            )

    def send_lux(self, lux: float) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_raw(f'{{"lux":{lux}}}'),
                self._loop,
            )

    def send_mouth(self, value: float) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_raw(json.dumps({"mouth": round(value, 3)})),
                self._loop,
            )

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def clear_interrupt(self) -> None:
        self.interrupt_event.clear()

    # ── Servers ──────────────────────────────────────────────────────────────

    def _run_servers(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        import websockets

        handler = _make_http_handler(WEB_DIR, self)
        http_server = http.server.HTTPServer(("localhost", HTTP_PORT), handler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()

        async with websockets.serve(self._ws_handler, "localhost", WS_PORT):
            await asyncio.Future()

    async def _ws_handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({"state": self._state.value}))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    cmd = msg.get("cmd")
                    if cmd == "interrupt":
                        self.interrupt_event.set()
                        print("[display] interrupt requested")
                    elif cmd == "vision":
                        self.vision_trigger.set()
                        print("[display] vision trigger requested")
                    elif cmd == "camera_open":
                        self.camera_open.set()
                    elif cmd == "camera_close":
                        self.camera_open.clear()
                except Exception:
                    pass
        finally:
            self._clients.discard(ws)

    async def _broadcast(self, state_value: str) -> None:
        await self._broadcast_raw(json.dumps({"state": state_value}))

    async def _broadcast_raw(self, msg: str) -> None:
        if not self._clients:
            return
        await asyncio.gather(
            *[client.send(msg) for client in self._clients],
            return_exceptions=True,
        )

    # ── Chromium ──────────────────────────────────────────────────────────────

    def _launch_chromium(self) -> None:
        url = f"http://localhost:{HTTP_PORT}/index.html?v={int(time.time())}"
        cmd = [
            "chromium-browser",
            "--noerrdialogs", "--disable-infobars", "--kiosk",
            "--disable-session-crashed-bubble", "--disable-restore-session-state",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-web-security", "--allow-file-access-from-files",
            "--password-store=basic",
            f"--app={url}",
        ]
        import shutil
        for binary in ["chromium-browser", "chromium"]:
            if not shutil.which(binary):
                continue
            cmd[0] = binary
            subprocess.Popen(
                ["nice", "-n", "15"] + cmd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY") or ":0"},
            )
            return


def _make_http_handler(directory: str, display: "Display"):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_GET(self):
            if self.path.split("?")[0] == "/snapshot.jpg":
                self._serve_snapshot()
            else:
                super().do_GET()

        def _serve_snapshot(self):
            try:
                from modules.camera import snapshot_jpeg
                data = snapshot_jpeg()
            except Exception:
                data = None

            if data is None:
                self.send_response(503)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            super().end_headers()

        def log_message(self, *_):
            pass

    return Handler
