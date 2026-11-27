import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scenetiq.deployment.display import live_display, render_job_bar, render_upload_bar
from scenetiq.runtime_log import JobProgress, UploadProgress, parse_progress

SERVER_APP = "scenetiq.server:build_app"
LOG_CONFIG_PATH = Path(__file__).with_name("logging.json")

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
CLOUDFLARED_PATH = Path("/usr/local/bin/cloudflared")


def server_environment(
    token: str, config: Mapping[str, Any], base: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(
        SCENETIQ_TOKEN=token,
        SCENETIQ_CONFIG=json.dumps(config),
        TQDM_DISABLE="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
    )
    return env


def server_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "--factory",
        SERVER_APP,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-config",
        str(LOG_CONFIG_PATH),
    ]


def find_server_pids(port: int, proc_root: Path = Path("/proc")) -> list[int]:
    if not proc_root.is_dir():
        return []
    pids = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:  # the process exited while we were looking, or is not ours to read
            continue
        if SERVER_APP in args and "--port" in args and str(port) in args:
            pids.append(int(entry.name))
    return sorted(pids)


def stop_server(
    port: int, timeout: float = 15.0, proc_root: Path = Path("/proc"), poll: float = 0.2
) -> list[int]:
    stopped = find_server_pids(port, proc_root)
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        remaining = find_server_pids(port, proc_root)
        for pid in remaining:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
        deadline = time.monotonic() + timeout
        while remaining and time.monotonic() < deadline:
            time.sleep(poll)
            remaining = find_server_pids(port, proc_root)
        if not remaining:
            return stopped
    raise TimeoutError(f"backend on port {port} still running after kill: {remaining}")


class RunningServer:
    def __init__(
        self,
        process: subprocess.Popen[str],
        max_lines: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._process = process
        self._clock = clock
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._line_count = 0
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._activity: JobProgress | UploadProgress | None = None
        self._activity_started = 0.0
        self._activity_ended: float | None = None
        self._readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for raw in self._process.stdout:
            line = raw.rstrip()
            progress = parse_progress(line)
            with self._lock:
                if progress is not None:
                    self._record_progress(progress)
                else:
                    self._lines.append(line)
                    self._line_count += 1

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for raw in self._process.stderr:
            with self._lock:
                self._stderr_tail.append(raw.rstrip())

    def _record_progress(self, progress: JobProgress | UploadProgress) -> None:
        current = self._activity
        if isinstance(progress, UploadProgress):
            same = isinstance(current, UploadProgress) and current.capture_id == progress.capture_id
            # Views arrive in parallel, so their reports can overtake each other.
            if (
                same
                and isinstance(current, UploadProgress)
                and progress.received < current.received
            ):
                return
            ended = progress.received >= progress.total
        else:
            same = isinstance(current, JobProgress) and current.job_id == progress.job_id
            ended = progress.status != "running"
        now = self._clock()
        if not same:
            self._activity_started = now
            self._activity_ended = None
        if ended:
            self._activity_ended = now
        self._activity = progress

    def poll(self) -> int | None:
        return self._process.poll()

    def new_lines(self, cursor: int) -> tuple[list[str], int]:
        with self._lock:
            first_kept = self._line_count - len(self._lines)
            lines = list(self._lines)[max(cursor - first_kept, 0) :]
            return lines, self._line_count

    def status_line(self) -> str:
        with self._lock:
            end = self._clock() if self._activity_ended is None else self._activity_ended
            elapsed = end - self._activity_started
            if isinstance(self._activity, UploadProgress):
                return render_upload_bar(self._activity, elapsed)
            return render_job_bar(self._activity, elapsed)

    def crash_output(self) -> str:
        for reader in self._readers:
            reader.join(timeout=2)
        with self._lock:
            return "\n".join([*list(self._lines)[-10:], *self._stderr_tail])


def start_server(env: dict[str, str], port: int) -> RunningServer:
    stopped = stop_server(port)
    if stopped:
        print(f"stopped previous server {stopped}; a job it was running is now marked failed")
    process = subprocess.Popen(
        server_command(port),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return RunningServer(process)


def watch(
    server: RunningServer,
    show: Callable[[str], None] | None = None,
    poll: float = 1.0,
) -> None:
    draw = show or live_display()
    cursor = 0
    try:
        while True:
            code = server.poll()
            output = server.crash_output() if code is not None else ""
            lines, cursor = server.new_lines(cursor)
            for line in lines:
                print(line, flush=True)
            draw(server.status_line())
            if code is not None:
                raise RuntimeError(f"backend exited with code {code}. Last output:\n{output}")
            time.sleep(poll)
    except KeyboardInterrupt:
        print("stopped watching; the backend is still running. Run this cell again to resume.")


def wait_until_ready(server: RunningServer, port: int, token: str, timeout: float = 120.0) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/scenes", headers={"Authorization": f"Bearer {token}"}
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = server.poll()
        if code is not None:
            output = server.crash_output()
            raise RuntimeError(f"backend exited with code {code}. Last output:\n{output}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"backend on port {port} did not become ready in {timeout:.0f}s")


def install_cloudflared() -> Path:
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
    CLOUDFLARED_PATH.chmod(0o755)
    return CLOUDFLARED_PATH


def tunnel_command(binary: Path, tunnel_token: str) -> list[str]:
    return [str(binary), "tunnel", "--no-autoupdate", "run", "--token", tunnel_token]


def start_named_tunnel(
    tunnel_token: str, binary: Path | None = None, grace: float = 5.0
) -> subprocess.Popen[bytes]:
    command = tunnel_command(binary or install_cloudflared(), tunnel_token)
    tunnel = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        code = tunnel.poll()
        if code is not None:
            raise RuntimeError(
                f"cloudflared exited with code {code}; check CLOUDFLARE_TUNNEL_TOKEN"
            )
        time.sleep(0.1)
    return tunnel
