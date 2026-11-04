"""Helpers the Colab bootstrap notebook calls to run the backend behind a tunnel.

The notebook only wires these together; it holds no logic of its own (ADR 0004).
"""

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

from scenetiq.config import Settings
from scenetiq.runtime_log import JobProgress, UploadProgress, parse_progress

SERVER_APP = "scenetiq.server:build_app"
LOG_CONFIG_PATH = Path(__file__).with_name("colab_logging.json")

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
CLOUDFLARED_PATH = Path("/usr/local/bin/cloudflared")

INSTALL_STAGES = (
    "Install uv",
    "Resolve packages",
    "Download and build",
    "Install packages",
    "Done",
)
"""Stages of :func:`install_packages`, in order, as shown on the progress bar."""


def _bar(fraction: float, width: int) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _clock(elapsed: float) -> str:
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes:d}:{seconds:02d}"


def install_command(repo: Path, pipeline: str) -> list[str]:
    """Return the uv command that installs SceneTIQ (and the GPU stack for ``"real"``).

    It installs into Colab's own Python (``--system``) so its CUDA torch is kept.
    """
    cmd = ["uv", "pip", "install", "--system", "-e", str(repo)]
    if pipeline == "real":
        cmd += ["-r", str(repo / "requirements-gpu.txt")]
    return cmd


class InstallProgress:
    """Tracks a uv install from the lines it prints and renders a progress bar.

    uv prints ``Resolved``, then ``Downloading``/``Building`` and their
    ``Downloaded``/``Built`` counterparts, then ``Prepared`` and ``Installed``
    (or only ``Audited`` when nothing changes). Small packages are downloaded
    silently, so the per-package counts cover only the ones uv names.
    """

    def __init__(self) -> None:
        """Start at the first stage, with nothing prepared."""
        self.stage = 0
        self.started: list[str] = []
        self.finished = 0
        self.current = ""

    def begin_resolve(self) -> None:
        """Mark uv as available, so resolving is under way."""
        self.stage = max(self.stage, 1)

    def feed(self, line: str) -> None:
        """Update the state from one line of uv output."""
        text = line.strip()
        word, _, rest = text.partition(" ")
        name = rest.split(" ")[0] if rest else ""
        if word == "Resolved":
            self.stage = max(self.stage, 2)
        elif word in ("Downloading", "Building"):
            self.stage = max(self.stage, 2)
            self.started.append(name)
            self.current = f"{word.lower()} {name}"
        elif word in ("Downloaded", "Built"):
            self.finished += 1
            self.current = ""
        elif word == "Prepared":
            self.stage = max(self.stage, 3)
            self.current = ""
        elif word in ("Installed", "Audited"):
            self.stage = len(INSTALL_STAGES) - 1
            self.current = ""

    def fraction(self) -> float:
        """Return overall completion in ``[0, 1]``, counting progress inside a stage."""
        last = len(INSTALL_STAGES) - 1
        within = 0.0
        if self.stage == 2 and self.started:
            within = self.finished / len(self.started)
        return min(1.0, (self.stage + within) / last)

    def render(self, elapsed: float, width: int = 24) -> str:
        """Return the progress bar and the current stage as one line of text."""
        bar = _bar(self.fraction(), width)
        step = f"{self.stage + 1}/{len(INSTALL_STAGES)} {INSTALL_STAGES[self.stage]}"
        detail = ""
        if self.stage == 2 and self.started:
            detail = f" · {self.finished}/{len(self.started)} named packages"
        if self.current:
            detail += f" · {self.current}"
        return f"[{bar}] {step}{detail} · {_clock(elapsed)}"


def live_display() -> Callable[[str], None]:
    """Return a function that redraws one text line in the notebook output.

    Falls back to ``print`` outside IPython.
    """
    try:
        from IPython.display import Pretty, display
    except ImportError:
        return print
    handle = display(Pretty(""), display_id=True)
    if handle is None:
        return print
    return lambda text: handle.update(Pretty(text))


def install_packages(
    repo: Path,
    pipeline: str,
    show: Callable[[str], None] = print,
    tail_lines: int = 40,
) -> None:
    """Install SceneTIQ with uv, reporting progress through ``show``.

    Raises:
        RuntimeError: If uv fails; the message holds the last ``tail_lines`` of its output.
    """
    progress = InstallProgress()
    start = time.monotonic()
    show(progress.render(0.0))
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    progress.begin_resolve()
    show(progress.render(time.monotonic() - start))

    tail: deque[str] = deque(maxlen=tail_lines)
    env = dict(os.environ, UV_NO_PROGRESS="1")
    proc = subprocess.Popen(
        install_command(repo, pipeline),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line.rstrip())
        progress.feed(line)
        show(progress.render(time.monotonic() - start))
    code = proc.wait()
    if code != 0:
        last = "\n".join(tail)
        raise RuntimeError(f"uv exited with {code}. Last lines:\n{last}")


def configured_pipeline(config: Mapping[str, Any]) -> str:
    """Return the Pipeline the notebook's ``CONFIG`` selects, checking all of ``CONFIG``.

    Raises:
        ValueError: If ``CONFIG`` is invalid, so a mistake shows before anything is installed.
    """
    return Settings.from_overrides("", config).pipeline


def server_environment(
    token: str, config: Mapping[str, Any], base: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment for the backend process, with model progress bars turned off.

    Args:
        token: Shared bearer token.
        config: The notebook's ``CONFIG``: overrides of the defaults in ``scenetiq.config``.
        base: Environment to extend; defaults to the current process environment.
    """
    env = dict(os.environ if base is None else base)
    env.update(
        SCENETIQ_TOKEN=token,
        SCENETIQ_CONFIG=json.dumps(config),
        TQDM_DISABLE="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
    )
    return env


def server_command(port: int) -> list[str]:
    """Return the command that starts the backend on ``port``, under ``colab_logging.json``."""
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
    """Return the ids of running backend processes started for ``port``, sorted.

    Matches on the command line, so it also finds a server left behind by a restarted
    notebook kernel. An exited process that has not been reaped has an empty command
    line and is not reported. Returns nothing where ``proc_root`` does not exist.

    Args:
        port: The port the backend was started on.
        proc_root: The Linux process table.
    """
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
    """Stop every backend process started for ``port`` and wait until they have gone.

    Each one gets ``timeout`` seconds to shut down, then is killed; a job it was running
    stops with it, and the next server to start marks that job failed.

    Args:
        port: The port the backend was started on.
        timeout: Seconds to wait after each signal.
        proc_root: The Linux process table.
        poll: Seconds between checks.

    Returns:
        The ids of the processes that were stopped.

    Raises:
        TimeoutError: If a process is still there after being killed.
    """
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


def render_job_bar(progress: JobProgress | None, elapsed: float, width: int = 24) -> str:
    """Return one line showing how far a job has come.

    The bar counts finished Stages, so it fills up only when a job succeeds.

    Args:
        progress: The latest progress of the latest job, or ``None`` if none has started.
        elapsed: Seconds the job has been running, or ran for.
        width: Length of the bar in characters.
    """
    if progress is None:
        return f"[{_bar(0.0, width)}] no job yet"
    if progress.status == "succeeded":
        fraction = 1.0
        state = "succeeded"
    else:
        fraction = max(progress.step - 1, 0) / progress.total
        if progress.status == "failed":
            state = f"failed at {progress.stage}" if progress.stage else "failed"
        else:
            state = f"{progress.step}/{progress.total} {progress.stage}"
    return f"[{_bar(fraction, width)}] {state} · scene {progress.scene_id[:8]} · {_clock(elapsed)}"


def render_upload_bar(progress: UploadProgress, elapsed: float, width: int = 24) -> str:
    """Return one line showing how many Views of a Capture have arrived.

    Args:
        progress: The latest progress of the upload.
        elapsed: Seconds since its first View arrived, until its last one did.
        width: Length of the bar in characters.
    """
    bar = _bar(progress.received / progress.total, width)
    count = f"upload {progress.received}/{progress.total} Views"
    return f"[{bar}] {count} · capture {progress.capture_id[:8]} · {_clock(elapsed)}"


class RunningServer:
    """The backend process, with its output read continuously so that it never blocks on a pipe.

    Stdout holds what ``colab_logging.json`` lets through: request lines, errors and upload
    and job progress. Stderr holds everything else (libraries, warnings) and is kept only to explain
    a crash.

    Args:
        process: The backend, started with text-mode pipes for stdout and stderr.
        max_lines: How many log lines to keep; older ones are dropped.
        clock: Source of the time used to measure a job.
    """

    def __init__(
        self,
        process: subprocess.Popen[str],
        max_lines: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start reading the process's output in background threads."""
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
        """Return the process's exit code, or ``None`` while it runs."""
        return self._process.poll()

    def new_lines(self, cursor: int) -> tuple[list[str], int]:
        """Return the log lines after ``cursor`` that are still kept, and the next cursor.

        Args:
            cursor: 0 for every kept line, else the cursor from the previous call.
        """
        with self._lock:
            first_kept = self._line_count - len(self._lines)
            lines = list(self._lines)[max(cursor - first_kept, 0) :]
            return lines, self._line_count

    def status_line(self) -> str:
        """Return the progress bar of the latest upload or job, whichever reported last."""
        with self._lock:
            end = self._clock() if self._activity_ended is None else self._activity_ended
            elapsed = end - self._activity_started
            if isinstance(self._activity, UploadProgress):
                return render_upload_bar(self._activity, elapsed)
            return render_job_bar(self._activity, elapsed)

    def crash_output(self) -> str:
        """Return the output that explains why the process exited, once it has."""
        for reader in self._readers:
            reader.join(timeout=2)
        with self._lock:
            return "\n".join([*list(self._lines)[-10:], *self._stderr_tail])


def start_server(env: dict[str, str], port: int) -> RunningServer:
    """Replace any backend running on ``port`` with a new one."""
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
    """Show the progress bar and print the log until the server exits or the cell is stopped.

    Stopping leaves the server running; calling this again prints the kept lines once more.

    Args:
        server: The process from :func:`start_server`.
        show: Redraws the progress bar in place; defaults to :func:`live_display`.
        poll: Seconds between refreshes.

    Raises:
        RuntimeError: If the server exits.
    """
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
    """Block until the backend answers an authenticated request.

    Args:
        server: The process from :func:`start_server`; if it exits, an answer on ``port``
            could only come from some other process.
        port: The port the backend listens on.
        token: Shared bearer token.
        timeout: Seconds to wait.

    Raises:
        RuntimeError: If ``server`` exits first.
        TimeoutError: If it does not answer within ``timeout`` seconds.
    """
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
    """Download the cloudflared binary unless it is already installed; return its path."""
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
    CLOUDFLARED_PATH.chmod(0o755)
    return CLOUDFLARED_PATH


def tunnel_command(binary: Path, tunnel_token: str) -> list[str]:
    """Return the command that runs the named tunnel with ``tunnel_token``."""
    return [str(binary), "tunnel", "--no-autoupdate", "run", "--token", tunnel_token]


def start_named_tunnel(
    tunnel_token: str, binary: Path | None = None, grace: float = 5.0
) -> subprocess.Popen[bytes]:
    """Run the Cloudflare named tunnel whose ingress is configured in the dashboard.

    The tunnel must route its public hostname to ``http://localhost:<port>``. Its output is
    discarded, so only an early exit is noticed here; a later failure shows as an offline
    backend in the frontend.

    Args:
        tunnel_token: The tunnel's token.
        binary: The cloudflared executable; downloaded if not given.
        grace: Seconds to watch for an early exit.

    Raises:
        RuntimeError: If cloudflared exits within ``grace`` seconds, as it does on a bad token.
    """
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
