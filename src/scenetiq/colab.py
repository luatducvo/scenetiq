"""Helpers the Colab bootstrap notebook calls to run the backend behind a tunnel.

The notebook only wires these together; it holds no logic of its own (ADR 0004).
"""

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

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
        filled = round(self.fraction() * width)
        bar = "█" * filled + "░" * (width - filled)
        step = f"{self.stage + 1}/{len(INSTALL_STAGES)} {INSTALL_STAGES[self.stage]}"
        detail = ""
        if self.stage == 2 and self.started:
            detail = f" · {self.finished}/{len(self.started)} named packages"
        if self.current:
            detail += f" · {self.current}"
        minutes, seconds = divmod(int(elapsed), 60)
        return f"[{bar}] {step}{detail} · {minutes:d}:{seconds:02d}"


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
    log_path: Path,
    show: Callable[[str], None] = print,
    tail_lines: int = 40,
) -> None:
    """Install SceneTIQ with uv, reporting progress through ``show``.

    The full uv output goes to ``log_path``.

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

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=tail_lines)
    env = dict(os.environ, UV_NO_PROGRESS="1")
    with log_path.open("w", encoding="utf-8") as log:
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
            log.write(line)
            tail.append(line.rstrip())
            progress.feed(line)
            show(progress.render(time.monotonic() - start))
        code = proc.wait()
    if code != 0:
        last = "\n".join(tail)
        raise RuntimeError(f"uv exited with {code}; see {log_path}. Last lines:\n{last}")


def server_environment(
    token: str, storage_root: Path, pipeline: str, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment for the backend process.

    Args:
        token: Shared bearer token.
        storage_root: Where Scenes are stored (the runtime's local disk on Colab).
        pipeline: ``"fake"`` or ``"real"``.
        base: Environment to extend; defaults to the current process environment.
    """
    env = dict(os.environ if base is None else base)
    env.update(
        SCENETIQ_TOKEN=token,
        SCENETIQ_STORAGE_ROOT=str(storage_root),
        SCENETIQ_PIPELINE=pipeline,
    )
    return env


def server_command(port: int) -> list[str]:
    """Return the command that starts the backend on ``port``, with timestamped logs."""
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


def start_server(env: dict[str, str], port: int, log_path: Path) -> subprocess.Popen[bytes]:
    """Replace any backend running on ``port`` with a new one, logging to ``log_path``.

    The log lasts across runtimes, so each start appends a line that marks where it begins.
    """
    stopped = stop_server(port)
    if stopped:
        print(f"stopped previous server {stopped}; a job it was running is now marked failed")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("ab") as log:
        log.write(f"\n=== server start {started_at} UTC, port {port} ===\n".encode())
        log.flush()
        return subprocess.Popen(server_command(port), env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_until_ready(
    server: subprocess.Popen[bytes], port: int, token: str, timeout: float = 120.0
) -> None:
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
            raise RuntimeError(f"backend exited with code {code}; see its log")
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


def start_named_tunnel(
    tunnel_token: str, log_path: Path, binary: Path | None = None
) -> subprocess.Popen[bytes]:
    """Run the Cloudflare named tunnel whose ingress is configured in the dashboard.

    The tunnel must route its public hostname to ``http://localhost:<port>``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    exe = str(binary or install_cloudflared())
    return subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "run", "--token", tunnel_token],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
