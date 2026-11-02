"""Helpers the Colab bootstrap notebook calls to run the backend behind a tunnel.

The notebook only wires these together; it holds no logic of its own (ADR 0004).
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
CLOUDFLARED_PATH = Path("/usr/local/bin/cloudflared")


def server_environment(
    token: str, storage_root: Path, pipeline: str, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment for the backend process.

    Args:
        token: Shared bearer token.
        storage_root: Where Scenes persist (a Google Drive folder on Colab).
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
    """Return the command that starts the backend on ``port``."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "--factory",
        "scenetiq.server:build_app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def start_server(env: dict[str, str], port: int, log_path: Path) -> subprocess.Popen[bytes]:
    """Start the backend in the background, logging to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    return subprocess.Popen(server_command(port), env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_until_ready(port: int, token: str, timeout: float = 120.0) -> None:
    """Block until the backend answers an authenticated request.

    Raises:
        TimeoutError: If it does not answer within ``timeout`` seconds.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/scenes", headers={"Authorization": f"Bearer {token}"}
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
