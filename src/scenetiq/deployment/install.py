import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scenetiq.config import Settings
from scenetiq.deployment.display import elapsed_clock, progress_bar

INSTALL_STAGES = (
    "Install uv",
    "Resolve packages",
    "Download and build",
    "Install packages",
    "Done",
)


def torch_family(lock: str | Path) -> list[str]:
    text = lock.read_text(encoding="utf-8") if isinstance(lock, Path) else lock
    names = (package["name"] for package in tomllib.loads(text).get("package", []))
    return sorted(
        name
        for name in names
        if name in ("torch", "torchvision", "triton") or name.startswith(("nvidia-", "cuda-"))
    )


def held_constraints(version: Callable[[str], str] = importlib.metadata.version) -> str:
    lines = []
    for name in ("torch", "torchvision"):
        try:
            installed = version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        lines.append(f"{name}=={installed.partition('+')[0]}")
    return "".join(f"{line}\n" for line in lines)


def export_command(repo: Path, output: Path, excluded: list[str]) -> list[str]:
    cmd = [
        "uv", "export", "--project", str(repo), "--frozen", "--extra", "gpu",
        "--no-hashes", "--no-emit-project", "--output-file", str(output),
    ]  # fmt: skip
    for name in excluded:
        cmd += ["--no-emit-package", name]
    return cmd


def install_command(
    repo: Path,
    pipeline: str,
    requirements: Path | None = None,
    constraints: Path | None = None,
) -> list[str]:
    cmd = ["uv", "pip", "install", "--system", "-e", str(repo)]
    if pipeline == "real":
        if requirements is None or constraints is None:
            raise ValueError("the real Pipeline needs the exported lock and the held constraints")
        cmd += [
            "--no-build-isolation-package", "sam-2",
            "-r", str(requirements), "-c", str(constraints),
        ]  # fmt: skip
    return cmd


def install_environment(pipeline: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base, UV_NO_PROGRESS="1")
    if pipeline == "real":
        env["SAM2_BUILD_CUDA"] = "0"
    return env


class InstallProgress:
    def __init__(self) -> None:
        self.stage = 0
        self.started: list[str] = []
        self.finished = 0
        self.current = ""

    def begin_resolve(self) -> None:
        self.stage = max(self.stage, 1)

    def feed(self, line: str) -> None:
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
        last = len(INSTALL_STAGES) - 1
        within = 0.0
        if self.stage == 2 and self.started:
            within = self.finished / len(self.started)
        return min(1.0, (self.stage + within) / last)

    def render(self, elapsed: float, width: int = 24) -> str:
        bar = progress_bar(self.fraction(), width)
        step = f"{self.stage + 1}/{len(INSTALL_STAGES)} {INSTALL_STAGES[self.stage]}"
        detail = ""
        if self.stage == 2 and self.started:
            detail = f" · {self.finished}/{len(self.started)} named packages"
        if self.current:
            detail += f" · {self.current}"
        return f"[{bar}] {step}{detail} · {elapsed_clock(elapsed)}"


def _run_uv(
    command: list[str],
    progress: InstallProgress,
    start: float,
    show: Callable[[str], None],
    tail_lines: int,
    env: dict[str, str],
) -> None:
    tail: deque[str] = deque(maxlen=tail_lines)
    proc = subprocess.Popen(
        command,
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


def install_packages(
    repo: Path,
    pipeline: str,
    show: Callable[[str], None] = print,
    tail_lines: int = 40,
) -> None:
    progress = InstallProgress()
    start = time.monotonic()
    show(progress.render(0.0))
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    progress.begin_resolve()
    show(progress.render(time.monotonic() - start))

    env = install_environment(pipeline)
    if pipeline != "real":
        _run_uv(install_command(repo, pipeline), progress, start, show, tail_lines, env)
        return
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / "requirements.txt"
        constraints = Path(scratch) / "held.txt"
        constraints.write_text(held_constraints(), encoding="utf-8")
        excluded = torch_family(repo / "uv.lock")
        _run_uv(
            export_command(repo, requirements, excluded), progress, start, show, tail_lines, env
        )
        command = install_command(repo, pipeline, requirements, constraints)
        _run_uv(command, progress, start, show, tail_lines, env)


def configured_pipeline(config: Mapping[str, Any]) -> str:
    return Settings.from_overrides("", config).pipeline
