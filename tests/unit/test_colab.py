import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from uvicorn.logging import AccessFormatter

from scenetiq import colab
from scenetiq.colab import InstallProgress, install_command, server_command, server_environment


def test_server_environment_sets_backend_config_and_keeps_base():
    env = server_environment("tok", Path("/content/scenetiq-data"), "real", base={"PATH": "/bin"})
    assert env == {
        "PATH": "/bin",
        "SCENETIQ_TOKEN": "tok",
        "SCENETIQ_STORAGE_ROOT": str(Path("/content/scenetiq-data")),
        "SCENETIQ_PIPELINE": "real",
    }


def test_server_command_runs_the_app_factory_on_localhost():
    cmd = server_command(8123)
    assert cmd[1:] == [
        "-m", "uvicorn", "--factory", "scenetiq.server:build_app",
        "--host", "127.0.0.1", "--port", "8123",
        "--log-config", str(colab.LOG_CONFIG_PATH),
    ]  # fmt: skip


def test_log_config_puts_a_timestamp_on_access_lines():
    config = json.loads(colab.LOG_CONFIG_PATH.read_text(encoding="utf-8"))
    access = config["formatters"]["access"]
    formatter = AccessFormatter(fmt=access["fmt"], datefmt=access["datefmt"], use_colors=False)
    formatter.converter = time.gmtime
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:0", "GET", "/health", "1.1", 200), None,
    )  # fmt: skip
    record.created = datetime(2026, 11, 4, 7, 5, 9, tzinfo=UTC).timestamp()
    assert formatter.format(record) == (
        '2026-11-04 07:05:09 INFO:     1.2.3.4:0 - "GET /health HTTP/1.1" 200 OK'
    )


def _fake_process(proc_root: Path, pid: int, *args: str) -> Path:
    entry = proc_root / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes("\0".join(args).encode() + b"\0")
    return entry


def _fake_server(proc_root: Path, pid: int, port: int) -> Path:
    return _fake_process(
        proc_root, pid, "python", "-m", "uvicorn", "--factory", "scenetiq.server:build_app",
        "--host", "127.0.0.1", "--port", str(port),
    )  # fmt: skip


def test_find_server_pids_matches_only_the_backend_on_that_port(tmp_path):
    _fake_server(tmp_path, 900, 8001)
    _fake_server(tmp_path, 44, 8001)
    _fake_server(tmp_path, 45, 9000)
    _fake_process(tmp_path, 46, "python", "-m", "uvicorn", "other:app", "--port", "8001")
    (_fake_process(tmp_path, 47) / "cmdline").write_bytes(b"")  # exited, not yet reaped
    (tmp_path / "self").mkdir()

    assert colab.find_server_pids(8001, tmp_path) == [44, 900]


def test_find_server_pids_is_empty_without_a_process_table(tmp_path):
    assert colab.find_server_pids(8001, tmp_path / "missing") == []


def test_stop_server_returns_once_the_process_exits_on_the_first_signal(monkeypatch, tmp_path):
    entry = _fake_server(tmp_path, 44, 8001)
    signalled: list[int] = []

    def kill(pid: int, _sig: int) -> None:
        signalled.append(pid)
        shutil.rmtree(entry)

    monkeypatch.setattr(colab.os, "kill", kill)
    assert colab.stop_server(8001, timeout=5, proc_root=tmp_path, poll=0.01) == [44]
    assert signalled == [44]


def test_stop_server_kills_a_process_that_ignores_the_first_signal(monkeypatch, tmp_path):
    entry = _fake_server(tmp_path, 44, 8001)
    signalled: list[int] = []

    def kill(pid: int, _sig: int) -> None:
        signalled.append(pid)
        if len(signalled) == 2:
            shutil.rmtree(entry)

    monkeypatch.setattr(colab.os, "kill", kill)
    assert colab.stop_server(8001, timeout=0.05, proc_root=tmp_path, poll=0.01) == [44]
    assert signalled == [44, 44]


def test_stop_server_raises_when_a_process_survives_the_kill(monkeypatch, tmp_path):
    _fake_server(tmp_path, 44, 8001)
    monkeypatch.setattr(colab.os, "kill", lambda *_: None)
    with pytest.raises(TimeoutError, match=r"\[44\]"):
        colab.stop_server(8001, timeout=0.02, proc_root=tmp_path, poll=0.01)


def test_stop_server_does_nothing_when_no_server_runs(tmp_path):
    assert colab.stop_server(8001, proc_root=tmp_path) == []


def test_start_server_stops_the_old_one_and_marks_the_new_start_in_the_log(monkeypatch, tmp_path):
    stopped_ports: list[int] = []
    monkeypatch.setattr(colab, "stop_server", lambda port: stopped_ports.append(port) or [])
    monkeypatch.setattr(colab, "server_command", lambda _: [sys.executable, "-c", "print('hello')"])
    log = tmp_path / "logs" / "server.log"
    log.parent.mkdir()
    log.write_text("earlier run\n", encoding="utf-8")

    server = colab.start_server(dict(os.environ), 8001, log)
    assert server.wait(timeout=30) == 0

    assert stopped_ports == [8001]
    lines = log.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "earlier run"
    assert re.fullmatch(
        r"=== server start \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC, port 8001 ===", lines[2]
    )
    assert lines[3].strip() == "hello"


class _AlwaysOk(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def other_server() -> Iterator[int]:
    """An unrelated server that answers every request, as an old backend would."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _AlwaysOk)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    thread.join()


def test_wait_until_ready_fails_when_our_server_died_though_the_port_answers(other_server):
    server = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    server.wait(timeout=30)
    with pytest.raises(RuntimeError, match="code 3"):
        colab.wait_until_ready(server, other_server, "tok", timeout=5)


def test_wait_until_ready_returns_when_our_server_runs_and_answers(other_server):
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        colab.wait_until_ready(server, other_server, "tok", timeout=5)
    finally:
        server.kill()
        server.wait(timeout=30)


def test_install_command_adds_gpu_requirements_only_for_real_pipeline():
    repo = Path("/content/scenetiq")
    assert install_command(repo, "fake") == [
        "uv", "pip", "install", "--system", "-e", str(repo),
    ]  # fmt: skip
    assert install_command(repo, "real")[-2:] == ["-r", str(repo / "requirements-gpu.txt")]


def test_install_progress_counts_named_packages_while_preparing():
    progress = InstallProgress()
    progress.begin_resolve()
    for line in [
        "Resolved 6 packages in 451ms",
        "Downloading numpy (12.1MiB)",
        " Downloaded numpy",
        "Building sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f",
    ]:
        progress.feed(line)
    # Stage 2 of 0..4, with 1 of 2 named packages done: (2 + 0.5) / 4.
    assert progress.fraction() == 0.625
    assert progress.render(65.0, width=8) == (
        "[█████░░░] 3/5 Download and build · 1/2 named packages · building sam-2 · 1:05"
    )


def test_install_progress_reaches_done_after_installed_or_audited():
    for last in ["Installed 6 packages in 467ms", "Audited 3 packages in 12ms"]:
        progress = InstallProgress()
        progress.feed("Resolved 6 packages in 451ms")
        progress.feed("Prepared 6 packages in 3.09s")
        assert progress.fraction() == 0.75
        progress.feed(last)
        assert progress.render(5.0, width=4) == "[████] 5/5 Done · 0:05"


def test_install_packages_raises_with_uv_output_tail_and_logs_it(monkeypatch, tmp_path):
    script = "print('Resolved 1 package in 1ms'); print('error: boom'); raise SystemExit(1)"
    monkeypatch.setattr(colab.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(colab, "install_command", lambda *_: [sys.executable, "-c", script])
    shown: list[str] = []
    log = tmp_path / "logs" / "install.log"

    with pytest.raises(RuntimeError, match="error: boom"):
        colab.install_packages(Path("/repo"), "real", log, show=shown.append)

    assert "error: boom" in log.read_text(encoding="utf-8")
    assert shown[0].startswith("[░")
    assert "1/5 Install uv" in shown[0]
    assert "3/5 Download and build" in shown[-1]
