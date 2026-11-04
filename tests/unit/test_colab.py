import io
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from uvicorn.logging import AccessFormatter

from scenetiq import colab
from scenetiq.colab import (
    InstallProgress,
    RunningServer,
    install_command,
    render_job_bar,
    render_upload_bar,
    server_command,
    server_environment,
)
from scenetiq.config import Settings
from scenetiq.runtime_log import JobProgress, UploadProgress, format_progress


def test_server_environment_sets_backend_config_and_keeps_base():
    env = server_environment(
        "tok", {"view_stride": 3, "da3": {"process_res": 392}}, base={"PATH": "/bin"}
    )
    assert env == {
        "PATH": "/bin",
        "SCENETIQ_TOKEN": "tok",
        "SCENETIQ_CONFIG": '{"view_stride": 3, "da3": {"process_res": 392}}',
        "TQDM_DISABLE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }


def test_backend_reads_the_config_cell_it_is_given(monkeypatch: pytest.MonkeyPatch):
    env = server_environment("tok", {"view_stride": 3, "da3": {"process_res": 392}}, base={})
    monkeypatch.setenv("SCENETIQ_TOKEN", env["SCENETIQ_TOKEN"])
    monkeypatch.setenv("SCENETIQ_CONFIG", env["SCENETIQ_CONFIG"])
    settings = Settings.from_env()
    assert settings.view_stride == 3
    assert settings.da3.process_res == 392


@pytest.mark.parametrize(("config", "expected"), [({"pipeline": "real"}, "real"), ({}, "fake")])
def test_configured_pipeline_follows_the_config_cell(config: dict, expected: str):
    assert colab.configured_pipeline(config) == expected


def test_mistake_in_the_config_cell_is_found_before_installing_anything():
    with pytest.raises(ValueError, match="view_stide"):
        colab.configured_pipeline({"view_stide": 7})


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


def _script_server(script: str, **kwargs: Any) -> RunningServer:
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return RunningServer(process, **kwargs)


def _progress_line(job_id: str, status: str, stage: str, step: int) -> str:
    return format_progress(JobProgress(job_id, "a1b2c3d4e5f6", status, stage, step, total=6))


def test_start_server_stops_the_old_one_and_keeps_only_stderr_out_of_the_log(monkeypatch):
    stopped_ports: list[int] = []
    script = "import sys; print('hello'); print('library noise', file=sys.stderr)"
    monkeypatch.setattr(colab, "stop_server", lambda port: stopped_ports.append(port) or [])
    monkeypatch.setattr(colab, "server_command", lambda _: [sys.executable, "-c", script])

    server = colab.start_server(dict(os.environ), 8001)

    assert server.crash_output() == "hello\nlibrary noise"  # waits for the output to be read
    assert stopped_ports == [8001]
    assert server.new_lines(0) == (["hello"], 1)


def test_new_lines_returns_only_what_is_new_and_drops_the_oldest_past_the_limit():
    server = _script_server("for i in range(5): print(i)", max_lines=3)
    server.crash_output()

    assert server.new_lines(0) == (["2", "3", "4"], 5)
    assert server.new_lines(4) == (["4"], 5)
    assert server.new_lines(5) == ([], 5)


def test_a_full_pipe_never_blocks_the_server_while_nobody_watches():
    # 200 kB is more than an OS pipe buffers, on both stdout and stderr.
    script = "import sys\nfor i in range(20000): print('x' * 9); print('y' * 9, file=sys.stderr)"
    server = _script_server(script)
    server.crash_output()
    deadline = time.monotonic() + 30
    while server.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.poll() == 0


def test_status_line_measures_a_job_and_stops_the_clock_when_it_ends():
    now = [100.0]
    script = (
        "import sys\n"
        f"print({_progress_line('j1', 'running', 'superpoints', 3)!r}, flush=True)\n"
        "sys.stdin.readline()\n"
        f"print({_progress_line('j1', 'succeeded', 'embedding', 6)!r}, flush=True)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    server = RunningServer(process, clock=lambda: now[0])
    assert server.status_line().endswith("no job yet")

    deadline = time.monotonic() + 30
    while "superpoints" not in server.status_line() and time.monotonic() < deadline:
        time.sleep(0.01)
    now[0] = 234.0
    assert server.status_line().endswith("3/6 superpoints · scene a1b2c3d4 · 2:14")

    assert process.stdin is not None
    process.stdin.write("\n")
    process.stdin.close()
    server.crash_output()
    now[0] = 999.0
    assert server.status_line().endswith("succeeded · scene a1b2c3d4 · 2:14")
    assert server.new_lines(0) == ([], 0)  # progress is not a log line


def test_status_line_follows_the_latest_job():
    script = (
        f"print({_progress_line('j1', 'failed', 'masking', 4)!r})\n"
        f"print({_progress_line('j2', 'running', 'fusing', 2)!r})\n"
    )
    server = _script_server(script, clock=lambda: 10.0)
    server.crash_output()
    assert server.status_line().endswith("2/6 fusing · scene a1b2c3d4 · 0:00")


def test_log_config_shows_errors_with_their_traceback_and_nothing_quieter():
    script = (
        "import json, logging, logging.config\n"
        f"logging.config.dictConfig(json.load(open({str(colab.LOG_CONFIG_PATH)!r})))\n"
        "log = logging.getLogger('scenetiq.jobs')\n"
        "log.warning('quiet warning')\n"
        "try:\n"
        "    raise ValueError('kaput')\n"
        "except ValueError:\n"
        "    log.exception('job j1 failed')\n"
    )
    server = _script_server(script)
    server.crash_output()
    output = "\n".join(server.new_lines(0)[0])
    assert "job j1 failed" in output
    assert "ValueError: kaput" in output
    assert "quiet warning" not in output


@pytest.mark.parametrize(
    ("status", "stage", "step", "expected"),
    [
        ("running", "reconstructing", 1, "[░░░░░░] 1/6 reconstructing · scene a1b2c3d4 · 1:05"),
        ("running", "superpoints", 3, "[██░░░░] 3/6 superpoints · scene a1b2c3d4 · 1:05"),
        ("succeeded", "embedding", 6, "[██████] succeeded · scene a1b2c3d4 · 1:05"),
        ("failed", "masking", 4, "[███░░░] failed at masking · scene a1b2c3d4 · 1:05"),
        ("failed", "", 0, "[░░░░░░] failed · scene a1b2c3d4 · 1:05"),
    ],
)
def test_render_job_bar_counts_finished_stages(status, stage, step, expected):
    progress = JobProgress("j1", "a1b2c3d4e5f6", status, stage, step, total=6)
    assert render_job_bar(progress, 65.0, width=6) == expected


def test_render_job_bar_before_any_job():
    assert render_job_bar(None, 0.0, width=6) == "[░░░░░░] no job yet"


@pytest.mark.parametrize(
    ("received", "expected"),
    [
        (1, "[░░░░░░] upload 1/12 Views · capture c0ffee12 · 0:42"),
        (6, "[███░░░] upload 6/12 Views · capture c0ffee12 · 0:42"),
        (12, "[██████] upload 12/12 Views · capture c0ffee12 · 0:42"),
    ],
)
def test_render_upload_bar_counts_received_views(received, expected):
    progress = UploadProgress("c0ffee1234", received, 12)
    assert render_upload_bar(progress, 42.0, width=6) == expected


def _upload_line(capture_id: str, received: int, total: int = 10) -> str:
    return format_progress(UploadProgress(capture_id, received, total))


def test_status_line_ignores_an_upload_report_that_was_overtaken():
    script = f"print({_upload_line('c1', 5)!r})\nprint({_upload_line('c1', 4)!r})\n"
    server = _script_server(script, clock=lambda: 0.0)
    server.crash_output()
    assert "upload 5/10 Views" in server.status_line()
    assert server.new_lines(0) == ([], 0)  # progress is not a log line


def test_status_line_stops_the_upload_clock_at_the_last_view_and_moves_on_to_the_job():
    now = [100.0]
    script = (
        "import sys\n"
        f"print({_upload_line('c1', 1, 2)!r}, flush=True)\n"
        "sys.stdin.readline()\n"
        f"print({_upload_line('c1', 2, 2)!r}, flush=True)\n"
        "sys.stdin.readline()\n"
        f"print({_progress_line('j1', 'running', 'fusing', 2)!r}, flush=True)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    server = RunningServer(process, clock=lambda: now[0])
    assert process.stdin is not None

    def wait_for(text: str) -> str:
        deadline = time.monotonic() + 30
        while text not in server.status_line() and time.monotonic() < deadline:
            time.sleep(0.01)
        return server.status_line()

    assert "upload 1/2 Views" in wait_for("upload 1/2")

    now[0] = 130.0
    process.stdin.write("\n")
    process.stdin.flush()
    assert wait_for("upload 2/2").endswith("upload 2/2 Views · capture c1 · 0:30")

    now[0] = 500.0
    assert server.status_line().endswith("0:30")

    process.stdin.write("\n")
    process.stdin.close()
    assert wait_for("fusing").endswith("2/6 fusing · scene a1b2c3d4 · 0:00")
    server.crash_output()


def test_watch_prints_the_log_and_reports_why_the_server_exited(capsys):
    script = (
        "import sys\n"
        "print('POST /scenes 201')\n"
        f"print({_progress_line('j1', 'running', 'masking', 4)!r})\n"
        "print('boom', file=sys.stderr)\n"
        "raise SystemExit(3)\n"
    )
    bars: list[str] = []

    with pytest.raises(RuntimeError, match=r"code 3[\s\S]*boom"):
        colab.watch(_script_server(script), show=bars.append, poll=0.01)

    assert capsys.readouterr().out == "POST /scenes 201\n"
    assert "4/6 masking" in bars[-1]


def test_watch_can_be_stopped_and_resumed_without_touching_the_server(capsys):
    script = "import sys; print('POST /scenes 201', flush=True); sys.stdin.readline()"
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    server = RunningServer(process)
    deadline = time.monotonic() + 30
    while server.new_lines(0)[1] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    def interrupt(_: str) -> None:
        raise KeyboardInterrupt

    try:
        colab.watch(server, show=interrupt)
        first = capsys.readouterr().out
        colab.watch(server, show=interrupt)
        second = capsys.readouterr().out
        assert server.poll() is None
    finally:
        process.kill()
        process.wait(timeout=30)
        assert process.stdin is not None
        process.stdin.close()

    assert first.startswith("POST /scenes 201\nstopped watching")
    assert second == first


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_lines(server: RunningServer, wanted: str) -> list[str]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        lines = server.new_lines(0)[0]
        if any(wanted in line for line in lines):
            return lines
        time.sleep(0.05)
    raise AssertionError(f"{wanted!r} never appeared in {server.new_lines(0)[0]}")


def test_real_server_logs_requests_and_errors_only_and_hides_successful_polling(tmp_path):
    token = "tok"
    port = _free_port()
    env = server_environment(token, {"storage_root": str(tmp_path), "pipeline": "fake"})
    process = subprocess.Popen(
        server_command(port),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    server = RunningServer(process)
    try:
        colab.wait_until_ready(server, port, token, timeout=60)
        auth = {"Authorization": f"Bearer {token}"}
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers=auth) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/jobs/unknown").status_code == 404
            capture_id = client.post("/captures", json={"view_count": 10}).json()["capture_id"]
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, format="PNG")
            for index in range(10):
                resp = client.post(
                    f"/captures/{capture_id}/views",
                    data={"index": str(index)},
                    files={"image": ("view.png", buf.getvalue())},
                )
                assert resp.status_code == 204
            deadline = time.monotonic() + 30
            while "upload 10/10" not in server.status_line() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert f"upload 10/10 Views · capture {capture_id[:8]}" in server.status_line()

            body = client.post("/scenes", json={"name": "Kitchen", "capture_id": capture_id}).json()
            job_url = f"/jobs/{body['job_id']}"
            assert client.get(job_url).status_code == 200

            deadline = time.monotonic() + 30
            while "succeeded" not in server.status_line() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert f"succeeded · scene {body['scene_id'][:8]}" in server.status_line()

            lines = _wait_for_lines(server, "POST /scenes")
    finally:
        process.terminate()
        process.wait(timeout=30)

    requests = [line for line in lines if " - " in line]
    assert len(requests) == len(lines), lines  # no startup banner or any other INFO line
    assert any('"GET /jobs/unknown HTTP/1.1" 404' in line for line in requests)
    assert any('"POST /captures HTTP/1.1" 201' in line for line in requests)
    assert not any("/health" in line for line in requests)
    assert not any(job_url in line for line in requests)


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
    script = "import sys; print('startup failed', file=sys.stderr); raise SystemExit(3)"
    server = _script_server(script)
    deadline = time.monotonic() + 30
    while server.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match=r"code 3[\s\S]*startup failed"):
        colab.wait_until_ready(server, other_server, "tok", timeout=5)


def test_wait_until_ready_returns_when_our_server_runs_and_answers(other_server):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        colab.wait_until_ready(RunningServer(process), other_server, "tok", timeout=5)
    finally:
        process.kill()
        process.wait(timeout=30)


def test_start_named_tunnel_reports_an_early_exit_without_leaking_the_token(monkeypatch):
    monkeypatch.setattr(
        colab, "tunnel_command", lambda *_: [sys.executable, "-c", "raise SystemExit(4)"]
    )
    with pytest.raises(RuntimeError, match="code 4") as excinfo:
        colab.start_named_tunnel("secret", binary=Path("cloudflared"), grace=30)
    assert "secret" not in str(excinfo.value)


def test_start_named_tunnel_returns_a_tunnel_that_stays_up(monkeypatch):
    monkeypatch.setattr(
        colab, "tunnel_command", lambda *_: [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    tunnel = colab.start_named_tunnel("secret", binary=Path("cloudflared"), grace=0.2)
    try:
        assert tunnel.poll() is None
    finally:
        tunnel.kill()
        tunnel.wait(timeout=30)


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


def test_install_packages_raises_with_uv_output_tail(monkeypatch):
    script = "print('Resolved 1 package in 1ms'); print('error: boom'); raise SystemExit(1)"
    monkeypatch.setattr(colab.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(colab, "install_command", lambda *_: [sys.executable, "-c", script])
    shown: list[str] = []

    with pytest.raises(RuntimeError, match="error: boom"):
        colab.install_packages(Path("/repo"), "real", show=shown.append)

    assert shown[0].startswith("[░")
    assert "1/5 Install uv" in shown[0]
    assert "3/5 Download and build" in shown[-1]
