import sys
from pathlib import Path

import pytest

from scenetiq import colab
from scenetiq.colab import InstallProgress, install_command, server_command, server_environment


def test_server_environment_sets_backend_config_and_keeps_base():
    env = server_environment("tok", Path("/drive/scenetiq"), "real", base={"PATH": "/bin"})
    assert env == {
        "PATH": "/bin",
        "SCENETIQ_TOKEN": "tok",
        "SCENETIQ_STORAGE_ROOT": str(Path("/drive/scenetiq")),
        "SCENETIQ_PIPELINE": "real",
    }


def test_server_command_runs_the_app_factory_on_localhost():
    cmd = server_command(8123)
    assert cmd[1:] == [
        "-m", "uvicorn", "--factory", "scenetiq.server:build_app",
        "--host", "127.0.0.1", "--port", "8123",
    ]  # fmt: skip


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
