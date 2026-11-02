from pathlib import Path

from scenetiq.colab import server_command, server_environment


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
