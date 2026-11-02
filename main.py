"""Cross-platform project commands (stdlib only), not an app entrypoint. Run `uv run main.py --help` from the repository root."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
TEST_DB = "scene-test-db"
TEST_DB_PORT = "15432"
TEST_DATABASE_URL = f"postgresql://scene:scene@127.0.0.1:{TEST_DB_PORT}/scene"

COMMANDS = {}


def command(help_text, *arguments, name=None, passthrough=False):
    """Registers a command under its function's name; `arguments` are (flags, options) for argparse."""

    def register(handler):
        COMMANDS[name or handler.__name__.replace("_", "-")] = (
            handler,
            help_text + ("; extra arguments go to the tool" if passthrough else ""),
            arguments,
            passthrough,
        )
        return handler

    return register


# Processes


def run(*argv, env=None, check=True):
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, cwd=ROOT, env={**os.environ, **(env or {})}, check=check)


def python(*arguments, env=None, check=True):
    return run(sys.executable, "-m", *arguments, env=env, check=check)


def require_docker():
    try:
        running = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except FileNotFoundError:
        running = False
    if not running:
        sys.exit(
            "Docker is not running. Start Docker Desktop (or the Docker daemon) and try again; "
            "`uv run main.py test-unit` runs the tests that need no database."
        )


# Checks


def ensure_test_database():
    """A fresh test database each run, so it always has the current backend/migrations/001_init.sql."""
    require_docker()
    subprocess.run(["docker", "rm", "-f", TEST_DB], capture_output=True)
    keys = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    credentials = [flag for key in keys for flag in ("-e", f"{key}=scene")]
    schema = f"{BACKEND / 'migrations'}:/docker-entrypoint-initdb.d:ro"
    container = ["--name", TEST_DB, "-p", f"127.0.0.1:{TEST_DB_PORT}:5432", "-v", schema]
    run("docker", "run", "-d", *container, *credentials, "pgvector/pgvector:pg16")
    # Over TCP: the server that runs initdb.d listens only on its socket, so this waits for the schema.
    ready = ["docker", "exec", TEST_DB, "pg_isready", "-U", "scene", "-h", "127.0.0.1"]
    for _ in range(60):
        if subprocess.run(ready, capture_output=True).returncode == 0:
            return
        time.sleep(1)
    sys.exit("Test database did not become ready")


def pytest(*arguments, env=None):
    # Pytest's temporary Workspaces live outside the repository.
    basetemp = Path(tempfile.gettempdir()) / "scene-pytest"
    python("pytest", "-p", "no:cacheprovider", f"--basetemp={basetemp}", *arguments, env=env)


def ruff(subcommand, arguments):
    # Findings are the output, not a crash: exit with Ruff's status and no traceback.
    sys.exit(python("ruff", subcommand, "backend", "main.py", *arguments, check=False).returncode)


@command("run every test against a disposable test database (needs Docker)", passthrough=True)
def test(args):
    ensure_test_database()
    pytest(*args.extra_args, env={"TEST_DATABASE_URL": TEST_DATABASE_URL})


@command("run the tests that need no database or Docker", passthrough=True)
def test_unit(args):
    pytest("-m", "not db", *args.extra_args)


@command("check the Python code with Ruff", passthrough=True)
def lint(args):
    ruff("check", args.extra_args)


@command("format the Python code with Ruff", name="format", passthrough=True)
def format_code(args):
    ruff("format", args.extra_args)


@command("export the API contract to backend/openapi/openapi.json")
def openapi(_):
    python("scene_api.export_openapi", env={"DATABASE_URL": "postgresql://unused@localhost/unused"})


def remove_artifact(path):
    """Delete a generated path only when its resolved target stays in the repository."""
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing to clean a path outside the repository: {path}")
    if path.is_symlink() or path.is_junction():
        return 0
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()
    else:
        return 0
    print(f"Removed {path.relative_to(ROOT)}")
    return 1


def clean_tree(root, caches, suffixes, *, skip=()):
    """Walk source folders without entering dependencies, data or linked directories."""

    def raise_walk_error(error):
        raise error

    if root.is_symlink() or root.is_junction():
        raise ValueError(f"Refusing to traverse a linked directory: {root}")
    removed = 0
    protected = {".venv", "venv", "env", "ENV", "node_modules", ".git", *skip}
    for folder, dirs, files in os.walk(root, onerror=raise_walk_error):
        dirs[:] = [
            d
            for d in dirs
            if d not in protected
            and not (Path(folder) / d).is_symlink()
            and not (Path(folder) / d).is_junction()
        ]
        for directory in [d for d in dirs if d in caches]:
            removed += remove_artifact(Path(folder) / directory)
            dirs.remove(directory)
        for filename in files:
            if filename.endswith(suffixes):
                removed += remove_artifact(Path(folder) / filename)
    return removed


@command(
    "remove local Python or React caches and build artifacts",
    (["target"], {"choices": ("python", "react"), "help": "which toolchain to clean"}),
)
def clean(args):
    try:
        if args.target == "python":
            removed = clean_tree(
                ROOT,
                {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache"},
                (".pyc", ".pyo"),
                skip={"frontend", "models", "data"},
            )
            if shutil.which("uv"):
                run("uv", "cache", "clean", "--force")
            else:
                print("uv was not found on PATH; its cache was not cleaned.")
        else:
            frontend = ROOT / "frontend"
            outputs = {"dist", "build", "out", ".next", ".vite", ".cache"}
            removed = clean_tree(
                frontend,
                set(),
                (".tsbuildinfo",),
                skip=outputs,
            )
            for artifact in (
                *sorted(outputs),
                "node_modules/.cache",
                "node_modules/.vite",
                "node_modules/.vite-temp",
                "next-env.d.ts",
            ):
                removed += remove_artifact(frontend / artifact)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Could not finish cleaning {args.target}: {error}")
    print(f"Cleaned {removed} {args.target} artifact locations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (handler, help_text, arguments, _) in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text)
        for flags, options in arguments:
            subparser.add_argument(*flags, **options)
        subparser.set_defaults(handler=handler)
    args, extra = parser.parse_known_args()
    if extra and not COMMANDS[args.command][3]:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra_args = extra
    args.handler(args)


if __name__ == "__main__":
    main()
