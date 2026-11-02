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
ENV_FILE = BACKEND / ".env"
TEST_DB = "scene-test-db"
TEST_DB_PORT = "15432"
TEST_DATABASE_URL = f"postgresql://scene:scene@127.0.0.1:{TEST_DB_PORT}/scene"
# Profiles of compose.yaml; `up` starts PARTS unless told which.
PARTS = ("backend", "frontend")
PROFILES = (*PARTS, "local-gpu")

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


IMAGE_ARGUMENTS = ((["--image"], {}), (["--extras"], {"default": "", "choices": ["", "gpu"]}))


# Processes


def run(*argv, env=None, check=True):
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, cwd=ROOT, env={**os.environ, **(env or {})}, check=check)


def output(*argv):
    return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def python(*arguments, env=None, check=True):
    return run(sys.executable, "-m", *arguments, env=env, check=check)


def env_values():
    """backend/.env as a dict, created from .env.example on first use."""
    if not ENV_FILE.exists():
        shutil.copyfile(BACKEND / ".env.example", ENV_FILE)
        print(f"Created {ENV_FILE.relative_to(ROOT)} from .env.example; fill in the keys you need.")
    values = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def compose(*arguments, profiles=PROFILES):
    env_values()
    flags = [flag for profile in profiles for flag in ("--profile", profile)]
    files = ["-f", str(ROOT / "compose.yaml"), "--env-file", str(ENV_FILE)]
    return run("docker", "compose", *files, *flags, *arguments)


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


def models_dir():
    return (BACKEND / env_values().get("HOST_MODELS_DIR", "./models")).resolve()


# Stack


@command("create the host model cache folder")
def init_storage(_=None):
    folder = models_dir()
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Model cache: {folder}")


@command(
    "start the backend (Postgres, scene-api, scene-worker) and the frontend",
    (["parts"], {"nargs": "*", "choices": PARTS, "help": "start only these profiles"}),
)
def up(args):
    init_storage()
    compose("up", "-d", "--build", profiles=args.parts or PARTS)


@command("also start scene-compute on the local GPU")
def up_local_gpu(_):
    init_storage()
    compose("up", "-d", "--build")


@command("stop every service, keeping all data")
def down(_):
    compose("down")


@command("delete all Scenes and the database, keep model weights")
def reset(_):
    compose("down", "-v")


@command("fetch model weights into the host cache")
def download_models(_):
    init_storage()
    module = ["python", "-m", "scene_compute.download_models"]
    compose("run", "--rm", "scene-compute", *module, profiles=["local-gpu"])


@command("show Docker volume and model cache sizes")
def disk_usage(_):
    run("docker", "system", "df", "-v", check=False)
    folder = models_dir()
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if folder.exists() else 0
    print(f"Model cache {folder}: {size / 2**30:.2f} GiB")


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


# GPU image


def image_name(args):
    """`--image` wins; otherwise COMPUTE_IMAGE_REPO:<variant>-<git short sha>, so live checks name an exact build."""
    if args.image:
        return args.image
    repository = env_values().get("COMPUTE_IMAGE_REPO")
    if not repository or "<" in repository:
        sys.exit("Pass --image or set COMPUTE_IMAGE_REPO in backend/.env")
    revision = output("git", "rev-parse", "--short", "HEAD").strip()
    return f"{repository}:{args.extras or 'fake'}-{revision}"


@command("build the GPU image", *IMAGE_ARGUMENTS)
def compute_image(args):
    dockerfile = str(ROOT / "Dockerfile.compute")
    platform = ["--platform", "linux/amd64", "--build-arg", f"EXTRAS={args.extras}"]
    run("docker", "build", *platform, "-f", dockerfile, "-t", image_name(args), str(ROOT))


@command("build and push the GPU image", *IMAGE_ARGUMENTS)
def push_compute_image(args):
    compute_image(args)
    image = image_name(args)
    run("docker", "push", image)
    template = "{{range .RepoDigests}}{{println .}}{{end}}"
    digests = output("docker", "image", "inspect", "--format", template, image).split()
    digest = next((d for d in digests if d.startswith(image.rsplit(":", 1)[0] + "@")), "unknown")
    print(f"Pushed {image}\nDigest {digest}")


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
