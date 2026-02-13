#!/usr/bin/env python3
"""
Cross-platform start script for the Jamuna Bankline Prediction System.

Usage:
  Linux / macOS : python start.py   OR   python3 start.py
  Windows       : python start.py   (in cmd / PowerShell / Git Bash)

What it does:
  1. Checks whether backend Python deps are installed (.venv)
     → Runs `uv sync` if not.
  2. Checks whether frontend Node deps are installed (frontend/node_modules)
     → Runs `pnpm install` inside frontend/ if not.
  3. Starts FastAPI backend on :8000
  4. Starts Next.js dev server on :3000
  5. Waits; kills both on Ctrl-C.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Colours (disabled on Windows unless ANSI is supported) ───────────────────
_USE_COLOR = sys.stdout.isatty() and (os.name != "nt" or os.environ.get("TERM"))

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def info(msg: str) -> None:  print(_c("36", f"[info]  ") + msg)
def ok(msg: str) -> None:    print(_c("32", f"[ok]    ") + msg)
def warn(msg: str) -> None:  print(_c("33", f"[warn]  ") + msg)
def err(msg: str) -> None:   print(_c("31", f"[error] ") + msg)
def head(msg: str) -> None:  print(_c("1;34", f"\n=== {msg} ==="))

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
VENV = ROOT / ".venv"

IS_WIN = os.name == "nt"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> None:
    """Run a command, streaming output, raising on failure."""
    info("Running: " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if check and result.returncode != 0:
        err(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _venv_has_package(pkg: str) -> bool:
    """Check if a package is installed in the project venv."""
    if IS_WIN:
        site_pkgs = VENV / "Lib" / "site-packages"
    else:
        # site-packages is under lib/pythonX.Y/site-packages
        lib = VENV / "lib"
        if not lib.exists():
            return False
        candidates = list(lib.glob("python*/site-packages"))
        site_pkgs = candidates[0] if candidates else lib / "site-packages"
    return (site_pkgs / pkg).exists() or any(site_pkgs.glob(f"{pkg}-*.dist-info"))


def _node_modules_ok() -> bool:
    """Return True if frontend/node_modules looks complete."""
    nm = FRONTEND / "node_modules"
    return nm.exists() and (nm / ".package-lock.json").exists() or (nm / ".modules.yaml").exists()


# ── Backend dep check ─────────────────────────────────────────────────────────

REQUIRED_PYTHON = "3.12"


def ensure_python_312() -> None:
    """Ensure Python 3.12 is available via uv (installs it if needed)."""
    result = subprocess.run(
        ["uv", "python", "find", REQUIRED_PYTHON],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok(f"Python {REQUIRED_PYTHON} found: {result.stdout.strip()}")
        return
    warn(f"Python {REQUIRED_PYTHON} not found. Installing via uv…")
    _run(["uv", "python", "install", REQUIRED_PYTHON], cwd=ROOT)
    ok(f"Python {REQUIRED_PYTHON} installed")


def ensure_backend_deps() -> None:
    head("Backend dependencies")

    if not _cmd_exists("uv"):
        err(
            "uv not found. Install it first:\n"
            "  Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  Windows:     winget install --id=astral-sh.uv  OR  pip install uv"
        )
        sys.exit(1)

    # Ensure the pinned Python version exists before syncing
    ensure_python_312()

    if _venv_has_package("fastapi") or _venv_has_package("pydantic"):
        ok("Python venv already set up — skipping install")
        return

    warn("Python dependencies not found. Running `uv sync` (this may take a while)…")
    _run(["uv", "sync"], cwd=ROOT)
    ok("Backend dependencies installed")


# ── Frontend dep check ────────────────────────────────────────────────────────

def ensure_frontend_deps() -> None:
    head("Frontend dependencies")

    if not _cmd_exists("pnpm"):
        err(
            "pnpm not found. Install it first:\n"
            "  npm install -g pnpm\n"
            "  OR: corepack enable && corepack prepare pnpm@latest --activate"
        )
        sys.exit(1)

    modules = FRONTEND / "node_modules"
    # Consider installed if node_modules/.pnpm exists (pnpm layout) or package dir for next exists
    next_pkg = modules / "next"
    if modules.exists() and next_pkg.exists():
        ok("Node modules already installed — skipping install")
        return

    warn("Node modules not found. Running `pnpm install`…")
    _run(["pnpm", "install"], cwd=FRONTEND)
    ok("Frontend dependencies installed")


# ── Start servers ─────────────────────────────────────────────────────────────

def _free_port(port: int) -> None:
    """Kill any process currently listening on *port* (Linux/macOS only)."""
    if IS_WIN:
        return  # fuser not available; skip on Windows
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True,
        )
        pids = result.stdout.split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except Exception:
                pass
        if pids:
            time.sleep(1)  # give processes a moment to exit
    except FileNotFoundError:
        pass  # fuser not installed — skip


def start_servers() -> None:
    head("Starting servers")

    # Kill any stale processes from a previous run
    info("Freeing ports 8000 and 3000…")
    _free_port(8000)
    _free_port(3000)
    # Remove stale Next.js dev lock if present
    lock = FRONTEND / ".next" / "dev" / "lock"
    if lock.exists():
        lock.unlink()

    # Resolve the uv-managed Python / uvicorn
    if IS_WIN:
        uvicorn_cmd = ["uv", "run", "uvicorn"]
        pnpm_cmd = ["pnpm.cmd"]
    else:
        uvicorn_cmd = ["uv", "run", "uvicorn"]
        pnpm_cmd = ["pnpm"]

    backend_cmd = uvicorn_cmd + [
        "src.serving.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    frontend_cmd = pnpm_cmd + ["dev"]

    info("Starting FastAPI  → http://localhost:8000")
    backend = subprocess.Popen(backend_cmd, cwd=str(ROOT))

    info("Starting Next.js  → http://localhost:3000")
    frontend = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND))

    print()
    print(_c("1", "  Backend  →") + " http://localhost:8000")
    print(_c("1", "  Frontend →") + " http://localhost:3000")
    print(_c("1", "  API Docs →") + " http://localhost:8000/docs")
    print()
    print(_c("33", "  Press Ctrl-C to stop both servers."))
    print()

    # ── Cleanup on exit ───────────────────────────────────────────────────────
    _stopping = False

    def _stop(signum=None, frame=None) -> None:
        nonlocal _stopping
        if _stopping:
            return
        _stopping = True
        # Reset handlers to defaults so re-entrant signals don't loop
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print("\nStopping servers…")
        for proc in (backend, frontend):
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in (backend, frontend):
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        ok("Servers stopped.")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Wait for either process to exit
    while True:
        be_code = backend.poll()
        fe_code = frontend.poll()
        if be_code is not None:
            warn(f"Backend exited with code {be_code}")
            _stop()
            break
        if fe_code is not None:
            warn(f"Frontend exited with code {fe_code}")
            _stop()
            break
        time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_c("1;36", "\n=== Jamuna Bankline Prediction System ===\n"))

    ensure_backend_deps()
    ensure_frontend_deps()
    start_servers()
