#!/usr/bin/env python3
"""Quick host/environment checks for the MPAIL2 + UWLab OmniReset workflow."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _status(ok: bool | None) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "WARN"


def _print_result(label: str, ok: bool | None, detail: str) -> None:
    print(f"[{_status(ok):>4}] {label}: {detail}")


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, ""
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode, output


def _check_python() -> None:
    version = sys.version_info
    ok = (version.major, version.minor) in {(3, 10), (3, 11)}
    _print_result(
        "Python",
        ok,
        f"{platform.python_version()} at {sys.executable} (recommended: 3.10 for core, 3.11 for OmniReset)",
    )


def _check_gpu() -> None:
    code, output = _run(["nvidia-smi"])
    if code == 127:
        _print_result("GPU driver", False, "`nvidia-smi` not found")
        return
    if code != 0:
        detail = output.splitlines()[0] if output else "`nvidia-smi` failed"
        _print_result("GPU driver", False, detail)
        return
    first = output.splitlines()[0] if output else "nvidia-smi ok"
    _print_result("GPU driver", True, first)


def _check_disk() -> None:
    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / (1024**3)
    ok = free_gb >= 60
    hint = "healthy" if free_gb >= 100 else "usable, but tighter than ideal"
    _print_result("Free disk", ok, f"{free_gb:.1f} GB available ({hint}; target >= 60 GB, better >= 100 GB)")


def _check_memory() -> None:
    try:
        import psutil  # type: ignore
    except ImportError:
        _print_result("Memory", None, "Install `psutil` to report RAM precisely")
        return

    total_gb = psutil.virtual_memory().total / (1024**3)
    ok = total_gb >= 24
    _print_result("Memory", ok, f"{total_gb:.1f} GB RAM detected (24 GB+ recommended)")


def _check_repo_layout() -> None:
    root = Path(__file__).resolve().parents[1].parent
    expected = {
        "mpail2": root / "mpail2",
        "UWLab": root / "UWLab",
        "WheeledLab-research": root / "WheeledLab-research",
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        _print_result("Workspace layout", False, f"Missing: {', '.join(missing)}")
        return
    _print_result("Workspace layout", True, "Found mpail2, UWLab, and WheeledLab-research")


def _check_import(name: str) -> None:
    code = (
        "import importlib.util; "
        f"spec = importlib.util.find_spec('{name}'); "
        "print('ok' if spec is not None else 'missing')"
    )
    rc, output = _run([sys.executable, "-c", code])
    if rc != 0:
        _print_result(f"Import {name}", False, "interpreter check failed")
        return
    ok = output.strip() == "ok"
    _print_result(f"Import {name}", ok, "available" if ok else "missing in current interpreter")


def main() -> None:
    print("MPAIL2 / OmniReset readiness check")
    print(f"Working directory: {Path.cwd()}")
    print("")
    _check_repo_layout()
    _check_python()
    _check_gpu()
    _check_memory()
    _check_disk()
    print("")
    for module_name in ("torch", "hydra", "omegaconf", "uwlab_tasks", "mpail2"):
        _check_import(module_name)
    print("")
    print("Notes:")
    print("- Current host development is easiest with two environments: `mpail2-core` (Py3.10) and `mpail2-omnireset` (Py3.11).")
    print("- Devcontainers are optional; do not use them as the first step for Isaac Sim bring-up.")


if __name__ == "__main__":
    main()
