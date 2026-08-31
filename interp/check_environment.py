"""Report interpretation-environment readiness without downloading anything.

This script uses only the Python standard library.  It never imports heavyweight
ML packages, reads API keys, installs dependencies, or contacts the network.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any


SCHEMA_VERSION = "1.0"
PACKAGES = ("torch", "transformers", "huggingface-hub", "jlens")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _nvidia_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"executable_found": False, "query_succeeded": False, "gpus": []}

    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "executable_found": True,
            "query_succeeded": False,
            "gpus": [],
            "error_type": type(exc).__name__,
        }

    gpus: list[dict[str, Any]] = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                memory_mib: int | None = int(fields[1])
            except ValueError:
                memory_mib = None
            gpus.append(
                {
                    "name": fields[0],
                    "memory_total_mib": memory_mib,
                    "driver_version": fields[2],
                }
            )

    report: dict[str, Any] = {
        "executable_found": True,
        "query_succeeded": completed.returncode == 0,
        "gpus": gpus,
    }
    if completed.returncode != 0:
        report["return_code"] = completed.returncode
    return report


def build_report() -> dict[str, Any]:
    release = platform.release().lower()
    return {
        "schema_version": SCHEMA_VERSION,
        "safety": {
            "network_calls": False,
            "downloads": False,
            "heavy_package_imports": False,
            "api_keys_read": False,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "running_under_wsl": "microsoft" in release or "WSL_INTEROP" in os.environ,
        },
        "tools": {
            "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
            "nvcc_found": shutil.which("nvcc") is not None,
        },
        "packages": _package_versions(),
        "nvidia": _nvidia_report(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect local J-lens rehearsal readiness without downloads or imports."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a compact text summary.",
    )
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        python = report["python"]
        platform_info = report["platform"]
        print(f"Python: {python['version']} ({python['implementation']})")
        print(
            f"Platform: {platform_info['system']} {platform_info['release']} "
            f"[{platform_info['machine']}]"
        )
        print(f"Running under WSL: {platform_info['running_under_wsl']}")
        print(f"Packages: {report['packages']}")
        print(f"NVIDIA: {report['nvidia']}")
        print(f"nvcc found: {report['tools']['nvcc_found']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
