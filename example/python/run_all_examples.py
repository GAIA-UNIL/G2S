#!/usr/bin/env python3
"""Run all Python examples."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
PYTHON_BUILD_DIR = REPO_ROOT / "build" / "python-build"
CURRENT_ROOT = SCRIPT_DIR
LEGACY_ROOT = REPO_ROOT / "legacy_example" / "python"
EXPECTED_FAILURES = {
    Path("example") / "python" / "reporting" / "reporting_probe.py",
    Path("legacy_example") / "python" / "reporting_probe.py",
}
EXPECTED_FAILURE_MARKERS = {
    Path("example") / "python" / "reporting" / "reporting_probe.py": "error probe submitted with job id",
    Path("legacy_example") / "python" / "reporting_probe.py": "error probe submitted with job id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every Python example under example/python and legacy_example/python."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="maximum seconds allowed for each example (default: 900)",
    )
    parser.add_argument(
        "--only",
        metavar="TEXT",
        help="run only examples whose relative path contains TEXT",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="run only examples under example/python",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="run only examples under legacy_example/python",
    )
    args = parser.parse_args()
    if args.current_only and args.legacy_only:
        parser.error("--current-only and --legacy-only cannot be used together")
    return args


def discover_examples(
    only: Optional[str],
    current_only: bool,
    legacy_only: bool,
) -> List[Path]:
    examples: List[Path] = []
    roots: List[Path] = []
    if not legacy_only:
        roots.append(CURRENT_ROOT)
    if not current_only and LEGACY_ROOT.exists():
        roots.append(LEGACY_ROOT)

    for root in roots:
        for path in root.rglob("*.py"):
            if path == SCRIPT_PATH:
                continue
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT)
            if only and only not in relative.as_posix():
                continue
            examples.append(path)
    return sorted(examples)


def build_environment() -> Dict[str, str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    entries = [str(PYTHON_BUILD_DIR)] if PYTHON_BUILD_DIR.exists() else []
    if python_path:
        entries.append(python_path)
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run_example(path: Path, timeout: float, env: Dict[str, str]) -> Tuple[str, float]:
    relative = path.relative_to(REPO_ROOT)
    expected_failure = relative in EXPECTED_FAILURES
    start = time.monotonic()
    print(f"[RUN] {relative}", flush=True)

    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=path.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        print(f"[TIMEOUT] {relative} after {duration:.1f}s", flush=True)
        return "timeout", duration

    duration = time.monotonic() - start
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")

    if completed.returncode == 0:
        if expected_failure:
            print(f"[UNEXPECTED PASS] {relative} in {duration:.1f}s", flush=True)
            return "unexpected_pass", duration
        print(f"[PASS] {relative} in {duration:.1f}s", flush=True)
        return "pass", duration

    if expected_failure:
        combined_output = f"{completed.stdout}\n{completed.stderr}"
        marker = EXPECTED_FAILURE_MARKERS.get(relative)
        if marker and marker not in combined_output:
            print(
                f"[FAIL] {relative} in {duration:.1f}s "
                "(expected failure marker was not reached)",
                flush=True,
            )
            return "fail", duration
        print(f"[EXPECTED FAIL] {relative} in {duration:.1f}s", flush=True)
        return "expected_fail", duration

    print(
        f"[FAIL] {relative} in {duration:.1f}s "
        f"(exit code {completed.returncode})",
        flush=True,
    )
    return "fail", duration


def main() -> int:
    args = parse_args()
    examples = discover_examples(args.only, args.current_only, args.legacy_only)
    if not examples:
        print("No Python examples matched.")
        return 1

    print(f"Running {len(examples)} Python example(s) from {REPO_ROOT}")
    env = build_environment()
    counts = {
        "pass": 0,
        "expected_fail": 0,
        "fail": 0,
        "timeout": 0,
        "unexpected_pass": 0,
    }

    for example in examples:
        status, _ = run_example(example, args.timeout, env)
        counts[status] += 1

    print("")
    print(
        "Summary: "
        f"{counts['pass']} passed, "
        f"{counts['expected_fail']} expected failure(s), "
        f"{counts['fail']} failed, "
        f"{counts['timeout']} timed out, "
        f"{counts['unexpected_pass']} unexpected pass(es)."
    )

    if counts["fail"] or counts["timeout"] or counts["unexpected_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
