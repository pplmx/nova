#!/usr/bin/env python3
"""Check the numbers in README.md against a live ctest count.

README's Testing section claimed "505 tests ... 99%+ passing" while the suite had grown to 2668
registered tests. Nothing was checking the figure, so it drifted by a factor of five and a reader
had no way to tell which number was true without running ctest themselves.

This reads the number out of the README and compares it with `ctest -N` in a build directory. It
deliberately checks only the count, not "all passing": pass/fail is what the test run reports, and
duplicating it here would be a second source of truth that can drift on its own.

Usage:
    python3 scripts/check_readme_facts.py [build-dir]

Exits 1 with a readable message when they disagree, so CI can fail on it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

# "**2668 registered tests across multiple test suites, all passing:**"
PATTERN = re.compile(r"\*\*(\d{2,})\s+registered tests\b")


def readme_count() -> int:
    text = README.read_text(encoding="utf-8")
    match = PATTERN.search(text)
    if match is None:
        sys.exit(
            "README.md no longer states a 'N registered tests' figure in its Testing section; "
            "update this check or restore the sentence"
        )
    return int(match.group(1))


def ctest_count(build_dir: Path) -> int:
    result = subprocess.run(
        ["ctest", "-N"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ctest -N failed in {build_dir}: {result.stderr.strip()}")

    match = re.search(r"Total Tests:\s*(\d+)", result.stdout)
    if match is None:
        sys.exit(f"could not find 'Total Tests:' in ctest -N output from {build_dir}")
    return int(match.group(1))


def main(argv: list[str]) -> int:
    build_dir = Path(argv[1]) if len(argv) > 1 else REPO / "build-ninja"
    if not build_dir.is_dir():
        sys.exit(f"build directory not found: {build_dir}")

    claimed = readme_count()
    actual = ctest_count(build_dir)

    if claimed != actual:
        print(
            f"README.md says {claimed} registered tests but ctest reports {actual}.\n"
            f"Update the Testing section of README.md.",
            file=sys.stderr,
        )
        return 1

    print(f"README test count matches ctest: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
