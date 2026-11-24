#!/usr/bin/env python3
"""Compile every complete example that appears in docs/tutorials.

A tutorial's job is to be copied and run. Nothing was checking that, so
docs/tutorials/01-quick-start.md shipped for a long time with an example that could not be built at
all: it qualified types as nova::memory::Buffer when the real namespace is cuda::memory, and it called
Buffer::host_data(), Buffer::sync_to_device() and algo::reduce(), none of which exist.

Only fenced ```cpp blocks that contain both an #include and an int main are checked. The other blocks
in these files are fragments -- a call, a declaration, a config snippet -- that are not translation
units on their own and cannot be compiled in isolation without inventing surrounding code.

Usage:
    python3 scripts/check_tutorial_snippets.py [--include DIR ...] [--keep-build-dir DIR]

Exits non-zero if any example fails to compile, listing each failure with the compiler's first
diagnostic. Requires g++ and the CUDA headers; pass --include to add paths.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

TUTORIAL_DIR = pathlib.Path("docs/tutorials")
BLOCK = re.compile(r"```cpp\n(.*?)```", re.S)


def complete_examples(directory: pathlib.Path):
    """Yield (source_path, index, code) for blocks that are standalone translation units."""
    for md in sorted(directory.glob("*.md")):
        for index, code in enumerate(BLOCK.findall(md.read_text())):
            if "#include" in code and "int main" in code:
                yield md, index, code


def first_diagnostic(output: str) -> str:
    for line in output.splitlines():
        if " error: " in line:
            return line.strip()
    return output.strip().splitlines()[0] if output.strip() else "(no output)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", action="append", default=[], help="extra -I directory (repeatable)")
    parser.add_argument("--keep-build-dir", help="write the extracted .cpp files here instead of a temp dir")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    tutorial_dir = root / TUTORIAL_DIR
    if not tutorial_dir.is_dir():
        print(f"error: {tutorial_dir} not found", file=sys.stderr)
        return 2

    includes = [root / "include", pathlib.Path("/usr/local/cuda/include")]
    includes.extend(pathlib.Path(p) for p in args.include)
    include_flags = [f"-I{p}" for p in includes]

    workdir = pathlib.Path(args.keep_build_dir) if args.keep_build_dir else pathlib.Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)

    failures = []
    checked = 0
    try:
        for md, index, code in complete_examples(tutorial_dir):
            checked += 1
            name = f"{md.stem}_{index}.cpp"
            source = workdir / name
            source.write_text(code)

            result = subprocess.run(
                ["g++", "-std=c++20", "-fsyntax-only", *include_flags, str(source)],
                capture_output=True,
                text=True,
            )
            status = "ok" if result.returncode == 0 else "FAIL"
            print(f"  [{status}] {md.relative_to(root)} block {index} -> {name}")
            if result.returncode != 0:
                failures.append((f"{md.relative_to(root)} block {index}", first_diagnostic(result.stderr)))
    finally:
        if not args.keep_build_dir:
            shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} of {checked} tutorial example(s) do not compile:")
        for where, why in failures:
            print(f"  {where}\n    {why}")
        print()
        print("A tutorial's examples are what readers copy. Fix the example, or move the block out of")
        print("the \\`\\`\\`cpp fence if it is a fragment rather than a program.")
        return 1

    print(f"All {checked} complete tutorial example(s) compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
