#!/usr/bin/env python3
"""Compile the cpp examples under docs/ and report the ones that fail for a reason other than
undefined context variables.

Nine rounds of checking these by hand found nine defects: a README memory example that named an
include it did not have, an ARCHITECTURE snippet using cuda::error where every header declares
nova::error, a SPARSE example calling a cg() function that does not exist, a QUANTIZATION example
passing nine arguments to a six-argument forward(), and so on. Each was found by compiling the block,
and none of them was visible by reading it.

The hard part is that most blocks in docs/ are fragments: they reference A, config, calibration_data,
stream -- names the surrounding prose introduces but no block defines. Those are not defects, and
requiring every block to compile would mean inventing declarations for a dozen variables and burying
the shape the example exists to show.

So this checks something narrower. A block fails the check only when the compiler reports an error
that is NOT one of:

  - "X was not declared in this scope"           (undefined context variable)
  - "X has not been declared"                    (same, for a type)
  - "X is not a member of std"                   (missing standard header, reported separately)
  - "expected primary-expression before ..."     (usually a knock-on from an undefined name)
  - "X does not name a type"                     (same)

What remains is the class that actually broke: a wrong argument type, a wrong arity, a member that
does not exist on a known class, or a missing project header. Those are reported with the block's
file and index.

Usage:
    python3 scripts/check_doc_examples.py            # report
    python3 scripts/check_doc_examples.py --strict   # also fail on missing standard headers
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"```cpp\n(.*?)```", re.S)

# Errors that mean "this block is a fragment", not "this block is wrong".
FRAGMENT_PATTERNS = [
    re.compile(r"‘\w+’ was not declared in this scope"),
    re.compile(r"‘\w+’ has not been declared"),
    re.compile(r"‘\w+’ does not name a type"),
    re.compile(r"expected primary-expression before"),
    re.compile(r"expected ‘;’ before"),
    re.compile(r"’ was not declared in this scope"),
    re.compile(r"’ has not been declared"),
    re.compile(r"\b[A-Za-z_]\w*\b was not declared"),
    re.compile(r"invalid use of incomplete type"),
]

# Reported separately: the block uses a standard type without including its header. Worth knowing
# but not the same class of defect, because the fix is mechanical and local.
STD_HEADER_PATTERN = re.compile(r"‘(\w+)’ is not a member of ‘std’")


def is_fragment_error(message: str) -> bool:
    return any(p.search(message) for p in FRAGMENT_PATTERNS)


def compile_block(code: str, include_dirs: list[str]) -> list[str]:
    """Return the compiler's error lines for this block wrapped in a function body."""
    lines = code.splitlines()
    includes = [l for l in lines if l.strip().startswith("#include")]
    body = [l for l in lines if not l.strip().startswith("#include")]

    # A block that already has its own main is a complete program; wrapping it again produces
    # "a function-definition is not allowed here" and hides the real diagnostics.
    if any("int main" in line for line in lines):
        source = "\n".join(lines) + "\n"
    else:
        source = "\n".join(includes) + "\nint main() {\n" + "\n".join(body) + "\nreturn 0; }\n"

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as handle:
        handle.write(source)
        path = handle.name

    try:
        result = subprocess.run(
            ["g++", "-std=c++20", "-fsyntax-only", *include_dirs, path],
            capture_output=True,
            text=True,
        )
    finally:
        pathlib.Path(path).unlink(missing_ok=True)

    return [line.split("error: ", 1)[1] for line in result.stderr.splitlines() if " error: " in line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=REPO)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--strict", action="store_true", help="also fail on missing standard headers")
    parser.add_argument("--skip", action="append", default=[], help="docs/ path prefix to skip")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    include_dirs = [f"-I{root / 'include'}", "-I/usr/local/cuda/include"]
    include_dirs.extend(f"-I{p}" for p in args.include)

    defects = 0
    std_header_gaps = 0
    checked = 0
    for md in sorted((root / "docs").rglob("*.md")):
        relative = str(md.relative_to(root))
        if any(relative.startswith(skip) for skip in args.skip):
            continue
        for index, code in enumerate(BLOCK.findall(md.read_text(encoding="utf-8"))):
            if "#include" not in code:
                continue
            checked += 1
            errors = compile_block(code, include_dirs)
            real = [e for e in errors if not is_fragment_error(e) and not STD_HEADER_PATTERN.search(e)]
            std = [e for e in errors if STD_HEADER_PATTERN.search(e)]

            if std:
                std_header_gaps += 1
                if args.strict:
                    print(f"{relative} block {index}: missing standard header -- {std[0]}", file=sys.stderr)
            if real:
                defects += 1
                print(f"{relative} block {index}:", file=sys.stderr)
                for message in real[:2]:
                    print(f"    {message}", file=sys.stderr)

    if defects or (args.strict and std_header_gaps):
        return 1

    print(f"doc examples: {checked} blocks with includes checked, {defects} defects, {std_header_gaps} missing standard headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
