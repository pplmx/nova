#!/usr/bin/env python3
"""Check that every #include in README's cpp blocks names a header that exists.

README's code blocks are fragments -- none has an int main -- so they cannot be compiled as written.
What they can be checked for is the thing that actually broke: a block showed how to build a
cuda::memory::MemoryPool while including only cuda/memory/buffer.h, which does not provide that type.
Anyone copying the example got a compile error, and nothing compared the two.

This walks every ```cpp block in README.md, collects its #include lines and checks each one resolves
to a file under include/. It does not check that the block uses only what it includes: that needs a
compilable translation unit, which a fragment is not.

Usage:
    python3 scripts/check_readme_includes.py [--root DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

INCLUDE_BLOCK = re.compile(r"```cpp\n(.*?)```", re.S)
INCLUDE_LINE = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.M)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=pathlib.Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    readme = root / "README.md"
    if not readme.is_file():
        print(f"README.md not found under {root}", file=sys.stderr)
        return 1

    text = readme.read_text(encoding="utf-8")
    blocks = INCLUDE_BLOCK.findall(text)
    if not blocks:
        print("no cpp blocks found in README.md; this check would pass vacuously", file=sys.stderr)
        return 1

    checked = 0
    missing: list[tuple[int, str]] = []
    for index, block in enumerate(blocks):
        for header in INCLUDE_LINE.findall(block):
            # cuda/* and parallel/* live under include/; system headers do not.
            candidate = root / "include" / header
            if not candidate.is_file():
                # Only report headers that look like project headers, so <vector> etc. do not trip it.
                if "/" in header and not header.startswith(("cuda/", "parallel/", "image/")):
                    continue
                missing.append((index, header))
            checked += 1

    if missing:
        for index, header in missing:
            print(f"README.md cpp block {index}: #include \"{header}\" does not exist", file=sys.stderr)
        return 1

    print(f"README includes resolve: {checked} checked across {len(blocks)} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
