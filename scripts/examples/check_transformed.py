#!/usr/bin/env python3
"""Assert that an example kernel actually changed a PGM image.

The examples were registered as ctest entries with only their exit status checked,
which is too weak: a kernel that returned success without touching a pixel passed.
This compares the pixel payload of the input and output, ignoring the P5 header.

Usage: check_transformed.py <input.pgm> <output.pgm>
Exit 0 when the payloads differ and both are well formed, 1 otherwise.
"""

import sys


def payload(path):
    with open(path, "rb") as handle:
        blob = handle.read()
    # P5\n<width> <height>\n<maxval>\n<pixels>
    parts = blob.split(b"\n", 3)
    if len(parts) != 4 or parts[0] != b"P5":
        raise ValueError(f"{path}: not a binary PGM (P5)")
    header = parts[1].split()
    if len(header) != 2:
        raise ValueError(f"{path}: malformed dimensions {parts[1]!r}")
    width, height = int(header[0]), int(header[1])
    pixels = parts[3]
    if len(pixels) != width * height:
        raise ValueError(f"{path}: {len(pixels)} bytes for {width}x{height}")
    return pixels


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    try:
        before = payload(argv[1])
        after = payload(argv[2])
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if before == after:
        print(f"FAIL: {argv[2]} is byte-identical to {argv[1]}; the kernel changed nothing",
              file=sys.stderr)
        return 1
    changed = sum(1 for a, b in zip(before, after) if a != b)
    print(f"OK: {changed}/{len(before)} pixels changed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
