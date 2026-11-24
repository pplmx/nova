#!/usr/bin/env python3
"""Build and run the complete examples in docs/tutorials/ using the project's own CMake.

check_tutorial_snippets.py only runs g++ -fsyntax-only. That catches a wrong namespace or a removed
member, but not a program that compiles and then fails at run time -- which is exactly what happened
to the all-reduce example in docs/tutorials/02-multi-gpu.md: it called a blocking collective once from
the main thread instead of once per rank thread, so on a multi-GPU host it threw from a worker and
terminated, while syntax-checking it was perfectly happy.

Building these needs the project's link configuration (device-link objects, the whole library set,
CUDA:: targets). Reproducing that by hand does not work: an earlier attempt with a standalone
CMakeLists could not resolve __cudaRegisterLinkedBinary, and adding -rdc=true did not fix it. So this
injects a temporary target into examples/CMakeLists.txt, builds it with the existing build tree and
removes the target again.

Usage:
    python3 scripts/run_tutorial_snippets.py --build-dir build-nccl
    python3 scripts/run_tutorial_snippets.py --build-dir build-ninja --timeout 60

Requires a configured build directory. Examples that need an NCCL build are skipped automatically
when NCCl is unavailable, and the reason is printed rather than counted as a pass.
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

TARGET_MARKER = "# --- run_tutorial_snippets.py temporary target ---"

TARGET_TEMPLATE = """
""" + TARGET_MARKER + """
add_executable({target} ${SNIPPET_SOURCE})
target_link_libraries({target} PRIVATE cuda_impl CUDA::cudart CUDA::cublas CUDA::cublasLt
    CUDA::cusolver CUDA::cusparse CUDA::curand CUDA::cufft)
target_include_directories({target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
"""


def complete_examples(directory: pathlib.Path):
    for md in sorted(directory.glob("*.md")):
        for index, code in enumerate(BLOCK.findall(md.read_text(encoding="utf-8"))):
            if "#include" in code and "int main" in code:
                yield md, index, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", default="build-ninja")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--root", default=REPO)
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    build_dir = (root / args.build_dir).resolve()
    if not build_dir.is_dir():
        print(f"error: build directory not found: {build_dir}", file=sys.stderr)
        return 2

    cmake_file = root / "examples" / "CMakeLists.txt"
    original = cmake_file.read_text(encoding="utf-8")
    if TARGET_MARKER in original:
        print("error: examples/CMakeLists.txt still contains a previous run's target", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    workdir = pathlib.Path(tempfile.mkdtemp())
    try:
        for i, (md, index, code) in enumerate(complete_examples(root / "docs" / "tutorials")):
            label = f"{md.relative_to(root)} block {index}"
            name = f"snippet_{i}"
            source = workdir / f"{name}.cpp"
            source.write_text(code, encoding="utf-8")

            # Register the target, configure so CMake picks up the new source, build and run.
            cmake_file.write_text(
                original + TARGET_TEMPLATE.replace("{target}", name), encoding="utf-8"
            )
            configure = subprocess.run(
                ["cmake", "-G", "Ninja", "-B", str(build_dir), "-DSNIPPET_SOURCE=" + str(source)],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if configure.returncode != 0:
                failures.append((label, "cmake configure failed: " + configure.stderr.strip()[-200:]))
                continue

            build = subprocess.run(
                ["ninja", "-C", str(build_dir), name], capture_output=True, text=True
            )
            if build.returncode != 0:
                first = next((l for l in build.stderr.splitlines() if " error: " in l), "build failed")
                failures.append((label, first.strip()[:200]))
                continue

            binary = build_dir / "bin" / name
            try:
                run = subprocess.run(
                    [str(binary)], capture_output=True, text=True, timeout=args.timeout
                )
            except subprocess.TimeoutExpired:
                failures.append((label, f"did not finish within {args.timeout}s"))
                continue

            output = (run.stdout + run.stderr).strip()
            if run.returncode != 0:
                # A multi-GPU example in a build without NCCL cannot work; report it as skipped
                # with the reason instead of pretending the example is broken.
                if "requires an initialized NCCL context" in output:
                    skipped.append((label, "needs -DNOVA_ENABLE_NCCL=ON"))
                else:
                    failures.append((label, output.splitlines()[-1][:200] if output else "non-zero exit"))
            else:
                print(f"  [ok]     {label}")
    finally:
        cmake_file.write_text(original, encoding="utf-8")
        subprocess.run(["cmake", "-G", "Ninja", "-B", str(build_dir)], cwd=root, capture_output=True)
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    for label, reason in skipped:
        print(f"  [skip]   {label}: {reason}")
    for label, reason in failures:
        print(f"  [FAIL]   {label}: {reason}", file=sys.stderr)

    print(f"tutorial programs: {len(failures)} failed, {len(skipped)} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
