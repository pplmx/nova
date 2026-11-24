# Architecture review

A structural read of this tree, with every claim paired to a command that reproduces it. Written
after an earlier version of this review stated four things that turned out to be false; the
corrections are listed at the end rather than quietly removed, because the failure mode is worth
recognising.

Scope: `include/` and `src/` as of the commit that adds this file. Numbers move; re-run the commands.

## Size

| Measure                       | Value                    | How to reproduce                                                                           |
| ----------------------------- | ------------------------ | ------------------------------------------------------------------------------------------ |
| Product code                  | 55,128 lines / 314 files | `find include src -name '*.h*' -o -name '*.cpp' -o -name '*.cu' \| xargs wc -l \| tail -1` |
| Test code                     | 62,717 lines             | `find tests -name '*.c*' \| xargs wc -l \| tail -1`                                        |
| Modules under `include/cuda/` | 35                       | `ls -d include/cuda/*/ \| wc -l`                                                           |
| Test cases                    | 2,646                    | `ctest --test-dir build-ninja -N \| tail -1`                                               |

Tests outweigh product code. That is the single most important number here: it means the tree is
cheap to change, which is what makes the rest of this document actionable rather than theoretical.

## The three findings that are real

### 1. Three circular module dependencies

```text
device     <-> memory
device     <-> memory <-> stream
distributed <-> nccl
```

Reproduce with the script below. Each arrow is an `#include "cuda/<module>/..."` across a module
boundary:

- `include/cuda/device/cublas_context.h` includes `cuda/memory/buffer.h` and `buffer-inl.h`
- `include/cuda/memory/buffer-inl.h` and `kv_cache_allocator.h` include `cuda/device/error.h`
- `include/cuda/distributed/matmul.h` and `sequence_parallel.h` include `cuda/nccl/*`
- `include/cuda/nccl/nccl_types.h` includes `cuda/distributed/common.h`

```python
import pathlib, re, collections
mods = {p.name for p in pathlib.Path('include/cuda').iterdir() if p.is_dir()}
edges = collections.defaultdict(set)
for f in pathlib.Path('include/cuda').rglob('*.h*'):
    src = f.relative_to('include/cuda').parts[0]
    for inc in re.findall(r'#include [<"]cuda/(\w+)/', f.read_text(errors='replace')):
        if inc != src and inc in mods:
            edges[src].add(inc)
for a, out in edges.items():
    for b in out:
        if a in edges.get(b, ()):
            print(a, '<->', b)
```

Why it matters: a cycle means neither module can be understood, tested, or extracted without the
other. `device/error.h` sits underneath `memory` and also depends on it, so the two form a knot
rather than a stack.

**Not urgent.** The build works and nothing observably misbehaves. It is the thing to fix first if
this tree ever grows a second consumer, because the knot is what would block extraction.

### 2. `observability` mixes two namespace roots

Six of its seven headers open `namespace cuda::...`; `logger.hpp` opens `namespace nova::logging`.

```text
cuda:  bandwidth_tracker.h inference_nvtx.h kernel_stats.h nvtx_extensions.h
       occupancy_analyzer.h timeline.h
nova:  logger.hpp
```

`logger.hpp` is not in `nova::observability` at all, only filed under that directory. A reader
looking for the logger's namespace by its path finds the wrong one.

Everything else is consistent: 26 modules are `cuda::*`, 7 are `nova::*`, and no module mixes apart
from this one. The split between the two roots is a naming disagreement, not a defect — `nova::` is
used by the newer modules (checkpoint, quantize, sparse, preemption, error, comm, gnn).

### 3. Module coupling is concentrated in two places

Fan-in (how many modules include it) — these are the foundations:

```text
memory 25, device 21, stream 11, nccl 10
```

Fan-out (how many modules it pulls in) — these are the risk:

```text
neural 12, algo 11, inference 8, production 8
```

`neural` is the widest consumer and `memory` the most depended on. The two lists barely intersect,
which is the healthy shape: foundations are stable and leaves are busy. The cycles above are what
stops it being fully layered.

## What is not a problem, despite appearances

### Header-only modules are not dead code

Six modules have headers and no `.cpp`: `async`, `benchmark`, `gnn`, `observability`, `stream`,
`testing`. All are template- or inline-based by design and are included from real translation units:

| Module        | Including files |
| ------------- | --------------- |
| observability | 20              |
| stream        | 21              |
| testing       | 18              |
| async         | 5               |
| benchmark     | 5               |
| gnn           | 1               |

`stream/stream.h` in particular is the base type most of the tree is written against — 32 files
include it and `stream::Stream` appears in 199 places, including the public signatures of
`flash_attention`, `beam_search` and `block_manager`.

### Three stream wrappers is not four copies of one thing

`stream/stream.h` is the primitive (RAII around `cudaStreamCreate`). `async/stream_manager.h` (3
includers) and `production/priority_stream.h` (4 includers) are narrow additions on top of it. The
count looks like duplication but the dependency direction is one-way and the layering is real.

### Core dumps

`.gitignore` has covered `core.*` and `*.core` since before this review (line 166). Fifteen core
files had accumulated in the repository root from earlier crash-testing; they were never tracked and
are now deleted. Nothing to change in the ignore rules.

## Corrections to the previous version of this document

Kept because the pattern is more useful than the tidiness:

| Claimed                                           | Actual                                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| "`stream/` is a dead module, delete 183 lines"    | 32 includers, 199 uses in signatures. Deleting it breaks the public API.                          |
| "`.gitignore` has no `core.*` rule"               | The rule is at `.gitignore:166` and works.                                                        |
| "`quantize`, `sparse`, `gnn` are bare namespaces" | All three are `nova::` — `quantize` and `gnn` use nested `namespace nova { namespace quantize {`. |
| "No circular dependencies"                        | Three, listed above.                                                                              |
| "74 commits"                                      | 921.                                                                                              |

Four of the five came from the same mistake: a single-line `grep` was treated as proof of a global
fact. `grep 'namespace quantize'` does not match `namespace nova {` followed by `namespace quantize
{`; `grep 'Stream::'` in `src/` does not find a type used by name in a header. The corrections above
were each found by running the thing rather than pattern-matching it.

## If something is to be done

Ordered by ratio of benefit to risk, with the honest caveat that only the first two are clearly
worth doing:

1. **Move `observability/logger.hpp` to `nova::observability`** (or move the file to a `logging/`
   directory). One file, removes the only mixed-root module.
2. **Break the two cycles.** `device/error.h` is a leaf concept that `memory` needs; moving the
   error types into their own module, or having `memory` use the canonical `CUDA_CHECK` from
   `device/error.h` without the reverse include, would make the layers acyclic.
3. **Unify the namespace roots.** 26 modules are `cuda::*` and 7 are `nova::*`. Unification touches
   248 files and 1,020 `cuda::` references and fixes no defect. It is a large diff for consistency
   alone; do it only if the tree is about to be published or handed to another team.

Not on this list: merging the stream wrappers, and deleting header-only modules. Both were
candidates in the previous version and both are wrong.

## What is missing that would matter more than any of this

The tutorials' code examples are not built by CI. `docs/tutorials/01-quick-start.md` shipped an
example that did not compile at all — wrong namespaces (`nova::memory::Buffer` for what is
`cuda::memory::Buffer`) and calls to methods that do not exist (`host_data()`, `sync_to_device()`,
and an `algo::reduce()` that is really `reduce_sum()`). It was fixed, but nothing stops the next
edit from rotting the same way. Extracting fenced `cpp` blocks from `docs/` and compiling them is a
small CI job and would have caught this years earlier.
