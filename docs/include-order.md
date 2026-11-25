# Include order: Nova headers before libcu++ or Thrust

Nova's public headers do not compile if `<cuda/std/...>` or any Thrust header has already been
included in the same translation unit. Include Nova first and the problem does not occur.

## What happens

```cpp
#include <cuda/std/cstdint>            // or #include <thrust/device_vector.h>
#include <cuda/nccl/nccl_context.h>
```

with nvcc produces:

```text
nccl_context.h(285): error: namespace "cuda::std" has no member "mutex"
nccl_context.h(285): error: namespace "cuda::std" has no member "lock_guard"
nccl_context.h(75):  error: namespace "cuda::std" has no member class "runtime_error"
```

57 errors from that two-line file. With `#include <thrust/device_vector.h>` instead of the libcu++
header, 23 errors from the same cause.

## Why

Nova declares everything inside `namespace cuda::...`, and writes `std::vector`, `std::mutex`,
`std::string` and so on without qualification. Unqualified lookup for `std` searches the enclosing
namespaces first. libcu++ defines `cuda::std`. So:

- **Nova header first**: when the `cuda::` namespace is being filled in, no `cuda::std` exists yet, and
  `std::` resolves to `::std`. Everything compiles.
- **libcu++ or Thrust first**: `cuda::std` already exists, it wins the lookup inside `namespace cuda::`,
  and `std::mutex` becomes `cuda::std::mutex`, which libcu++ does not provide.

Minimal reproduction with no project headers involved:

```cpp
#include <cuda/std/cstdint>
#include <mutex>
namespace cuda { namespace std_shim { } }
namespace cuda::nccl {
    void check() { std::lock_guard<std::mutex> lock(*new std::mutex); }
}
```

nvcc reports the same two lookup errors. Delete the `std_shim` line and it compiles.

## Workaround

Put Nova's includes first:

```cpp
#include <cuda/memory/buffer.h>        // Nova first
#include <cuda/nccl/nccl_context.h>

#include <thrust/device_vector.h>      // libcu++ / Thrust after
#include <cuda/std/cstdint>
```

Verified both ways with nvcc: 0 errors when Nova comes first, 57 (libcu++) and 23 (Thrust) when it does
not.

If a header is precompiled or force-included by your build and you cannot reorder, the alternative is to
qualify the uses inside Nova as `::std::`, which requires editing the headers.

## Nova no longer creates `cuda::std` itself

`include/cuda/quantize/fp8_types.hpp` was the only place in the tree that included anything under
`<cuda/std/>` -- it needed `numeric_limits` for `fp8_to_float()`, which is `__host__ __device__`, and
`std::numeric_limits` is not available in device code. That single include was enough to put
`cuda::std` into any translation unit that touched FP8, and from then on every other Nova header in
that unit failed. Including `fp8_types.hpp` and then `nccl_context.h` failed; it now compiles.

It uses `__builtin_huge_valf()` and `__builtin_nanf("")` instead -- both nvcc built-ins, both valid in
device code. Verified by compiling and by the 119 FP8 and quantization tests.

What this does **not** fix: a user who includes `<cuda/std/...>` or a Thrust header *before* Nova's
headers still hits the same shadowing, because `cuda::std` comes from libcu++ in that case. Include
order remains the contract for that case, and the section below still applies.

## Scope

While `fp8_types.hpp` pulled libcu++ in, 99 of the 177 headers under `include/` failed this way (measured
by compiling each with `<cuda/std/cstdint>` in front), across 13 modules -- `performance` (13), `nccl`
(12), `neural` (11), `production` (9), `algo` (8), `distributed` (7), `testing` (5), `memory` (4),
`inference` (4), `pipeline` (3), and others. Counted by compiling each header with
`#include <cuda/std/cstdint>` placed before it.

The same shadowing is why the `ci` preset's unity build fails; see
[build-performance.md](build-performance.md#the-ci-preset-currently-fails-to-build).

## Why this is documented rather than fixed

Two fixes exist and both need a decision that belongs to whoever owns the project's naming conventions:

1. **Qualify every use as `::std::`.** Unambiguous and local to each line, but it is 1,405 occurrences
   across 92 headers and was measured to reach 75 files before the scan finished -- a large mechanical
   change that wants review in batches.
2. **Add `namespace std = ::std;` inside each affected `cuda::` namespace.** About 30-60 lines instead
   of 1,405 edits, and verified to work on a reproduction. It also injects a `std` alias into `cuda::`,
   which may collide with what libcu++ users expect from `cuda::std`.

Until one of those lands, include order is the contract.
