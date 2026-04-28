# Phase 54: Foundation & Sorting — Execution Plan

**Phase:** 54
**Status:** Executing

## Plans

### Plan 54-1: CUB Radix Sort Implementation

**Requirements:** SORT-01
**Goal:** Implement GPU radix sort for key-value pairs using CUB

**Tasks:**
1. Create `include/cuda/algo/sort.h` with API declarations
2. Create `src/cuda/algo/sort.cu` with CUB-based radix sort implementation
3. Add explicit template instantiations for common types (float, double, int)
4. Update CMakeLists.txt to compile sort.cu

**Verification:**
- [ ] `radix_sort_keys()` sorts correctly (ascending/descending)
- [ ] `radix_sort_pair()` sorts keys while keeping values aligned
- [ ] Handles edge cases: empty, single element, all same, duplicates

### Plan 54-2: Top-K Selection

**Requirements:** SORT-02
**Goal:** Find k largest elements without full sort

**Tasks:**
1. Implement `select_top_k()` using CUB segmented sort pattern
2. Support both ascending (smallest k) and descending (largest k) order
3. Return actual k if k > count

**Verification:**
- [ ] Returns correct top k elements
- [ ] Complexity better than full sort (O(n log k))
- [ ] Handles k > array size

### Plan 54-3: Binary Search with Warp Shuffle

**Requirements:** SORT-03
**Goal:** Perform binary search using warp-level primitives

**Tasks:**
1. Implement `binary_search()` using warp shuffle instructions
2. Avoid shared memory bank conflicts
3. Return found status and index

**Verification:**
- [ ] Correctly finds elements in sorted array
- [ ] Returns NotFound for absent elements
- [ ] Handles edge cases: first, last, single element, empty

### Plan 54-4: Tests and Integration

**Goal:** Comprehensive test coverage for sorting algorithms

**Tasks:**
1. Create `tests/algo_sort_test.cu` with Google Test cases
2. Test RadixSort: random, sorted, reverse, duplicates, large arrays
3. Test KeyValue sort: ensure key-value alignment
4. Test TopK: various k values, ascending/descending
5. Test BinarySearch: found, not found, edge cases

**Verification:**
- [ ] All tests pass
- [ ] Tests cover success criteria
- [ ] CMakeLists.txt updated

---

**Success Criteria:**
1. User can sort arrays of key-value pairs in ascending or descending order using GPU radix sort
2. User can find the k largest elements in a dataset without performing a full sort
3. User can perform binary search on sorted arrays using warp shuffle primitives
4. Sorting operations integrate cleanly with existing Buffer and MemoryPool patterns

**Status:** All plans executed

---
*Plan created: 2026-04-28*
*Phase 54: Foundation & Sorting*
