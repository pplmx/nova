# Phase 66: Retry System - Plan

## Requirements
- RT-01: Exponential backoff with configurable base delay
- RT-02: Jitter implementation (full/decorrelated)
- RT-03: Circuit breaker pattern with threshold configuration
- RT-04: Retry policy composition and chaining

## Implementation Plan

### 1. Retry Policy
- exponential_backoff_policy struct
- Configurable base_delay, multiplier, max_delay, max_attempts

### 2. Jitter
- full_jitter policy function
- decorrelated_jitter policy function

### 3. Circuit Breaker
- circuit_breaker class with states: closed, open, half_open
- Threshold configuration for failure count and reset timeout

### 4. Policy Chain
- retry_policy concept for composable policies
- retry_executor that chains policies

## Files to Create
- `include/cuda/error/retry.hpp`
- `src/cuda/error/retry.cpp`
- `tests/retry_test.cpp`
