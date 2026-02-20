@CLAUDE.md

## Critical Rules

- **NEVER run benchmarks in parallel.** Benchmarks must be run sequentially (1-thread first, then 4-thread, etc.). Running them concurrently causes interference between benchmark processes, producing unreliable and misleading results.
- **Record strided-rs git hash when updating README benchmark results.** Before running benchmarks, commit all changes in strided-rs and record the git hash. Include it in the README alongside the results (e.g. in the environment line or notes). This ensures reproducibility and makes it possible to trace which code produced which results.
- **Always pin CPU cores with `taskset` for ALL benchmarks (Rust and Julia, including 1T).** Use `taskset -c 0` (1T), `taskset -c 0-3` (4T), `taskset -c 0-7` (8T) to bind benchmark processes to specific cores within the same L3/CCD domain. Without pinning, threads migrate across CCDs on AMD EPYC, causing L3 cache misses and up to 24% performance degradation. Before running, check `top` or `ps -eo pid,psr,%cpu,comm --sort=-%cpu | head` for CPU-intensive processes from other users, and avoid their CCD (use `lscpu -e` to map core→L3 domain).
