@CLAUDE.md

## Critical Rules

- **NEVER run benchmarks in parallel.** Benchmarks must be run sequentially (1-thread first, then 4-thread, etc.). Running them concurrently causes interference between benchmark processes, producing unreliable and misleading results.
