# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kernel benchmarks for [strided-rs](https://github.com/tensor4all/strided-rs)
(`strided-perm`, `strided-kernel`, `strided-fused`, `strided-view`). Each
benchmark is a `[[bin]]` under `benchmarks/strided_benchmarks/<name>/` with its
own README describing the cases, the baselines and how to run it. The einsum
benchmark half was removed with the strided-rs einsum crates.

## Build & Run Commands

```bash
cargo build --release [--features parallel] [--features hptt]
cargo run --release --bin permute
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Cargo.toml` references the strided crates via local paths
(`../strided-rs/`); clone strided-rs as a sibling directory.

## Benchmark Execution Rules

**NEVER run multiple benchmarks concurrently.** Run thread-count variants one
at a time. See `AGENTS.md` for pinning and hash-recording rules.
