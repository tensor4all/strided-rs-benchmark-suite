# Einsum Benchmarks

This directory contains the full-suite einsum benchmark result tables and
focused einsum case studies. The repository root README intentionally links
here instead of embedding large result tables.

## Pages

| Page | Scope |
|---|---|
| [Full suite results](#benchmark-instances) | Repository-level `strided-opteinsum` versus OMEinsum.jl results. |
| [TN light 415 late-step case study](tn_light_415_late_step/README.md) | One late binary contraction from `tensornetwork_permutation_light_415`. |

## Benchmark Instances

Instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite.
Selection is per-category (see
[Export benchmark metadata](../../README.md#1-export-benchmark-metadata)); dtype is
float64 or complex128; tensors are zero-filled at runtime.

| Instance | Category | Tensors | Dims | Typical shapes | Steps | log10(FLOPS) | log2(SIZE) |
|----------|----------|--------:|------|----------------|------:|-------------:|------------:|
| `gm_queen5_5_3.wcsp` | Graphical model | 160 | 2D | 3×3 | 159 | 9.75 | 26.94 |
| `lm_batch_likelihood_brackets_4_4d` | Language model | 84 | 2–4D | 4×4, 4×4×4, 7×1996 | 83 | 8.37 | 18.96 |
| `lm_batch_likelihood_sentence_3_12d` | Language model | 38 | 2–4D | 11×11, 11×11×11, 100×1100 | 37 | 9.20 | 20.86 |
| `lm_batch_likelihood_sentence_4_4d` | Language model | 84 | 2–4D | 4×4, 4×4×4, 7×1900 | 83 | 8.46 | 18.89 |
| `str_matrix_chain_multiplication_100` | Structured | 100 | 2D | 21×478 to 511×507 | 99 | 8.48 | 17.26 |
| `str_mps_varying_inner_product_200` | Structured (MPS) | 200 | 2D | varying | 199 | 8.31 | 15.48 |
| `str_nw_mera_closed_120` | Structured (MERA) | 120 | 2D | 3×3, etc. | 119 | 10.66 | 25.02 |
| `str_nw_mera_open_26` | Structured (MERA) | 26 | 2D | 3×3, etc. | 25 | 10.49 | 25.36 |
| `tensornetwork_permutation_light_415` | Tensor network | 415 | 2D | 2×2 (uniform) | 414 | 9.65 | 24.0 |
| `tensornetwork_permutation_focus_step409_316` | Tensor network (focused) | 316 | 2D | 2×2 (uniform) | 315 | 9.65 | 24.0 |

- **Graphical model (gm_*)**: e.g. WCSP / constraint networks; many small 2D factors (e.g. 3×3), full contraction to scalar.
- **Language model (lm_*)**: many small multi-dimensional tensors (3D/4D) with large batch dimensions; many steps with small GEMM kernels.
- **Structured — matrix chain (str_matrix_chain_*)**: large 2D matrices; each step is one large GEMM.
- **Structured — MPS (str_mps_*)**: matrix product state–style networks; varying inner dimensions, many 2D contractions.
- **Structured — MERA (str_nw_mera_*)**: tensor networks from multi-scale entanglement renormalization; many small 3×3-like tensors, heavy contraction.
- **Tensor network (tensornetwork_permutation_light_415)**: lightweight variant (~5 s vs ~40 s); 415 tensors extracted from the full instance via BFS-connected subgraph. Create via `scripts/create_lightweight_instance.py` (see [Optional: Create a lightweight tensor network instance](../../README.md#optional-create-a-lightweight-tensor-network-instance)).
- **Tensor network focused (tensornetwork_permutation_focus_step409_316)**: focused subtree instance for profiling original late bottleneck steps (408/409). Contains 316 tensors and keeps the high-cost intermediate (`log2_size = 24`) while reducing total contractions.

## Benchmark Results

- [Apple Silicon M2](#apple-silicon-m2)
- [AMD EPYC 7713P](#amd-epyc-7713p)

### Apple Silicon M2

Environment: Apple Silicon M2. Median ± IQR (ms) of 15 runs (3 warmup). Julia BLAS: OpenBLAS (lbt). Run date: 2026-02-20.

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **2054.330 ± 58.939** | 2429.877 ± 66.449 | - |
| lm_batch_likelihood_brackets_4_4d | **11.597 ± 0.096** | 13.082 ± 0.012 | 16.325 ± 40.165 |
| lm_batch_likelihood_sentence_3_12d | **40.257 ± 0.995** | 41.400 ± 1.965 | 55.301 ± 69.093 |
| lm_batch_likelihood_sentence_4_4d | **12.936 ± 0.112** | 14.012 ± 0.028 | 17.239 ± 5.973 |
| str_matrix_chain_multiplication_100 | 10.183 ± 0.165 | **9.338 ± 0.180** | 12.425 ± 0.369 |
| str_mps_varying_inner_product_200 | **10.580 ± 0.059** | 12.005 ± 0.336 | 15.732 ± 11.607 |
| str_nw_mera_closed_120 | 1029.458 ± 13.583 | **994.700 ± 11.886** | 1169.895 ± 68.600 |
| str_nw_mera_open_26 | 668.449 ± 7.164 | **658.564 ± 2.241** | 905.097 ± 40.212 |
| tensornetwork_permutation_focus_step409_316 | **204.914 ± 1.585** | 208.848 ± 1.175 | 242.716 ± 10.592 |
| tensornetwork_permutation_light_415 | **203.023 ± 1.199** | 208.206 ± 3.764 | 242.415 ± 3.871 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **802.483 ± 3.670** | 888.207 ± 9.465 | - |
| lm_batch_likelihood_brackets_4_4d | **12.671 ± 0.064** | 14.821 ± 0.163 | 16.430 ± 10.745 |
| lm_batch_likelihood_sentence_3_12d | 41.237 ± 0.392 | **40.456 ± 0.982** | 55.733 ± 74.042 |
| lm_batch_likelihood_sentence_4_4d | **13.916 ± 0.052** | 15.943 ± 0.199 | 16.873 ± 5.622 |
| str_matrix_chain_multiplication_100 | 10.376 ± 0.568 | **9.394 ± 0.161** | 12.317 ± 0.562 |
| str_mps_varying_inner_product_200 | **10.177 ± 0.166** | 11.824 ± 0.370 | 14.082 ± 0.875 |
| str_nw_mera_closed_120 | 1037.025 ± 12.973 | **1007.038 ± 15.957** | 1107.183 ± 69.989 |
| str_nw_mera_open_26 | 668.463 ± 3.169 | **656.337 ± 13.777** | 913.331 ± 26.657 |
| tensornetwork_permutation_focus_step409_316 | **205.090 ± 1.225** | 209.114 ± 1.121 | 246.776 ± 8.711 |
| tensornetwork_permutation_light_415 | **202.232 ± 2.898** | 209.465 ± 4.182 | 245.172 ± 1.571 |

#### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 2142.267 ± 21.018 | **1937.117 ± 7.069** | - |
| lm_batch_likelihood_brackets_4_4d | **10.287 ± 0.456** | 11.791 ± 0.187 | 14.973 ± 23.651 |
| lm_batch_likelihood_sentence_3_12d | **18.962 ± 0.489** | 21.107 ± 0.430 | 53.829 ± 28.348 |
| lm_batch_likelihood_sentence_4_4d | **10.554 ± 1.242** | 11.852 ± 0.165 | 35.352 ± 29.476 |
| str_matrix_chain_multiplication_100 | 8.021 ± 0.228 | **7.796 ± 0.119** | 26.935 ± 18.299 |
| str_mps_varying_inner_product_200 | **11.756 ± 0.273** | 15.154 ± 0.231 | 16.429 ± 12.281 |
| str_nw_mera_closed_120 | 310.146 ± 4.654 | **302.529 ± 3.577** | 402.899 ± 27.089 |
| str_nw_mera_open_26 | 193.329 ± 2.267 | **192.600 ± 1.033** | 352.312 ± 22.305 |
| tensornetwork_permutation_focus_step409_316 | 137.439 ± 2.193 | 141.690 ± 1.012 | **131.448 ± 19.086** |
| tensornetwork_permutation_light_415 | 135.437 ± 1.367 | 140.215 ± 1.627 | **119.559 ± 44.121** |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 581.936 ± 10.790 | **524.492 ± 29.697** | - |
| lm_batch_likelihood_brackets_4_4d | **11.302 ± 0.267** | 12.708 ± 0.394 | 14.751 ± 13.813 |
| lm_batch_likelihood_sentence_3_12d | **19.485 ± 0.335** | 20.652 ± 0.362 | 46.534 ± 24.080 |
| lm_batch_likelihood_sentence_4_4d | **11.836 ± 0.308** | 13.375 ± 0.263 | 13.368 ± 10.760 |
| str_matrix_chain_multiplication_100 | **7.353 ± 0.370** | 7.805 ± 0.131 | 8.419 ± 0.239 |
| str_mps_varying_inner_product_200 | **11.328 ± 0.092** | 15.592 ± 0.299 | 13.365 ± 11.631 |
| str_nw_mera_closed_120 | 320.486 ± 2.468 | **315.732 ± 6.818** | 344.773 ± 23.264 |
| str_nw_mera_open_26 | **193.917 ± 1.573** | 194.489 ± 2.752 | 367.635 ± 9.190 |
| tensornetwork_permutation_focus_step409_316 | 136.560 ± 1.751 | 142.500 ± 2.283 | **117.932 ± 2.795** |
| tensornetwork_permutation_light_415 | 135.696 ± 1.032 | 141.111 ± 0.816 | **120.311 ± 2.997** |

**Notes:**
- `-` in tables indicates the instance was skipped (e.g. strided-opteinsum skips operands with duplicate axis labels). Skipped instances are printed as **SKIP** with the reason on stderr.
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM). **strided-rs OpenBLAS** uses OpenBLAS via `cblas-sys`. Julia uses OpenBLAS via libblastrampoline (lbt).
- **strided-rs** and **OMEinsum.jl** (omeinsum_path mode) use the same pre-computed contraction path for fair comparison.
- At 4T, **OpenBLAS outperforms faer** on most BLAS-heavy instances (gm, lm, str_nw_mera, str_matrix_chain), while **faer leads on tensor network instances** (`tensornetwork_permutation_light_415`: 69ms faer vs 74ms blas vs 112ms Julia).
- Tensor network instances show significant improvement with HPTT-based permutation and source-order copy optimizations in strided-rs (e.g. `tensornetwork_permutation_light_415`: 166ms vs 249ms Julia at 1T, 69ms vs 112ms at 4T).

### AMD EPYC 7713P

Environment: AMD EPYC 7713P (64-Core, Zen 3). Median ± IQR (ms) of 15 runs (3 warmup). CPU pinned via `taskset` to same CCD (L3 domain 0: cores 0–7). Rust OpenBLAS: 0.3.29 (local build). Julia 1.12.5, BLAS: lbt (OpenBLAS 0.3.29). strided-rs: [`ea37986`](https://github.com/tensor4all/strided-rs/commit/ea37986). Run date: 2026-02-21.

#### 1 thread (`taskset -c 0`)

##### Strategy: opt_flops

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 5580.061 ± 55.340 | **5401.830 ± 47.513** | - |
| lm_batch_likelihood_brackets_4_4d | **17.239 ± 1.859** | 20.771 ± 1.466 | 29.845 ± 1.570 |
| lm_batch_likelihood_sentence_3_12d | **67.371 ± 5.203** | 77.955 ± 2.621 | 97.359 ± 20.856 |
| lm_batch_likelihood_sentence_4_4d | **17.928 ± 1.476** | 24.252 ± 1.081 | 35.284 ± 38.163 |
| str_matrix_chain_multiplication_100 | **12.804 ± 1.074** | 14.127 ± 1.107 | 17.396 ± 40.098 |
| str_mps_varying_inner_product_200 | **13.210 ± 0.956** | 15.925 ± 1.798 | 41.056 ± 70.141 |
| str_nw_mera_closed_120 | 1446.066 ± 6.914 | **1428.330 ± 9.268** | 1555.513 ± 181.316 |
| str_nw_mera_open_26 | 942.305 ± 25.243 | **921.522 ± 17.555** | 1309.899 ± 41.183 |
| tensornetwork_permutation_focus_step409_316 | **437.128 ± 9.447** | 470.365 ± 7.069 | 516.757 ± 4.771 |
| tensornetwork_permutation_light_415 | **447.583 ± 10.860** | 467.062 ± 10.766 | 519.872 ± 4.235 |

##### Strategy: opt_size

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **1692.086 ± 24.120** | 1700.402 ± 25.158 | - |
| lm_batch_likelihood_brackets_4_4d | **17.062 ± 1.233** | 21.720 ± 1.991 | 34.557 ± 9.437 |
| lm_batch_likelihood_sentence_3_12d | **51.703 ± 3.573** | 62.269 ± 3.211 | 256.770 ± 170.132 |
| lm_batch_likelihood_sentence_4_4d | **19.978 ± 1.678** | 27.441 ± 2.219 | 36.019 ± 42.593 |
| str_matrix_chain_multiplication_100 | 12.951 ± 0.944 | **12.933 ± 0.901** | 18.268 ± 75.934 |
| str_mps_varying_inner_product_200 | **13.551 ± 0.868** | 17.355 ± 1.329 | 36.992 ± 13.746 |
| str_nw_mera_closed_120 | 1282.228 ± 21.226 | **1225.115 ± 10.019** | 1355.883 ± 17.972 |
| str_nw_mera_open_26 | 943.912 ± 14.200 | **934.619 ± 12.021** | 1309.120 ± 89.740 |
| tensornetwork_permutation_focus_step409_316 | **437.922 ± 5.051** | 468.172 ± 7.457 | 494.669 ± 5.075 |
| tensornetwork_permutation_light_415 | **445.542 ± 9.662** | 472.461 ± 9.604 | 491.960 ± 9.025 |

#### 4 threads (`taskset -c 0-3`)

##### Strategy: opt_flops

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 5512.784 ± 86.130 | **3992.797 ± 37.984** | - |
| lm_batch_likelihood_brackets_4_4d | **16.078 ± 1.869** | 17.391 ± 1.153 | 28.501 ± 45.772 |
| lm_batch_likelihood_sentence_3_12d | **41.998 ± 2.088** | 44.292 ± 1.188 | 45.954 ± 1.573 |
| lm_batch_likelihood_sentence_4_4d | **15.635 ± 2.168** | 19.498 ± 2.522 | 62.263 ± 37.157 |
| str_matrix_chain_multiplication_100 | **6.123 ± 2.618** | 9.920 ± 0.783 | 14.161 ± 53.199 |
| str_mps_varying_inner_product_200 | **15.488 ± 0.801** | 19.116 ± 0.042 | 48.746 ± 78.943 |
| str_nw_mera_closed_120 | 657.926 ± 10.324 | **602.683 ± 8.358** | 904.907 ± 120.875 |
| str_nw_mera_open_26 | 406.797 ± 7.727 | **351.793 ± 4.958** | 820.224 ± 21.907 |
| tensornetwork_permutation_focus_step409_316 | 391.689 ± 8.052 | **355.948 ± 5.827** | 478.484 ± 13.308 |
| tensornetwork_permutation_light_415 | 392.532 ± 7.361 | **358.282 ± 5.811** | 480.983 ± 7.111 |

##### Strategy: opt_size

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1479.842 ± 24.958 | **1226.907 ± 13.498** | - |
| lm_batch_likelihood_brackets_4_4d | **15.696 ± 0.270** | 17.842 ± 0.177 | 62.620 ± 11.448 |
| lm_batch_likelihood_sentence_3_12d | **31.610 ± 1.179** | 36.315 ± 0.499 | 263.916 ± 189.462 |
| lm_batch_likelihood_sentence_4_4d | **17.918 ± 1.681** | 22.698 ± 0.228 | 67.977 ± 41.570 |
| str_matrix_chain_multiplication_100 | **6.071 ± 0.057** | 9.543 ± 0.899 | 25.478 ± 85.027 |
| str_mps_varying_inner_product_200 | **15.055 ± 1.423** | 16.503 ± 0.155 | 44.975 ± 12.542 |
| str_nw_mera_closed_120 | 497.744 ± 17.759 | **461.474 ± 5.895** | 683.364 ± 43.011 |
| str_nw_mera_open_26 | 414.631 ± 10.215 | **358.834 ± 5.199** | 819.899 ± 68.758 |
| tensornetwork_permutation_focus_step409_316 | 392.074 ± 7.082 | **352.531 ± 7.305** | 378.413 ± 5.702 |
| tensornetwork_permutation_light_415 | 396.933 ± 4.428 | **355.005 ± 8.554** | 381.927 ± 2.255 |

#### 8 threads (`taskset -c 0-7`)

##### Strategy: opt_flops

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 5965.004 ± 110.575 | **3782.546 ± 52.632** | - |
| lm_batch_likelihood_brackets_4_4d | **15.989 ± 0.126** | 18.307 ± 0.245 | 83.654 ± 10.822 |
| lm_batch_likelihood_sentence_3_12d | **38.293 ± 1.894** | 38.537 ± 0.317 | 41.621 ± 43.829 |
| lm_batch_likelihood_sentence_4_4d | **15.075 ± 0.248** | 19.392 ± 0.188 | 63.517 ± 47.558 |
| str_matrix_chain_multiplication_100 | **9.870 ± 2.674** | 10.435 ± 0.118 | 11.153 ± 30.160 |
| str_mps_varying_inner_product_200 | 20.818 ± 2.882 | **16.208 ± 0.094** | 37.386 ± 32.399 |
| str_nw_mera_closed_120 | 565.986 ± 2.954 | **449.963 ± 0.789** | 798.578 ± 86.413 |
| str_nw_mera_open_26 | 330.528 ± 1.863 | **247.799 ± 0.907** | 737.971 ± 55.092 |
| tensornetwork_permutation_focus_step409_316 | 385.981 ± 6.197 | **343.721 ± 2.568** | 487.492 ± 36.500 |
| tensornetwork_permutation_light_415 | 391.726 ± 8.114 | **343.222 ± 3.441** | 445.071 ± 78.367 |

##### Strategy: opt_size

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1638.325 ± 14.515 | **1151.342 ± 6.523** | - |
| lm_batch_likelihood_brackets_4_4d | **14.910 ± 0.227** | 18.084 ± 0.213 | 70.936 ± 27.905 |
| lm_batch_likelihood_sentence_3_12d | **25.754 ± 0.292** | 32.220 ± 0.346 | 137.867 ± 117.311 |
| lm_batch_likelihood_sentence_4_4d | **18.174 ± 0.182** | 22.580 ± 0.150 | 69.235 ± 31.532 |
| str_matrix_chain_multiplication_100 | 11.265 ± 2.975 | **6.431 ± 0.064** | 11.298 ± 36.080 |
| str_mps_varying_inner_product_200 | **16.028 ± 0.418** | 16.607 ± 0.114 | 42.210 ± 22.552 |
| str_nw_mera_closed_120 | 400.335 ± 3.110 | **313.646 ± 2.226** | 548.971 ± 56.314 |
| str_nw_mera_open_26 | 337.987 ± 1.216 | **255.177 ± 1.293** | 704.828 ± 81.067 |
| tensornetwork_permutation_focus_step409_316 | 388.465 ± 3.823 | **342.513 ± 2.243** | 463.811 ± 45.327 |
| tensornetwork_permutation_light_415 | 389.834 ± 5.560 | **338.575 ± 7.816** | 427.575 ± 30.165 |

**Notes:**
- All benchmarks pinned to cores within the same CCD (L3 domain) via `taskset`. Without pinning, rayon threads migrate across CCDs on AMD EPYC, causing up to 24% performance degradation due to L3 cache misses.
- **strided-rs OpenBLAS** uses OpenBLAS 0.3.29 (locally built, see [Rust setup](../../README.md#rust)). **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM).
- **OMEinsum.jl** uses OpenBLAS 0.3.29 bundled with Julia 1.12.5 via libblastrampoline (lbt).
- `gm_queen5_5_3.wcsp` skipped by Julia due to a `MethodError` (3D tensor not supported by Matrix constructor).
- At 1T, **faer outperforms blas** on most instances due to lower per-call overhead for small GEMMs, while **blas leads on large MERA instances** where GEMM size is larger.
- At 4T/8T, **blas (OpenBLAS 0.3.29) outperforms both faer and OMEinsum.jl** on GEMM-heavy instances. OpenBLAS internal threading provides strong scaling (e.g. `str_nw_mera_open_26`: 1T 922ms → 4T 352ms → 8T 248ms). **faer still leads on small-GEMM workloads** (lm, str_matrix_chain) even at 4T/8T.
- **OMEinsum.jl degrades at 4T/8T** on several instances (e.g. `lm_brackets` 1T: 30ms → 8T: 84ms). This is likely because Julia's `permutedims!` is not parallelized, so tensor permutation becomes a serial bottleneck while BLAS threads compete for cache.
- faer 4T/8T shows regression on `gm_queen5_5_3.wcsp` (5580ms → 5965ms at 8T) due to rayon threading overhead on many small GEMMs with negligible parallelizable work.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) — tensor network contraction benchmarks (PyTorch, OMEinsum.jl)
