# =============================================================================
# Julia benchmark runner for the strided-rs benchmark suite
# =============================================================================
#
# Compares OMEinsum.jl against the same einsum instances and contraction paths
# used by the Rust runner (strided-opteinsum). Instances are loaded from
# data/instances/*.json (metadata only; tensors are zero-filled at runtime).
#
# Modes:
#   - omeinsum_path: same pre-computed path as Rust (fair kernel comparison).
#   - omeinsum_opt:  OMEinsum's optimize_code with TreeSA() (optimizer path).
#
# Strategies (from JSON): opt_flops (minimize FLOPS) and opt_size (minimize
# largest intermediate). Each strategy has its own contraction path.
#
# =============================================================================

using Printf
using Statistics: median
using LinearAlgebra: BLAS
using OMEinsum
using OMEinsumContractionOrders
using JSON

# ---------------------------------------------------------------------------
# Format string parsing
# ---------------------------------------------------------------------------
# Splits an einsum format string "ab,bc->ac" into per-operand index lists
# and the output index list. Used to build DynamicEinCode and to drive
# path-based execution.

function parse_format_string(s::AbstractString)
    parts = split(s, "->")
    input_str = parts[1]
    output_str = parts[2]
    input_indices = [collect(String(op)) for op in split(input_str, ",")]
    output_indices = collect(String(output_str))
    return input_indices, output_indices
end

# ---------------------------------------------------------------------------
# OMEinsum: Pre-computed path execution (omeinsum_path mode)
# ---------------------------------------------------------------------------
# Uses the exact same contraction path as the Rust runner (from JSON path).
# Each step [i, j] contracts the i-th and j-th tensors in the current list
# (opt_einsum/cotengra convention; indices are 0-based in JSON, 1-based here).
# Output indices of each pairwise contraction are derived from the final
# output and from indices still needed by remaining tensors.

"""
    run_with_path(tensors, input_indices, output_indices, path)

Execute the full einsum by following the pre-computed `path`. Each step
contracts two tensors and appends the result. The final tensor is permuted
to match `output_indices` if necessary.
"""
function run_with_path(tensors, input_indices, output_indices, path)
    current = collect(zip(tensors, input_indices))

    for (a, b) in path
        i, j = min(a, b) + 1, max(a, b) + 1  # 0-indexed → 1-indexed

        tj, ij = current[j]
        deleteat!(current, j)
        ti, ii = current[i]
        deleteat!(current, i)

        # Determine which indices to keep: those needed by remaining tensors or final output
        remaining_needed = Set{Char}(output_indices)
        for (_, idx) in current
            union!(remaining_needed, idx)
        end
        # Preserve first-seen order from ii then ij
        pair_output = Char[]
        seen = Set{Char}()
        for c in Iterators.flatten((ii, ij))
            if c in remaining_needed && !(c in seen)
                push!(pair_output, c)
                push!(seen, c)
            end
        end

        code = DynamicEinCode([ii, ij], pair_output)
        result = einsum(code, (ti, tj))
        push!(current, (result, pair_output))
    end

    final_tensor, final_ids = current[1]

    # Permute to match expected output order if needed
    if final_ids != output_indices
        perm = [findfirst(==(c), final_ids) for c in output_indices]
        final_tensor = permutedims(final_tensor, perm)
    end

    return final_tensor
end

# ---------------------------------------------------------------------------
# OMEinsum: Optimizer-based execution (omeinsum_opt mode)
# ---------------------------------------------------------------------------
# Builds a DynamicEinCode and optimizes it with TreeSA(); the contraction
# order is chosen by OMEinsum, not by the JSON path. Useful for comparing
# optimizer quality vs. the pre-computed path.

"""
    run_with_optimizer(tensors, input_indices, output_indices, shapes)

Build size_dict from shapes, optimize the contraction order with TreeSA(),
then execute. The optimization cost is not included in benchmark timing
(opt_code is built once per instance in benchmark_instance).
"""
function run_with_optimizer(tensors, input_indices, output_indices, shapes)
    code = DynamicEinCode(input_indices, output_indices)

    # Build size dict from indices and shapes
    size_dict = Dict{Char,Int}()
    for (idx, shape) in zip(input_indices, shapes)
        for (c, s) in zip(idx, shape)
            size_dict[c] = s
        end
    end

    opt_code = optimize_code(code, size_dict, TreeSA())
    return opt_code(tensors...)
end

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
# create_tensors: build zero-filled arrays from JSON shapes (float64 or complex128).
# benchmark_instance: run warmup (2 runs), then 5 timed runs; return median (ms)
# or (nothing, error_string) if the run fails (e.g. unsupported index pattern).

function create_tensors(shapes, dtype::AbstractString)
    if dtype == "float64"
        return [zeros(Float64, s...) for s in shapes]
    elseif dtype == "complex128"
        return [zeros(ComplexF64, s...) for s in shapes]
    else
        error("unsupported dtype: $dtype")
    end
end

function benchmark_instance(instance, strategy::AbstractString, mode::Symbol)
    # Julia is column-major; use row-major format_string and shapes from JSON
    # (same as NumPy convention; shapes match the index order in format_string).
    format_str = instance["format_string"]
    shapes = [Tuple(s) for s in instance["shapes"]]
    dtype = instance["dtype"]
    path_meta = instance["paths"][strategy]
    path = [Tuple(p) for p in path_meta["path"]]

    input_indices, output_indices = parse_format_string(format_str)
    @assert length(input_indices) == instance["num_tensors"]

    # Build the computation closure (takes pre-allocated tensors as argument).
    # Tensor allocation is excluded from timing to match the Rust runner.
    run_fn = if mode == :omeinsum_path
        # Same path as Rust: fair comparison of kernel performance.
        (tensors) -> run_with_path(tensors, input_indices, output_indices, path)
    elseif mode == :omeinsum_opt
        # OMEinsum optimizer (TreeSA); path built once, not timed.
        code = DynamicEinCode(input_indices, output_indices)
        size_dict = Dict{Char,Int}()
        for (idx, shape) in zip(input_indices, shapes)
            for (c, s) in zip(idx, shape)
                size_dict[c] = s
            end
        end
        opt_code = optimize_code(code, size_dict, TreeSA())
        (tensors) -> opt_code(tensors...)
    else
        error("unknown mode: $mode")
    end

    # Warmup (2 runs); failures (e.g. MethodError) return (nothing, error_string)
    try
        for _ in 1:2
            tensors = create_tensors(shapes, dtype)
            run_fn(tensors)
        end
    catch e
        return nothing, string(e)
    end

    # Timed runs: 5 runs, report median in ms
    num_runs = 5
    durations = Float64[]
    for _ in 1:num_runs
        tensors = create_tensors(shapes, dtype)
        t0 = time_ns()
        result = run_fn(tensors)
        elapsed = (time_ns() - t0) / 1e6  # ns -> ms
        push!(durations, elapsed)
    end

    return median(durations), nothing
end

# ---------------------------------------------------------------------------
# Main: load instances, optional filter, run all mode × strategy combinations
# ---------------------------------------------------------------------------

function load_instances()
    data_dir = joinpath(@__DIR__, "..", "data", "instances")
    json_files = sort(filter(f -> endswith(f, ".json"), readdir(data_dir)))
    return [JSON.parsefile(joinpath(data_dir, f)) for f in json_files]
end

function main()
    data_dir = joinpath(@__DIR__, "..", "data", "instances")
    instances = load_instances()

    # Optional: run only one instance (e.g. BENCH_INSTANCE=str_nw_mera_closed_120)
    filter_name = get(ENV, "BENCH_INSTANCE", "")
    if !isempty(filter_name)
        instances = filter(i -> i["name"] == filter_name, instances)
        if isempty(instances)
            @error "BENCH_INSTANCE=$filter_name: no matching instance found"
            exit(1)
        end
    end

    println("Julia einsum benchmark suite")
    println("==================================")
    println("Loaded $(length(instances)) instances from $data_dir")
    println("Julia threads: $(Threads.nthreads()), BLAS threads: $(BLAS.get_num_threads()), BLAS vendor: $(BLAS.vendor())")
    println("OMP_NUM_THREADS=$(get(ENV, "OMP_NUM_THREADS", "unset")), JULIA_NUM_THREADS=$(get(ENV, "JULIA_NUM_THREADS", "unset"))")
    println("Timing: median of 5 runs (2 warmup)")

    strategies = ["opt_flops", "opt_size"]
    modes = [:omeinsum_path, :omeinsum_opt]  # same path as Rust; optimizer path

    for mode in modes
        for strategy in strategies
            println()
            println("Mode: $mode / Strategy: $strategy")
            @printf("%-50s %8s %10s %12s %12s\n",
                "Instance", "Tensors", "log10FLOPS", "log2SIZE", "Median (ms)")
            println("-"^96)

            for instance in instances
                path_meta = instance["paths"][strategy]
                median_ms, err = benchmark_instance(instance, strategy, mode)
                if median_ms === nothing
                    @printf("%-50s %8d %10.2f %12.2f %12s\n",
                        instance["name"],
                        instance["num_tensors"],
                        path_meta["log10_flops"],
                        path_meta["log2_size"],
                        "SKIP")
                    println("  reason: $err")
                else
                    @printf("%-50s %8d %10.2f %12.2f %12.3f\n",
                        instance["name"],
                        instance["num_tensors"],
                        path_meta["log10_flops"],
                        path_meta["log2_size"],
                        median_ms)
                end
            end
        end
    end
end

main()
