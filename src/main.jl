using Printf
using Statistics: median
using LinearAlgebra: BLAS
using OMEinsum
using OMEinsumContractionOrders
using TensorOperations
using JSON

# ---------------------------------------------------------------------------
# Format string parsing
# ---------------------------------------------------------------------------

function parse_format_string(s::AbstractString)
    parts = split(s, "->")
    input_str = parts[1]
    output_str = parts[2]
    input_indices = [collect(String(op)) for op in split(input_str, ",")]
    output_indices = collect(String(output_str))
    return input_indices, output_indices
end

# ---------------------------------------------------------------------------
# OMEinsum: Pre-computed path execution
# ---------------------------------------------------------------------------

"""
Execute einsum following a pre-computed contraction path (opt_einsum convention).
Each step [i, j] contracts tensors at those indices in the current list.
Higher index is removed first; result is appended to the end.
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
# OMEinsum: optimizer execution
# ---------------------------------------------------------------------------

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
# TensorOperations: Full network contraction via ncon
# ---------------------------------------------------------------------------

"""
Execute einsum using TensorOperations.ncon for the full tensor network.
Contraction order is determined by TensorOperations internally.
"""
function run_with_tensorops_ncon(tensors, input_indices, output_indices)
    char_to_int = Dict{Char,Int}()

    # Output indices get negative labels (-1, -2, ...)
    for (k, c) in enumerate(output_indices)
        char_to_int[c] = -k
    end

    # Contracted indices get positive labels
    # Indices appearing in only one tensor and not in output get extra negative labels
    pos = 0
    extra_neg = length(output_indices)
    extra_output = Char[]

    for idx in input_indices
        for c in idx
            if !haskey(char_to_int, c)
                count = sum(c in other_idx for other_idx in input_indices)
                if count >= 2
                    pos += 1
                    char_to_int[c] = pos
                else
                    extra_neg += 1
                    char_to_int[c] = -extra_neg
                    push!(extra_output, c)
                end
            end
        end
    end

    network = [Int[char_to_int[c] for c in idx] for idx in input_indices]

    result = ncon(tensors, network)

    # Sum over extra dimensions (indices in only one tensor, not in output)
    if !isempty(extra_output)
        extra_dims = Tuple(length(output_indices) + k for k in 1:length(extra_output))
        result = dropdims(sum(result, dims=extra_dims), dims=extra_dims)
    end

    return result
end

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

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
    # Julia is column-major natively: use original format_string and shapes
    format_str = instance["format_string"]
    shapes = [Tuple(s) for s in instance["shapes"]]
    dtype = instance["dtype"]
    path_meta = instance["paths"][strategy]
    path = [Tuple(p) for p in path_meta["path"]]

    input_indices, output_indices = parse_format_string(format_str)
    @assert length(input_indices) == instance["num_tensors"]

    run_fn = if mode == :omeinsum_path
        () -> begin
            tensors = create_tensors(shapes, dtype)
            run_with_path(tensors, input_indices, output_indices, path)
        end
    elseif mode == :omeinsum_opt
        # Pre-optimize once (optimization cost not included in timing)
        code = DynamicEinCode(input_indices, output_indices)
        size_dict = Dict{Char,Int}()
        for (idx, shape) in zip(input_indices, shapes)
            for (c, s) in zip(idx, shape)
                size_dict[c] = s
            end
        end
        opt_code = optimize_code(code, size_dict, TreeSA())
        () -> begin
            tensors = create_tensors(shapes, dtype)
            opt_code(tensors...)
        end
    elseif mode == :tensorops
        () -> begin
            tensors = create_tensors(shapes, dtype)
            run_with_tensorops_ncon(tensors, input_indices, output_indices)
        end
    else
        error("unknown mode: $mode")
    end

    # Warmup
    try
        for _ in 1:2
            run_fn()
        end
    catch e
        return nothing, string(e)
    end

    # Timed runs
    num_runs = 5
    durations = Float64[]
    for _ in 1:num_runs
        t0 = time_ns()
        result = run_fn()
        elapsed = (time_ns() - t0) / 1e6  # ms
        push!(durations, elapsed)
    end

    return median(durations), nothing
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function load_instances()
    data_dir = joinpath(@__DIR__, "..", "data", "instances")
    json_files = sort(filter(f -> endswith(f, ".json"), readdir(data_dir)))
    return [JSON.parsefile(joinpath(data_dir, f)) for f in json_files]
end

function main()
    data_dir = joinpath(@__DIR__, "..", "data", "instances")
    instances = load_instances()

    println("Julia einsum benchmark suite")
    println("==================================")
    println("Loaded $(length(instances)) instances from $data_dir")
    println("Julia threads: $(Threads.nthreads()), BLAS threads: $(BLAS.get_num_threads()), BLAS vendor: $(BLAS.vendor())")
    println("OMP_NUM_THREADS=$(get(ENV, "OMP_NUM_THREADS", "unset")), JULIA_NUM_THREADS=$(get(ENV, "JULIA_NUM_THREADS", "unset"))")
    println("Timing: median of 5 runs (2 warmup)")

    strategies = ["opt_flops", "opt_size"]
    modes = [:omeinsum_path, :omeinsum_opt, :tensorops]

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
