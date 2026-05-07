using Pkg
Pkg.activate("Gnn_environment")
cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")
include("../src/InferenceUtils.jl")

using .InferenceUtils
using .Self_GNN_PM
using Flux
using GraphNeuralNetworks
using JSON
using MLUtils: DataLoader
using BSON
using Glob

const PATH_DATA = "test_systems/data-opf/"
const RESULTS_EPOCHS_DIR = "results_epochs"
const OUTPUT_DIR = "results/model_selection"
const BATCH_PROGRESS_PATH = joinpath(OUTPUT_DIR, "evaluation_progress_all_systems.json")
const FROM_SAMPLES = 1
const TO_SAMPLES = 5
const NNFEATURES = 6
const NEFEATURES = 5
const HEADS = 1
const HIDDEN_DIM = 1024

function load_checkpoint_model!(model, model_file::String)
    checkpoint = BSON.load(model_file)

    for key in (:model_semisupervised, :model_selfsupervised, :model_fullsupervised, :model_epoch)
        if haskey(checkpoint, key)
            Flux.loadparams!(model, Flux.params(checkpoint[key]))
            return model
        end
    end

    error("No supported model key found in $model_file")
end

function extract_epoch(model_file::String, prefix::String)
    pattern = Regex("^" * prefix * "(\\d+)\\.bson\$")
    match_result = match(pattern, basename(model_file))
    match_result === nothing && return nothing
    return parse(Int, match_result.captures[1])
end

function build_loader_for_system(system_name::String; from_samples::Int=FROM_SAMPLES, to_samples::Int=TO_SAMPLES)
    ref, x, e, y, _ = process_system_powermodels(system_name, PATH_DATA)
    graphs = data_graph_powermodels(ref, x, e, y, from_samples, to_samples, GNNGraph[])
    isempty(graphs) && error("No graphs available for $system_name in samples $from_samples:$to_samples")

    graph_loader = DataLoader(
        graphs,
        batchsize=1,
        shuffle=false,
        collate=xs -> GraphWithRef(batch(xs), ref)
    )

    probe_batch = nothing
    for batch in graph_loader
        probe_batch = batch
        break
    end
    probe_batch === nothing && error("Unable to build probe batch for $system_name")

    dc_vals, _, _ = InferenceUtils.compute_dc_cost(PATH_DATA, system_name, ref)
    init_val = ACPM(probe_batch.data)["results"]["solution"]

    return graph_loader, dc_vals, init_val
end

function build_model(model_type::Symbol)
    if model_type == :self || model_type == :semi
        return Self_GNN_PM.Self_supervised_GNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    end

    error("Unsupported model type: $model_type")
end

function evaluate_checkpoint(model_type::Symbol, model_file::String, graph_loader, dc_vals, init_val;
    from_samples::Int=FROM_SAMPLES, to_samples::Int=TO_SAMPLES)
    model = build_model(model_type)
    load_checkpoint_model!(model, model_file)
    Flux.testmode!(model, true)

    pred, real, _ = InferenceUtils.evaluate_best_model_HF(model, graph_loader, init_val)

    df = InferenceUtils.summarize_results(real, pred, dc_vals[from_samples:to_samples, :])
    mean_error = Float64(df[!, 2][3])

    return Dict(
        "mean_error" => mean_error,
        "summary_table" => Dict(
            string(df[1, 1]) => Float64(df[1, 2]),
            string(df[2, 1]) => Float64(df[2, 2]),
            string(df[3, 1]) => Float64(df[3, 2])
        )
    )
end

function build_model_result_summary(model_type::Symbol, evaluations, errors=Dict{String,Any}[])
    sorted_evaluations = sort(copy(evaluations), by=x -> x["mean_error"])
    best_model = isempty(sorted_evaluations) ? nothing : sorted_evaluations[1]

    return Dict(
        "model_type" => String(model_type),
        "num_models_evaluated" => length(evaluations),
        "num_errors" => length(errors),
        "evaluations_by_epoch" => evaluations,
        "sorted_by_mean_error" => sorted_evaluations,
        "errors" => errors,
        "best_model" => best_model,
        "best_epoch" => isnothing(best_model) ? nothing : best_model["epoch"],
        "best_mean_error" => isnothing(best_model) ? nothing : best_model["mean_error"],
        "best_model_path" => isnothing(best_model) ? nothing : best_model["model_path"]
    )
end

function collect_epoch_results(system_name::String, model_type::Symbol, graph_loader, dc_vals, init_val;
    start_epoch::Int=1,
    end_epoch::Int=0,
    from_samples::Int=FROM_SAMPLES,
    to_samples::Int=TO_SAMPLES,
    on_progress=nothing)
    model_dir = joinpath(RESULTS_EPOCHS_DIR, system_name)
    prefix = model_type == :self ? "Model_self_epoch" : "Model_semi_epoch"
    model_files = Glob.glob(prefix * "*.bson", model_dir)

    checkpoints = Tuple{Int,String}[]
    for model_file in model_files
        epoch = extract_epoch(model_file, prefix)
        epoch === nothing && continue
        if epoch >= start_epoch && (end_epoch == 0 || epoch <= end_epoch)
            push!(checkpoints, (epoch, model_file))
        end
    end
    sort!(checkpoints, by=first)

    evaluations = Dict{String,Any}[]
    errors = Dict{String,Any}[]
    for (epoch, model_file) in checkpoints
        println("Evaluating $(uppercase(String(model_type))) $system_name epoch $epoch")
        try
            checkpoint_result = evaluate_checkpoint(
                model_type,
                model_file,
                graph_loader,
                dc_vals,
                init_val;
                from_samples=from_samples,
                to_samples=to_samples
            )

            push!(evaluations, Dict(
                "epoch" => epoch,
                "mean_error" => checkpoint_result["mean_error"],
                "model_path" => model_file,
                "metrics" => checkpoint_result["summary_table"]
            ))
        catch err
            push!(errors, Dict(
                "epoch" => epoch,
                "model_path" => model_file,
                "error" => sprint(showerror, err)
            ))
            println("Failed evaluating $(uppercase(String(model_type))) $system_name epoch $epoch")
            println(sprint(showerror, err))
        end

        if !isnothing(on_progress)
            on_progress(build_model_result_summary(model_type, evaluations, errors))
        end
    end

    return build_model_result_summary(model_type, evaluations, errors)
end

function evaluate_system(system_name::String;
    eval_self::Bool=true,
    eval_semi::Bool=true,
    start_epoch::Int=1,
    end_epoch_self::Int=0,
    end_epoch_semi::Int=0,
    from_samples::Int=FROM_SAMPLES,
    to_samples::Int=TO_SAMPLES,
    progress_path=nothing)
    graph_loader, dc_vals, init_val = build_loader_for_system(
        system_name;
        from_samples=from_samples,
        to_samples=to_samples
    )

    system_result = Dict(
        "system_name" => system_name,
        "from_samples" => from_samples,
        "to_samples" => to_samples,
        "status" => "running"
    )

    save_progress = function ()
        if !isnothing(progress_path)
            save_json(progress_path, system_result)
        end
    end

    save_progress()

    if eval_self
        system_result["self"] = collect_epoch_results(
            system_name,
            :self,
            graph_loader,
            dc_vals,
            init_val;
            start_epoch=start_epoch,
            end_epoch=end_epoch_self,
            from_samples=from_samples,
            to_samples=to_samples,
            on_progress=model_result -> begin
                system_result["self"] = model_result
                save_progress()
            end
        )
        save_progress()
    end

    if eval_semi
        system_result["semi"] = collect_epoch_results(
            system_name,
            :semi,
            graph_loader,
            dc_vals,
            init_val;
            start_epoch=start_epoch,
            end_epoch=end_epoch_semi,
            from_samples=from_samples,
            to_samples=to_samples,
            on_progress=model_result -> begin
                system_result["semi"] = model_result
                save_progress()
            end
        )
        save_progress()
    end

    system_result["status"] = "completed"
    save_progress()
    return system_result
end

function summarize_best_models(all_results)
    summary = Dict(
        "self" => Dict{String,Any}[],
        "semi" => Dict{String,Any}[]
    )

    for result in all_results
        for model_type in ("self", "semi")
            haskey(result, model_type) || continue
            best_model = result[model_type]["best_model"]
            best_model === nothing && continue

            push!(summary[model_type], Dict(
                "system_name" => result["system_name"],
                "best_epoch" => best_model["epoch"],
                "best_mean_error" => best_model["mean_error"],
                "best_model_path" => best_model["model_path"]
            ))
        end
    end

    sort!(summary["self"], by=x -> x["best_mean_error"])
    sort!(summary["semi"], by=x -> x["best_mean_error"])
    return summary
end

function discover_systems(results_dir::String=RESULTS_EPOCHS_DIR)
    isdir(results_dir) || error("Results directory not found: $results_dir")
    return sort(filter(name -> isdir(joinpath(results_dir, name)), readdir(results_dir)))
end

function save_json(path::String, data)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
end

function load_best_model_info(system_name::String, model_type::Symbol;
    summary_dir::String=OUTPUT_DIR)
    per_system_path = joinpath(summary_dir, "evaluation_results_$(system_name).json")
    isfile(per_system_path) || error("Evaluation summary not found for $system_name at $per_system_path")

    summary_data = JSON.parsefile(per_system_path)
    model_key = String(model_type)
    haskey(summary_data, model_key) || error("No $model_key results found for $system_name")

    best_model = summary_data[model_key]["best_model"]
    best_model === nothing && error("No best $model_key model stored for $system_name")

    return best_model
end

function load_best_model(system_name::String, model_type::Symbol;
    summary_dir::String=OUTPUT_DIR)
    best_model = load_best_model_info(system_name, model_type; summary_dir=summary_dir)
    model_path = best_model["model_path"]
    isfile(model_path) || error("Best model checkpoint not found at $model_path")

    model = build_model(model_type)
    load_checkpoint_model!(model, model_path)
    Flux.testmode!(model, true)

    return model, best_model
end

function evaluate_all_systems(;
    systems=discover_systems(),
    eval_self::Bool=true,
    eval_semi::Bool=true,
    start_epoch::Int=1,
    end_epoch_self::Int=0,
    end_epoch_semi::Int=0,
    from_samples::Int=FROM_SAMPLES,
    to_samples::Int=TO_SAMPLES)
    mkpath(OUTPUT_DIR)

    all_results = Dict{String,Any}[]
    batch_progress = Dict(
        "systems" => systems,
        "from_samples" => from_samples,
        "to_samples" => to_samples,
        "completed_systems" => String[],
        "failed_systems" => Dict{String,Any}[],
        "results" => all_results,
        "best_models_by_error" => Dict("self" => Any[], "semi" => Any[]),
        "status" => "running"
    )
    save_json(BATCH_PROGRESS_PATH, batch_progress)

    for system_name in systems
        println("========== Evaluating system $system_name ==========")
        per_system_path = joinpath(OUTPUT_DIR, "evaluation_results_$(system_name).json")
        try
            system_result = evaluate_system(
                system_name;
                eval_self=eval_self,
                eval_semi=eval_semi,
                start_epoch=start_epoch,
                end_epoch_self=end_epoch_self,
                end_epoch_semi=end_epoch_semi,
                from_samples=from_samples,
                to_samples=to_samples,
                progress_path=per_system_path
            )
            push!(all_results, system_result)
            push!(batch_progress["completed_systems"], system_name)
        catch err
            failure = Dict(
                "system_name" => system_name,
                "error" => sprint(showerror, err)
            )
            push!(batch_progress["failed_systems"], failure)
            save_json(per_system_path, Dict(
                "system_name" => system_name,
                "status" => "error",
                "error" => sprint(showerror, err)
            ))
            println("Failed evaluating system $system_name")
            println(sprint(showerror, err))
        end

        batch_progress["best_models_by_error"] = summarize_best_models(all_results)
        save_json(BATCH_PROGRESS_PATH, batch_progress)
    end

    summary = summarize_best_models(all_results)
    batch_output = Dict(
        "systems" => systems,
        "from_samples" => from_samples,
        "to_samples" => to_samples,
        "results" => all_results,
        "best_models_by_error" => summary
    )

    save_json(joinpath(OUTPUT_DIR, "evaluation_results_all_systems.json"), batch_output)
    batch_progress["best_models_by_error"] = summary
    batch_progress["status"] = "completed"
    save_json(BATCH_PROGRESS_PATH, batch_progress)
    println("Saved batch evaluation to $(joinpath(OUTPUT_DIR, "evaluation_results_all_systems.json"))")

    return batch_output
end


evaluate_all_systems()
