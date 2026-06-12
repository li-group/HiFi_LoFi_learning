using Pkg

Pkg.activate("Gnn_environment")
cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")
include("../src/InferenceUtils.jl")
using .Self_GNN_PM
using .InferenceUtils

using BSON
using BSON: @save
using DataFrames
using Dates
using Flux
using Flux.Optimise: update!
using GraphNeuralNetworks
using GraphNeuralNetworks: batch
using JSON
using MLUtils: DataLoader
using Random

const NNFEATURES = 6
const NEFEATURES = 5
const HEADS = 1
const HIDDEN_DIM = 1024

const BATCHSIZE = 1
const INIT_EPOCHS = 100
const TRAIN_EPOCHS = 5000
const USE_TEST_DATA = false
const TRAIN_SAMPLE_RANGE = 1:1
const RANDOM_SEED = 20

#? Should we change the range of times?
const SYSTEM_TIME_BUDGETS = [
    # ("pglib_opf_case500_goc", 5),
    # ("pglib_opf_case1354_pegase", 30),
    # ("pglib_opf_case2000_goc", 120),
    # ("pglib_opf_case4661_sdet", 120),
    # ("pglib_opf_case13659_pegase", 180),
    ("pglib_opf_case30000_goc", 240),
]

# const SYSTEM_TIME_BUDGETS = [
#     ("pglib_opf_case500_goc", 5),
#     # ("pglib_opf_case1354_pegase", 60),
#     # ("pglib_opf_case2000_goc", 120),
#     # ("pglib_opf_case4661_sdet", 120),
#     # ("pglib_opf_case13659_pegase", 240),
#     # ("pglib_opf_case30000_goc", 360),
# ]

const PATH_DATA = "test_systems/data-opf/"
const BATCH_PROGRESS_PATH = "results/self_batch_progress.json"
const MODEL_SELECTION_DIR = "results/model_selection"
const HF_EVAL_FROM_SAMPLE = 1
const HF_EVAL_TO_SAMPLE = 5
const SAVE_ALL_EPOCH_MODELS = false

function build_loader(ref, x, e, y, sample_range; batchsize::Int, shuffle::Bool)
    graphs = data_graph_powermodels(
        ref,
        x,
        e,
        y,
        first(sample_range),
        last(sample_range),
        GNNGraph[]
    )

    return DataLoader(
        graphs,
        batchsize=batchsize,
        shuffle=shuffle,
        collate=xs -> GraphWithRef(batch(xs), ref)
    )
end

function relative_objective_error(predicted_obj, true_obj)
    scale = max(abs(true_obj), 1e-8)
    return abs(predicted_obj - true_obj) / scale * 100
end

function pretrain_initialization!(model, train_loader)
    opt = Flux.AdamW(1e-4)
    ps = Flux.params(model)

    for epoch in 1:INIT_EPOCHS
        for batch in train_loader
            loss, grads = Flux.withgradient(ps) do
                loss_fn(model, batch)
            end
            update!(opt, ps, grads)
            println("Init epoch $epoch - Loss: $loss")
        end
    end

    return model
end

function save_batch_progress(progress_dict)
    mkpath(dirname(BATCH_PROGRESS_PATH))
    open(BATCH_PROGRESS_PATH, "w") do io
        JSON.print(io, progress_dict)
    end
end

function save_json(path::String, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload)
    end
end

function load_json(path::String)
    return JSON.parsefile(path; use_mmap=!Sys.iswindows())
end

function build_hf_eval_context(ref, x, e, y, system_name::String, n_samples::Int;
    from_samples::Int=HF_EVAL_FROM_SAMPLE,
    to_samples::Int=HF_EVAL_TO_SAMPLE)
    eval_from = from_samples
    eval_to = min(to_samples, n_samples)
    eval_to >= eval_from || error("No samples available for HF evaluation in $system_name")

    graphs = data_graph_powermodels(ref, x, e, y, eval_from, eval_to, GNNGraph[])
    isempty(graphs) && error("No HF evaluation graphs available for $system_name")

    graph_loader = DataLoader(
        graphs,
        batchsize=1,
        shuffle=false,
        collate=xs -> GraphWithRef(batch(xs), ref)
    )

    probe_batch = nothing
    for graph in graph_loader
        probe_batch = graph
        break
    end
    probe_batch === nothing && error("Unable to build HF evaluation probe batch for $system_name")

    dc_vals, _, _ = InferenceUtils.compute_dc_cost(PATH_DATA, system_name, ref)
    # init_val = ACPM(probe_batch.data)["results"]["solution"]

    return Dict(
        "graph_loader" => graph_loader,
        "dc_vals" => dc_vals,
        # "init_val" => init_val,
        "from_samples" => eval_from,
        "to_samples" => eval_to
    )
end

function build_hf_eval_model(train_model)
    eval_model = Self_supervised_GNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    # Flux.loadparams!(eval_model, Flux.params(train_model))
    #! use it instead loadparams!
    for (p, x) in zip(Flux.params(eval_model), Flux.params(train_model))
        size(p) == size(x) ||
            error("Expected param size $(size(p)), got $(size(x))")
        copyto!(p, x)
    end




    # Flux.testmode!(eval_model, true)
    return eval_model
end

function evaluate_current_hf_model(train_model, eval_context)
    eval_model = build_hf_eval_model(train_model)
    from_samples = eval_context["from_samples"]
    to_samples = eval_context["to_samples"]

    pred, real, _ = InferenceUtils.evaluate_best_model_HF(
        eval_model,
        eval_context["graph_loader"]
    )
    df = InferenceUtils.summarize_results(real, pred, eval_context["dc_vals"][from_samples:to_samples, :])

    return Dict(
        "mean_error" => Float64(df[!, 2][3]),
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

function update_model_selection_summary!(system_name::String, model_type::Symbol, model_summary;
    from_samples::Int,
    to_samples::Int,
    status::String)
    summary_path = joinpath(MODEL_SELECTION_DIR, "evaluation_results_$(system_name).json")
    system_summary = isfile(summary_path) ? load_json(summary_path) : Dict{String,Any}()

    system_summary["system_name"] = system_name
    system_summary["from_samples"] = from_samples
    system_summary["to_samples"] = to_samples
    system_summary["status"] = status
    system_summary[String(model_type)] = model_summary

    save_json(summary_path, system_summary)
    return summary_path
end

function train_single_system(system_name::String, total_minutes::Int)
    Random.seed!(RANDOM_SEED)

    ref_train, x_train, e_train, y_train, _ = process_system_powermodels(
        system_name,
        PATH_DATA;
        use_test_data=USE_TEST_DATA
    )

    n_samples = size(y_train, 2)
    train_range = TRAIN_SAMPLE_RANGE
    last(train_range) <= n_samples || error("Training range exceeds available samples.")

    println("System: $system_name")
    println("Training samples: $(first(train_range)):$(last(train_range))")

    train_loader = build_loader(ref_train, x_train, e_train, y_train, train_range; batchsize=BATCHSIZE, shuffle=false)
    eval_context = build_hf_eval_context(ref_train, x_train, e_train, y_train, system_name, n_samples)

    probe_batch = nothing
    for batch in train_loader
        probe_batch = batch
        break
    end
    probe_batch === nothing && error("Training loader is empty.")

    model_initialization = InizitilizationGNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    pretrain_initialization!(model_initialization, train_loader)
    model_initialization(probe_batch)

    model_self = Self_supervised_GNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    # Flux.loadparams!(model_self, Flux.params(model_initialization))
    #! use it instead loadparams!
    for (p, x) in zip(Flux.params(model_self), Flux.params(model_initialization))
        size(p) == size(x) ||
            error("Expected param size $(size(p)), got $(size(x))")
        copyto!(p, x)
    end

    opt = Flux.AdamW(1e-6)
    ps = Flux.params(model_self)

    empty!(fwd_times_dc)
    empty!(bwd_times_dc)
    empty!(fwd_times_ac)
    empty!(bwd_times_ac)

    # results_ac_init = ACPM(probe_batch.data)["results"]["solution"]
    model_self(probe_batch)

    epoch_losses = Float64[]
    epoch_times = Float64[]
    epoch_mean_errors = Float64[]
    evaluation_times = Float64[]
    evaluation_errors = Dict{String,Any}[]
    stop_epoch = 0
    best_epoch = 0
    best_mean_error = Inf

    epoch_model_dir = joinpath("results_epochs", system_name)
    best_model_path = joinpath(epoch_model_dir, "Model_self_best.bson")
    metrics_path = joinpath("results", "times_metrics_self_$(system_name).json")
    mkpath(epoch_model_dir)
    mkpath(dirname(metrics_path))

    model_evaluations = Dict{String,Any}[]
    selection_summary_path = update_model_selection_summary!(
        system_name,
        :self,
        build_model_result_summary(:self, model_evaluations, evaluation_errors);
        from_samples=eval_context["from_samples"],
        to_samples=eval_context["to_samples"],
        status="running"
    )

    for epoch in 1:TRAIN_EPOCHS
        total_loss = 0.0
        num_batches = 0

        epoch_time = @elapsed begin
            # Flux.trainmode!(model_self)
            for batch in train_loader
                batch_loss, grads = Flux.withgradient(ps) do
                    model_self(batch)
                end
                update!(opt, ps, grads)
                total_loss += batch_loss
                num_batches += 1
            end
        end

        num_batches > 0 || error("Training loader is empty.")
        epoch_avg_loss = total_loss / num_batches

        push!(epoch_losses, epoch_avg_loss)
        push!(epoch_times, epoch_time)

        println(
            "Epoch $epoch | train loss = $(round(epoch_avg_loss, sigdigits=6))" *
            " | time = $(round(epoch_time, digits=4)) s"
        )

        epoch_result = nothing
        eval_start = time()
        try
            epoch_result = evaluate_current_hf_model(model_self, eval_context)
            eval_time = time() - eval_start
            push!(epoch_mean_errors, epoch_result["mean_error"])
            push!(evaluation_times, eval_time)

            evaluation_entry = Dict(
                "epoch" => epoch,
                "mean_error" => epoch_result["mean_error"],
                "evaluation_time" => eval_time,
                "model_path" => nothing,
                "metrics" => epoch_result["summary_table"],
                "saved_as_best" => false
            )

            if epoch_result["mean_error"] < best_mean_error
                best_mean_error = epoch_result["mean_error"]
                best_epoch = epoch
                evaluation_entry["model_path"] = best_model_path
                evaluation_entry["saved_as_best"] = true

                model_selfsupervised = deepcopy(model_self)
                @save best_model_path model_selfsupervised
                println("Updated best SELF model -> $best_model_path")
                println("Best HF mean error = $(round(best_mean_error, sigdigits=6)) at epoch $best_epoch")
            end

            push!(model_evaluations, evaluation_entry)
            println("HF evaluation time = $(round(eval_time, digits=4)) s")
        catch err
            eval_time = time() - eval_start
            push!(evaluation_errors, Dict(
                "epoch" => epoch,
                "evaluation_time" => eval_time,
                "error" => sprint(showerror, err)
            ))
            push!(epoch_mean_errors, NaN)
            push!(evaluation_times, eval_time)
            println("HF evaluation failed at epoch $epoch for $system_name")
            println("HF evaluation time before failure = $(round(eval_time, digits=4)) s")
            println(sprint(showerror, err))
        end

        if SAVE_ALL_EPOCH_MODELS
            model_selfsupervised = deepcopy(model_self)
            epoch_model_path = joinpath(epoch_model_dir, "Model_self_epoch$(epoch).bson")
            @save epoch_model_path model_selfsupervised
            println("Saved epoch model to $epoch_model_path")
        end

        update_model_selection_summary!(
            system_name,
            :self,
            build_model_result_summary(:self, model_evaluations, evaluation_errors);
            from_samples=eval_context["from_samples"],
            to_samples=eval_context["to_samples"],
            status="running"
        )

        training_elapsed_seconds = sum(epoch_times)
        training_elapsed_minutes = training_elapsed_seconds / 60
        if training_elapsed_minutes >= total_minutes
            stop_epoch = epoch
            println("Training stopped at epoch $epoch due to training time limit ($(round(training_elapsed_minutes, digits=4)) min).")
            break
        end
    end

    if stop_epoch == 0
        stop_epoch = length(epoch_losses)
    end

    times_df = DataFrame(
        Epoch_Time=sum(epoch_times),
        Evaluation_Time=sum(evaluation_times),
        Forward_Time_DC=sum(fwd_times_dc),
        Backward_Time_DC=sum(bwd_times_dc),
        Forward_Time_AC=sum(fwd_times_ac),
        Backward_Time_AC=sum(bwd_times_ac)
    )
    times_df.Total_Time = times_df.Forward_Time_DC .+ times_df.Backward_Time_DC .+ times_df.Forward_Time_AC .+ times_df.Backward_Time_AC
    println(times_df)

    metrics_dict = Dict(
        "System" => system_name,
        "Train_Range" => [first(train_range), last(train_range)],
        "Epoch_Time" => epoch_times,
        "Total_Training_Time" => sum(epoch_times),
        "Training_Time_Minutes" => sum(epoch_times) / 60,
        "Evaluation_Time" => evaluation_times,
        "Total_Evaluation_Time" => sum(evaluation_times),
        "Epoch_Losses" => epoch_losses,
        "Epoch_Mean_Error" => epoch_mean_errors,
        "Epoch_Stop" => stop_epoch,
        "Epoch_Model_Dir" => epoch_model_dir,
        "Best_Epoch" => best_epoch == 0 ? nothing : best_epoch,
        "Best_Mean_Error" => isfinite(best_mean_error) ? best_mean_error : nothing,
        "Best_Model_Path" => isfinite(best_mean_error) ? best_model_path : nothing,
        "Selection_Summary_Path" => selection_summary_path,
        "HF_Eval_Range" => [eval_context["from_samples"], eval_context["to_samples"]],
        "Forward_Time_DC" => fwd_times_dc,
        "Backward_Time_DC" => bwd_times_dc,
        "Forward_Time_AC" => fwd_times_ac,
        "Backward_Time_AC" => bwd_times_ac,
        "TOTAL_MINUTES" => total_minutes,
        "Status" => "completed"
    )

    open(metrics_path, "w") do io
        JSON.print(io, metrics_dict)
    end

    update_model_selection_summary!(
        system_name,
        :self,
        build_model_result_summary(:self, model_evaluations, evaluation_errors);
        from_samples=eval_context["from_samples"],
        to_samples=eval_context["to_samples"],
        status="completed"
    )

    return metrics_dict
end

function run_batch_training()
    progress = Dict(
        "updated_at" => string(Dates.now()),
        "systems" => Dict{String,Any}()
    )

    for (system_name, total_minutes) in SYSTEM_TIME_BUDGETS
        progress["updated_at"] = string(Dates.now())
        progress["systems"][system_name] = Dict(
            "status" => "running",
            "total_minutes" => total_minutes
        )
        save_batch_progress(progress)

        try
            metrics = train_single_system(system_name, total_minutes)
            progress["systems"][system_name] = Dict(
                "status" => "completed",
                "total_minutes" => total_minutes,
                "epoch_stop" => metrics["Epoch_Stop"],
                "best_epoch" => metrics["Best_Epoch"],
                "best_model_path" => metrics["Best_Model_Path"],
                "selection_summary_path" => metrics["Selection_Summary_Path"],
                "metrics_path" => joinpath("results", "times_metrics_self_$(system_name).json")
            )
        catch err
            progress["systems"][system_name] = Dict(
                "status" => "error",
                "total_minutes" => total_minutes,
                "error" => sprint(showerror, err),
                "backtrace" => sprint(show, catch_backtrace())
            )
            save_batch_progress(progress)
            println("Training failed for $system_name")
            println(sprint(showerror, err))
            continue
        end

        progress["updated_at"] = string(Dates.now())
        save_batch_progress(progress)
    end

    return progress
end

run_batch_training()
