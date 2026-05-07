using Pkg
Pkg.activate("Gnn_environment")
cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")
using .Self_GNN_PM

using BSON: @load
using BSON
using Flux
using GraphNeuralNetworks
using GraphNeuralNetworks: batch
using JSON
using MLUtils: DataLoader
using Statistics

const PATH_DATA = "test_systems/data-ogf/"
const SAVE_DIR = "inference_results"
const INFERENCE_CONFIGS = [
    (mode="transductive", model_system="gaslib-40-E", evaluation_system="gaslib-40-E"),
    (mode="transductive", model_system="gaslib-135-F", evaluation_system="gaslib-135-F"),
    (mode="inductive", model_system="gaslib-40-E", evaluation_system="gaslib-135-F"),
]

const NNFEATURES = 4
const NEFEATURES = 3
const HEADS = 5
const HIDDEN_DIM = 512
const FROM_SAMPLES = 1
const TO_SAMPLES = 1000

function load_models(model_system::String)
    self_model_file = joinpath("results", "Model_self_best_GM_$model_system.bson")
    semi_model_file = joinpath("results", "Model_semi_best_GM_$model_system.bson")

    @load self_model_file model_selfsupervised
    @load semi_model_file model_semisupervised

    gnn_inference = InizitilizationGNN_GM(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    truncated_model = Semi_Supervised_GNN_GM(
        NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM
    )

    Flux.loadparams!(truncated_model, Flux.params(model_selfsupervised))
    Flux.loadparams!(gnn_inference, Flux.params(model_selfsupervised))

    Flux.testmode!(model_selfsupervised, true)
    Flux.testmode!(model_semisupervised, true)
    Flux.testmode!(truncated_model, true)
    Flux.testmode!(gnn_inference, true)

    return model_selfsupervised, model_semisupervised, gnn_inference, truncated_model
end

function summarize_metric(metric_name, values)
    return Dict(
        "Metric" => metric_name,
        "Min" => minimum(values),
        "Mean" => mean(values),
        "Max" => maximum(values),
    )
end

function timed_call(f)
    start_time = time()
    result = f()
    return result, time() - start_time
end

function copy_timing_cache(timing_cache::Dict)
    return Dict(
        "times_gnn_only" => copy(timing_cache["times_gnn_only"]),
        "pred_time" => copy(timing_cache["pred_time"]),
        "times_crdwp" => copy(timing_cache["times_crdwp"]),
        "times_lrdwp" => copy(timing_cache["times_lrdwp"]),
        "times_wp" => copy(timing_cache["times_wp"]),
    )
end

function build_timing_summary(timing_data::Dict)
    return [
        summarize_metric("times_gnn_only", timing_data["times_gnn_only"]),
        summarize_metric("pred_time", timing_data["pred_time"]),
        summarize_metric("times_crdwp", timing_data["times_crdwp"]),
        summarize_metric("times_lrdwp", timing_data["times_lrdwp"]),
        summarize_metric("times_wp", timing_data["times_wp"]),
    ]
end

function run_inference(
    mode::String,
    model_system::String,
    evaluation_system::String;
    timing_cache::Union{Nothing,Dict}=nothing
)
    println(
        "Running $mode inference with model $model_system on system $evaluation_system"
    )

    model_selfsupervised, model_semisupervised, gnn_inference, truncated_model = load_models(model_system)

    ref, graphs = process_system_gasmodels(
        evaluation_system,
        PATH_DATA;
        from_numgraph=FROM_SAMPLES,
        to_numgraph=TO_SAMPLES,
        use_test_data=true
    )

    graph_loader = DataLoader(
        graphs,
        batchsize=1,
        shuffle=false,
        collate=xs -> GraphWithRef(batch(xs), ref)
    )

    results = BSON.load(joinpath(PATH_DATA, "data_test_$evaluation_system.bson"))

    data_probe = collect(graph_loader)
    true_value_init = Self_GNN_PM.WP(data_probe[1].data, results[:results_test]["delivery"][1, :])

    cost_hf_self = []
    cost_lf_self = []
    cost_total_self = []
    fg_pred_self = []
    slack_fg1_pred_self = []
    slack_fg2_pred_self = []
    max_slack_pred_self = []
    flag_pred_self = []

    cost_hf_semi = []
    cost_lf_semi = []
    cost_total_semi = []
    fg_pred_semi = []
    slack_fg1_pred_semi = []
    slack_fg2_pred_semi = []
    max_slack_pred_semi = []
    flag_pred_semi = []

    slack_fg1_lf = []
    slack_fg2_lf = []
    max_slack_lf = []
    slack_fg1_xmodel = []
    slack_fg2_xmodel = []
    max_slack_xmodel = []
    cost_pred_base = []
    cost_pred_convex = []
    fg_vars_hf = []
    flag_lf = []
    flag_xmodel = []

    local_timing_cache = timing_cache === nothing ? Dict(
        "times_gnn_only" => Float64[],
        "pred_time" => Float64[],
        "times_crdwp" => Float64[],
        "times_lrdwp" => Float64[],
        "times_wp" => Float64[],
    ) : copy_timing_cache(timing_cache)

    reuse_timing_cache = timing_cache !== nothing

    for (i, graph) in enumerate(graph_loader)
        if !reuse_timing_cache
            _, gnn_only_time = timed_call(() -> gnn_inference(graph))
            push!(local_timing_cache["times_gnn_only"], gnn_only_time)
        end

        prediction_results_self = model_selfsupervised(graph, true_value_init)
        if reuse_timing_cache
            prediction_semi = model_semisupervised(graph)
        else
            prediction_semi, prediction_semi_time = timed_call(() -> model_semisupervised(graph))
            push!(local_timing_cache["pred_time"], prediction_semi_time)
        end
        prediction_results_semi = Self_GNN_PM.WPLayer_GM(graph, prediction_semi, true_value_init)
        base = Self_GNN_PM.WPLayer_GM(graph, results[:results_test]["fg_LRWP"][i, :], true_value_init)
        base_convex = Self_GNN_PM.WPLayer_GM(graph, results[:results_test]["fg_CRDWP"][i, :], true_value_init)

        if !reuse_timing_cache
            _, crdwp_time = timed_call(() -> CRDWP(graph.data, graph.graph.x[1, :]))
            push!(local_timing_cache["times_crdwp"], crdwp_time)

            _, lrdwp_time = timed_call(() -> LRDWP(graph.data, graph.graph.x[1, :]))
            push!(local_timing_cache["times_lrdwp"], lrdwp_time)

            _, wp_time = timed_call(() -> WP(graph.data, graph.graph.x[1, :]))
            push!(local_timing_cache["times_wp"], wp_time)
        end

        push!(cost_hf_self, prediction_results_self["cost_hf"])
        push!(cost_lf_self, prediction_results_self["cost_lf"])
        push!(cost_total_self, prediction_results_self["objective"])
        push!(fg_pred_self, prediction_results_self["fg_target"])
        push!(slack_fg1_pred_self, prediction_results_self["slack_fg1"])
        push!(slack_fg2_pred_self, prediction_results_self["slack_fg2"])
        push!(max_slack_pred_self, prediction_results_self["max_slack"])
        push!(flag_pred_self, prediction_results_self["flag"])

        push!(cost_hf_semi, prediction_results_semi["cost_hf"])
        push!(cost_lf_semi, prediction_results_semi["cost_lf"])
        push!(cost_total_semi, prediction_results_semi["objective"])
        push!(fg_pred_semi, prediction_results_semi["fg_target"])
        push!(slack_fg1_pred_semi, prediction_results_semi["slack_fg1"])
        push!(slack_fg2_pred_semi, prediction_results_semi["slack_fg2"])
        push!(max_slack_pred_semi, prediction_results_semi["max_slack"])
        push!(flag_pred_semi, prediction_results_semi["flag"])

        push!(slack_fg1_lf, base["slack_fg1"])
        push!(slack_fg2_lf, base["slack_fg2"])
        push!(max_slack_lf, base["max_slack"])
        push!(slack_fg1_xmodel, base_convex["slack_fg1"])
        push!(slack_fg2_xmodel, base_convex["slack_fg2"])
        push!(max_slack_xmodel, base_convex["max_slack"])
        push!(fg_vars_hf, base["fg"])
        push!(cost_pred_base, base["objective"])
        push!(cost_pred_convex, base_convex["objective"])
        push!(flag_lf, base["flag"])
        push!(flag_xmodel, base_convex["flag"])
    end

    obj_hf = vec(results[:results_test]["Obj_HF"][FROM_SAMPLES:TO_SAMPLES])
    error_lf = abs.(obj_hf .- cost_pred_base) ./ obj_hf .* 100
    error_pred_self = abs.(obj_hf .- cost_total_self) ./ obj_hf .* 100
    error_pred_semi = abs.(obj_hf .- cost_total_semi) ./ obj_hf .* 100
    error_xmodel = abs.(obj_hf .- cost_pred_convex) ./ obj_hf .* 100

    sorted_max_slack_lf = sort(max_slack_lf)
    sorted_max_slack_self = sort(max_slack_pred_self)
    sorted_max_slack_semi = sort(max_slack_pred_semi)
    sorted_max_slack_xmodel = sort(max_slack_xmodel)
    percentiles = collect(1:100)

    percentile_values_max_slack_lf = [quantile(sorted_max_slack_lf, p / 100) for p in percentiles]
    percentile_values_max_slack_self = [quantile(sorted_max_slack_self, p / 100) for p in percentiles]
    percentile_values_max_slack_semi = [quantile(sorted_max_slack_semi, p / 100) for p in percentiles]
    percentile_values_max_slack_xmodel = [quantile(sorted_max_slack_xmodel, p / 100) for p in percentiles]

    return Dict(
        "inference_mode" => mode,
        "model_system" => model_system,
        "evaluation_system" => evaluation_system,
        "system_name" => evaluation_system,
        "from_samples" => FROM_SAMPLES,
        "to_samples" => TO_SAMPLES,
        "cost_real" => obj_hf,
        "cost_hf_self" => cost_hf_self,
        "cost_lf_self" => cost_lf_self,
        "cost_total_self" => cost_total_self,
        "fg_pred_self" => fg_pred_self,
        "slack_fg1_pred_self" => slack_fg1_pred_self,
        "slack_fg2_pred_self" => slack_fg2_pred_self,
        "max_slack_pred_self" => max_slack_pred_self,
        "flag_pred_self" => flag_pred_self,
        "cost_hf_semi" => cost_hf_semi,
        "cost_lf_semi" => cost_lf_semi,
        "cost_total_semi" => cost_total_semi,
        "fg_pred_semi" => fg_pred_semi,
        "slack_fg1_pred_semi" => slack_fg1_pred_semi,
        "slack_fg2_pred_semi" => slack_fg2_pred_semi,
        "max_slack_pred_semi" => max_slack_pred_semi,
        "flag_pred_semi" => flag_pred_semi,
        "fg_vars_hf" => fg_vars_hf,
        "slack_fg1_lf" => slack_fg1_lf,
        "slack_fg2_lf" => slack_fg2_lf,
        "max_slack_lf" => max_slack_lf,
        "slack_fg1_xmodel" => slack_fg1_xmodel,
        "slack_fg2_xmodel" => slack_fg2_xmodel,
        "max_slack_xmodel" => max_slack_xmodel,
        "cost_pred_base" => cost_pred_base,
        "cost_pred_convex" => cost_pred_convex,
        "flag_lf" => flag_lf,
        "flag_xmodel" => flag_xmodel,
        "times_gnn_only" => local_timing_cache["times_gnn_only"],
        "pred_time" => local_timing_cache["pred_time"],
        "times_crdwp" => local_timing_cache["times_crdwp"],
        "times_lrdwp" => local_timing_cache["times_lrdwp"],
        "times_wp" => local_timing_cache["times_wp"],
        "timing_summary" => build_timing_summary(local_timing_cache),
        "error_summary" => [
            summarize_metric("error_lf", error_lf),
            summarize_metric("error_pred_self", error_pred_self),
            summarize_metric("error_pred_semi", error_pred_semi),
            summarize_metric("error_xmodel", error_xmodel),
        ],
        "max_slack_percentiles" => Dict(
            "Percentile" => percentiles,
            "LRWP" => percentile_values_max_slack_lf,
            "PRED_SELF" => percentile_values_max_slack_self,
            "PRED_SEMI" => percentile_values_max_slack_semi,
            "CRWP" => percentile_values_max_slack_xmodel,
        ),
        "timing_reused" => reuse_timing_cache,
    )
end

function build_save_path(mode::String, model_system::String, evaluation_system::String)
    if mode == "transductive"
        return joinpath(SAVE_DIR, "evaluation_results_$evaluation_system.json")
    end

    return joinpath(
        SAVE_DIR,
        "evaluation_results_$(mode)_$(model_system)_to_$(evaluation_system).json"
    )
end

mkpath(SAVE_DIR)
timing_cache_by_system = Dict{String,Dict}()

for config in INFERENCE_CONFIGS
    export_data = run_inference(
        config.mode,
        config.model_system,
        config.evaluation_system;
        timing_cache=get(timing_cache_by_system, config.evaluation_system, nothing)
    )

    if !haskey(timing_cache_by_system, config.evaluation_system)
        timing_cache_by_system[config.evaluation_system] = Dict(
            "times_gnn_only" => copy(export_data["times_gnn_only"]),
            "pred_time" => copy(export_data["pred_time"]),
            "times_crdwp" => copy(export_data["times_crdwp"]),
            "times_lrdwp" => copy(export_data["times_lrdwp"]),
            "times_wp" => copy(export_data["times_wp"]),
        )
    end

    save_path = build_save_path(
        config.mode,
        config.model_system,
        config.evaluation_system
    )

    open(save_path, "w") do io
        JSON.print(io, export_data)
    end

    println("Saved inference results to: $save_path")
end

