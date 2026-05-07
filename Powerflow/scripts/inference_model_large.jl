using Pkg

Pkg.activate("Gnn_environment")
cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")
using .Self_GNN_PM

using BSON
using Dates
using Distributions
using Flux
using GraphNeuralNetworks
using GraphNeuralNetworks: batch
using JSON
using MLUtils: DataLoader
using Random

const SYSTEM_NAME = "pglib_opf_case30000_goc"
const PATH_DATA = "test_systems/data-opf/"
const RESULTS_HF_DIR = "results_HF"
const RESULTS_EPOCHS_DIR = "results_epochs"

const NNFEATURES = 6
const NEFEATURES = 5
const HEADS = 1
const HIDDEN_DIM = 1024

const INFERENCE_RANDOM_SEED = 20
const INFERENCE_LOAD_SAMPLES = 20
const BATCHSIZE = 1
const REGENERATE_DEMAND_BANK = false

const MODEL_PATH = joinpath(RESULTS_EPOCHS_DIR, "$(SYSTEM_NAME)_large", "Model_self_large_best.bson")
const DEMAND_BANK_PATH = joinpath(PATH_DATA, "inference_demands_$(SYSTEM_NAME).json")
const RESULTS_JSON_PATH = joinpath(RESULTS_HF_DIR, "evaluation_self_large_$(SYSTEM_NAME).json")

function save_json(path::String, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload)
    end
end

matrix_to_nested_rows(values::AbstractMatrix) = [Float64.(collect(values[i, :])) for i in axes(values, 1)]

function nested_rows_to_matrix(values)
    isempty(values) && return zeros(0, 0)
    return reduce(vcat, [reshape(Float64.(row), 1, :) for row in values])
end

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

function generate_truncated_loads(data::Dict{String,Any}, nsamples::Int; seed::Int)
    rng = MersenneTwister(seed)
    nbus = length(keys(data["bus"]))

    scale_P = 1 .+ rand(rng, truncated(Normal(0), -0.3, 0.3), nbus, nsamples)
    scale_Q = 1 .+ rand(rng, truncated(Normal(0), -0.15, 0.0), nbus, nsamples)

    Pd_base = zeros(nbus)
    Qd_base = zeros(nbus)

    bus_ids = sort(parse.(Int, keys(data["bus"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    for (_, load_data) in data["load"]
        bus_id = load_data["load_bus"]
        idx = bus_idx_map[bus_id]
        Pd_base[idx] = load_data["pd"]
        Qd_base[idx] = load_data["qd"]
    end

    P_load = scale_P' .* Pd_base'
    Q_load = scale_Q' .* Qd_base'

    return P_load, Q_load
end

function load_or_generate_demand_bank(system_name::String, nsamples::Int; regenerate::Bool=false)
    if isfile(DEMAND_BANK_PATH) && !regenerate
        payload = JSON.parsefile(DEMAND_BANK_PATH)
        stored_samples = length(payload["P_load"])
        if payload["system_name"] == system_name && stored_samples >= nsamples
            println("Using saved inference demand bank from $DEMAND_BANK_PATH")
            payload["P_load"] = nested_rows_to_matrix(payload["P_load"])
            payload["Q_load"] = nested_rows_to_matrix(payload["Q_load"])
            return payload
        end
        println("Existing inference demand bank does not match requested size; regenerating.")
    end

    case_path = joinpath(PATH_DATA, "$(system_name).m")
    data = load_system_powermodels(case_path)
    P_load, Q_load = generate_truncated_loads(data, nsamples; seed=INFERENCE_RANDOM_SEED)

    payload = Dict(
        "system_name" => system_name,
        "seed" => INFERENCE_RANDOM_SEED,
        "created_at" => string(Dates.now()),
        "num_samples" => nsamples,
        "P_load" => matrix_to_nested_rows(P_load),
        "Q_load" => matrix_to_nested_rows(Q_load)
    )

    mkpath(dirname(DEMAND_BANK_PATH))
    save_json(DEMAND_BANK_PATH, payload)
    println("Saved inference demand bank to $DEMAND_BANK_PATH")

    payload["P_load"] = P_load
    payload["Q_load"] = Q_load
    return payload
end

function build_graph_tensors_from_loads(data::Dict{String,Any}, P_load::AbstractMatrix, Q_load::AbstractMatrix)
    nsamples = size(P_load, 1)
    ngen = length(keys(data["gen"]))
    demand = hcat(P_load, Q_load)
    # In 30k inference we do not have AC true labels; keep a zero placeholder
    # only because the graph builder expects a y matrix.
    dummy_dispatch = zeros(nsamples, ngen)

    x, e, y, _ = process_system_from_data_contingencies(data, dummy_dispatch, demand)
    return x, e, y
end

function load_large_self_model(model_path::String)
    isfile(model_path) || error("Large self-supervised checkpoint not found at $model_path")

    checkpoint = BSON.load(model_path)
    model = Self_GNN_PM.Self_supervised_GNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)

    for key in (:model_self_large, :model_selfsupervised, :model_epoch)
        if haskey(checkpoint, key)
            Flux.loadparams!(model, Flux.params(checkpoint[key]))
            Flux.testmode!(model, true)
            return model
        end
    end

    error("No supported model key found in $model_path")
end

function summarize_time(values::AbstractVector{<:Real})
    isempty(values) && return Dict(
        "count" => 0,
        "mean" => nothing,
        "minimum" => nothing,
        "maximum" => nothing
    )

    values_f = Float64.(values)
    return Dict(
        "count" => length(values_f),
        "mean" => sum(values_f) / length(values_f),
        "minimum" => minimum(values_f),
        "maximum" => maximum(values_f)
    )
end

function build_results_summary(results)
    valid_mask = results["flag"]
    valid_objectives = [
        results["objective"][i]
        for i in eachindex(results["objective"])
        if valid_mask[i] && isfinite(results["objective"][i])
    ]
    valid_cost_slack = [
        results["cost_slack"][i]
        for i in eachindex(results["cost_slack"])
        if valid_mask[i] && isfinite(results["cost_slack"][i])
    ]
    valid_max_slack = [
        results["max_slack"][i]
        for i in eachindex(results["max_slack"])
        if valid_mask[i] && isfinite(results["max_slack"][i])
    ]

    return Dict(
        "num_samples" => length(results["flag"]),
        "num_success" => count(identity, results["flag"]),
        "num_errors" => length(results["errors"]),
        "success_ratio" => isempty(results["flag"]) ? 0.0 : count(identity, results["flag"]) / length(results["flag"]),
        "pred_time" => summarize_time(results["pred_time"]),
        "objective_mean" => isempty(valid_objectives) ? nothing : sum(valid_objectives) / length(valid_objectives),
        "cost_slack_mean" => isempty(valid_cost_slack) ? nothing : sum(valid_cost_slack) / length(valid_cost_slack),
        "max_slack_mean" => isempty(valid_max_slack) ? nothing : sum(valid_max_slack) / length(valid_max_slack)
    )
end

function evaluate_large_self_model(data_probe, model, ac_init, demand_bank)
    results = Dict{String,Any}(
        "system_name" => SYSTEM_NAME,
        "model_path" => MODEL_PATH,
        "demand_bank_path" => DEMAND_BANK_PATH,
        "created_at" => string(Dates.now()),
        "sample_indices" => collect(1:length(data_probe)),
        "P_load" => matrix_to_nested_rows(demand_bank["P_load"]),
        "Q_load" => matrix_to_nested_rows(demand_bank["Q_load"]),
        "flag" => Bool[],
        "pred_time" => Float64[],
        "objective" => Float64[],
        "cost_slack" => Float64[],
        "max_slack" => Float64[],
        "Pg_hf" => Vector{Any}(),
        "Pg_pred_lf" => Vector{Any}(),
        "errors" => Vector{Any}()
    )

    for (idx, probe) in enumerate(data_probe)
        start_time = time()
        elapsed = 0.0

        try
            sample_results = model(probe, ac_init)
            elapsed = time() - start_time

            push!(results["flag"], get(sample_results, "flag", false))
            push!(results["pred_time"], elapsed)
            push!(results["objective"], Float64(get(sample_results, "objective", NaN)))
            push!(results["cost_slack"], Float64(get(sample_results, "Cost_slack", NaN)))
            push!(results["max_slack"], Float64(get(sample_results, "max_slack", NaN)))
            push!(results["Pg_hf"], get(sample_results, "Pg", nothing))
            push!(results["Pg_pred_lf"], get(sample_results, "Pg_pred_lf", nothing))
        catch err
            elapsed = time() - start_time
            push!(results["flag"], false)
            push!(results["pred_time"], elapsed)
            push!(results["objective"], NaN)
            push!(results["cost_slack"], NaN)
            push!(results["max_slack"], NaN)
            push!(results["Pg_hf"], nothing)
            push!(results["Pg_pred_lf"], nothing)
            push!(results["errors"], Dict(
                "sample" => idx,
                "error" => sprint(showerror, err)
            ))
            println("Sample $idx failed during self-large inference")
            println(sprint(showerror, err))
        end
    end

    results["summary"] = build_results_summary(results)
    return results
end

function run_large_self_inference()
    Random.seed!(INFERENCE_RANDOM_SEED)

    demand_bank = load_or_generate_demand_bank(
        SYSTEM_NAME,
        INFERENCE_LOAD_SAMPLES;
        regenerate=REGENERATE_DEMAND_BANK
    )

    case_path = joinpath(PATH_DATA, "$(SYSTEM_NAME).m")
    ref = load_system_powermodels(case_path)
    x, e, y = build_graph_tensors_from_loads(ref, demand_bank["P_load"], demand_bank["Q_load"])
    data_loader = build_loader(ref, x, e, y, 1:INFERENCE_LOAD_SAMPLES; batchsize=BATCHSIZE, shuffle=false)
    data_probe = collect(data_loader)

    isempty(data_probe) && error("No graphs available for large-system inference.")

    model = load_large_self_model(MODEL_PATH)
    ac_init = ACPM(deepcopy(ref))["results"]["solution"]

    results = evaluate_large_self_model(data_probe, model, ac_init, demand_bank)

    mkpath(dirname(RESULTS_JSON_PATH))
    save_json(RESULTS_JSON_PATH, results)

    println("Saved large self inference results to $RESULTS_JSON_PATH")

    return results
end

function main()
    run_large_self_inference()
    return nothing
end


main()
