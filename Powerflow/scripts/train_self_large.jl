using Pkg

Pkg.activate("Gnn_environment")
cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")
using .Self_GNN_PM

using BSON
using BSON: @save
using Dates
using Distributions
using Flux
using Flux.Losses: mse
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

const SYSTEM_NAME = haskey(ENV, "SELF_LARGE_SYSTEM") ? ENV["SELF_LARGE_SYSTEM"] : "pglib_opf_case30000_goc"
const PATH_DATA = "test_systems/data-opf/"

const BATCHSIZE = 1
const INIT_EPOCHS = 100
const TRAIN_EPOCHS = 5000
const TRAIN_LOAD_SAMPLES = 1
const VAL_LOAD_SAMPLES = 5
const TOTAL_MINUTES = 720
const RANDOM_SEED = 20
const REGENERATE_LOAD_BANK = false

const RESULTS_DIR = "results"
const EPOCH_MODEL_DIR = joinpath("results_epochs", "$(SYSTEM_NAME)_large")
const METRICS_PATH = joinpath(RESULTS_DIR, "times_metrics_self_large_$(SYSTEM_NAME).json")
const BEST_MODEL_PATH = joinpath(EPOCH_MODEL_DIR, "Model_self_large_best.bson")

demand_bank_path(system_name::String) = joinpath(PATH_DATA, "demands_$(system_name).bson")

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

function save_json(path::String, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload)
    end
end

function load_or_generate_demand_bank(system_name::String, total_samples::Int; regenerate::Bool=false)
    bank_path = demand_bank_path(system_name)

    if isfile(bank_path) && !regenerate
        payload = BSON.load(bank_path)[:payload]
        stored_samples = size(payload["P_load"], 1)
        if payload["system_name"] == system_name && stored_samples >= total_samples
            println("Using saved demand bank from $bank_path")
            return payload
        end
        println("Existing demand bank does not match requested size; regenerating.")
    end

    case_path = joinpath(PATH_DATA, "$(system_name).m")
    data = load_system_powermodels(case_path)
    P_load, Q_load = generate_truncated_loads(data, total_samples; seed=RANDOM_SEED)

    payload = Dict(
        "system_name" => system_name,
        "seed" => RANDOM_SEED,
        "created_at" => string(Dates.now()),
        "total_samples" => total_samples,
        "train_range" => [1, TRAIN_LOAD_SAMPLES],
        "val_range" => [TRAIN_LOAD_SAMPLES + 1, total_samples],
        "P_load" => P_load,
        "Q_load" => Q_load
    )

    mkpath(dirname(bank_path))
    @save bank_path payload
    println("Saved demand bank to $bank_path")

    return payload
end

function build_graph_tensors_from_loads(data::Dict{String,Any}, P_load::AbstractMatrix, Q_load::AbstractMatrix)
    nsamples = size(P_load, 1)
    ngen = length(keys(data["gen"]))
    demand = hcat(P_load, Q_load)
    dummy_dispatch = zeros(nsamples, ngen)

    x, e, y, _ = process_system_from_data_contingencies(data, dummy_dispatch, demand)
    return x, e, y
end

function initialization_target(gw)
    nbus = length(keys(gw.data["bus"]))
    return ones(nbus, gw.graph.num_graphs)
end

function initialization_loss(model, gw)
    return mse(model(gw), initialization_target(gw))
end

function pretrain_initialization!(model, train_loader)
    opt = Flux.AdamW(1e-4)
    ps = Flux.params(model)

    for epoch in 1:INIT_EPOCHS
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader
            loss, grads = Flux.withgradient(ps) do
                initialization_loss(model, batch)
            end
            update!(opt, ps, grads)
            total_loss += loss
            num_batches += 1
        end

        num_batches > 0 || error("Initialization loader is empty.")
        println("Init epoch $epoch - Loss: $(total_loss / num_batches)")
    end

    return model
end

function evaluate_validation_set(model, val_loader, ac_init)
    Flux.testmode!(model, true)

    total_objective = 0.0
    total_cost_slack = 0.0
    total_max_slack = 0.0
    feasible_count = 0
    num_batches = 0

    for batch in val_loader
        results = model(batch, ac_init)
        total_objective += results["objective"]
        total_cost_slack += get(results, "Cost_slack", 0.0)
        total_max_slack += get(results, "max_slack", 0.0)
        feasible_count += get(results, "flag", false) ? 1 : 0
        num_batches += 1
    end

    num_batches > 0 || error("Validation loader is empty.")

    return Dict(
        "mean_objective" => total_objective / num_batches,
        "mean_cost_slack" => total_cost_slack / num_batches,
        "mean_max_slack" => total_max_slack / num_batches,
        "feasible_ratio" => feasible_count / num_batches,
        "num_samples" => num_batches
    )
end

function first_batch(loader)
    for batch in loader
        return batch
    end
    error("Loader is empty.")
end

function train_large_self_supervised(system_name::String)
    Random.seed!(RANDOM_SEED)

    total_samples = TRAIN_LOAD_SAMPLES + VAL_LOAD_SAMPLES
    bank_path = demand_bank_path(system_name)
    demand_bank = load_or_generate_demand_bank(
        system_name,
        total_samples;
        regenerate=REGENERATE_LOAD_BANK
    )

    case_path = joinpath(PATH_DATA, "$(system_name).m")
    base_data = load_system_powermodels(case_path)

    P_load = demand_bank["P_load"]
    Q_load = demand_bank["Q_load"]
    x_all, e_all, y_all = build_graph_tensors_from_loads(base_data, P_load, Q_load)

    train_range = 1:TRAIN_LOAD_SAMPLES
    val_range = (TRAIN_LOAD_SAMPLES + 1):(TRAIN_LOAD_SAMPLES + VAL_LOAD_SAMPLES)

    ref_train = deepcopy(base_data)
    ref_val = deepcopy(base_data)
    train_loader = build_loader(ref_train, x_all, e_all, y_all, train_range; batchsize=BATCHSIZE, shuffle=true)
    val_loader = build_loader(ref_val, x_all, e_all, y_all, val_range; batchsize=1, shuffle=false)

    probe_batch = first_batch(train_loader)
    ac_init = ACPM(deepcopy(base_data))["results"]["solution"]

    model_initialization = InizitilizationGNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    pretrain_initialization!(model_initialization, train_loader)
    model_initialization(probe_batch)

    model_self = Self_supervised_GNN(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
    Flux.loadparams!(model_self, Flux.params(model_initialization))

    opt = Flux.AdamW(1e-6)
    ps = Flux.params(model_self)

    empty!(fwd_times_dc)
    empty!(bwd_times_dc)
    empty!(fwd_times_ac)
    empty!(bwd_times_ac)

    model_self(probe_batch, ac_init)

    epoch_losses = Float64[]
    epoch_times = Float64[]
    val_objectives = Float64[]
    val_cost_slacks = Float64[]
    val_max_slacks = Float64[]
    val_feasible_ratios = Float64[]
    stop_epoch = 0
    best_epoch = 0
    best_val_objective = Inf
    best_val_summary = Dict{String,Any}()

    mkpath(EPOCH_MODEL_DIR)
    mkpath(dirname(METRICS_PATH))

    start_time = time()

    println("System: $system_name")
    println("Demand bank: $bank_path")
    println("Train samples: $(first(train_range)):$(last(train_range))")
    println("Validation samples: $(first(val_range)):$(last(val_range))")

    for epoch in 1:TRAIN_EPOCHS
        total_loss = 0.0
        num_batches = 0

        epoch_time = @elapsed begin
            Flux.trainmode!(model_self, true)
            for batch in train_loader
                batch_loss, grads = Flux.withgradient(ps) do
                    model_self(batch, ac_init)
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

        val_summary = evaluate_validation_set(model_self, val_loader, ac_init)
        push!(val_objectives, val_summary["mean_objective"])
        push!(val_cost_slacks, val_summary["mean_cost_slack"])
        push!(val_max_slacks, val_summary["mean_max_slack"])
        push!(val_feasible_ratios, val_summary["feasible_ratio"])

        val_objective = val_summary["mean_objective"]
        val_max_slack = val_summary["mean_max_slack"]

        println(
            "Epoch $epoch | train loss = $(round(epoch_avg_loss, sigdigits=6))" *
            " | val objective = $(round(val_objective, sigdigits=6))" *
            " | val max slack = $(round(val_max_slack, sigdigits=6))" *
            " | time = $(round(epoch_time, digits=4)) s"
        )

        if val_summary["mean_objective"] < best_val_objective
            best_val_objective = val_summary["mean_objective"]
            best_epoch = epoch
            best_val_summary = deepcopy(val_summary)

            model_self_large = deepcopy(model_self)
            @save BEST_MODEL_PATH model_self_large
            println("Updated best SELF large model -> $BEST_MODEL_PATH")
        end

        elapsed_minutes = (time() - start_time) / 60.0
        if elapsed_minutes >= TOTAL_MINUTES
            stop_epoch = epoch
            println("Training stopped at epoch $epoch due to time limit.")
            break
        end
    end

    if stop_epoch == 0
        stop_epoch = length(epoch_losses)
    end

    metrics = Dict(
        "system_name" => system_name,
        "demand_bank_path" => bank_path,
        "train_range" => [first(train_range), last(train_range)],
        "val_range" => [first(val_range), last(val_range)],
        "train_samples" => TRAIN_LOAD_SAMPLES,
        "val_samples" => VAL_LOAD_SAMPLES,
        "epoch_losses" => epoch_losses,
        "epoch_times" => epoch_times,
        "val_objectives" => val_objectives,
        "val_cost_slacks" => val_cost_slacks,
        "val_max_slacks" => val_max_slacks,
        "val_feasible_ratios" => val_feasible_ratios,
        "epoch_stop" => stop_epoch,
        "best_epoch" => best_epoch == 0 ? nothing : best_epoch,
        "best_val_objective" => isfinite(best_val_objective) ? best_val_objective : nothing,
        "best_val_summary" => isempty(best_val_summary) ? nothing : best_val_summary,
        "best_model_path" => best_epoch == 0 ? nothing : BEST_MODEL_PATH,
        "forward_time_dc" => fwd_times_dc,
        "backward_time_dc" => bwd_times_dc,
        "forward_time_ac" => fwd_times_ac,
        "backward_time_ac" => bwd_times_ac,
        "total_minutes" => TOTAL_MINUTES,
        "created_at" => string(Dates.now())
    )

    save_json(METRICS_PATH, metrics)
    println("Saved metrics to $METRICS_PATH")

    return metrics
end

function main()
    train_large_self_supervised(SYSTEM_NAME)
    return nothing
end


main()
