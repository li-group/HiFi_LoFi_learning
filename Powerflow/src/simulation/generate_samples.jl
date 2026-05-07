using Pkg

Pkg.activate("Gnn_environment")
include("../Self_GNN_PM.jl")
using .Self_GNN_PM

using BSON
using Distributions
using Gurobi
using Ipopt
using JSON
using JuMP
using MadNLP
using PowerModels
using Random

export run_ac_dc_batch_from_data

const RANDOM_SEED = 20
const PATH_DATA = "test_systems/data-opf/"
const DEFAULT_TRAIN_SAMPLES = 30
const SAMPLE_POOL_MULTIPLIER = 3
const MIN_EXTRA_POOL = 1000

const SYSTEM_NAMES = [
    "pglib_opf_case500_goc",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2000_goc",
    "pglib_opf_case4661_sdet",
    "pglib_opf_case13659_pegase"
]

function generate_truncated_scales(data::Dict, nsamples::Int)
    nbus = length(keys(data["bus"]))

    scale_P = 1 .+ rand(truncated(Normal(0), -0.3, 0.3), nbus, nsamples)
    scale_Q = 1 .+ rand(truncated(Normal(0), -0.15, 0.0), nbus, nsamples)

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

    P_samples = scale_P' .* Pd_base'
    Q_samples = scale_Q' .* Qd_base'

    return P_samples, Q_samples
end

function run_ac_dc_batch_from_data(data_orig, P_samples, Q_samples, nsamples, solver=Ipopt.Optimizer)
    total_available, nbus = size(P_samples)

    data0 = deepcopy(data_orig)
    spec0 = PowerModels.instantiate_model(data0, PowerModels.ACRPowerModel, PowerModels.build_opf)
    ngen = length(spec0.ref[:it][:pm][:nw][0][:gen])

    Demand = zeros(nsamples, 2 * nbus)
    DispatchAC = zeros(nsamples, ngen)
    DispatchDC = zeros(nsamples, ngen)
    DualAC = zeros(nsamples, nbus)
    DualDC = zeros(nsamples, nbus)
    TimeAC = zeros(nsamples)
    TimeDC = zeros(nsamples)
    ObjAC = zeros(nsamples)
    ObjDC = zeros(nsamples)

    valid_k = 0
    j = 1

    @info "Buscando $nsamples muestras válidas (total disponibles = $total_available)..."

    while valid_k < nsamples && j <= total_available
        @info "Evaluando muestra $j del total disponible $total_available"
        Pd_i, Qd_i = P_samples[j, :], Q_samples[j, :]

        data_dc = deepcopy(data_orig)
        t1 = time()
        res_dc = nothing
        t_dc = 0.0
        dc_success = false

        try
            res_dc = DCPM(data_dc, Pd_i, Qd_i; solver=Gurobi.Optimizer)
            t_dc = time() - t1
            dc_success = res_dc["flag"]
        catch e
            @warn "Muestra $j falló en DC con error: $e"
            j += 1
            continue
        end

        if !dc_success
            @warn "Muestra $j descartada (DC sin solución)"
            j += 1
            continue
        end

        data_ac = deepcopy(data_orig)
        t0 = time()
        res_ac = ACPM(data_ac, Pd_i, Qd_i; solver=solver)
        t_ac = time() - t0

        if res_ac["flag"]
            valid_k += 1
            Demand[valid_k, :] = vcat(Pd_i, Qd_i)
            DispatchAC[valid_k, :] = res_ac["Pg"]
            DispatchDC[valid_k, :] = res_dc["Pg"]
            TimeAC[valid_k] = t_ac
            TimeDC[valid_k] = t_dc
            ObjAC[valid_k] = res_ac["objective"]
            ObjDC[valid_k] = res_dc["objective"]
        else
            @warn "Muestra $j descartada (AC sin solución)"
        end

        j += 1
    end

    if valid_k < nsamples
        @warn "Solo se obtuvieron $valid_k muestras válidas de $nsamples solicitadas."
        Demand = Demand[1:valid_k, :]
        DispatchAC = DispatchAC[1:valid_k, :]
        DispatchDC = DispatchDC[1:valid_k, :]
        DualAC = DualAC[1:valid_k, :]
        DualDC = DualDC[1:valid_k, :]
        TimeAC = TimeAC[1:valid_k]
        TimeDC = TimeDC[1:valid_k]
        ObjAC = ObjAC[1:valid_k]
        ObjDC = ObjDC[1:valid_k]
    end

    return Dict(
        "Demand" => Demand,
        "DispatchAC" => DispatchAC,
        "DispatchDC" => DispatchDC,
        "DualAC" => DualAC,
        "DualDC" => DualDC,
        "TimeAC" => TimeAC,
        "TimeDC" => TimeDC,
        "ObjAC" => ObjAC,
        "ObjDC" => ObjDC,
        "n_valid" => valid_k
    )
end

function target_train_samples(system_name::String)
    return DEFAULT_TRAIN_SAMPLES
end

available_pool_size(target::Int) = max(target * SAMPLE_POOL_MULTIPLIER, target + MIN_EXTRA_POOL)

function generate_training_dataset(data::Dict, system_name::String, nsamples::Int, seed::Int)
    Random.seed!(seed)
    limit_sample = available_pool_size(nsamples)
    P_samples, Q_samples = generate_truncated_scales(data, limit_sample)
    results = run_ac_dc_batch_from_data(data, P_samples, Q_samples, nsamples)

    output_file = joinpath(PATH_DATA, "data_$(system_name).bson")
    BSON.@save output_file results
    println("Saved training dataset for $system_name -> $output_file")
    println("Valid samples: $(results["n_valid"]) / $nsamples")

    return results
end

function generate_all_training_datasets()
    summary = Dict{String,Any}()

    for (idx, system_name) in enumerate(SYSTEM_NAMES)
        println("Generating training dataset for $system_name")
        file_name = joinpath(PATH_DATA, "$(system_name).m")
        data = load_system_powermodels(file_name)
        train_samples = target_train_samples(system_name)

        train_results = generate_training_dataset(
            data,
            system_name,
            train_samples,
            RANDOM_SEED + 10 * idx
        )

        summary[system_name] = Dict(
            "train_target_samples" => train_samples,
            "train_valid" => train_results["n_valid"],
            "train_file" => joinpath(PATH_DATA, "data_$(system_name).bson")
        )
    end

    open(joinpath(PATH_DATA, "generation_summary.json"), "w") do io
        JSON.print(io, summary)
    end

    return summary
end

generate_all_training_datasets()
