using Pkg
Pkg.activate("Gnn_environment")
cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")

using .Self_GNN_PM
using Distributions
using Flux
using BSON
using BSON: @load
using GraphNeuralNetworks
using Gurobi
using Ipopt
using JSON
using MLUtils: DataLoader
using PowerModels
using Random


const PATH_DATA = "test_systems/data-opf/"
const MODEL_SELECTION_DIR = "results/model_selection"
const RESULTS_EPOCHS_DIR = "results_epochs"

const INFERENCE_TARGET_VALID_SAMPLES = 1000 # How many valid samples generate
const INFERENCE_EVAL_SAMPLES = 1000 # How many valid samples evaluate

const INFERENCE_EVAL_START = 1
const INFERENCE_RANDOM_SEED = 20
const INFERENCE_CANDIDATE_POOL_SIZE = 3000
const SAVE_GENERATED_TEST_DATA = true
const MAX_ACCEPTED_AC_TIME_SEC = 60.0

const NNFEATURES = 6
const NEFEATURES = 5
const HEADS = 1
const HIDDEN_DIM = 1024
const INFERENCE_TIMES_DIR = "results_HF"

const TRANSDUCTIVE_SYSTEMS = [
    # "pglib_opf_case500_goc",
    # "pglib_opf_case1354_pegase",
    # "pglib_opf_case2000_goc",
    # "pglib_opf_case4661_sdet",
    "pglib_opf_case13659_pegase",
    # "pglib_opf_case30000_goc"
]
const INDUCTIVE_SOURCE_SYSTEM = "pglib_opf_case500_goc"
const INDUCTIVE_TARGET_SYSTEMS = [
    "pglib_opf_case1354_pegase",
    # "pglib_opf_case2000_goc",
    # "pglib_opf_case4661_sdet",
    # "pglib_opf_case13659_pegase",
    # "pglib_opf_case30000_goc"
]

function expected_checkpoint_key(model_type::Symbol)
    if model_type == :self
        return :model_selfsupervised
    elseif model_type == :semi
        return :model_semisupervised
    else
        error("Unsupported model type: $model_type")
    end
end

function load_checkpoint_model!(model, model_file::String, model_type::Symbol)
    checkpoint = BSON.load(model_file)
    expected_key = expected_checkpoint_key(model_type)

    haskey(checkpoint, expected_key) || error(
        "Expected key $expected_key not found in $model_file. " *
        "Available keys: $(collect(keys(checkpoint)))"
    )

    target_params = collect(Flux.params(model))
    source_params = collect(Flux.params(checkpoint[expected_key]))
    length(target_params) == length(source_params) || error(
        "Parameter count mismatch loading $model_type from $model_file: " *
        "target has $(length(target_params)), checkpoint has $(length(source_params))"
    )

    for (idx, (p, x)) in enumerate(zip(target_params, source_params))
        size(p) == size(x) ||
            error("Parameter $idx size mismatch loading $model_type from $model_file: expected $(size(p)), got $(size(x))")
        copyto!(p, x)
    end

    println("Loaded $(uppercase(String(model_type))) checkpoint key $expected_key from $model_file ($(length(source_params)) parameter arrays)")
    return model
end


#! Build models for inference
function build_best_model()
    return Self_GNN_PM.Self_supervised_GNN_EXA(NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM)
end

function build_gnn_inference_model(source_model)
    gnn_inference = Self_GNN_PM.InizitilizationGNN(
        NNFEATURES,
        NEFEATURES,
        HEADS,
        HIDDEN_DIM
    )
    #Flux.loadparams!(gnn_inference, Flux.params(source_model))
    #! use it instead loadparams!
    for (p, x) in zip(Flux.params(gnn_inference), Flux.params(source_model))
        size(p) == size(x) ||
            error("Expected param size $(size(p)), got $(size(x))")
        copyto!(p, x)
    end

    Flux.testmode!(gnn_inference, true)
    return gnn_inference
end

function build_truncated_inference_model(source_model)
    truncated_model = Self_GNN_PM.Self_supervised_GNN_EXA_truncated(
        NNFEATURES,
        NEFEATURES,
        HEADS,
        HIDDEN_DIM
    )

    for (p, x) in zip(Flux.params(truncated_model), Flux.params(source_model))
        size(p) == size(x) ||
            error("Expected param size $(size(p)), got $(size(x))")
        copyto!(p, x)
    end

    Flux.testmode!(truncated_model, true)
    return truncated_model
end



function best_checkpoint_path(system_name::String, model_type::Symbol)
    file_name =
        if model_type == :self
            "Model_self_best.bson"
        elseif model_type == :semi
            "Model_semi_best.bson"
        else
            error("Unsupported model type: $model_type")
        end

    return joinpath(RESULTS_EPOCHS_DIR, system_name, file_name)
end

function load_best_model_info(system_name::String, model_type::Symbol;
    summary_dir::String=MODEL_SELECTION_DIR)
    per_system_path = joinpath(summary_dir, "evaluation_results_$(system_name).json")
    isfile(per_system_path) || error("Evaluation summary not found for $system_name at $per_system_path")

    summary_data = JSON.parsefile(per_system_path)
    status = get(summary_data, "status", "unknown")
    model_key = String(model_type)
    haskey(summary_data, model_key) || error(
        "No $model_key results found for $system_name. " *
        "Summary status is '$status' at $per_system_path"
    )

    best_model = summary_data[model_key]["best_model"]
    best_model === nothing && error(
        "No best $model_key model stored for $system_name. " *
        "Summary status is '$status' at $per_system_path"
    )

    return best_model
end

function load_best_model(system_name::String, model_type::Symbol;
    summary_dir::String=MODEL_SELECTION_DIR)
    best_model = load_best_model_info(system_name, model_type; summary_dir=summary_dir)
    model_path = best_checkpoint_path(system_name, model_type)
    isfile(model_path) || error("Best model checkpoint not found at $model_path")

    model = build_best_model()
    load_checkpoint_model!(model, model_path, model_type)
    Flux.testmode!(model, true)

    best_model = copy(best_model)
    best_model["model_path"] = model_path

    return model, best_model
end

function candidate_pool_size(target_valid_samples::Int)
    return max(INFERENCE_CANDIDATE_POOL_SIZE, target_valid_samples)
end

function inference_test_bson_path(system_name::String)
    return dataset_bson_path(PATH_DATA, system_name; use_test_data=true)
end

function generate_inference_scales(data::Dict, nsamples::Int)
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

function run_ac_dc_batch_for_inference(data_orig, P_samples, Q_samples, nsamples, solver=Ipopt.Optimizer)
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

    while valid_k < nsamples && j <= total_available
        Pd_i, Qd_i = P_samples[j, :], Q_samples[j, :]

        data_dc = deepcopy(data_orig)
        t1 = time()
        res_dc = nothing
        t_dc = 0.0
        dc_success = false

        try
            res_dc = DCPMExa(data_dc, Pd_i, Qd_i)
            t_dc = time() - t1
            dc_success = res_dc["flag"]
        catch
            j += 1
            continue
        end

        if !dc_success
            j += 1
            continue
        end

        data_ac = deepcopy(data_orig)
        t0 = time()
        res_ac = ACPMExa(data_ac, Pd_i, Qd_i)
        t_ac = time() - t0

        if res_ac["flag"] && t_ac < MAX_ACCEPTED_AC_TIME_SEC
            valid_k += 1
            Demand[valid_k, :] = vcat(Pd_i, Qd_i)
            DispatchAC[valid_k, :] = res_ac["Pg"]
            DispatchDC[valid_k, :] = res_dc["Pg"]
            TimeAC[valid_k] = t_ac
            TimeDC[valid_k] = t_dc
            ObjAC[valid_k] = res_ac["objective"]
            ObjDC[valid_k] = res_dc["objective"]
        elseif res_ac["flag"]
            println(
                "Discarding inference sample $j because AC solve time " *
                "$(round(t_ac, digits=3))s reached/exceeded the " *
                "$(MAX_ACCEPTED_AC_TIME_SEC)s limit."
            )
        end

        j += 1
    end

    if valid_k < nsamples
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

function depure_results_by_ac_time(results::Dict; max_ac_time_sec::Float64=MAX_ACCEPTED_AC_TIME_SEC)
    haskey(results, "TimeAC") || return results, Int(get(results, "n_valid", 0)), 0

    time_ac = Float64.(results["TimeAC"])
    keep = findall(t -> isfinite(t) && t < max_ac_time_sec, time_ac)
    dropped = length(time_ac) - length(keep)
    dropped == 0 && return results, length(keep), 0

    cleaned = Dict{String,Any}()
    n_rows = length(time_ac)
    for (key, value) in results
        if value isa AbstractMatrix && size(value, 1) == n_rows
            cleaned[key] = value[keep, :]
        elseif value isa AbstractVector && length(value) == n_rows
            cleaned[key] = value[keep]
        elseif key == "n_valid"
            cleaned[key] = length(keep)
        else
            cleaned[key] = value
        end
    end
    cleaned["n_valid"] = length(keep)

    println(
        "Depured inference dataset by TimeAC < $(max_ac_time_sec)s: " *
        "kept $(length(keep)) / $n_rows, dropped $dropped."
    )
    return cleaned, length(keep), dropped
end

function generate_test_results(system_name::String, target_valid_samples::Int; seed::Int=INFERENCE_RANDOM_SEED)
    file_name = joinpath(PATH_DATA, "$(system_name).m")
    data = load_system_powermodels(file_name)

    Random.seed!(seed)
    pool_size = candidate_pool_size(target_valid_samples)
    P_samples, Q_samples = generate_inference_scales(data, pool_size)
    results = run_ac_dc_batch_for_inference(data, P_samples, Q_samples, target_valid_samples)

    return data, results
end

function save_generated_test_results(system_name::String, generated_results)
    output_file = dataset_bson_path(PATH_DATA, system_name; use_test_data=true)
    BSON.@save output_file results = generated_results
    println("Saved generated test dataset for $system_name -> $output_file")
    return output_file
end

function load_or_generate_test_results(system_name::String, target_valid_samples::Int; seed::Int=INFERENCE_RANDOM_SEED)
    saved_test_file = inference_test_bson_path(system_name)

    if isfile(saved_test_file)
        saved_payload = BSON.load(saved_test_file)
        generated_results = saved_payload[:results]
        generated_results, _, dropped = depure_results_by_ac_time(generated_results)
        if dropped > 0 && SAVE_GENERATED_TEST_DATA
            save_generated_test_results(system_name, generated_results)
        end
        n_valid = Int(get(generated_results, "n_valid", 0))

        if n_valid >= target_valid_samples
            println("Loading saved test dataset for $system_name from $saved_test_file")
            file_name = joinpath(PATH_DATA, "$(system_name).m")
            ref = load_system_powermodels(file_name)
            return ref, generated_results, saved_test_file
        end

        println(
            "Saved test dataset for $system_name has only $n_valid valid samples; " *
            "regenerating to reach $target_valid_samples."
        )
    end

    ref, generated_results = generate_test_results(system_name, target_valid_samples; seed=seed)
    saved_test_file = save_generated_test_results(system_name, generated_results)
    return ref, generated_results, saved_test_file
end

function build_evaluation_context(system_name::String;
    target_valid_samples::Int=INFERENCE_TARGET_VALID_SAMPLES,
    eval_samples::Int=INFERENCE_EVAL_SAMPLES,
    seed::Int=INFERENCE_RANDOM_SEED)
    ref, generated_results, saved_test_file = load_or_generate_test_results(
        system_name,
        target_valid_samples;
        seed=seed
    )

    n_valid = Int(generated_results["n_valid"])
    n_valid > 0 || error("No valid inference samples were generated for $system_name")

    actual_from = INFERENCE_EVAL_START
    actual_to = min(n_valid, actual_from + eval_samples - 1)

    if actual_to < actual_from
        error("No valid samples available to evaluate for $system_name")
    end

    if n_valid < eval_samples
        println(
            "Warning: only $n_valid valid test samples were generated for $system_name, " *
            "so inference will evaluate $n_valid samples instead of $eval_samples."
        )
    end

    x, e, y, _ = process_system_from_data_contingencies(
        ref,
        generated_results["DispatchAC"],
        generated_results["Demand"]
    )
    graphs = data_graph_powermodels(ref, x, e, y, actual_from, actual_to, GNNGraph[])
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
    probe_batch === nothing && error("No graphs available for $system_name")

    # ac_base_result = ACPM(probe_batch.data)["results"]["solution"]

    context = Dict(
        "ref" => ref,
        "graph_loader" => graph_loader,
        "data_probe" => collect(graph_loader),
        "pg_dc" => generated_results["DispatchDC"],
        # "ac_base_result" => ac_base_result,
        "generated_results" => generated_results,
        "saved_test_file" => saved_test_file,
        "from_samples" => actual_from,
        "to_samples" => actual_to,
        "target_valid_samples" => target_valid_samples,
        "evaluated_samples" => actual_to - actual_from + 1,
        "candidate_pool_size" => candidate_pool_size(target_valid_samples)
    )
    return context
end

function summarize_timing_vector(values::AbstractVector)
    isempty(values) && return Dict{String,Any}(
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

function save_inference_times(stage_name::String, evaluation_system::String, model_system::String, timing_payload;
    use_inductive_output::Bool=false)
    mkpath(INFERENCE_TIMES_DIR)
    suffix = use_inductive_output ? "inductive" : "transductive"
    output_path = joinpath(
        INFERENCE_TIMES_DIR,
        "inference_times_$(suffix)_$(evaluation_system)_from_$(model_system).json"
    )

    open(output_path, "w") do io
        JSON.print(io, timing_payload)
    end

    println("Saved inference times for $stage_name -> $output_path")
    return output_path
end

function timing_source_model(model_bundle)
    if haskey(model_bundle, "semi")
        return model_bundle["semi"]
    elseif haskey(model_bundle, "self")
        return model_bundle["self"]
    else
        error("No model available for inference timing.")
    end
end

function evaluate_inference_times(data_probe, model_bundle)
    source_model = timing_source_model(model_bundle)
    gnn_inference = build_gnn_inference_model(source_model)
    truncated_inference = build_truncated_inference_model(source_model)

    # ac_time = Float64[]
    # dc_time = Float64[]
    pred_time = Float64[]
    solo_gnn_time = Float64[]

    for probe in data_probe
        pd = vec(Array(probe.graph.x[1, :]))
        qd = vec(Array(probe.graph.x[2, :]))

        # push!(ac_time, @elapsed ACPMExa(deepcopy(probe.data), pd, qd))
        # push!(dc_time, @elapsed DCPM(deepcopy(probe.data), pd, qd))
        push!(pred_time, @elapsed truncated_inference(probe))
        push!(solo_gnn_time, @elapsed gnn_inference(probe))
    end

    return Dict(
        # "ac_time" => ac_time,
        # "dc_time" => dc_time,
        "pred_time" => pred_time,
        "solo_gnn_time" => solo_gnn_time,
        "summary" => Dict(
            # "ac_time" => summarize_timing_vector(ac_time),
            # "dc_time" => summarize_timing_vector(dc_time),
            "pred_time" => summarize_timing_vector(pred_time),
            "solo_gnn_time" => summarize_timing_vector(solo_gnn_time)
        )
    )
end

function load_best_models_for_system(model_system::String; model_types=(:self, :semi))
    model_bundle = Dict{String,Any}()

    if :self in model_types
        model_self, best_self_info = load_best_model(model_system, :self)
        Flux.testmode!(model_self, true)
        model_bundle["self"] = model_self
        model_bundle["self_info"] = best_self_info
    end

    if :semi in model_types
        model_semi, best_semi_info = load_best_model(model_system, :semi)
        Flux.testmode!(model_semi, true)
        model_bundle["semi"] = model_semi
        model_bundle["semi_info"] = best_semi_info
    end

    if haskey(model_bundle, "self_info") && haskey(model_bundle, "semi_info")
        self_path = abspath(model_bundle["self_info"]["model_path"])
        semi_path = abspath(model_bundle["semi_info"]["model_path"])
        self_path != semi_path || error(
            "SELF and SEMI resolved to the same checkpoint for $model_system: $self_path"
        )
    end

    return model_bundle
end

function print_model_selection(stage_name::String, evaluation_system::String, model_system::String, model_bundle)
    println("========== $stage_name ==========")
    println("Evaluation system: $evaluation_system")
    println("Model source system: $model_system")
    if haskey(model_bundle, "self_info")
        println("SELF best epoch: $(model_bundle["self_info"]["epoch"])")
        println("SELF mean error: $(model_bundle["self_info"]["mean_error"])")
        println("SELF path: $(model_bundle["self_info"]["model_path"])")
    end
    if haskey(model_bundle, "semi_info")
        println("SEMI best epoch: $(model_bundle["semi_info"]["epoch"])")
        println("SEMI mean error: $(model_bundle["semi_info"]["mean_error"])")
        println("SEMI path: $(model_bundle["semi_info"]["model_path"])")
    end
end

function run_inference_stage(stage_name::String, evaluation_system::String, model_system::String;
    use_inductive_output::Bool=false,
    target_valid_samples::Int=INFERENCE_TARGET_VALID_SAMPLES,
    eval_samples::Int=INFERENCE_EVAL_SAMPLES,
    model_types=(:self, :semi))
    model_bundle = load_best_models_for_system(model_system; model_types=model_types)
    context = build_evaluation_context(
        evaluation_system;
        target_valid_samples=target_valid_samples,
        eval_samples=eval_samples
    )

    print_model_selection(stage_name, evaluation_system, model_system, model_bundle)

    results_hf = Self_GNN_PM.evaluate_model_HF(
        get(model_bundle, "self", nothing),
        get(model_bundle, "semi", nothing),
        [],
        context["graph_loader"],
        context["pg_dc"][context["from_samples"]:context["to_samples"], :],
        evaluation_system;
        use_inductive_output=use_inductive_output
    )

    return Dict(
        "stage" => stage_name,
        "evaluation_system" => evaluation_system,
        "model_system" => model_system,
        "self_info" => get(model_bundle, "self_info", nothing),
        "semi_info" => get(model_bundle, "semi_info", nothing),
        "target_valid_samples" => context["target_valid_samples"],
        "candidate_pool_size" => context["candidate_pool_size"],
        "num_generated_test_samples" => context["generated_results"]["n_valid"],
        "num_evaluated_samples" => context["evaluated_samples"],
        "saved_test_file" => context["saved_test_file"],
        "results_hf" => results_hf
    )
end

function run_inference_times_stage(evaluation_system::String, model_system::String=evaluation_system;
    stage_name::String="inference_times",
    use_inductive_output::Bool=false,
    target_valid_samples::Int=INFERENCE_TARGET_VALID_SAMPLES,
    eval_samples::Int=INFERENCE_EVAL_SAMPLES,
    model_types=(:self, :semi))
    model_bundle = load_best_models_for_system(model_system; model_types=model_types)
    context = build_evaluation_context(
        evaluation_system;
        target_valid_samples=target_valid_samples,
        eval_samples=eval_samples
    )

    print_model_selection(stage_name, evaluation_system, model_system, model_bundle)

    inference_times = evaluate_inference_times(
        context["data_probe"],
        model_bundle
    )
    inference_times_payload = Dict(
        "stage" => stage_name,
        "evaluation_system" => evaluation_system,
        "model_system" => model_system,
        "num_samples" => length(context["data_probe"]),
        "times" => inference_times
    )
    inference_times_path = save_inference_times(
        stage_name,
        evaluation_system,
        model_system,
        inference_times_payload;
        use_inductive_output=use_inductive_output
    )

    return Dict(
        "stage" => stage_name,
        "evaluation_system" => evaluation_system,
        "model_system" => model_system,
        "self_info" => get(model_bundle, "self_info", nothing),
        "semi_info" => get(model_bundle, "semi_info", nothing),
        "target_valid_samples" => context["target_valid_samples"],
        "candidate_pool_size" => context["candidate_pool_size"],
        "num_generated_test_samples" => context["generated_results"]["n_valid"],
        "num_evaluated_samples" => context["evaluated_samples"],
        "saved_test_file" => context["saved_test_file"],
        "inference_times_path" => inference_times_path,
        "inference_times" => inference_times
    )
end

function run_inference_times_for_systems(systems=TRANSDUCTIVE_SYSTEMS)
    results = Dict{String,Any}()

    for system_name in systems
        try
            results[system_name] = run_inference_times_stage(
                system_name,
                system_name
            )
        catch err
            results[system_name] = Dict(
                "stage" => "inference_times",
                "evaluation_system" => system_name,
                "model_system" => system_name,
                "status" => "error",
                "error" => sprint(showerror, err)
            )
            println("Skipping inference timing system $system_name")
            println(sprint(showerror, err))
        end
    end

    return results
end

function run_transductive_stage(systems=TRANSDUCTIVE_SYSTEMS)
    results = Dict{String,Any}()

    for system_name in systems
        try
            results[system_name] = run_inference_stage(
                "transductive",
                system_name,
                system_name;
                use_inductive_output=false,
                model_types=(:self, :semi)
            )
        catch err
            results[system_name] = Dict(
                "stage" => "transductive",
                "evaluation_system" => system_name,
                "model_system" => system_name,
                "status" => "error",
                "error" => sprint(showerror, err)
            )
            println("Skipping transductive system $system_name")
            println(sprint(showerror, err))
        end
    end

    return results
end

function run_inductive_stage(source_system::String=INDUCTIVE_SOURCE_SYSTEM, target_systems=INDUCTIVE_TARGET_SYSTEMS)
    results = Dict{String,Any}()

    for target_system in target_systems
        try
            results[target_system] = run_inference_stage(
                "inductive",
                target_system,
                source_system;
                use_inductive_output=true,
                model_types=(:self, :semi)
            )
        catch err
            results[target_system] = Dict(
                "stage" => "inductive",
                "evaluation_system" => target_system,
                "model_system" => source_system,
                "status" => "error",
                "error" => sprint(showerror, err)
            )
            println("Skipping inductive system $target_system")
            println(sprint(showerror, err))
        end
    end

    return results
end

function run_single_inductive_target(target_system::String;
    source_system::String=INDUCTIVE_SOURCE_SYSTEM,
    target_valid_samples::Int=INFERENCE_TARGET_VALID_SAMPLES,
    eval_samples::Int=INFERENCE_EVAL_SAMPLES)
    println("========== inductive single target ==========")
    println("Source system: $source_system")
    println("Target system: $target_system")

    return run_inference_stage(
        "inductive",
        target_system,
        source_system;
        use_inductive_output=true,
        target_valid_samples=target_valid_samples,
        eval_samples=eval_samples,
        model_types=(:self, :semi)
    )
end

function run_inductive_stage_one_by_one(source_system::String=INDUCTIVE_SOURCE_SYSTEM, target_systems=INDUCTIVE_TARGET_SYSTEMS;
    target_valid_samples::Int=INFERENCE_TARGET_VALID_SAMPLES,
    eval_samples::Int=INFERENCE_EVAL_SAMPLES)
    results = Dict{String,Any}()

    for target_system in target_systems
        println("--------------------------------------------------")
        println("Running inductive target: $target_system")
        results[target_system] = run_single_inductive_target(
            target_system;
            source_system=source_system,
            target_valid_samples=target_valid_samples,
            eval_samples=eval_samples
        )
    end

    return results
end

# transductive_results = run_transductive_stage()
# inductive_results = run_inductive_stage()
# inference_times_results = run_inference_times_for_systems(TRANSDUCTIVE_SYSTEMS)

a = BSON.load("C:\\Users\\and78\\GNN_Diff_HF-LF-Alternative-to-create-model-ONCE\\GPU_HF_LF_POWERFLOW\\test_systems\\data-opf\\data_test_pglib_opf_case30000_goc.bson")
   

using JSON

path = joinpath("results_HF", "evaluation_HF_pglib_opf_case30000_goc.json")
data = JSON.parsefile(path)


# load the data file
data_30 = PowerModels.parse_file("C:\\Users\\and78\\GNN_Diff_HF-LF-Alternative-to-create-model-ONCE\\GPU_HF_LF_POWERFLOW\\test_systems\\data-opf\\pglib_opf_case30000_goc.m")

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)

# use build_ref to filter out inactive components
ref_data = PowerModels.build_ref(data)[:it][:pm][:nw][0]

# println(keys(data))

sol_exa = ACPMExa(data)

pm = PowerModels.instantiate_model(
    data,
    PowerModels.DCPPowerModel,   # or ACPPowerModel, SOCWRPowerModel, etc.
    PowerModels.build_opf
)
using PowerModels
using Gurobi
jump_model = pm.model
result = optimize_model!(pm, optimizer=Gurobi.Optimizer)



nvars = JuMP.num_variables(jump_model)
ncons = JuMP.num_constraints(jump_model; count_variable_in_set_constraints=true)

println("Number of variables = ", nvars)
println("Number of constraints = ", ncons)
