module InferenceUtils

using ..Self_GNN_PM
using Flux, PowerModels, ChainRulesCore, MAT, Statistics
using DataFrames
export evaluate_best_model, compute_dc_cost, summarize_results, evaluate_model_beta, calculate_cost_feas, evaluate_model_HF, evaluate_best_model_HF
using BSON
using JuMP
using JSON


function evaluate_best_model(model, loader)
    predictions = Float64[]
    real_values = Float64[]
    Pg_predicted = []
    pfr_predicted = []
    slacks_on = falses(length(loader))

    for (i, graph) in enumerate(loader)
        data = graph.data

        ref_data = ChainRulesCore.ignore_derivatives() do
            PowerModels.build_ref(data; ref_extensions=Dict())
        end[:it][:pm][:nw][0]

        start_time = time()  # tiempo en segundos desde la época UNIX

        results = model(graph)

        end_time = time()
        elapsed_time = end_time - start_time


        Pg_pred = results["Pg_values"]

        # if sum(results["slacks"]) > 0
        #     graph.graph.x[1, :] .-= results["slacks"]
        #     results = model(graph)
        #     Pg_pred = results["Pg_values"]
        #     slacks_on[i] = true
        # end
        pfr = JuMP.value.(results["model"][:p])

        push!(Pg_predicted, Pg_pred)
        push!(pfr_predicted, pfr)

        cost_pred = sum(gen["cost"][1] * Pg_pred[i]^2 + gen["cost"][2] * Pg_pred[i] + gen["cost"][3]
                        for (i, gen) in ref_data[:gen])
        push!(predictions, cost_pred)

        Pg_real = graph.graph.y
        gen_ids = sort(collect(keys(ref_data[:gen])))

        cost_real = sum(
            ref_data[:gen][g]["cost"][1] * Pg_real[i]^2 +
            ref_data[:gen][g]["cost"][2] * Pg_real[i] +
            ref_data[:gen][g]["cost"][3]
            for (i, g) in enumerate(gen_ids)
        )

        push!(real_values, cost_real)
    end

    return predictions, real_values, slacks_on, Pg_predicted, pfr_predicted
end


function evaluate_best_model_HF(model, loader)
    predictions = Float64[]
    real_values = Float64[]
    Pg_predicted = []


    for (i, graph) in enumerate(loader)
        data = graph.data

        ref_data = ChainRulesCore.ignore_derivatives() do
            PowerModels.build_ref(data; ref_extensions=Dict())
        end[:it][:pm][:nw][0]

        start_time = time()  # tiempo en segundos desde la época UNIX

        results = model(graph)

        end_time = time()
        elapsed_time = end_time - start_time


        Pg_pred = results["Pg"]

        # if sum(results["slacks"]) > 0
        #     graph.graph.x[1, :] .-= results["slacks"]
        #     results = model(graph)
        #     Pg_pred = results["Pg_values"]
        #     slacks_on[i] = true
        # end
        # pfr = JuMP.value.(results["model"][:p])

        push!(Pg_predicted, Pg_pred)
        # push!(pfr_predicted, pfr)

        cost_pred = results["objective"]
        push!(predictions, cost_pred)

        Pg_real = graph.graph.y
        gen_ids = sort(collect(keys(ref_data[:gen])))

        cost_real = sum(
            ref_data[:gen][g]["cost"][1] * Pg_real[i]^2 +
            ref_data[:gen][g]["cost"][2] * Pg_real[i] +
            ref_data[:gen][g]["cost"][3]
            for (i, g) in enumerate(gen_ids)
        )

        push!(real_values, cost_real)
    end

    return predictions, real_values, Pg_predicted
end

# function evaluate_model_HF(model_self, model_self_trunc, model_semi, model_semi_trunc,
#     loader, Pg_dc, system_name)

#     # Containers for both models
#     results_all = Dict(
#         "self" => Dict(
#             "predictions" => Float64[],
#             "real_values" => Float64[],
#             "cost_sl" => Float64[],
#             "Pg_predicted" => [],
#             "cost_pg" => Float64[],
#             "sum_slack_truncated" => Float64[],
#             "cost_dc" => Float64[],
#             "cost_slack" => Float64[],
#             "elapsed_time" => Float64[]   # ⏱ new field for elapsed time
#         ),
#         "semi" => Dict(
#             "predictions" => Float64[],
#             "real_values" => Float64[],
#             "cost_sl" => Float64[],
#             "Pg_predicted" => [],
#             "cost_pg" => Float64[],
#             "sum_slack_truncated" => Float64[],
#             "cost_dc" => Float64[],
#             "cost_slack" => Float64[],
#             "elapsed_time" => Float64[]   # ⏱ new field for elapsed time
#         )
#     )

#     save_path = "evaluation_HF_$(system_name).json"
#     # save_path = "evaluation_HF_projection_$(system_name).json"

#     for (i, graph) in enumerate(loader)
#         data = graph.data
#         ref_data = ChainRulesCore.ignore_derivatives() do
#             PowerModels.build_ref(data; ref_extensions=Dict())
#         end[:it][:pm][:nw][0]

#         gen_ids = sort(collect(keys(ref_data[:gen])))
#         Pg_real = graph.graph.y

#         # --- Evaluate both models ---
#         for (label, model, model_trunc) in [
#             ("self", model_self, model_self_trunc),
#             ("semi", model_semi, model_semi_trunc)
#         ]
#             start_time = time()

#             results = model(graph)
#             # results_trunc = model_trunc(graph)
#             results_DC = ACPMLayer_Array(graph, Pg_dc[i,:])

#             elapsed = time() - start_time  # ⏱ compute elapsed time

#             # --- Extract and store values ---
#             # Sum_slack = results_trunc["Sum_slacks"]
#             Pg_pred = results["Pg"]
#             cost_pred = results["objective"]
#             cost_slack = results["Cost_slack"]
#             cost_pg = results["Cost_Pg"]

#             cost_real = sum(
#                 ref_data[:gen][g]["cost"][1] * Pg_real[j]^2 +
#                 ref_data[:gen][g]["cost"][2] * Pg_real[j] +
#                 ref_data[:gen][g]["cost"][3]
#                 for (j, g) in enumerate(gen_ids)
#             )

#             push!(results_all[label]["predictions"], cost_pred)
#             push!(results_all[label]["real_values"], cost_real)
#             push!(results_all[label]["cost_sl"], cost_slack)
#             push!(results_all[label]["Pg_predicted"], Pg_pred)
#             push!(results_all[label]["cost_pg"], cost_pg)
#             push!(results_all[label]["cost_dc"], results_DC["objective"])
#             push!(results_all[label]["cost_slack"], results_DC["Cost_slack"])
#             # push!(results_all[label]["elapsed_time"], elapsed)  # ✅ store elapsed time
#         end

#         # 💾 Save progress after each iteration
#         progress_data = Dict(
#             "system_name" => system_name,
#             "iteration" => i,
#             "results" => results_all
#         )

#         open(save_path, "w") do io
#             JSON.print(io, progress_data)
#         end

#         println("Saved progress iteration $i → $save_path")
#     end

#     # Return both results separately for convenience
#     return results_all["self"]
# end



function evaluate_model_HF(model_self, model_semi,
    ac_init, loader, Pg_dc, system_name; use_inductive_output::Bool=false)

    results_all = Dict{String,Any}(
        "cost_real" => Float64[],
        "Pg_real" => Vector{Any}(),
        "dc" => Vector{Any}(),
        "self" => Dict{String,Any}(
            "predictions" => Float64[],
            "real_values" => Float64[],
            "cost_sl" => Float64[],
            "Pg_pred_lf" => [],
            "Pg_hf" => [],
            "cost_pg" => Float64[],
            "max_slack" => Float64[],
            "pos_slack" => [],
            "neg_slack" => [],
            "sum_slack_truncated" => Float64[],
            "cost_dc" => Float64[],
            "cost_slack" => Float64[],
            "elapsed_time" => Float64[],
            "hf_results" => Vector{Any}()
        ),
        "semi" => Dict{String,Any}(
            "predictions" => Float64[],
            "real_values" => Float64[],
            "cost_sl" => Float64[],
            "Pg_pred_lf" => [],
            "Pg_hf" => [],
            "cost_pg" => Float64[],
            "max_slack" => Float64[],
            "pos_slack" => [],
            "neg_slack" => [],
            "sum_slack_truncated" => Float64[],
            "cost_dc" => Float64[],
            "cost_slack" => Float64[],
            "elapsed_time" => Float64[],
            "hf_results" => Vector{Any}()
        )
    )

    is_available_model(model) = !(model === nothing || ismissing(model) || (model isa AbstractVector && isempty(model)))
    is_available_model(model_self) || delete!(results_all, "self")
    is_available_model(model_semi) || delete!(results_all, "semi")

    function compute_percentiles(values)
        isempty(values) && return Dict{String,Float64}()
        sorted_values = sort(Float64.(values))
        n = length(sorted_values)
        return Dict(
            string(p) => sorted_values[clamp(ceil(Int, p / 100 * n), 1, n)]
            for p in 1:100
        )
    end
    compute_real_cost(ref_data, pg_values) = begin
        gen_ids = sort(collect(keys(ref_data[:gen])))
        sum(
            ref_data[:gen][g]["cost"][1] * pg_values[j]^2 +
            ref_data[:gen][g]["cost"][2] * pg_values[j] +
            ref_data[:gen][g]["cost"][3]
            for (j, g) in enumerate(gen_ids)
        )
    end

    function json_ready(value)
        if value === nothing || value isa Number || value isa AbstractString || value isa Bool
            return value
        elseif value isa AbstractArray
            return json_ready.(Array(value))
        elseif value isa Tuple
            return [json_ready(v) for v in value]
        elseif value isa AbstractDict
            return Dict(string(k) => json_ready(v) for (k, v) in value)
        else
            return value
        end
    end

    slack_sum(value) =
        value === nothing ? nothing :
        value isa Number ? value :
        sum(Array(value))

    solver_succeeded(results) = get(results, "flag", false) == 1

    hf_snapshot(results) = Dict(
        "flag" => json_ready(results["flag"]),
        "objective" => json_ready(results["objective"]),
        "Pg" => json_ready(results["Pg"]),
        "Pg_pred_lf" => json_ready(get(results, "Pg_pred_lf", nothing)),
        "Cost_slack_all" => json_ready(get(results, "Cost_slack_all", nothing)),
        "Cost_Pg" => json_ready(get(results, "Cost_Pg", nothing)),
        "pos_slack" => json_ready(get(results, "pos_slack", nothing)),
        "neg_slack" => json_ready(get(results, "neg_slack", nothing)),
        "sum_pos_slack" => json_ready(slack_sum(get(results, "pos_slack", get(results, "sum_pos_slack", nothing)))),
        "sum_neg_slack" => json_ready(slack_sum(get(results, "neg_slack", get(results, "sum_neg_slack", nothing)))),
        "max_slack" => json_ready(get(results, "max_slack", nothing)),
        "Cost_slack" => json_ready(get(results, "Cost_slack", nothing)),
        "Dual" => json_ready(get(results, "Dual", nothing))
    )
    function evaluate_model_branch(label, model, graph)
        if label == "self" || label == "semi"
            results = model(graph)
            return results, get(results, "Pg_pred_lf", nothing)
        else
            error("Unsupported label $label in evaluate_model_HF.")
        end
    end

    save_path =
        if use_inductive_output
            joinpath("results_HF", "evaluation_HF_inductive_$(system_name).json")
        else
            joinpath("results_HF", "evaluation_HF_$(system_name).json")
        end
    mkpath(dirname(save_path))

    for (i, graph) in enumerate(loader)
        data = graph.data
        ref_data = ChainRulesCore.ignore_derivatives() do
            PowerModels.build_ref(data; ref_extensions=Dict())
        end[:it][:pm][:nw][0]

        Pg_real = graph.graph.y

        # results_dc = ACPMLayer_Array(graph, Pg_dc[i, :])
        println(size(graph.graph.x[1, :]))
        results_dc = ACPMLayerExa(graph, Pg_dc[i, :])

        if !solver_succeeded(results_dc)
            println("Discarding HF sample $i because DC flag != 1. flag = $(get(results_dc, "flag", nothing))")
            continue
        end

        cost_real = compute_real_cost(ref_data, Pg_real)

        push!(results_all["cost_real"], json_ready(cost_real))
        push!(results_all["Pg_real"], json_ready(Pg_real))
        push!(results_all["dc"], hf_snapshot(results_dc))

        model_specs = []
        if is_available_model(model_self)
            push!(model_specs, ("self", model_self))
        end
        if is_available_model(model_semi)
            push!(model_specs, ("semi", model_semi))
        end
        isempty(model_specs) && error("No valid model was provided to evaluate_model_HF.")

        for (label, model) in model_specs
            start_time = time()
            results, Pg_pred_lf = evaluate_model_branch(label, model, graph)
            elapsed = time() - start_time

            Pg_slack = results["Pg"]
            cost_pred = results["objective"]
            cost_slack = results["Cost_slack"]
            cost_pg = results["Cost_Pg"]
            max_slack = get(results, "max_slack", 0.0)
            pos_slack = get(results, "pos_slack", nothing)
            neg_slack = get(results, "neg_slack", nothing)

            push!(results_all[label]["predictions"], json_ready(cost_pred))
            push!(results_all[label]["real_values"], json_ready(cost_real))
            push!(results_all[label]["cost_sl"], json_ready(cost_slack))
            push!(results_all[label]["Pg_pred_lf"], json_ready(Pg_pred_lf))
            push!(results_all[label]["Pg_hf"], json_ready(Pg_slack))
            push!(results_all[label]["cost_pg"], json_ready(cost_pg))
            push!(results_all[label]["max_slack"], json_ready(max_slack))
            push!(results_all[label]["pos_slack"], json_ready(pos_slack))
            push!(results_all[label]["neg_slack"], json_ready(neg_slack))
            push!(results_all[label]["cost_dc"], json_ready(results_dc["objective"]))
            push!(results_all[label]["cost_slack"], json_ready(results_dc["Cost_slack"]))
            push!(results_all[label]["elapsed_time"], elapsed)
            push!(results_all[label]["hf_results"], hf_snapshot(results))
        end

        # dc_max_slack = [
        #     get(results_dc_i, "max_slack", 0.0)
        #     for results_dc_i in results_all["dc"]
        # ]
        # results_all["dc_max_slack"] = dc_max_slack
        # results_all["dc_max_slack_percentiles"] = compute_percentiles(dc_max_slack)
        # if haskey(results_all, "self")
        #     results_all["self"]["max_slack_percentiles"] = compute_percentiles(results_all["self"]["max_slack"])
        # end
        # if haskey(results_all, "semi")
        #     results_all["semi"]["max_slack_percentiles"] = compute_percentiles(results_all["semi"]["max_slack"])
        # end

        progress_data = Dict(
            "system_name" => system_name,
            "iteration" => i,
            "results" => results_all
        )

        open(save_path, "w") do io
            JSON.print(io, json_ready(progress_data))
        end

        println("Saved progress iteration $i → $save_path")
    end

    return results_all
end




# function evaluate_model_beta(model, loader, pg_ac)
#     predictions = Float64[]
#     real_values = Float64[]
#     objectives = []
#     slacks_on = falses(length(loader))

#     for (i, graph) in enumerate(loader)
#         data = graph.data

#         ref_data = ChainRulesCore.ignore_derivatives() do
#             PowerModels.build_ref(data; ref_extensions=Dict())
#         end[:it][:pm][:nw][0]

#         results = model(graph)
#         Pg_pred = results["Pg_values"]

#         if sum(results["slacks"]) > 0
#             graph.graph.x[1, :] .-= results["slacks"]
#             results = model(graph)
#             Pg_pred = results["Pg_values"]
#             slacks_on[i] = true
#         end

#         push!(objectives, Pg_pred)

#         cost_pred = sum(gen["cost"][1] * Pg_pred[i]^2 + gen["cost"][2] * Pg_pred[i] + gen["cost"][3]
#                         for (i, gen) in ref_data[:gen])
#         push!(predictions, cost_pred)

#         Pg_real = pg_ac[:, i]
#         cost_real = sum(gen["cost"][1] * Pg_real[i]^2 + gen["cost"][2] * Pg_real[i] + gen["cost"][3]
#                         for (i, gen) in ref_data[:gen])
#         push!(real_values, cost_real)
#     end

#     return predictions, real_values, slacks_on
# end

function compute_dc_cost(data_path, case_system, data)

    # file_path = joinpath(data_path, "data_test_$(case_system).bson")
    file_path = joinpath(data_path, "data_$(case_system).bson")
    results = BSON.load(file_path)

    ref_data = ChainRulesCore.ignore_derivatives() do
        PowerModels.build_ref(data; ref_extensions=Dict())
    end[:it][:pm][:nw][0]
    gen_ids = sort(collect(keys(ref_data[:gen])))
    mat_data = results[:results]["DispatchDC"]  # tamaño (n_samples, n_gens)
    dc_cost_pm = results[:results]["ObjDC"]
    #mat_data = pg


    nsamples = size(mat_data, 1)

    # Inicializar vector de costos por muestra
    dc_costs = zeros(nsamples)

    for (i, g) in enumerate(gen_ids)
        cost = ref_data[:gen][g]["cost"]
        c2, c1, c0 = cost
        #println(cost)
        pg = mat_data[:, i]  # columna i: generación del generador g en todas las muestras

        dc_costs .+= c2 .* pg .^ 2 .+ c1 .* pg .+ c0  # vectorizado por fila
    end

    return dc_costs, mat_data, dc_cost_pm  # Vector de tamaño (n_samples,)
end


function summarize_results(real, pred, dc)

    err_self = abs.(real .- pred) ./ real .* 100
    err_dc = abs.(real .- dc) ./ real .* 100

    cor_self = cor(real, pred)
    cor_dc = cor(real, dc)

    return DataFrame(
        Metric=["Maximum", "Minimum", "Mean", "Correlation"],
        Error_Self=[maximum(err_self), minimum(err_self), mean(err_self), cor_self],
        Error_DC=[maximum(err_dc), minimum(err_dc), mean(err_dc), cor_dc]
    )
end

function calculate_cost_feas(Pg, data)

    ref_data = ChainRulesCore.ignore_derivatives() do
        PowerModels.build_ref(data; ref_extensions=Dict())
    end[:it][:pm][:nw][0]
    gen_ids = sort(collect(keys(ref_data[:gen])))


    nsamples = size(Pg, 1)

    # Inicializar vector de costos por muestra
    feas_costs = zeros(nsamples)

    for (i, g) in enumerate(gen_ids)
        cost = ref_data[:gen][g]["cost"]
        c2, c1, c0 = cost
        pg = Pg[:, i]
        #pg = Pg # columna i: generación del generador g en todas las muestras

        feas_costs .+= c2 .* pg .^ 2 .+ c1 .* pg .+ c0  # vectorizado por fila
    end
    return feas_costs
end

end
