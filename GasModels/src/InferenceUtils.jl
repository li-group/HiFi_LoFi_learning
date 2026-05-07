module InferenceUtils

using ..DataLoader: GraphWithRef
using Flux, PowerModels, ChainRulesCore, MAT, Statistics, GasModels
using DataFrames
export evaluate_model, compute_dc_cost, summarize_results, evaluate_model_beta, calculate_cost_feas, evaluate_model_GM
using BSON


"""    evaluate_model(model, loader)
    Evaluates a model on a data loader.
    - `model`: The model to evaluate Semi - Self supervised.
    - `loader`: A DataLoader containing the graphs to evaluate.
    Returns predictions, real values, slacks_on, and Pg_predicted.
"""

function evaluate_model(model, loader)
    predictions = Float64[]
    real_values = Float64[]
    Pg_predicted = []
    slacks_on = falses(length(loader))

    for (i, graph) in enumerate(loader)
        data = graph.data

        ref_data = ChainRulesCore.ignore_derivatives() do
            PowerModels.build_ref(data; ref_extensions=Dict())
        end[:it][:pm][:nw][0]

        results = model(graph)
        Pg_pred = results["Pg_values"]

        # if sum(results["slacks"]) > 0
        #     graph.graph.x[1, :] .-= results["slacks"]
        #     results = model(graph)
        #     Pg_pred = results["Pg_values"]
        #     slacks_on[i] = true
        # end

        push!(Pg_predicted, Pg_pred)

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

    return predictions, real_values, slacks_on, Pg_predicted
end


function evaluate_model_GM(model, loader, results, from_samples, to_samples)

    cost_self = []
    fg_self = []
    # ft_self = []
    #fg_predicted = []
    # slacks_on = falses(length(loader))

    Obj_HF = results[:results]["Obj_HF"][from_samples:to_samples, :]
    Obj_LF = results[:results]["Obj_LF"][from_samples:to_samples, :]

    # ft_HF = results[:results]["ft_HF"][from_samples:to_samples, :]
    # ft_LF = results[:results]["ft_LF"][from_samples:to_samples, :]
    fg_HF = results[:results]["fg_HF"][from_samples:to_samples, :]
    fg_LF = results[:results]["fg_LF"][from_samples:to_samples, :]

    for (i, graph) in enumerate(loader)
        data = graph.data

        ref_data = GasModels.build_ref(data)[:it][:gm][:nw][0]

        economic_weighting = get_economic_weighting(data)

        transfer_set = [i for (i, x) in ref_data[:transfer] if x["is_dispatchable"] == 1]
        prod_set = [i for (i, x) in ref_data[:receipt] if x["is_dispatchable"] == 1]

        prod_prices = Dict(i => get(ref_data[:receipt][i], "offer_price", 1.0) for i in prod_set)
        transfer_prices = Dict(i =>
            ref_data[:transfer][i]["withdrawal_min"] >= 0.0 ?
            get(ref_data[:transfer][i], "bid_price", 1.0) :
            -get(ref_data[:transfer][i], "offer_price", 1.0) for i in transfer_set)


        fg_pred, ft_pred = model(graph)

        fg_pred = Array(fg_pred)
        ft_pred = Array(ft_pred)


        cost_pred = (
            -sum(transfer_prices[k] * ft_pred[k] for k in keys(transfer_prices)) +
            sum(prod_prices[k] * fg_pred[k] for k in keys(prod_prices))
        )

        push!(fg_self, fg_pred)
        push!(ft_self, ft_pred)


        push!(cost_self, cost_pred)
    end

    fg_self_array = hcat(fg_self...)
    ft_self_array = hcat(ft_self...)

    return Obj_HF, Obj_LF, cost_self, ft_HF, ft_LF, fg_HF, fg_LF, fg_self_array, ft_self_array
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

"""   compute_dc_cost(data_path, case_system, data)
    Computes the DC cost for a given power system case.
    - `data_path`: The path to the data files.
    - `case_system`: The name of the power system case.
    - `data`: The data dictionary containing the power system information.
    Returns a tuple containing the DC costs, matrix data, and DC cost from PowerModels.
"""

function compute_dc_cost(data_path, case_system, data)

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


"""    calculate_cost_feas(Pg, data)
    Calculates the cost of feasible generation for a given power system.
    - `Pg`: A matrix of generation values for each generator across samples.
    - `data`: The data dictionary containing the power system information.
    Returns a vector of costs for each sample.
"""

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


"""    summarize_results(real, pred, dc)
    Summarizes the results of a model evaluation.
    - `real`: The real values (ground truth).
    - `pred`: The predicted values from the model.
    - `dc`: The DC cost values.
    Returns a DataFrame with metrics such as maximum, minimum, mean, and correlation.
"""

function summarize_results(real, pred, dc)

    err_self = abs.(abs.(real .- pred) ./ real .* 100)
    err_dc = abs.(abs.(real .- dc) ./ real .* 100)

    cor_self = cor(real, pred)
    cor_dc = cor(real, dc)

    return DataFrame(
        Metric=["Maximum", "Minimum", "Mean", "Correlation"],
        Error_Self=[maximum(err_self), minimum(err_self), mean(err_self), cor_self],
        Error_DC=[maximum(err_dc), minimum(err_dc), mean(err_dc), cor_dc]
    )
end




end
