module Utils

export input, get_target, loss_fn, loss_semisupervised

using Flux
using JuMP
using Flux.Losses: mse


"""   get_target(gw)
    Extracts the target values from the graph `gw`.
    - `gw`: A GraphWithRef object containing the graph data.
    Returns a matrix of target values, where each column corresponds to a bus and each row corresponds to a batch.
"""

function get_target(gw)
    data = gw.data
    # nbus = length(keys(data["bus"]))
    # nbus = length(keys(data["pipe"]))
    # +length(keys(data["compressor"]))
    njunctinos = length(gw.graph.x[1,:])  # number of junctions in the graph
    nbranches = length(gw.graph.e[1,:])  # number of branches in the graph
    # println(nbus, " numero de junctions")
    nbatch = gw.graph.num_graphs
    # ini = [-0.011041343832952803
    #     -0.00739871968595069
    #     -0.0021347057401281807
    #     -0.0010189198700911362
    #     0
    #     0]
    # return ones(nbus, nbatch)
    return ones(njunctinos, nbatch), ones(nbranches, nbatch)
end

"""   loss_fn(model, gw)
    Computes the mean squared error loss between the model predictions and the target values usefull for initialization GNN.
    - `model`: The GNN model to be evaluated.
    - `gw`: A GraphWithRef object containing the graph data.
    Returns the mean squared error loss.
"""
function loss_fn(model, gw)
    ŷ = model(gw)
    ŷ_1 = ŷ[1]
    ŷ_2 = ŷ[2]
    # println("salida:        ", ŷ)
    y1,y2 = get_target(gw)
    #println(size(gw.graph.y))
    # println(size(ŷ), "tamaño de salida y target")
    # println("target:        ", y1)
    # println("target:        ", y2)
    # println("pred:          ", ŷ_1)
    return Flux.Losses.mse(100 * ŷ_1, 100 * y1)+ Flux.Losses.mse(100 * ŷ_2, 100 * y2)
end

"""   loss_semisupervised(model, gw)
    Computes the mean squared error loss for semi-supervised & full-supervised learning.
    - `model`: The GNN model to be evaluated.
    - `gw`: A GraphWithRef object containing the graph data.
    Returns the mean squared error loss.
"""

function loss_semisupervised(model, gw)
    m = Array(model(gw))
    # println(m)
    # println(m[1])
    # println(m[2])
    fg_pred = m
    # fg_pred = m
    # Targets verdaderos
    fg_target = gw.graph.y[1,:]

    # println("value_fg_ped", fg_pred)
    # println("value fg target", fg_target)
    # ft_target = gw.graph.y[2:6]

    # Normaliza tanto predicción como target (para que mse funcione)
    fg_pred_scaled = fg_pred ./ max.(fg_target, 1e-7)
    # ft_pred_scaled = ft_pred ./ max.(ft_target, 1e-7)


    fg_target_scaled = ones(length(fg_target))
    # ft_target_scaled = ones(length(ft_target))

    ŷ = vcat(fg_pred_scaled)
    y = vcat(fg_target_scaled)

    # println(ŷ)


    return Flux.Losses.mse(ŷ, y)
end

end
