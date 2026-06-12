module Utils

export input, get_target, loss_fn, loss_semisupervised

using Flux
using Flux.Losses: mse

abstracted_input = nothing

function input(x::AbstractMatrix)
    global abstracted_input = x
    return Float64.(x)
end

function get_target(gw)
    data = gw.data
    nbus = length(keys(data["bus"]))
    nbatch = gw.graph.num_graphs
    return ones(nbus, nbatch)
end
function loss_fn(model, gw)
    ŷ = model(gw)
    y = get_target(gw)
    println(size(gw.graph.y))
    return Flux.Losses.mse(ŷ, y)
end

function loss_semisupervised(model, gw)
    ŷ = Array(model(gw))
    y = gw.graph.y
    return Flux.Losses.mse((ŷ), y)
end

end
