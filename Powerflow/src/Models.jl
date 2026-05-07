module Models

export InizitilizationGNN, Self_supervised_GNN, Self_supervised_GNN_truncated, Semi_Supervised_GNN, Full_Supervised_GNN, Self_projection_supervised_GNN

using Flux
using GraphNeuralNetworks
using ..DataLoader: GraphWithRef
using ..Layers: DCPMLayer, DCPMLayerStruct, build_DCPMLayer,
    ACPMLayer, DCPMLayerBatch, ACPMLayerV2, ACPMLayerV2Struct, build_ACPMLayerV2, ACPM_PROJECTION, DCPMLayerTruncado

grad_log_dc = Vector{Dict}()
# Define the initialize model for the GNN (β_{i} ≈ 1)
struct InizitilizationGNN
    # in_x
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    #final_mlp3
end



function InizitilizationGNN(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    InizitilizationGNN(
        # input,
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim, add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2), add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4), add_self_loops=false, bias=false, heads=heads),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        #Dense(Int(hidden_dim) => 64),
        Dense(64 => 1),
    )
end

function (model::InizitilizationGNN)(gw)
    g = gw.graph
    data = gw.data
    x = g.x
    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus
    x = model.final_mlp(x) |> softplus
    x = model.final_mlp2(x) |> softplus
    #x = model.final_mlp3(x) |> softplus


    nbus = length(keys(data["bus"]))
    batch_size = Int(size(x, 2) / nbus)
    return reshape(x, nbus, batch_size)
end

# Define Self Supervised GNN model
struct Self_supervised_GNN
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    soft
    # diff::DCPMLayerStruct        # DC layer object
    diff   # DC layer object
    # acdiff::ACPMLayerV2Struct    # AC layer object (new)
    acdiff
end


function Self_supervised_GNN(nnfeatures::Int, nefeatures::Int,
    heads::Int, hidden_dim::Int)
    Self_supervised_GNN(
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim,
            add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2),
            add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4),
            add_self_loops=false, bias=false, heads=heads),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        Dense(64 => 1),
        softplus,
        # build_DCPMLayer(gw),      # build DC layer once
        DCPMLayer,
        # build_ACPMLayerV2(gw),    # build AC layer once
        ACPMLayer
    )
end



function (model::Self_supervised_GNN)(gw, ac_init)
    g = gw.graph
    data = gw.data
    x = g.x
    x1 = model.conv1(g, x, g.e) |> softplus
    x2 = model.conv2(g, x1, g.e) |> softplus
    x3 = model.conv3(g, x2, g.e) |> softplus
    x4 = model.final_mlp(x3) |> softplus
    x5 = model.final_mlp2(x4)

    nbus = length(keys(data["bus"]))
    batch_size = Int(size(x5, 2) / nbus)
    x6 = reshape(x5, nbus, batch_size)
    x7 = model.soft.(x6)

    Pd = reshape(g.x[1, :], nbus, batch_size)
    Gs = reshape(g.x[6, :], nbus, batch_size)

    rhs = Pd .* x7 .+ Gs

    # DC layer forward (reusing the model inside model.diff)
    # x8 = DCPMLayer(model.diff, rhs)   # instead of model.diff(gw, rhs)
    x8 = model.diff(gw, rhs)
    # AC layer remains the same
    # x = ACPMLayerV2(model.acdiff, gw, x8)
    x = model.acdiff(gw, x8, ac_init)
    # println("valor de ac", x["objective"])
    return x
end



# Define Self Supervised GNN model for inference without ACPM
struct Self_supervised_GNN_truncated
    # in_x
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    #final_mlp3
    #soft
    soft
    diff
end

function Self_supervised_GNN_truncated(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    Self_supervised_GNN_truncated(
        # input,
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim, add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2), add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4), add_self_loops=false, bias=false, heads=heads),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        #Dense(Int(hidden_dim) => 64),
        Dense(64 => 1),
        softplus,
        DCPMLayerTruncado
    )
end

function (model::Self_supervised_GNN_truncated)(gw)
    g = gw.graph
    data = gw.data
    # x = model.in_x(g.x)
    x = g.x
    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus
    x = model.final_mlp(x) |> softplus
    x = model.final_mlp2(x)
    #x = model.final_mlp3(x)


    nbus = length(keys(data["bus"]))
    batch_size = Int(size(x, 2) / nbus)
    x = reshape(x, nbus, batch_size)
    x = model.soft.(x)

    Pd = reshape(g.x[1, :], nbus, batch_size)
    Gs = reshape(g.x[6, :], nbus, batch_size)
    rhs = Pd .* x .+ Gs

    return model.diff(data, rhs)
end

# Define Semi Supervised GNN model
struct Semi_Supervised_GNN
    # in_x
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    #final_mlp3
    soft
    diff
end

function Semi_Supervised_GNN(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    Semi_Supervised_GNN(
        # input,
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim, add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2), add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4), add_self_loops=false, bias=false, heads=heads),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        #Dense(Int(hidden_dim) => 64),
        Dense(64 => 1),
        softplus,
        DCPMLayerBatch
        #DCPMLayer
    )
end

function (model::Semi_Supervised_GNN)(gw)
    g = gw.graph
    data = gw.data
    # x = model.in_x(g.x)
    x = g.x
    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus
    x = model.final_mlp(x) |> softplus
    x = model.final_mlp2(x)
    #x = model.final_mlp3(x)

    nbus = length(keys(data["bus"]))
    batch_size = Int(size(x, 2) / nbus)
    x = reshape(x, nbus, batch_size)
    x = model.soft.(x)

    Pd = reshape(g.x[1, :], nbus, batch_size)
    Gs = reshape(g.x[6, :], nbus, batch_size)

    rhs = Pd .* x .+ Gs
    x = model.diff(data, rhs)
    println("pasa por PG")
    return x
end

# Define Full Supervised GNN model
struct Full_Supervised_GNN
    in_x
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    #final_mlp3
    soft
    #relu
end

function Full_Supervised_GNN(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    Full_Supervised_GNN(
        input,
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim, add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2), add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4), add_self_loops=false, bias=false, heads=heads),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        #Dense(Int(hidden_dim) => 64),
        Dense(64 => 1),
        softplus
    )
end

function (model::Full_Supervised_GNN)(gw)
    g = gw.graph
    data = gw.data
    x = model.in_x(g.x)
    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus
    x = model.final_mlp(x) |> softplus
    x = model.final_mlp2(x)
    #x = model.final_mlp3(x)

    nbus = length(keys(data["bus"]))
    batch_size = Int(size(x, 2) / nbus)
    x = reshape(x, nbus, batch_size)
    x = model.soft.(x)
    return x
end

struct Self_projection_supervised_GNN
    # in_x
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    #final_mlp3
    soft
    diff
    acdiff
end

function Self_projection_supervised_GNN(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    Self_projection_supervised_GNN(
        # input,
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim, add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2), add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4), add_self_loops=false, bias=false, heads=heads),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        #Dense(Int(hidden_dim) => 64),
        Dense(64 => 1),
        softplus,
        DCPMLayer,
        ACPM_PROJECTION
    )
end

function (model::Self_projection_supervised_GNN)(gw)
    g = gw.graph
    data = gw.data
    x = g.x
    x1 = model.conv1(g, x, g.e) |> softplus
    x2 = model.conv2(g, x1, g.e) |> softplus
    x3 = model.conv3(g, x2, g.e) |> softplus
    x4 = model.final_mlp(x3) |> softplus
    x5 = model.final_mlp2(x4)
    #x = model.final_mlp3(x)

    nbus = length(keys(data["bus"]))
    batch_size = Int(size(x5, 2) / nbus)
    x6 = reshape(x5, nbus, batch_size)
    x7 = model.soft.(x6)

    Pd = reshape(g.x[1, :], nbus, batch_size)
    Gs = reshape(g.x[6, :], nbus, batch_size)

    rhs = Pd .* x7 .+ Gs

    # println("valor de β ", x7)
    x8 = model.diff(gw, rhs)
    x = model.acdiff(gw, x8)
    return x
end



Flux.@functor InizitilizationGNN
Flux.@functor Self_supervised_GNN
Flux.@functor Self_projection_supervised_GNN
Flux.@functor Self_supervised_GNN_truncated
Flux.@functor Semi_Supervised_GNN
Flux.@functor Full_Supervised_GNN

end

