module Models

export Semi_Supervised_GNN_GM, InizitilizationGNN_GM, Self_supervised_GNN_GM

using Flux
using GraphNeuralNetworks
#using ..Utils: input
using ..Layers: LRWPLayer_GM, WPLayer_GM, LRWPLayer_GM_truncated



###############################################################
# Struct: Self_supervised_GNN_GM
#
# Purpose:
#   Graph Neural Network architecture for GasModels that:
#
#     1) Learns node and edge latent representations
#     2) Predicts intermediate physical quantities
#     3) Embeds LF (LRWP) optimization layer
#     4) Embeds HF (WP) optimization layer
#
# This is a physics-aware, differentiable pipeline:
#
#   GNN → LF Layer → HF Layer → Final Output
#
###############################################################

struct Self_supervised_GNN_GM

    # Graph Attention Convolution layers
    conv1
    conv2
    conv3

    # Node prediction MLP
    final_mlp
    final_mlp2

    # Edge prediction MLP
    final_mlp_edge
    final_mlp2_edge

    # Activation function
    soft

    # Differentiable optimization layers
    diff_lf     # Low-fidelity layer (LRWP)
    diff_hf     # High-fidelity layer (WP)

end


###############################################################
# Constructor
#
# Inputs:
#   nnfeatures  -> number of node features
#   nefeatures  -> number of edge features
#   heads       -> number of attention heads
#   hidden_dim  -> latent embedding size
#
###############################################################

function Self_supervised_GNN_GM(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)

    Self_supervised_GNN_GM(

        #######################################################
        # GATv2 Layer 1
        #
        # Input: (node_features, edge_features)
        # Output: hidden_dim × heads
        #
        #######################################################
        GATv2Conv(
            (nnfeatures, nefeatures) => hidden_dim,
            add_self_loops=false,
            bias=false,
            heads=heads
        ),

        #######################################################
        # GATv2 Layer 2
        #
        # Input: heads * hidden_dim
        # Output: hidden_dim/2 × heads
        #######################################################
        GATv2Conv(
            (heads * hidden_dim, nefeatures) => Int(hidden_dim / 2),
            add_self_loops=false,
            bias=false,
            heads=heads
        ),

        #######################################################
        # GATv2 Layer 3
        #
        # Further reduces embedding dimension
        #######################################################
        GATv2Conv(
            (heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4),
            add_self_loops=false,
            bias=false,
            heads=heads
        ),

        #######################################################
        # Node-level prediction MLP
        #
        # Produces intermediate node quantities
        #######################################################
        Dense(Int(hidden_dim / 4 * heads) => 64),
        Dense(64 => 1),

        #######################################################
        # Edge-level prediction MLP
        #
        # Uses concatenated (xi, xj - xi)
        #######################################################
        Dense(Int(2 * hidden_dim / 4 * heads) => 64),
        Dense(64 => 1),

        #######################################################
        # Activation function (Softplus ensures positivity)
        #######################################################
        softplus,

        #######################################################
        # Optimization layers
        #######################################################
        LRWPLayer_GM,   # Low-fidelity (linear relaxed Weymouth)
        WPLayer_GM      # High-fidelity (nonlinear Weymouth)
    )
end


###############################################################
# Forward Pass
#
# Inputs:
#   gw       -> Graph wrapper (contains graph + GasModels data)
#   init_val -> initialization for HF layer
#
# Pipeline:
#
#   1) Message Passing (GATv2 layers)
#   2) Node & Edge prediction
#   3) LF optimization layer
#   4) HF optimization layer
#
###############################################################

function (model::Self_supervised_GNN_GM)(gw, init_val)

    g = gw.graph
    data = gw.data

    batch_size = Int(g.num_graphs)

    ###########################################################
    # 1) Message Passing
    ###########################################################

    x = g.x

    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus

    ###########################################################
    # 2) Build edge latent features
    #
    # For each edge:
    #   (xi, xj - xi)

    ###########################################################

    edge_features = apply_edges(
        (xi, xj, e) -> vcat(xi, xj .- xi),
        g,
        xi=x,
        xj=x,
        e=g.e
    )

    ###########################################################
    # 3) Node prediction head
    ###########################################################

    x = model.final_mlp(x) |> softplus
    x = model.final_mlp2(x)

    ###########################################################
    # 4) Edge prediction head
    ###########################################################

    edge_x = model.final_mlp_edge(edge_features) |> softplus
    edge_x = model.final_mlp2_edge(edge_x)

    ###########################################################
    # 5) Ensure positivity (physical feasibility)
    ###########################################################

    x = model.soft.(x)
    edge_x = model.soft.(edge_x)

    ###########################################################
    # 6) LF Optimization Layer
    #
    # Injects physics constraints (linear relaxed)
    # Produces physically consistent intermediate solution.
    ###########################################################

    x = model.diff_lf(gw, x, edge_x)

    ###########################################################
    # 7) HF Optimization Layer
    #
    # Refines LF solution using full nonlinear physics.
    # Fully differentiable via DiffOpt / custom rrules.
    ###########################################################

    x = model.diff_hf(gw, x, init_val)

    return x
end


#! Need modification (extra edge embeddings)
# Define Semi Supervised GNN model
struct Semi_Supervised_GNN_GM
    # Graph Attention Convolution layers
    conv1
    conv2
    conv3

    # Node prediction MLP
    final_mlp
    final_mlp2

    # Edge prediction MLP
    final_mlp_edge
    final_mlp2_edge

    # Activation function
    soft

    # Differentiable optimization layers
    diff_lf     # Low-fidelity layer (LRWP)
end

function Semi_Supervised_GNN_GM(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    Semi_Supervised_GNN_GM(GATv2Conv(
            (nnfeatures, nefeatures) => hidden_dim,
            add_self_loops=false,
            bias=false,
            heads=heads
        ), GATv2Conv(
            (heads * hidden_dim, nefeatures) => Int(hidden_dim / 2),
            add_self_loops=false,
            bias=false,
            heads=heads
        ), GATv2Conv(
            (heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4),
            add_self_loops=false,
            bias=false,
            heads=heads
        ),
        Dense(Int(hidden_dim / 4 * heads) => 64),
        Dense(64 => 1), Dense(Int(2 * hidden_dim / 4 * heads) => 64),
        Dense(64 => 1),
        softplus,
        LRWPLayer_GM,   # Low-fidelity (linear relaxed Weymouth)
    )
end

function (model::Semi_Supervised_GNN_GM)(gw)
    g = gw.graph
    data = gw.data

    batch_size = Int(g.num_graphs)

    x = g.x

    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus

    edge_features = apply_edges(
        (xi, xj, e) -> vcat(xi, xj .- xi),
        g,
        xi=x,
        xj=x,
        e=g.e
    )

    x = model.final_mlp(x) |> softplus
    x = model.final_mlp2(x)

    edge_x = model.final_mlp_edge(edge_features) |> softplus
    edge_x = model.final_mlp2_edge(edge_x)

    x = model.soft.(x)
    edge_x = model.soft.(edge_x)

    x = model.diff_lf(gw, x, edge_x)

    return x
end

# Define the initialize model for the GNN (β_{i} ≈ 1)
struct InizitilizationGNN_GM
    # in_x
    conv1
    conv2
    conv3
    final_mlp
    final_mlp2
    final_mlp_edge
    final_mlp2_edge
    #final_mlp3
end



function InizitilizationGNN_GM(nnfeatures::Int, nefeatures::Int, heads::Int, hidden_dim::Int)
    InizitilizationGNN_GM(
        # input,
        GATv2Conv((nnfeatures, nefeatures) => hidden_dim, add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * hidden_dim, nefeatures) => Int(hidden_dim / 2), add_self_loops=false, bias=false, heads=heads),
        GATv2Conv((heads * Int(hidden_dim / 2), nefeatures) => Int(hidden_dim / 4), add_self_loops=false, bias=false, heads=heads),
        # Dense(Int(2 * hidden_dim / 4 * heads) => 64),
        Dense(Int(hidden_dim / 4 * heads) => 64),

        #Dense(Int(hidden_dim) => 64),
        Dense(64 => 1),
        Dense(Int(2 * hidden_dim / 4 * heads) => 64),
        Dense(64 => 1),
    )
end

function (model::InizitilizationGNN_GM)(gw)
    g = gw.graph
    data = gw.data
    # x = model.in_x(g.x)
    x = g.x
    x = model.conv1(g, x, g.e) |> softplus
    x = model.conv2(g, x, g.e) |> softplus
    x = model.conv3(g, x, g.e) |> softplus
    # println("shape of edge_x0", x)
    # Apply apply_edges to get edge-level features


    # println("value edge", edge_features)
    # println("shape of edge_x1", size(edge_features))
    # Final MLPs applied to edge embeddings

    edge_features = apply_edges((xi, xj, e) -> vcat(xi, xj .- xi), g, xi=x, xj=x, e=g.e)
    x = model.final_mlp(x) |> softplus
    println(size(x))
    x = model.final_mlp2(x) |> softplus



    edge_x = model.final_mlp_edge(edge_features) |> softplus
    # println("shape of edge_x2", size(edge_x))

    edge_x = model.final_mlp2(edge_x) |> softplus


    # println("shape of edge_x3", size(edge_x))
    # x = model.final_mlp(x) |> softplus
    # x = model.final_mlp2(x) |> softplus
    # #x = model.final_mlp3(x) |> softplus
    npipes = length(keys(data["pipe"]))

    # + length(keys(data["compressor"]))
    njunctions = length(g.x[1, :])
    batch_size = Int(g.num_graphs)
    x = reshape(x, njunctions, batch_size)
    edge_x = reshape(edge_x, npipes, batch_size)

    # return reshape(edge_x, npipes, batch_size)
    return (x, edge_x)
end







Flux.@functor InizitilizationGNN_GM
Flux.@functor Self_supervised_GNN_GM
# Flux.@functor Self_supervised_GNN_GM_truncated
Flux.@functor Semi_Supervised_GNN_GM
end

