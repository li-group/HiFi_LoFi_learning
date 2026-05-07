using Pkg
Pkg.activate("Gnn_environment")
#cd(joinpath(@__DIR__, ".."))

include("../src/Self_GNN_PM.jl")
using .Self_GNN_PM

using BSON: @load,@save
using BSON
using Flux
using GraphNeuralNetworks
using GraphNeuralNetworks: batch
using JSON
using MLUtils: DataLoader
using Statistics

const PATH_DATA = "test_systems/data-ogf/"
const SAVE_DIR = "inference_results"
#const SYSTEM_NAMES = ["gaslib-40-E", "gaslib-135-F"]
const SYSTEM_NAMES = ["gaslib-40-E"]
const NNFEATURES = 4
const NEFEATURES = 3
const HEADS = 5
const HIDDEN_DIM = 512
for system_name in SYSTEM_NAMES
    self_model_file = joinpath("results", "Model_self_best_GM_$system_name.bson")
    
    #=@load self_model_file model_selfsupervised
    
    # OLD
    ps = Flux.params(model_selfsupervised)

    weights = Vector{Dict}(undef, length(ps))

    for (i, p) in enumerate(ps)
        weights[i] = Dict(
            "size" => size(p),
            "data" => Array(p)
        )
    end

    metadata = Dict(
        "num_params" => length(ps)
    )

    @save  "self_weights_$system_name.bson" weights metadata=#
    

    # NEW
    model = Self_supervised_GNN_GM(
        NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM
    )
    @load "self_weights_$system_name.bson" weights metadata

    ps = Flux.params(model)

    # --- Check number of parameters ---
    @assert length(ps) == metadata["num_params"] "Parameter count mismatch!"

    for (i, p) in enumerate(ps)
        w = weights[i]

        # --- Shape check ---
        @assert size(p) == w["size"] "Shape mismatch at param $i"

        # --- Assign ---
        p .= w["data"]
    end
end