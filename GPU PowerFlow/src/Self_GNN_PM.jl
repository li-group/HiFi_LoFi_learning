module Self_GNN_PM

export
    GraphWithRef, dataset_bson_path, load_system_powermodels, load_generation_data, process_system_powermodels, process_system_powermodels_beta, load_beta_values,
    DCPMLayer, DCPMLayerStruct, build_DCPMLayer, DCPMLayerTruncado, ACPMLayer, ACPMLayer_Array,
    InizitilizationGNN, Self_supervised_GNN, Self_supervised_GNN_truncated, Semi_Supervised_GNN, Full_Supervised_GNN, Self_projection_supervised_GNN, Self_supervised_GNN_EXA, Self_supervised_GNN_EXA_truncated, input, get_target, loss_fn, loss_semisupervised, data_graph_powermodels, ACPM, DCPM, run_ac_dc_batch_from_data, feasibility, process_system_from_data_contingencies, bwd_times_dc, fwd_times_dc, bwd_times_ac, fwd_times_ac, evaluate_best_model, evaluate_model_HF, evaluate_best_model_HF, compute_dc_cost, ACPMLayerExa, ACPMExa, DCPMExa, DCPMLayerExa

include("Utils.jl")
include("DataLoader.jl")
include("Layers.jl")
include("RRules.jl")
include("Models.jl")
include("InferenceUtils.jl")

#include("simulation/generate_samples.jl")

#using .BatchRunner

using .DataLoader
using .Layers
using .Models
using .Utils
using .InferenceUtils
using .RRules

end
