module Self_GNN_PM

export GraphWithRef,
    load_system_powermodels, load_generation_data, process_system_powermodels, process_system_powermodels_beta, load_beta_values,
    DCPMLayer, DCPMLayerTruncado, ACPMLayer,
    Semi_Supervised_GNN_GM, Self_supervised_GNN_GM,
    input, get_target, loss_fn, loss_semisupervised, data_graph_powermodels, data_graph_gasmodels, ACPM, DCPM, run_ac_dc_batch_from_data, feasibility, process_system_from_data_contingencies, bwd_times_lf, fwd_times_lf, bwd_times_hf, fwd_times_hf, process_system_gasmodels, load_system_gasmodels,
    LRWP, WP, InizitilizationGNN_GM, LRWPLayer_GM, WPLayer_GM, load_GM_data, Self_supervised_GNN_GM_truncated, evaluate_model_GM, Feasibility_GM, LRWPLayer_GM_truncated, WPLayer_GM_wt_objslack, LRDWP, CRDWP
include("Utils.jl")
include("DataLoader.jl")
include("Layers.jl")
include("Models.jl")
include("InferenceUtils.jl") # Take care here
#include("simulation/generate_samples.jl")

#using .BatchRunner

using .DataLoader
using .Layers
using .Models
using .Utils
using .InferenceUtils

end
