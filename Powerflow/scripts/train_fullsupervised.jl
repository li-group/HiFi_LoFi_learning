using Pkg

Pkg.activate("Gnn_environment")
include("../src/Self_GNN_PM.jl")
using .Self_GNN_PM

#using Self_GNN_PM
using Flux
using Ipopt
using MadNLP
using Flux.Optimise: update!
using MLUtils: DataLoader
using Plots
using GraphNeuralNetworks
using GraphNeuralNetworks: batch
using BSON: @save
using DataFrames
#export nnfeatures, nefeatures


# Parámetros
nnfeatures = 6
nefeatures = 5
batchsize = 128

# Cargar datos
 
system_name = "pglib_opf_case14_ieee"
system_name = "pglib_opf_case30_ieee"
system_name = "pglib_opf_case39_epri"
system_name = "pglib_opf_case57_ieee"
system_name = "pglib_opf_case118_ieee"
#system_name = "pglib_opf_case300_ieee"
#system_name = "pglib_opf_case1354_pegase"
#system_name = "pglib_opf_case2383wp_k"

path_data = "test_systems/"
systems_train = [(process_system_powermodels_beta(system_name, path_data))]

 #for name in system_names_train

# Un sistema de muestra para construir graphs
ref_train, x_train, e_train, y_train, edge_index_train = systems_train[1]


From_sample = 1
To_sample = 1
# Generar graphs
graphs_train = data_graph_powermodels(ref_train, x_train, e_train, y_train, From_sample, To_sample, GNNGraph[])

# Collate personalizado
const _ref_train = ref_train
function custom_collate(graph_batch)
    return GraphWithRef(batch(graph_batch), _ref_train)
end

train_loader = DataLoader(graphs_train, batchsize=batchsize, shuffle=true, collate=custom_collate)


# # Suponiendo que ya tienes esto listo:
# model_initialization = InizitilizationGNN(nnfeatures, nefeatures)

# # Crea el optimizador (puedes cambiar el learning rate)
# opt = Flux.Adam(1e-4)
# ps = Flux.params(model_initialization)

# for epoch in 1:100
#     for batch in train_loader
#         loss, gs = Flux.withgradient(ps) do
#             loss_fn(model_initialization, batch)
#         end
#         println("Epoch $epoch - Loss: $loss")
#         Flux.Optimise.update!(opt, ps, gs)
#     end
# end


# Suponiendo que ya tienes esto listo:
model_fullsupervised = Full_Supervised_GNN(nnfeatures, nefeatures)

# Flux.loadparams!(model_fullsupervised, Flux.params(model_initialization))

data_probe = []
for batch in train_loader
    push!(data_probe, batch)
end

a = Array(model_fullsupervised(data_probe[1]))
data_probe[1].graph.y
# Crea el optimizador (puedes cambiar el learning rate)
opt = Flux.AdamW(1e-4)
ps = Flux.params(model_fullsupervised)


using BSON, JSON, DataFrames, Flux, Plots

# Inicializa listas de tracking
epoch_losses = Float64[]
epoch_times = Float64[]

best_loss = Inf
best_m = nothing
best_ps = nothing
best_epoch = 0
patience_counter = 0

epochs = 3000
patience = 20
threshold = 1e-12

# Reset timers si usas fwd/bwd tracking
using .Self_GNN_PM: fwd_times_dc, bwd_times_dc, fwd_times_ac, bwd_times_ac
empty!(fwd_times_dc)
empty!(bwd_times_dc)
empty!(fwd_times_ac)
empty!(bwd_times_ac)

for epoch in 1:epochs
    total_loss = 0.0
    num_batches = 0

    epoch_time = @elapsed begin
        for batch in train_loader
            batch_loss, grads = Flux.withgradient(ps) do
                loss_semisupervised(model_fullsupervised, batch)
            end

            Flux.Optimise.update!(opt, ps, grads)
            total_loss += batch_loss
            num_batches += 1
        end
    end

    epoch_avg_loss = total_loss / num_batches
    push!(epoch_losses, epoch_avg_loss)
    push!(epoch_times, epoch_time)

    println("📊 Epoch $epoch - Avg Loss = $(round(epoch_avg_loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")

    if epoch_avg_loss < threshold
        println("🛑 Early stopping: loss below threshold ($threshold).")
        best_m = deepcopy(model_fullsupervised)
        best_ps = deepcopy(ps)
        best_loss = epoch_avg_loss
        best_epoch = epoch
        break
    elseif epoch_avg_loss < best_loss
        best_loss = epoch_avg_loss
        patience_counter = 0
        best_m = deepcopy(model_fullsupervised)
        best_ps = deepcopy(ps)
        best_epoch = epoch
        println("💾 New best model found at epoch $epoch with loss $best_loss")
    else
        patience_counter += 1
        println("📉 No improvement. Patience: $patience_counter / $patience")
        if patience_counter >= patience
            println("🛑 Early stopping: no improvement after $patience epochs.")
            break
        end
    end
end

# Actualiza modelo con el mejor
model_fullsupervised = best_m
ps = best_ps

# Guardar el mejor modelo entrenado
filename = "Model_fullsupervised_$system_name.bson"
@save "results/$filename" model_fullsupervised ps
println("✅ Final best model saved at epoch $best_epoch with loss $best_loss as $filename")

# Crear tabla de tiempos
times_df = DataFrame(Epoch_Time=sum(epoch_times),
    Forward_Time_DC=sum(fwd_times_dc),
    Backward_Time_DC=sum(bwd_times_dc),
    Forward_Time_AC=sum(fwd_times_ac),
    Backward_Time_AC=sum(bwd_times_ac)
)
times_df.Total_Time = times_df.Forward_Time_DC .+ times_df.Backward_Time_DC .+
                      times_df.Forward_Time_AC .+ times_df.Backward_Time_AC

println(times_df)

# Exportar métricas como JSON
metrics_dict = Dict(
    "Epoch_Time" => epoch_times,
    "Epoch_Losses" => epoch_losses,
    "Forward_Time_DC" => fwd_times_dc,
    "Backward_Time_DC" => bwd_times_dc,
    "Forward_Time_AC" => fwd_times_ac,
    "Backward_Time_AC" => bwd_times_ac,
    "Total_Time" => sum(fwd_times_dc) + sum(bwd_times_dc) + sum(fwd_times_ac) + sum(bwd_times_ac)
)

json_filename = "results/times_metrics_full_$system_name.json"
open(json_filename, "w") do io
    JSON.print(io, metrics_dict)
end
println("📁 Training metrics saved in: $json_filename")

# Gráfico de pérdida
plot(epoch_losses, xlabel="Epoch", ylabel="Loss", title="Full Supervised Training Loss")

a = Array(model_fullsupervised(data_probe[1]))

