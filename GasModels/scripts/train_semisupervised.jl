using Pkg
Pkg.activate("Gnn_environment")

include("../src/Self_GNN_PM.jl")
using .Self_GNN_PM

using Flux
using Flux.Optimise: update!
using MLUtils: DataLoader
using GraphNeuralNetworks
using GraphNeuralNetworks: batch
using BSON: @save
using DataFrames
using Random
using JSON

const SYSTEM_NAMES = ["gaslib-40-E", "gaslib-135-F"]
const PATH_DATA = "test_systems/data-ogf/"

const NNFEATURES = 4
const NEFEATURES = 3
const HEADS = 5
const HIDDEN_DIM = 512

const INIT_EPOCHS = 100
const EPOCHS = 1000
const BATCHSIZE = 1

const PATIENCE = 700
const THRESHOLD = 1e-20
const TOTAL_MINUTES_BY_SYSTEM = Dict(
    "gaslib-40-E" => 3,
    "gaslib-135-F" => 6,
)
const MAX_SAVES = 4

function build_train_loader(ref_train, graphs_train)
    function custom_collate(graph_batch)
        return GraphWithRef(batch(graph_batch), ref_train)
    end

    return DataLoader(
        graphs_train,
        batchsize=BATCHSIZE,
        shuffle=true,
        collate=custom_collate
    )
end

function train_semisupervised_system(system_name::String)
    println("Training semi-supervised model for $system_name")
    total_minutes = get(TOTAL_MINUTES_BY_SYSTEM, system_name, 240)
    save_every_sec = (total_minutes * 60) ÷ MAX_SAVES

    ref_train, graphs_train = process_system_gasmodels(
        system_name,
        PATH_DATA;
        from_numgraph=1,
        to_numgraph=1,
        use_test_data=false
    )

    train_loader = build_train_loader(ref_train, graphs_train)

    Random.seed!(20)

    model_initialization = InizitilizationGNN_GM(
        NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM
    )

    init_opt = Flux.AdamW(1e-4)
    init_ps = Flux.params(model_initialization)

    for epoch in 1:INIT_EPOCHS
        for batch in train_loader
            init_loss, init_grads = Flux.withgradient(init_ps) do
                loss_fn(model_initialization, batch)
            end

            Flux.Optimise.update!(init_opt, init_ps, init_grads)
            println("[$system_name] Init Epoch $epoch | Loss = $(round(init_loss, sigdigits=6))")
        end
    end

    model_semisupervised = Semi_Supervised_GNN_GM(
        NNFEATURES, NEFEATURES, HEADS, HIDDEN_DIM
    )

    Flux.loadparams!(model_semisupervised, Flux.params(model_initialization))

    opt = Flux.ADAMW(1e-5)
    ps = Flux.params(model_semisupervised)

    epoch_losses = Float64[]
    epoch_times = Float64[]
    best_loss = Inf
    best_epoch = 0
    stop_epoch = 0
    patience_counter = 0
    saves_done = 0

    empty!(fwd_times_lf)
    empty!(bwd_times_lf)
    empty!(fwd_times_hf)
    empty!(bwd_times_hf)

    start_time = time()
    #next_save_time = start_time + SAVE_EVERY_SEC

    for epoch in 1:EPOCHS
        total_loss = 0.0
        num_batches = 0

        epoch_time = @elapsed begin
            for batch in train_loader
                batch_loss, grads = Flux.withgradient(ps) do
                    loss_semisupervised(model_semisupervised, batch)
                end

                Flux.Optimise.update!(opt, ps, grads)
                total_loss += batch_loss
                num_batches += 1
            end
        end

        avg_loss = total_loss / num_batches
        push!(epoch_losses, avg_loss)
        push!(epoch_times, epoch_time)

        println("[$system_name] Epoch $epoch | Avg Loss = $(round(avg_loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0

            filename = "Model_semi_best_GM_$system_name.bson"
            @save joinpath("results", filename) model_semisupervised

            println("[$system_name] New best model at epoch $epoch")
        
        end


        current_time = time()

        elapsed_minutes = (current_time - start_time) / 60
        if elapsed_minutes >= total_minutes
            stop_epoch = epoch
            println("[$system_name] Training stopped at epoch $epoch due to time limit ($total_minutes minutes)")
            break
        end
    end

    if stop_epoch == 0
        stop_epoch = length(epoch_losses)
    end

    times_df = DataFrame(
        Epoch_Time=sum(epoch_times),
        Forward_Time_LF=sum(fwd_times_lf),
        Backward_Time_LF=sum(bwd_times_lf),
        Forward_Time_HF=sum(fwd_times_hf),
        Backward_Time_HF=sum(bwd_times_hf)
    )

    times_df.Total_Time =
        times_df.Forward_Time_LF .+
        times_df.Backward_Time_LF .+
        times_df.Forward_Time_HF .+
        times_df.Backward_Time_HF

    times_df.Untracked_Time = times_df.Epoch_Time .- times_df.Total_Time

    println(times_df)

    time_until_best = best_epoch > 0 ? sum(epoch_times[1:best_epoch]) : 0.0
    metrics_dict = Dict(
        "System_Name" => system_name,
        "Best_Epoch" => best_epoch,
        "Best_Loss" => best_loss,
        "Time_Until_Best_Epoch" => time_until_best,
        "Epoch_Losses" => epoch_losses,
        "Epoch_Times" => epoch_times,
        "Epoch_Stop" => stop_epoch,
        "Forward_Time_LF" => fwd_times_lf,
        "Backward_Time_LF" => bwd_times_lf,
        "Forward_Time_HF" => fwd_times_hf,
        "Backward_Time_HF" => bwd_times_hf,
        "GNN_Time" => times_df.Untracked_Time[1],
    )

    open(joinpath("results", "times_metrics_semi_$system_name.json"), "w") do io
        JSON.print(io, metrics_dict)
    end
end

for system_name in SYSTEM_NAMES
    train_semisupervised_system(system_name)
end
