using JuMP
using Gurobi
using DiffOpt
using Statistics
using JSON
using Flux
using Zygote
using CSV
using DataFrames

using MathOptInterface
using BSON: @save, @load
using BSON
const MOI = MathOptInterface
using Random
         

using ChainRulesCore
using Statistics
include("Models.jl")
include("DataLoader.jl")
include("Layers.jl")
include("Utils.jl")


const SYSTEM_TIME_BUDGETS = [
    ("instance_1_1", 20),
    ("instance_2_1", 60),
    ("instance_3_1", 60),
    ("instance_4_1", 300),
    ("instance_5_1", 600),
]



function self_supervised_training_pooling(base_instance,total_time)
    Random.seed!(20) 
    empty!(fwd_times_lf)
    empty!(bwd_times_lf)
    empty!(fwd_times_hf)
    empty!(bwd_times_hf)
    ## Data Loading
    #Nominal sample
    reference_instance = parse_instance("instances_final/"*base_instance*".json")
    g_ref, node_id_ref, ranges_ref, edge_maps_ref, X_ref, Eattr_ref =
        build_tripartite_graph(reference_instance; return_features=true)

    src_range, bl_range, trm_range = ranges_ref

    norm = fit_normalizer_from_features(X_ref, src_range, bl_range, trm_range, Eattr_ref)
    rel_ref = split_relation_graphs(g_ref)
    ref_data = (
        instance = reference_instance, 
        g = g_ref, 
        node_id = node_id_ref, 
        edge_maps = edge_maps_ref, 
        rel = rel_ref
    )

    # Samples from distribuiton for tuning
    n_tuning = 25
    
    processed_instances = []
    for i in 1:n_tuning
        path = "instances_gen_vary_train/"*base_instance*"_$i.json"
        inst = parse_instance(path)
        
        # Build static graph components 
        g, node_id, ranges, edge_maps = build_tripartite_graph(inst; normalizer=norm)
        rel = split_relation_graphs(g)
        
        # Store as a named tuple or custom struct
        push!(processed_instances, (
            instance = inst, 
            g = g, 
            node_id = node_id, 
            edge_maps = edge_maps, 
            rel = rel
        ))
    end

    batch_size = 5
    batches = collect(Base.Iterators.partition(processed_instances, batch_size))

    ## GNN loading and initialization
    model_selfsupervised = HeteroPoolingGNN(hdim=32)

    ps = Flux.params(model_selfsupervised)
    opt = Flux.Optimiser(Flux.ADAMW(1e-2))
    l = 100
    epoch = 1
    while l > 0.005
        l, gs = Flux.withgradient(ps) do
            loss_fn_warm_start(model_selfsupervised, ref_data)
        end

        #clip_grads!(gs; clip_norm=1.0)
        Flux.Optimise.update!(opt, ps, gs)

        println("epoch: $epoch, loss = $l ")
        epoch = epoch + 1
    end

    ## Training
    epoch_losses = Float64[]
    epoch_times = Float64[]
    best_loss = Inf
    best_epoch = 0
    n_epoch_conservative = 100
    n_epoch_aggressive = 100
    n_epoch_batch = 8

    opt = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.ADAMW(1e-3))

    for epoch in 1:n_epoch_conservative
        epoch_time = @elapsed begin
            loss, gs = Flux.withgradient(ps) do
                loss_fn_conservative_selfsupervised(model_selfsupervised, ref_data)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end
        push!(epoch_losses,loss)
        push!(epoch_times,epoch_time)
        println("Epoch $epoch | Loss = $(round(loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")
        open("results/pooling_"*base_instance*"_self_supervised_timed.csv", "a") do io
            write(io, "Epoch $epoch | Loss = $(round(loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s \n")
        end
    end

    for epoch in 1:n_epoch_aggressive
        epoch_time = @elapsed begin
            loss, gs = Flux.withgradient(ps) do
                loss_fn_aggressive_selfsupervised(model_selfsupervised, ref_data)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end
        push!(epoch_losses,loss)
        push!(epoch_times,epoch_time)
        println("Epoch $epoch | Loss = $(round(loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")
        open("results/pooling_"*base_instance*"_self_supervised_timed.csv", "a") do io
            write(io, "Epoch $epoch | Loss = $(round(loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s \n")
        end
    end

    epoch = 1
    
    while(sum(epoch_times)<=total_time*60)
        total_loss = 0.0
        epoch_time = @elapsed begin
            for (i, batch) in enumerate(batches)
                # Calculate gradients for the batch of 5
                loss, gs = Flux.withgradient(ps) do
                    loss_fn_batch_selfsupervised(model_selfsupervised, batch)
                end
                
                # Update model parameters
                Flux.Optimise.update!(opt, ps, gs)
                
                total_loss += loss*batch_size
            end
        end
        push!(epoch_losses,total_loss/n_tuning)
        push!(epoch_times,epoch_time)
        println("Epoch $epoch | Loss = $(round(total_loss/n_tuning, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")
        open("results/pooling_"*base_instance*"_self_supervised_timed.csv", "a") do io
            write(io, "Epoch $epoch | Loss = $(round(total_loss/n_tuning, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s \n")
        end
        epoch = epoch+1
        
    end
    filename = "results/Model_self_GM_pooling_tuned_timed_"*base_instance*".bson"
    @save filename model_selfsupervised
    times_df = DataFrame(
        Epoch_Time=sum(epoch_times),
        Forward_Time_LF=sum(fwd_times_lf),
        Backward_Time_LF=sum(bwd_times_lf),
        Forward_Time_HF=sum(fwd_times_hf),
        Backward_Time_HF=sum(bwd_times_hf)
    )
    println(fwd_times_lf)
    println(bwd_times_lf)
    println(fwd_times_hf)
    println(bwd_times_hf)
    times_df.Total_Time_1 =
    times_df.Forward_Time_LF .+
    times_df.Backward_Time_LF .+
    times_df.Forward_Time_HF .+
    times_df.Backward_Time_HF

    times_df.Untracked_Time = times_df.Epoch_Time .- times_df.Total_Time_1

    println(times_df)


    metrics_dict = Dict(
        "instance" => base_instance,
        "Epoch_Losses" => epoch_losses,
        "Epoch_Times" => epoch_times,
        "Forward_Time_LF" => fwd_times_lf,
        "Backward_Time_LF" => bwd_times_lf,
        "Forward_Time_HF" => fwd_times_hf,
        "Backward_Time_HF" => bwd_times_hf,
        "GNN_Time" => times_df.Untracked_Time[1],
    )
    open("results/times_metrics_self_pooling_timed_$base_instance.json", "w") do io
            JSON.print(io, metrics_dict)
    end
end

for (inst,total_time) in SYSTEM_TIME_BUDGETS
    self_supervised_training_pooling(inst,total_time)
end

