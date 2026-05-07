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
using BSON
using BSON: @save, @load
const MOI = MathOptInterface
using Random
       

using ChainRulesCore
using Statistics
include("Models.jl")
include("DataLoader.jl")
include("Layers.jl")
include("Utils.jl")




#=const SYSTEM_TIME_BUDGETS = [
    ("instance_1_1", 20),
    ("instance_2_1", 30),
    ("instance_3_1", 60),
    ("instance_4_1", 240),
    ("instance_5_1", 360),
]=#

const SYSTEM_TIME_BUDGETS = [
    
    ("instance_2_1", 60),
   # ("instance_4_1", 300),
     ("instance_5_1", 600)
]

function semi_supervised_training_pooling(base_instance,total_time)
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
    BSON.@load "instances_gen_vary_train/"*base_instance*"_q.bson" true_q
    ref_data = (
        instance = reference_instance, 
        g = g_ref, 
        node_id = node_id_ref, 
        edge_maps = edge_maps_ref, 
        rel = rel_ref,
        true_q = true_q
    )

    # Samples from distribuiton for tuning
    n_tuning = 25
    
    processed_instances = []
    for i in 1:n_tuning
        #global g,node_id,ranges,edge_maps
        path = "instances_gen_vary_train/"*base_instance*"_$i.json"
        inst = parse_instance(path)
        
        # Build static graph components 
        g, node_id, ranges, edge_maps = build_tripartite_graph(inst; normalizer=norm)
        rel = split_relation_graphs(g)
        BSON.@load "instances_gen_vary_train/"*base_instance*"_$i"*"_q.bson" true_q
        
        # Store as a named tuple or custom struct
        push!(processed_instances, (
            instance = inst, 
            g = g, 
            node_id = node_id, 
            edge_maps = edge_maps, 
            rel = rel,
            true_q = true_q
        ))
    end

    batch_size = 5
    batches = collect(Base.Iterators.partition(processed_instances, batch_size))

    ## GNN loading and initialization
    model_semisupervised = HeteroPoolingGNN(hdim=32)
    

    ps = Flux.params(model_semisupervised)
    opt = Flux.Optimiser(Flux.ADAMW(1e-2))
    l = 100
    epoch = 1
    while l > 0.005
        #global l,epoch
        l, gs = Flux.withgradient(ps) do
            loss_fn_warm_start(model_semisupervised, ref_data)
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
    n_epoch_1sample = 200
    n_epoch_batch = 8

    opt = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.ADAMW(1e-3))


    for epoch in 1:n_epoch_1sample
        #global epoch_losses,epoch_times
        epoch_time = @elapsed begin
            loss, gs = Flux.withgradient(ps) do
                loss_fn_semisupervised(model_semisupervised, ref_data)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end
        push!(epoch_losses,loss)
        push!(epoch_times,epoch_time)
        println("Epoch $epoch | Loss = $(round(loss, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")
        open("results/pooling_"*base_instance*"_semi_supervised_timed.csv", "a") do io
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
                    loss_fn_batch_semisupervised(model_semisupervised, batch)
                end
                
                # Update model parameters
                Flux.Optimise.update!(opt, ps, gs)
                
                total_loss += loss*batch_size
            end
        end
        
        push!(epoch_losses,total_loss/n_tuning)
        push!(epoch_times,epoch_time)
        println("Epoch $epoch | Loss = $(round(total_loss/n_tuning, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s")
        open("results/pooling_"*base_instance*"_semi_supervised_timed.csv", "a") do io
            write(io, "Epoch $epoch | Loss = $(round(total_loss/n_tuning, sigdigits=6)) | Time = $(round(epoch_time, digits=4)) s \n")
        end
        epoch = epoch+1
        
    end
    filename = "results/Model_semi_GM_pooling_tuned_timed_"*base_instance*".bson"
    @save filename model_semisupervised
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
    times_df.Total_Time =
    times_df.Forward_Time_LF .+
    times_df.Backward_Time_LF .+
    times_df.Forward_Time_HF .+
    times_df.Backward_Time_HF

    times_df.Untracked_Time = times_df.Epoch_Time .- times_df.Total_Time

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
    open("results/times_metrics_semi_pooling_timed_$base_instance.json", "w") do io
            JSON.print(io, metrics_dict)
    end
end



for (inst,total_time) in SYSTEM_TIME_BUDGETS
    semi_supervised_training_pooling(inst,total_time)
end



function inference_results(base_instance)
    Random.seed!(20)   
    reference_instance = parse_instance("instances_final/"*base_instance*".json")
    g_ref, node_id_ref, ranges_ref, edge_maps_ref, X_ref, Eattr_ref =
        build_tripartite_graph(reference_instance; return_features=true)

    src_range, bl_range, trm_range = ranges_ref

    norm = fit_normalizer_from_features(X_ref, src_range, bl_range, trm_range, Eattr_ref)
    n_tuning = 1
    n_test_samples = 1000
    #n_test_samples = 200
    evaluation_results = Dict()
    
    test_loss_mck = []
    test_feasibilty_metric_mck = []
    test_raw_slack_mck = []
    test_slack_used_mck = []
    test_time_mck = []
    test_loss_self = []
    test_feasibilty_metric_self = []
    test_raw_slack_self = []
    test_slack_used_self = []
    test_time_self = []
    
    test_ub = []
    @load "results/Model_self_GM_pooling_tuned_timed_"*base_instance*".bson" model_selfsupervised 
    #if(base_instance !="instance_5_1" )
        @load "results/Model_semi_GM_pooling_tuned_timed_"*base_instance*".bson" model_semisupervised 
        test_loss_semi= []
        test_feasibilty_metric_semi = []
        test_raw_slack_semi = []
        test_slack_used_semi = []
        test_time_semi = []
    #end
    i = 1
    j = 1

    while i <= n_test_samples
        try
            instance_vary = generate_variations_1(
                "instances_final/" * base_instance * ".json",
                1, 0, 1, 0;
                frac = 0.10,
                seed = j+25
            )

            path = "instances_gen_vary_train/" * base_instance * "_test.json"

            open(path, "w") do io
                JSON.print(io, instance_vary, 4)
            end

            instance = parse_instance(path)

            # Build static graph components
            g, node_id, ranges, edge_maps = build_tripartite_graph(instance; normalizer = norm)
            rel = split_relation_graphs(g)

            obj_true = mccormick_model_bound(instance)
            

            obj_true,q_mck,LF_time_mck = mccormick_model_bound_q(instance)
            loss_mck, feasibilty_metric_mck, raw_slack_mck, slack_used_mck,HF_time_mck =
                true_model_with_slacks_logged(instance, q_mck)
            push!(test_time_mck,[LF_time_mck,HF_time_mck])
            push!(test_ub, obj_true)
            push!(test_loss_mck, loss_mck)
            push!(test_feasibilty_metric_mck, feasibilty_metric_mck)
            push!(test_raw_slack_mck, raw_slack_mck)
            push!(test_slack_used_mck, slack_used_mck)

            loss_self, feasibilty_metric_self, raw_slack_self, slack_used_self,time_used_self =
                loss_fn_with_logging(model_selfsupervised, instance, rel, g, edge_maps, node_id)
            push!(test_time_self,time_used_self)
            push!(test_loss_self, loss_self)
            push!(test_feasibilty_metric_self, feasibilty_metric_self)
            push!(test_raw_slack_self, raw_slack_self)
            push!(test_slack_used_self, slack_used_self)

            #if base_instance != "instance_5_1"
                loss_semi, feasibilty_metric_semi, raw_slack_semi, slack_used_semi,time_used_semi =
                    loss_fn_with_logging(model_semisupervised, instance, rel, g, edge_maps, node_id)
                push!(test_time_semi,time_used_semi)
                push!(test_loss_semi, loss_semi)
                push!(test_feasibilty_metric_semi, feasibilty_metric_semi)
                push!(test_raw_slack_semi, raw_slack_semi)
                push!(test_slack_used_semi, slack_used_semi)
            #end

            println("✅ Sample $i succeeded with seed = $j")

            i += 1   # only increment on success

        catch e
            println("⚠️ Failed at seed = $j, retrying...")
            # do nothing, just retry with new seed
        end

        j += 1   # always increment seed
    end
    
    evaluation_results["test"] = Dict(
        "test_loss_mck" => test_loss_mck,
        "test_feasibilty_metric_mck" => test_feasibilty_metric_mck,
        "test_raw_slack_mck" => test_raw_slack_mck,
        "test_slack_used_mck" => test_slack_used_mck,
        "test_loss_self" => test_loss_self,
        "test_feasibilty_metric_self" => test_feasibilty_metric_self,
        "test_raw_slack_self" => test_raw_slack_self,
        "test_slack_used_self" => test_slack_used_self,
        "test_ub" => test_ub,
        "test_time_mck" => test_time_mck,
        "test_time_self" => test_time_self
    )
    percentiles = collect(1:100)
    #if(base_instance !="instance_5_1" )
        evaluation_results["test"]["test_loss_semi"] = test_loss_semi
        evaluation_results["test"]["test_feasibilty_metric_semi"] = test_feasibilty_metric_semi
        evaluation_results["test"]["test_raw_slack_semi"] = test_raw_slack_semi
        evaluation_results["test"]["test_slack_used_semi"] = test_slack_used_semi
        error_pred_semi_test = abs.(test_ub  .- test_loss_semi) ./ abs.(test_ub)  .* 100
        percentile_values_feasibilty_metric_semi_test = [quantile(sort(test_feasibilty_metric_semi), p / 100) for p in percentiles]
        evaluation_results["error_pred_semi_test"] = error_pred_semi_test
        evaluation_results["percentile_values_feasibilty_metric_semi_test"] = percentile_values_feasibilty_metric_semi_test
        evaluation_results["test"]["test_time_semi"] = test_time_semi
    #end

    error_pred_self_test = abs.(test_ub .- test_loss_self) ./ abs.(test_ub)  .* 100
    
    error_mck_test = abs.(test_ub  .- test_loss_mck) ./ abs.(test_ub)  .* 100
    
    

    percentile_values_feasibilty_metric_mck_test = [quantile(sort(test_feasibilty_metric_mck), p / 100) for p in percentiles]
    percentile_values_feasibilty_metric_self_test = [quantile(sort(test_feasibilty_metric_self), p / 100) for p in percentiles]
    
    evaluation_results["error_pred_self_test"] = error_pred_self_test
    
    evaluation_results["error_pred_mck_test"] = error_mck_test

    evaluation_results["percentile_values_feasibilty_metric_mck_test"] = percentile_values_feasibilty_metric_mck_test
    evaluation_results["percentile_values_feasibilty_metric_self_test"] = percentile_values_feasibilty_metric_self_test
    
    return evaluation_results 
end

base_instances_list = ["instance_2_1","instance_4_1","instance_5_1"]
for inst in base_instances_list
    export_data = inference_results(inst)
    save_path = joinpath("results", "evaluation_results_timed_$inst.json")
    open(save_path, "w") do io
        JSON.print(io, export_data)
    end
end