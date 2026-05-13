using Pkg
Pkg.activate("Gnn_environment")
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


base_instances_list = ["instance_5_1","instance_4_1","instance_3_1","instance_2_1","instance_1_1"]
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
    mck_slack = []
    self_slack = []
    semi_slack = []
    problem_instance = []
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
            loss_mck, feasibilty_metric_mck, raw_slack_mck, slack_used_mck,HF_time_mck,all_slack_mck =
                true_model_with_slacks_logged(instance, q_mck)
            push!(test_time_mck,[LF_time_mck,HF_time_mck])
            push!(test_ub, obj_true)
            push!(test_loss_mck, loss_mck)
            push!(test_feasibilty_metric_mck, feasibilty_metric_mck)
            push!(test_raw_slack_mck, raw_slack_mck)
            push!(test_slack_used_mck, slack_used_mck)
            push!(mck_slack,all_slack_mck)

            loss_self, feasibilty_metric_self, raw_slack_self, slack_used_self,time_used_self,all_slack_self =
                loss_fn_with_logging(model_selfsupervised, instance, rel, g, edge_maps, node_id)
            push!(test_time_self,time_used_self)
            push!(test_loss_self, loss_self)
            push!(test_feasibilty_metric_self, feasibilty_metric_self)
            push!(test_raw_slack_self, raw_slack_self)
            push!(test_slack_used_self, slack_used_self)
            push!(self_slack,all_slack_self)

            #if base_instance != "instance_5_1"
                loss_semi, feasibilty_metric_semi, raw_slack_semi, slack_used_semi,time_used_semi,all_slack_semi =
                    loss_fn_with_logging(model_semisupervised, instance, rel, g, edge_maps, node_id)
                push!(test_time_semi,time_used_semi)
                push!(test_loss_semi, loss_semi)
                push!(test_feasibilty_metric_semi, feasibilty_metric_semi)
                push!(test_raw_slack_semi, raw_slack_semi)
                push!(test_slack_used_semi, slack_used_semi)
                push!(semi_slack,all_slack_semi)
            #end

            println("✅ Sample $i succeeded with seed = $j")

            i += 1   # only increment on success

        catch e
            println("⚠️ Failed at seed = $j, retrying...")
            println("Error: ", e)
            # do nothing, just retry with new seed
            push!(problem_instance,j)
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
        "test_time_self" => test_time_self,
        "all_slack_mck" => mck_slack,
        "all_slack_self" => self_slack,
        "all_slack_semi" => semi_slack,
        "problem_instance" =>problem_instance

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
for inst in base_instances_list
    export_data = inference_results(inst)
    save_path = joinpath("results", "evaluation_results_timed_1_$inst.json")
    open(save_path, "w") do io
        JSON.print(io, export_data)
    end
end
