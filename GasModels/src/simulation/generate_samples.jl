###############################################################
# GasModels LF/HF Data Generation Script
#
# Purpose:
#   Generate paired Low-Fidelity (LF) and High-Fidelity (HF)
#   optimal gas flow solutions for supervised / self-supervised
#   GNN training.
#
#   LF  -> LRWP  (Linear Relaxed Weymouth approximation)
#   HF  -> WP    (Nonlinear Weymouth physics)
#
#   Only samples that are feasible for BOTH models are kept.
#
###############################################################

using Pkg
Pkg.activate("Gnn_environment")

# Load local project module (contains LRWP and WP definitions)
include("../Self_GNN_PM.jl")
using .Self_GNN_PM

# Optimization and modeling packages
using PowerModels, JuMP, MadNLP, Ipopt, Gurobi
using Distributions
using BSON
using GasModels
using Plots

export run_ac_dc_batch_from_data

###############################################################
# Function: generate_truncated_scales_GM
#
# Generates random demand perturbations for gas deliveries.
# Each delivery withdrawal is scaled using a truncated Gaussian.
#
# Input:
#   data      -> GasModels network dictionary
#   nsamples  -> number of random samples
#
# Output:
#   delivery_samples  -> nsamples × ndelivery matrix
#   delivery_base     -> base withdrawal values
#   delivery_ids      -> sorted delivery IDs
#
###############################################################

function generate_truncated_scales_GM(data::Dict, nsamples::Int)

    # Extract delivery IDs and sort them numerically
    delivery_ids = sort(parse.(Int, keys(data["delivery"])))
    ndel = length(delivery_ids)

    # Generate multiplicative perturbations:
    # 1 + ε where ε ~ truncated N(0, 0.1)
    # Truncated between [-2, 2] to avoid extreme scaling
    scale_delivery = 1 .+ rand(truncated(Normal(0, 0.1), -3, 3), ndel, nsamples)

    # Extract base nominal withdrawals
    delivery_base = zeros(ndel)
    for (i, id) in enumerate(delivery_ids)
        # println("id delivery ", id)
        delivery_base[i] = data["delivery"][string(id)]["withdrawal_nominal"]
    end

    # Apply scaling
    delivery_samples = scale_delivery .* delivery_base

    # Return as (nsamples × ndel)
    return delivery_samples', delivery_base, delivery_ids
end

###############################################################
# Function: run_ac_dc_batch_from_data_GM
#
# Runs LF (LRWP) and HF (WP) models for each delivery sample.
#
# Keeps only samples where:
#   - LF is feasible
#   - HF is feasible
#
# Stores:
#   - generator flows (fg)
#   - objectives
#   - runtime
#
###############################################################

function run_ac_dc_batch_from_data_GM(
    data_orig,
    delivery_samples,
    nsamples,
    solver=Ipopt.Optimizer
)

    total_available, njunction = size(delivery_samples)

    data0 = deepcopy(data_orig)
    nreceipt = length(data0["receipt"])

    ###########################################################
    # Preallocate
    ###########################################################

    delivery = zeros(nsamples, njunction)

    fg_HF = zeros(nsamples, nreceipt)
    fg_LRWP = zeros(nsamples, nreceipt)
    fg_LRDWP = zeros(nsamples, nreceipt)
    fg_CRDWP = zeros(nsamples, nreceipt)

    Time_HF = zeros(nsamples)
    Time_LRWP = zeros(nsamples)
    Time_LRDWP = zeros(nsamples)
    Time_CRDWP = zeros(nsamples)

    Obj_HF = zeros(nsamples)
    Obj_LRWP = zeros(nsamples)
    Obj_LRDWP = zeros(nsamples)
    Obj_CRDWP = zeros(nsamples)

    valid_k = 0
    j = 1

    @info "Searching $nsamples valid samples (available = $total_available)..."

    ###########################################################
    # Main loop
    ###########################################################

    while valid_k < nsamples && j <= total_available

        @info "Evaluating sample $j / $total_available"

        delivery_i = delivery_samples[j, :]

        ############################
        # 1) LRWP
        ############################
        data_lrwp = deepcopy(data_orig)

        res_lrwp = nothing
        t_lrwp = 0.0

        t1 = time()
        try
            res_lrwp = Self_GNN_PM.LRWP(data_lrwp, delivery_i)
            t_lrwp = time() - t1
            if !res_lrwp["flag"]
                j += 1
                continue
            end
        catch e
            @warn "Sample $j failed in LRWP: $e"
            j += 1
            continue
        end
        ############################
        # 2) LRDWP
        ############################
        data_lrdwp = deepcopy(data_orig)
        res_lrdwp = nothing
        t_lrdwp = 0.0

        t2 = time()
        try
            res_lrdwp = Self_GNN_PM.LRDWP(data_lrdwp, delivery_i)
            t_lrdwp = time() - t2
            if !res_lrdwp["flag"]
                j += 1
                continue
            end
        catch e
            @warn "Sample $j failed in LRDWP: $e"
            j += 1
            continue
        end
        ############################
        # 3) CRDWP
        ############################
        data_crdwp = deepcopy(data_orig)
        res_crdwp = nothing
        t_crdwp = 0.0

        t3 = time()
        try
            res_crdwp = Self_GNN_PM.CRDWP(data_crdwp, delivery_i)
            t_crdwp = time() - t3
            if !res_crdwp["flag"]
                j += 1
                continue
            end
        catch e
            @warn "Sample $j failed in CRDWP: $e"
            j += 1
            continue
        end

        ############################
        # 4) HF (WP)
        ############################
        data_hf = deepcopy(data_orig)
        t4 = time()
        res_hf = Self_GNN_PM.WP(data_hf, delivery_i)
        t_hf = time() - t4

        if res_hf["flag"]

            valid_k += 1
            delivery[valid_k, :] = delivery_i

            fg_HF[valid_k, :] = res_hf["fg"]
            fg_LRWP[valid_k, :] = res_lrwp["fg"]
            fg_LRDWP[valid_k, :] = res_lrdwp["fg"]
            fg_CRDWP[valid_k, :] = res_crdwp["fg"]

            Time_HF[valid_k] = t_hf
            Time_LRWP[valid_k] = t_lrwp
            Time_LRDWP[valid_k] = t_lrdwp
            Time_CRDWP[valid_k] = t_crdwp

            Obj_HF[valid_k] = res_hf["objective"]
            Obj_LRWP[valid_k] = res_lrwp["objective"]
            Obj_LRDWP[valid_k] = res_lrdwp["objective"]
            Obj_CRDWP[valid_k] = res_crdwp["objective"]

        else
            @warn "Sample $j discarded (HF infeasible)"
        end

        j += 1
    end

    ###########################################################
    # Shrink if needed
    ###########################################################

    if valid_k < nsamples
        delivery = delivery[1:valid_k, :]
        fg_HF = fg_HF[1:valid_k, :]
        fg_LRWP = fg_LRWP[1:valid_k, :]
        fg_LRDWP = fg_LRDWP[1:valid_k, :]
        fg_CRDWP = fg_CRDWP[1:valid_k, :]

        Time_HF = Time_HF[1:valid_k]
        Time_LRWP = Time_LRWP[1:valid_k]
        Time_LRDWP = Time_LRDWP[1:valid_k]
        Time_CRDWP = Time_CRDWP[1:valid_k]

        Obj_HF = Obj_HF[1:valid_k]
        Obj_LRWP = Obj_LRWP[1:valid_k]
        Obj_LRDWP = Obj_LRDWP[1:valid_k]
        Obj_CRDWP = Obj_CRDWP[1:valid_k]
    end

    ###########################################################
    # Return
    ###########################################################

    return Dict(
        "delivery" => delivery, "fg_HF" => fg_HF,
        "fg_LRWP" => fg_LRWP,
        "fg_LRDWP" => fg_LRDWP,
        "fg_CRDWP" => fg_CRDWP, "Time_HF" => Time_HF,
        "Time_LRWP" => Time_LRWP,
        "Time_LRDWP" => Time_LRDWP,
        "Time_CRDWP" => Time_CRDWP, "Obj_HF" => Obj_HF,
        "Obj_LRWP" => Obj_LRWP,
        "Obj_LRDWP" => Obj_LRDWP,
        "Obj_CRDWP" => Obj_CRDWP, "n_valid" => valid_k
    )
end

function split_results_gm(results::Dict, sample_range)
    split_dict = Dict{String,Any}()

    for key in [
        "delivery",
        "fg_HF",
        "fg_LRWP",
        "fg_LRDWP",
        "fg_CRDWP",
    ]
        split_dict[key] = results[key][sample_range, :]
    end

    for key in ["Time_HF", "Time_LRWP", "Time_LRDWP", "Time_CRDWP", "Obj_HF", "Obj_LRWP", "Obj_LRDWP", "Obj_CRDWP"]
        split_dict[key] = results[key][sample_range]
    end

    split_dict["n_valid"] = length(sample_range)
    return split_dict
end

###############################################################
# ===================== MAIN SCRIPT ===========================
###############################################################

# Folder containing GasModels .m files
path_data = "test_systems/data-ogf"

# Selected GasLib cases
system_names = ["gaslib-40-E", "gaslib-135-F"]

# Number of random perturbations generated
limit_sample = 6000

# Number of valid LF/HF pairs required
nsamples_train = 1000
nsamples_test = 1000
nsamples_total = nsamples_train + nsamples_test

for system_name in system_names
    @info "Generating samples for $system_name"

    file_name = joinpath(path_data, "$(system_name).m")
    data = Self_GNN_PM.load_system_gasmodels(file_name)

    delivery_samples, delivery_base, delivery_ids =
        generate_truncated_scales_GM(data, limit_sample)

    results_all = run_ac_dc_batch_from_data_GM(data, delivery_samples, nsamples_total)

    if results_all["n_valid"] < nsamples_total
        @warn "Skipping $system_name: only $(results_all["n_valid"]) valid samples were generated, but $nsamples_total are required."
        continue
    end

    train_range = 1:nsamples_train
    test_range = (nsamples_train + 1):nsamples_total

    results = split_results_gm(results_all, train_range)
    results_test = split_results_gm(results_all, test_range)

    BSON.@save joinpath(path_data, "data_$system_name.bson") results
    BSON.@save joinpath(path_data, "data_test_$system_name.bson") results_test

    @info "Saved train/test datasets for $system_name"
end




# sum(results["Time_HF"]) / 3600
# k = BSON.load("test_systems/data-ogf/data_$system_name.bson")

# results = k[:results]

# a = results["Obj_HF"]
# b = results["Obj_LRWP"]
# c = results["Obj_CRDWP"]
# d = results["Obj_LRDWP"]
# scatter(a, a)
# scatter!(a, b)
# scatter!(a, c)
# scatter!(a, d)

# error1 = abs.(a .- b) ./ a * 100













