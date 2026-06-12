module Layers

export DCPMLayerStruct, build_DCPMLayer, DCPMLayer,
    DCPMLayerTruncado, ACPMLayer,
    ACPM, DCPM, DCPMLayerBatch, feasibility, top_congested_lines, ACPMLayer_Array, ACPMLayerExa, ACPMExa, DCPMExa, DCPMLayerExa, fwd_times_dc, fwd_times_ac, bwd_times_dc, bwd_times_ac, grad_log_dc


#export DCPMLayer, DCPMLayerTruncado, ACPMLayer_v5,


using JuMP
using PowerModels
using DiffOpt
using MadNLP
using MadNLPGPU
using CUDA
using ExaModels
using Gurobi
using Ipopt
using ChainRulesCore
using MathOptInterface
# using HSL_jll
using ..DataLoader: GraphWithRef
# using ..Utils: fwd_times_dc, bwd_times_dc, fwd_times_ac, bwd_times_ac
const fwd_times_dc = Float64[]
const fwd_times_ac = Float64[]
const bwd_times_dc = Float64[]
const bwd_times_ac = Float64[]
grad_log_dc = Vector{Dict}()

const MOI = MathOptInterface

const GRB_ENV = Ref{Any}(nothing)

function build_silent_gurobi_env()
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            try
                return Gurobi.Env(; output_flag=0)
            catch
                return Gurobi.Env()
            end
        end
    end
end

function silent_gurobi_optimizer()
    if GRB_ENV[] === nothing
        GRB_ENV[] = build_silent_gurobi_env()
    end

    optimizer = Gurobi.Optimizer(GRB_ENV[])
    MOI.set(optimizer, MOI.RawOptimizerAttribute("OutputFlag"), 0)
    return optimizer
end


function DCPMLayer(gw::GraphWithRef, rhs::AbstractMatrix; model=direct_model(DiffOpt.diff_optimizer(silent_gurobi_optimizer)))
    data = gw.data


    ref_data = ChainRulesCore.ignore_derivatives() do
        PowerModels.build_ref(data; ref_extensions=Dict())
    end[:it][:pm][:nw][0]

    Pd = gw.graph.x[1, :]


    buses = sort(collect(keys(ref_data[:bus])))
    #    gens = sort(collect(keys(ref_data[:gen])))

    gens = sort(collect(keys(ref_data[:gen])))
    arcs_from = ref_data[:arcs_from]
    arcs_dc = collect(keys(ref_data[:arcs_dc]))


    #set_optimizer_attribute(model, "print_level", 0)
    set_silent(model)
    # Variables
    # @variable(model, β[i=1:length(buses)] in Parameter(beta[i]))
    @variable(model, va[i in buses])
    @variable(model, ref_data[:gen][i]["pmin"] <= pg[i in gens] <= ref_data[:gen][i]["pmax"])
    @variable(model, slack_pos[1:length(buses)] >= 0)
    # @variable(model, slack_neg[1:length(buses)] >= 0)
    @variable(model, -ref_data[:branch][a[1]]["rate_a"] <= p[a in arcs_from] <= ref_data[:branch][a[1]]["rate_a"])
    @variable(model, p_dc[a in arcs_dc])
    # <-- Slack positive variable for each bus

    # Build expressions for branch flows
    p_expr = Dict([((l, i, j), 1.0 * p[(l, i, j)]) for (l, i, j) in ref_data[:arcs_from]])
    p_expr = merge(p_expr, Dict([((l, j, i), -1.0 * p[(l, i, j)]) for (l, i, j) in ref_data[:arcs_from]]))

    # Build the objective: generation cost + slack penalty
    from_idx = Dict(arc[1] => arc for arc in ref_data[:arcs_from_dc])

    # Find maximum linear generation cost coefficient
    max_ck1 = maximum(gen["cost"][2] for (i, gen) in ref_data[:gen])

    @objective(model, Min,
        sum(gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3] for (i, gen) in ref_data[:gen]) +
        sum(dcline["cost"][1] * p_dc[from_idx[i]]^2 + dcline["cost"][2] * p_dc[from_idx[i]] + dcline["cost"][3] for (i, dcline) in ref_data[:dcline]) +
        10 * sum(max_ck1 * slack_pos[i] for i = 1:length(buses))  # <-- Penalizar slack
        # + 10 * max_ck1 * sum(slack_neg[i] for i = 1:length(buses))
    )

    # Constraints

    # Reference angle constraint
    for (i, bus) in ref_data[:ref_buses]
        @constraint(model, va[i] == 0)
    end

    # @constraint(model, BalanceeqP[i=1:length(buses)],
    #     sum(p_expr[a] for a in ref_data[:bus_arcs][buses[i]]) +
    #     sum(p_dc[a_dc] for a_dc in ref_data[:bus_arcs_dc][buses[i]]) ==
    #     sum(pg[g] for g in ref_data[:bus_gens][buses[i]]) -
    #     rhs[i] +
    #     slack[buses[i]]
    # )

    # println("funcion 1")

    # @constraint(model, BalanceeqP[i=1:length(buses)],
    #     sum(p_expr[a] for a in ref_data[:bus_arcs][buses[i]]) +
    #     sum(p_dc[a_dc] for a_dc in ref_data[:bus_arcs_dc][buses[i]]) ==
    #     sum(pg[g] for g in ref_data[:bus_gens][buses[i]]) -
    #     Pd[i] * β[i] + slack_pos[i] -
    #     sum(shunt["gs"] for shunt in [ref_data[:shunt][s] for s in ref_data[:bus_shunts][buses[i]]]) * 1.0^2
    # )

    @constraint(model, BalanceeqP[i=1:length(buses)],
        sum(p_expr[a] for a in ref_data[:bus_arcs][buses[i]]) +
        sum(p_dc[a_dc] for a_dc in ref_data[:bus_arcs_dc][buses[i]]) ==
        sum(pg[g] for g in ref_data[:bus_gens][buses[i]]) -
        rhs[i] + slack_pos[i])

    # println("funcion 2")

    # Branch flow physics and angle limits
    for (i, branch) in ref_data[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])

        p_fr = p[f_idx]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)

        @constraint(model, p_fr == -b * (va_fr - va_to))

        @constraint(model, va_fr - va_to <= branch["angmax"])
        @constraint(model, va_fr - va_to >= branch["angmin"])
    end

    # HVDC line constraints
    for (i, dcline) in ref_data[:dcline]
        f_idx = (i, dcline["f_bus"], dcline["t_bus"])
        t_idx = (i, dcline["t_bus"], dcline["f_bus"])

        @constraint(model, (1 - dcline["loss1"]) * p_dc[f_idx] + (p_dc[t_idx] - dcline["loss0"]) == 0)
    end
    # Solve the model
    #optimize!(model)
    #DiffOpt.track!(model)
    # println("funcion 3")

    optimize!(model)

    # println("funcion 4")
    #println(BalanceeqP)
    # println("Solver statuspDC:    ", termination_status(model))
    # println("Objective value pDC: ", objective_value(model))
    pg_dict = JuMP.value.(pg)


    # pg_vec = [pg_dict[g] for g in gens]

    # println("pg_dict: ", pg_vec)
    return pg_dict
end
# function DCPMLayer(gw::GraphWithRef, beta::AbstractMatrix; model=Model(() -> DiffOpt.diff_optimizer(Gurobi.Optimizer)))
#! =============================================================
#! Reusable DC-OPF layer for DiffOpt
#! =============================================================

# -------------------------------------------------------------------
# Reusable DC-OPF layer for DiffOpt, without @parameter
# -------------------------------------------------------------------

struct DCPMLayerStruct
    model::JuMP.Model
    pg         # JuMP container of generator variables (DenseAxisArray)
    balance    # Dict{Int, JuMP.ConstraintRef} for balance eqs
    buses::Vector{Int}
    gens::Vector{Int}
end

"""
    build_DCPMLayer(gw::GraphWithRef) -> DCPMLayerStruct

Build the DC-OPF model once. The power-balance constraints are written
with RHS = 0, and in each forward pass we update the RHS using
`JuMP.set_normalized_rhs` to encode the current `rhs`.
"""
function build_DCPMLayer(gw::GraphWithRef)
    data = gw.data

    ref_data = ChainRulesCore.ignore_derivatives() do
        PowerModels.build_ref(data; ref_extensions=Dict())
    end[:it][:pm][:nw][0]

    buses = sort(collect(keys(ref_data[:bus])))
    gens = sort(collect(keys(ref_data[:gen])))
    arcs_from = ref_data[:arcs_from]
    arcs_dc = collect(keys(ref_data[:arcs_dc]))

    model = DiffOpt.quadratic_diff_model(silent_gurobi_optimizer)
    JuMP.set_silent(model)

    # -------------------------
    # VARIABLES
    # -------------------------
    @variable(model, va[i in buses])
    @variable(model,
        ref_data[:gen][i]["pmin"] <= pg[i in gens] <= ref_data[:gen][i]["pmax"]
    )
    @variable(model, slack_pos[i in buses] >= 0)
    @variable(model,
        -ref_data[:branch][a[1]]["rate_a"] <= p[a in arcs_from] <= ref_data[:branch][a[1]]["rate_a"]
    )
    @variable(model, p_dc[a in arcs_dc])

    # -------------------------
    # BRANCH FLOW EXPRESSIONS
    # -------------------------
    p_expr = Dict{Tuple,JuMP.AffExpr}()
    for (l, i, j) in ref_data[:arcs_from]
        p_expr[(l, i, j)] = p[(l, i, j)]
        p_expr[(l, j, i)] = -p[(l, i, j)]
    end

    # -------------------------
    # OBJECTIVE (same as your original)
    # -------------------------
    # from_idx = Dict(arc[1] => arc for arc in ref_data[:arcs_from_dc])

    max_ck1 = maximum(gen["cost"][2] for (_, gen) in ref_data[:gen])

    # quad_eps = 1e-5

    from_idx = Dict(arc[1] => arc for arc in ref_data[:arcs_from_dc])

    @objective(model, Min,
        # generator cost
        sum(gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3]
            for (i, gen) in ref_data[:gen])
        +
        # HVDC cost (restored)
        sum(dcline["cost"][1] * p_dc[from_idx[i]]^2 +
            dcline["cost"][2] * p_dc[from_idx[i]] +
            dcline["cost"][3]
            for (i, dcline) in ref_data[:dcline])
        +
        # #(optionally) regularization term
        # quad_eps * sum(pg[i]^2 for i in gens)
        +
        # slack penalty
        10 * max_ck1 * sum(slack_pos[i] for i in buses)
    )



    # -------------------------
    # REFERENCE ANGLE
    # -------------------------
    for (i, _) in ref_data[:ref_buses]
        @constraint(model, va[i] == 0)
    end

    # -------------------------
    # BALANCE CONSTRAINTS
    # We write them as:
    #   flow + dc - gen + slack_pos == 0
    # and later we set RHS = -rhs[i] using set_normalized_rhs.
    # -------------------------
    balance = Dict{Int,JuMP.ConstraintRef}()

    for i in buses
        balance[i] = @constraint(model,
            sum(p_expr[a] for a in ref_data[:bus_arcs][i]) +
            sum(p_dc[a_dc] for a_dc in ref_data[:bus_arcs_dc][i]) -
            sum(pg[g] for g in ref_data[:bus_gens][i]) -
            slack_pos[i] == 0
        )
    end

    # -------------------------
    # BRANCH PHYSICS & ANGLE LIMITS
    # -------------------------
    for (i, branch) in ref_data[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])

        p_fr = p[f_idx]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)

        @constraint(model, p_fr == -b * (va_fr - va_to))
        @constraint(model, va_fr - va_to <= branch["angmax"])
        @constraint(model, va_fr - va_to >= branch["angmin"])
    end

    # -------------------------
    # HVDC LINE CONSTRAINTS
    # -------------------------
    for (i, dcline) in ref_data[:dcline]
        f_idx = (i, dcline["f_bus"], dcline["t_bus"])
        t_idx = (i, dcline["t_bus"], dcline["f_bus"])

        @constraint(model,
            (1 - dcline["loss1"]) * p_dc[f_idx] +
            (p_dc[t_idx] - dcline["loss0"]) == 0
        )
    end

    # Enable DiffOpt tracking

    return DCPMLayerStruct(model, pg, balance, buses, gens)
end


"""
    DCPMLayer(layer::DCPMLayerStruct, rhs)

Fast forward pass:
- updates the RHS of the power-balance constraints using rhs,
- solves DC-OPF with DiffOpt,
- returns pg values (same container shape as before).
"""
# function DCPMLayer(layer::DCPMLayerStruct, rhs::AbstractVecOrMat)
#     start_fwd = time()

#     # Ensure rhs is a vector aligned with `buses`
#     rhs_vec = vec(rhs)
#     @assert length(rhs_vec) == length(layer.buses) "rhs length must match number of buses"
#     # println("valor de rhs",rhs_vec)
#     # println(keys(layer.buses))
#     # Update the RHS of each balance constraint:
#     #   flow + dc - gen + slack_pos == -rhs[i]
#     for (k, bus) in enumerate(layer.buses)
#         # println("bus: ", bus, " rhs: ", rhs_vec[k])
#         JuMP.set_normalized_rhs(layer.balance[bus], -rhs_vec[k])
#     end

#     # Forward solve (IMPORTANT: use optimize! instead of DiffOpt.forward)
#     optimize!(layer.model)

#     fwd_time_ref = time() - start_fwd
#     push!(fwd_times_dc, fwd_time_ref)

#     println("Solver status pDC:    ", JuMP.termination_status(layer.model))
#     println("Objective value pDC: ", JuMP.objective_value(layer.model))

#     pg_vals = JuMP.value.(layer.pg)

#     return pg_vals
# end



#! =============================================================




# 

function ACPMLayer(gw::GraphWithRef, Pg_target, results_init; solver=optimizer_with_attributes(
    Ipopt.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => "ma27",
    "hsllib" => HSL_jll.libhsl_path
)
)
    data = gw.data
    x = gw.graph.x
    # y = gw.graph.y

    bus_init = results_init["bus"]
    gen_init = results_init["gen"]
    branch_init = results_init["branch"]



    # === Extraer Pd y Qd desde x ===
    nbus = length(keys(data["bus"]))
    Pd_full = x[1, :]
    Qd_full = x[2, :]

    # Paso 1: ordenar los IDs de buses y crear el mapa
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # Paso 2: luego, al actualizar las cargas:
    for (load_id, load_data) in data["load"]
        bus_id = load_data["load_bus"]
        idx = bus_idx_map[bus_id]
        data["load"][load_id]["pd"] = Pd_full[idx]
        data["load"][load_id]["qd"] = Qd_full[idx]
    end
    # gen_keys = sort(collect(keys(ref_data[:gen])))

    # === Construcción del modelo ===
    spec = PowerModels.instantiate_model(
        data,
        PowerModels.ACRPowerModel,
        PowerModels.build_opf
    )

    #println("Model slacks ON======")

    model = spec.model
    gen_ref = spec.ref[:it][:pm][:nw][0][:gen]
    # pg_vars = spec.var[:it][:pm][:nw][0][:pg]


    # Extract PowerModels variable references
    vr_vars = spec.var[:it][:pm][:nw][0][:vr]
    vi_vars = spec.var[:it][:pm][:nw][0][:vi]
    pg_vars = spec.var[:it][:pm][:nw][0][:pg]
    qg_vars = spec.var[:it][:pm][:nw][0][:qg]

    # === Initialize bus voltage values ===
    for (bus_id_str, vals) in bus_init
        bus_id = parse(Int, bus_id_str)
        if bus_id in keys(vr_vars)
            set_start_value(vr_vars[bus_id], vals["vr"])
        end
        if bus_id in keys(vi_vars)
            set_start_value(vi_vars[bus_id], vals["vi"])
        end
    end

    # === Initialize generator outputs ===
    for (gen_id_str, vals) in gen_init
        gen_id = parse(Int, gen_id_str)
        if gen_id in keys(pg_vars)
            set_start_value(pg_vars[gen_id], vals["pg"])
        end
        if gen_id in keys(qg_vars)
            set_start_value(qg_vars[gen_id], vals["qg"])
        end
    end

    @variables(model, begin
        slack_pg_pos[i in keys(gen_ref)] >= 0
        slack_pg_neg[i in keys(gen_ref)] >= 0
    end)

    @constraint(model, pg_aux[i in keys(gen_ref)],
        pg_vars[i] == Pg_target[i] + slack_pg_pos[i] - slack_pg_neg[i]
    )


    @objective(model, Min,
        sum(
            data["gen"][string(i)]["cost"][1] * pg_vars[i]^2 +
            data["gen"][string(i)]["cost"][2] * pg_vars[i]
            +
            data["gen"][string(i)]["cost"][3]
            for i in
            keys(gen_ref)
        )
        +
        sum(data["gen"][string(i)]["cost"][1] * (slack_pg_neg[i] + slack_pg_pos[i])^2 for i in keys(gen_ref))
        +
        sum(data["gen"][string(i)]["cost"][2] * (slack_pg_neg[i] + slack_pg_pos[i]) for i in keys(gen_ref))
    )



    set_silent(spec.model)
    set_time_limit_sec(spec.model, 1300.0)
    PowerModels.optimize_model!(spec; optimizer=solver)


    active_gen_ids = sort([
        gen_id for gen_id in keys(gen_ref)
        if data["gen"][string(gen_id)]["gen_status"] == 1
    ])

    dual_vec = [JuMP.dual(model[:pg_aux][i]) for i in active_gen_ids]


    slacks_neg = JuMP.value.(slack_pg_neg)
    slacks_pos = JuMP.value.(slack_pg_pos)
    # println(keys(gen_ref))
    pg_val = [JuMP.value(pg_vars[i]) for i in sort(collect(keys(gen_ref)))]


    Cost_slack = sum(data["gen"][string(i)]["cost"][1] * (slacks_neg[i] + slacks_pos[i])^2 for i in keys(gen_ref)) +
                 sum(data["gen"][string(i)]["cost"][2] * (slacks_neg[i] + slacks_pos[i]) for i in keys(gen_ref))


    Cost_Pg = sum(
        data["gen"][string(i)]["cost"][1] * Pg_target[i]^2 +
        data["gen"][string(i)]["cost"][2] * Pg_target[i]
        +
        data["gen"][string(i)]["cost"][3]
        for i in
        keys(gen_ref)
    )

    cost_slack_all = JuMP.objective_value(spec.model) - Cost_Pg

    slacks_pos_value = Array(JuMP.value.(slacks_pos))
    slacks_neg_value = Array(JuMP.value.(slacks_neg))
    max_slack = maximum(abs.(vcat(slacks_pos_value, slacks_neg_value)))


    return Dict(
        "flag" => (JuMP.termination_status(spec.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(spec.model),
        "Pg" => pg_val,
        "Pg_pred_lf" => Pg_target,
        "Cost_slack_all" => cost_slack_all,
        "Cost_Pg" => Cost_Pg,
        "pos_slack" => slack_pos_val,
        "neg_slack" => slack_neg_val,
        "max_slack" => max_slack * 100, #[MW],
        "Cost_slack" => Cost_slack,
        "Dual" => dual_vec
    )
end



function ACPMLayer_Array(gw::GraphWithRef, Pg_target; solver=optimizer_with_attributes(
    Ipopt.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => "ma27",
    "hsllib" => HSL_jll.libhsl_path
))
    start = time()
    data = gw.data
    x = gw.graph.x
    pg_init = gw.graph.y

    # === Extraer Pd y Qd desde x ===
    nbus = length(keys(data["bus"]))
    Pd_full = x[1, :]
    Qd_full = x[2, :]

    # Paso 1: ordenar los IDs de buses y crear el mapa
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # Paso 2: luego, al actualizar las cargas:
    for (load_id, load_data) in data["load"]
        bus_id = load_data["load_bus"]
        idx = bus_idx_map[bus_id]
        data["load"][load_id]["pd"] = Pd_full[idx]
        data["load"][load_id]["qd"] = Qd_full[idx]
    end
    # gen_keys = sort(collect(keys(ref_data[:gen])))

    # === Construcción del modelo ===
    spec = PowerModels.instantiate_model(
        data,
        PowerModels.ACRPowerModel,
        PowerModels.build_opf
    )

    #println("Model slacks ON======")

    model = spec.model
    gen_ref = spec.ref[:it][:pm][:nw][0][:gen]


    n_gens = length(gen_ref)
    gens = sort(collect(keys(gen_ref)))


    gen_ids = sort(collect(keys(spec.ref[:it][:pm][:nw][0][:gen])))

    for (i, g_id) in enumerate(gen_ids)
        var = spec.var[:it][:pm][:nw][0][:pg][g_id]
        set_start_value(var, pg_init[i])
    end
    pg_vars = spec.var[:it][:pm][:nw][0][:pg]

    # println(gen_ref)
    # Diccionario: gen_id => índice consecutivo
    gen_id_to_idx = Dict(gens[i] => i for i in 1:n_gens)


    @variables(model, begin
        slack_pg_pos[i in keys(gen_ref)] >= 0
        slack_pg_neg[i in keys(gen_ref)] >= 0
    end)

    @constraint(model, pg_aux[i in keys(gen_ref)],
        pg_vars[i] == Pg_target[gen_id_to_idx[i]] + slack_pg_pos[i] - slack_pg_neg[i]
    )


    @objective(model, Min,
        sum(
            data["gen"][string(i)]["cost"][1] * pg_vars[i]^2 +
            data["gen"][string(i)]["cost"][2] * pg_vars[i]
            +
            data["gen"][string(i)]["cost"][3]
            for i in
            keys(gen_ref)
        )
        +
        # 10 * sum(data["gen"][string(i)]["cost"][2] * (slack_pg_neg[i]^2 + slack_pg_pos[i]^2) for i in keys(gen_ref))

        # sum(data["gen"][string(i)]["cost"][2] * (slack_pg_neg[i] + slack_pg_pos[i]) for i in keys(gen_ref))

        sum(data["gen"][string(i)]["cost"][1] * (slack_pg_neg[i] + slack_pg_pos[i])^2 for i in keys(gen_ref))
        +
        sum(data["gen"][string(i)]["cost"][2] * (slack_pg_neg[i] + slack_pg_pos[i]) for i in keys(gen_ref))
        # +
        # sum(data["gen"][string(i)]["cost"][2] * (slack_pg_neg[i] + slack_pg_pos[i]) for i in keys(gen_ref))
    )



    set_silent(spec.model)
    set_time_limit_sec(spec.model, 1300.0)
    PowerModels.optimize_model!(spec; optimizer=solver)


    active_gen_ids = sort([
        gen_id for gen_id in keys(gen_ref)
        if data["gen"][string(gen_id)]["gen_status"] == 1
    ])

    dual_vec = [JuMP.dual(model[:pg_aux][i]) for i in active_gen_ids]





    elapsed = time() - start

    push!(fwd_times_ac, elapsed)


    slacks_neg = JuMP.value.(slack_pg_neg)
    slacks_pos = JuMP.value.(slack_pg_pos)
    # println(keys(gen_ref))
    pg_val = [JuMP.value(pg_vars[i]) for i in sort(collect(keys(gen_ref)))]


    Cost_slack = sum(data["gen"][string(i)]["cost"][1] * (slacks_neg[i] + slacks_pos[i])^2 for i in keys(gen_ref)) +
                 sum(data["gen"][string(i)]["cost"][2] * (slacks_neg[i] + slacks_pos[i]) for i in keys(gen_ref))

    Cost_Pg = sum(
        data["gen"][string(i)]["cost"][1] * JuMP.value(pg_vars[i])^2 +
        data["gen"][string(i)]["cost"][2] * JuMP.value(pg_vars[i])
        +
        data["gen"][string(i)]["cost"][3]
        for i in
        keys(gen_ref)
    )

    slacks_pos_value = Array(JuMP.value.(slacks_pos))
    slacks_neg_value = Array(JuMP.value.(slacks_neg))
    max_slack = maximum(abs.(vcat(slacks_pos_value, slacks_neg_value)))
    return Dict(
        "flag" => (JuMP.termination_status(spec.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(spec.model),
        "Pg" => pg_val,
        #"Vm" => [JuMP.value(spec.var[:it][:pm][:nw][0][:vm][i]) for (i, _) in spec.ref[:it][:pm][:nw][0][:bus]],
        #"Va" => [JuMP.value(spec.var[:it][:pm][:nw][0][:va][i]) for (i, _) in spec.ref[:it][:pm][:nw][0][:bus]],
        "max_slack" => max_slack * 100, #[MW],
        "Cost_Pg" => Cost_Pg,
        "pos_slack" => slacks_pos_value,
        "neg_slack" => slacks_neg_value,
        "Cost_slack" => Cost_slack,
        "Dual" => dual_vec
    )
end




#! =============================================================


function DCPMLayerTruncado(data::Dict, rhs::Matrix; model=direct_model(DiffOpt.diff_optimizer(silent_gurobi_optimizer)))

    start = time()
    ref_data = ChainRulesCore.ignore_derivatives() do
        PowerModels.build_ref(data; ref_extensions=Dict())
    end[:it][:pm][:nw][0]



    buses = sort(collect(keys(ref_data[:bus])))
    #    gens = sort(collect(keys(ref_data[:gen])))

    gens = sort(collect(keys(ref_data[:gen])))
    arcs_from = ref_data[:arcs_from]
    arcs_dc = collect(keys(ref_data[:arcs_dc]))


    #set_optimizer_attribute(model, "print_level", 0)
    set_silent(model)
    # Variables

    @variable(model, va[i in buses])
    @variable(model, ref_data[:gen][i]["pmin"] <= pg[i in gens] <= ref_data[:gen][i]["pmax"])
    @variable(model, slack[1:length(buses)] >= 0)
    @variable(model, -ref_data[:branch][a[1]]["rate_a"] <= p[a in arcs_from] <= ref_data[:branch][a[1]]["rate_a"])
    @variable(model, p_dc[a in arcs_dc])
    # <-- Slack positive variable for each bus

    # Build expressions for branch flows
    p_expr = Dict([((l, i, j), 1.0 * p[(l, i, j)]) for (l, i, j) in ref_data[:arcs_from]])
    p_expr = merge(p_expr, Dict([((l, j, i), -1.0 * p[(l, i, j)]) for (l, i, j) in ref_data[:arcs_from]]))

    # Build the objective: generation cost + slack penalty
    from_idx = Dict(arc[1] => arc for arc in ref_data[:arcs_from_dc])

    # Find maximum linear generation cost coefficient
    max_ck1 = maximum(gen["cost"][2] for (i, gen) in ref_data[:gen])

    @objective(model, Min,
        sum(gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3] for (i, gen) in ref_data[:gen])
        +
        10 * max_ck1 * sum(slack[i] for i = 1:length(buses))
    )

    # Constraints

    # Reference angle constraint
    for (i, bus) in ref_data[:ref_buses]
        @constraint(model, va[i] == 0)
    end

    @constraint(model, BalanceeqP[i=1:length(buses)],
        sum(p_expr[a] for a in ref_data[:bus_arcs][buses[i]]) +
        sum(p_dc[a_dc] for a_dc in ref_data[:bus_arcs_dc][buses[i]]) ==
        sum(pg[g] for g in ref_data[:bus_gens][buses[i]]) -
        rhs[i]
        +
        slack[i]
    )

    #println("buses: ", buses)
    # @constraint(model, BalanceeqP[i=1:length(buses)],
    #     sum(p_expr[a] for a in ref_data[:bus_arcs][buses[i]]) +
    #     sum(p_dc[a_dc] for a_dc in ref_data[:bus_arcs_dc][buses[i]]) ==
    #     sum(pg[g] for g in ref_data[:bus_gens][buses[i]]) -
    #     rhs[i]
    # )


    # Branch flow physics and angle limits
    for (i, branch) in ref_data[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])

        p_fr = p[f_idx]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)

        @constraint(model, p_fr == -b * (va_fr - va_to))

        @constraint(model, va_fr - va_to <= branch["angmax"])
        @constraint(model, va_fr - va_to >= branch["angmin"])
    end

    # HVDC line constraints
    for (i, dcline) in ref_data[:dcline]
        f_idx = (i, dcline["f_bus"], dcline["t_bus"])
        t_idx = (i, dcline["t_bus"], dcline["f_bus"])

        @constraint(model, (1 - dcline["loss1"]) * p_dc[f_idx] + (p_dc[t_idx] - dcline["loss0"]) == 0)
    end

    # Solve the model
    #set_silent(model)
    optimize!(model)
    #println(BalanceeqP)


    # println("Solver status: ", termination_status(model))
    #println("Objective value: ", objective_value(model))
    pg_dict = JuMP.value.(pg)
    slacks = JuMP.value.(slack)
    # println("objective_value: ", objective_value(model))


    Cost_slack = 10 * max_ck1 * sum(slacks[i] for i = 1:length(buses))
    Cost_Pg = sum(gen["cost"][1] * pg_dict[i]^2 + gen["cost"][2] * pg_dict[i] + gen["cost"][3] for (i, gen) in ref_data[:gen])
    Cost_total = objective_value(model)
    total_slack = sum(slacks)
    # pg_vec = [pg_dict[g] for g in gens]
    elapsed = time() - start
    # println("pg_dict: ", pg_vec)
    return Dict(
        "Pg_values" => pg_dict,
        "slacks" => slacks,
        "model" => model,
        "Cost_slack" => Cost_slack,
        "Cost_Pg" => Cost_Pg,
        "Cost_total" => Cost_total,
        "Sum_slacks" => total_slack,
        "time" => elapsed
    )
end


function ACPM(data, Pd=nothing, Qd=nothing; solver=optimizer_with_attributes(
    Ipopt.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => "ma27",
    "hsllib" => HSL_jll.libhsl_path
))
    # Paso 1: ordenar los IDs de buses y crear el mapa
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # step 2  Update the loads 
    if Pd !== nothing && Qd !== nothing
        for (load_id, load_data) in data["load"]
            bus_id = load_data["load_bus"]
            idx = bus_idx_map[bus_id]
            data["load"][load_id]["pd"] = Pd[idx]
            data["load"][load_id]["qd"] = Qd[idx]
        end
    end

    # === Build the model ===
    spec = PowerModels.instantiate_model(
        data,
        PowerModels.ACRPowerModel,
        PowerModels.build_opf
    )
    set_time_limit_sec(spec.model, 1100.0)
    set_silent(spec.model)
    results = PowerModels.optimize_model!(spec; optimizer=solver)

    # Sort the generators index
    ref_gen = spec.ref[:it][:pm][:nw][0][:gen]
    pg_da = spec.var[:it][:pm][:nw][0][:pg]
    gen_ids_sorted = sort(collect(keys(ref_gen)))
    Pg_ordered = [
        JuMP.value(pg_da[g_id])
        for g_id in gen_ids_sorted
    ]

    # println("satus ACOPF ", JuMP.termination_status(spec.model))
    return Dict(
        "flag" => (JuMP.termination_status(spec.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(spec.model),
        "Pg" => Pg_ordered,
        "results" => results
    )
end


function DCPM(data, Pd=nothing, Qd=nothing; solver=silent_gurobi_optimizer)


    # Paso 1: ordenar los IDs de buses y crear el mapa
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    if Pd !== nothing && Qd !== nothing
        for (load_id, load_data) in data["load"]
            bus_id = load_data["load_bus"]
            idx = bus_idx_map[bus_id]
            data["load"][load_id]["pd"] = Pd[idx]
            data["load"][load_id]["qd"] = Qd[idx]
        end
    end


    # === Construcción del modelo ===
    spec = PowerModels.instantiate_model(
        data,
        PowerModels.DCPPowerModel,
        PowerModels.build_opf
    )

    set_silent(spec.model)
    results = PowerModels.optimize_model!(spec; optimizer=solver)

    ref_gen = spec.ref[:it][:pm][:nw][0][:gen]

    pg_da = spec.var[:it][:pm][:nw][0][:pg]


    gen_ids_sorted = sort(collect(keys(ref_gen)))
    Pg_ordered = [
        JuMP.value(pg_da[g_id])
        for g_id in gen_ids_sorted
    ]

    # println("satus DCOPF ", JuMP.termination_status(spec.model))

    return Dict(
        "flag" => (JuMP.termination_status(spec.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(spec.model),
        "Pg" => Pg_ordered,
        "results" => results,
        #"Vm" => [JuMP.value(spec.var[:it][:pm][:nw][0][:vm][i]) for (i, _) in spec.ref[:it][:pm][:nw][0][:bus]],
        #"Va" => [JuMP.value(spec.var[:it][:pm][:nw][0][:va][i]) for (i, _) in spec.ref[:it][:pm][:nw][0][:bus]]
        #"Dual" => dual_vec
        "model" => spec.model
    )
end

function DCPMLayerBatch(data::Dict, rhs::AbstractMatrix; model=quadratic_diff_model(silent_gurobi_optimizer))

    ref_data = ChainRulesCore.ignore_derivatives() do
        PowerModels.build_ref(data; ref_extensions=Dict())
    end[:it][:pm][:nw][0]

    buses = sort(collect(keys(ref_data[:bus])))
    gens = sort(collect(keys(ref_data[:gen])))
    arcs_from = ref_data[:arcs_from]
    arcs_dc = collect(keys(ref_data[:arcs_dc]))
    S = size(rhs, 2)  # Tamaño del batch

    set_silent(model)

    # Variables por batch s
    @variable(model, va[i in buses, s in 1:S])
    @variable(model, pg[i in gens, s in 1:S],
        lower_bound = ref_data[:gen][i]["pmin"],
        upper_bound = ref_data[:gen][i]["pmax"])
    @variable(model, slack[i in 1:length(buses), s in 1:S] >= 0)
    @variable(model, p[a in arcs_from, s in 1:S],
        lower_bound = -ref_data[:branch][a[1]]["rate_a"],
        upper_bound = ref_data[:branch][a[1]]["rate_a"])
    @variable(model, p_dc[a in arcs_dc, s in 1:S])

    # Build expressions for branch flows por s
    p_expr = Dict()
    for s in 1:S, (l, i, j) in arcs_from
        p_expr[(s, l, i, j)] = p[(l, i, j), s]
        p_expr[(s, l, j, i)] = -p[(l, i, j), s]
    end

    from_idx = Dict(arc[1] => arc for arc in ref_data[:arcs_from_dc])
    max_ck1 = maximum(gen["cost"][2] for (i, gen) in ref_data[:gen])

    @objective(model, Min,
        sum(
            sum(gen["cost"][1] * pg[i, s]^2 + gen["cost"][2] * pg[i, s] + gen["cost"][3]
                for (i, gen) in ref_data[:gen]) +
            sum(dcline["cost"][1] * p_dc[from_idx[i], s]^2 + dcline["cost"][2] * p_dc[from_idx[i], s] + dcline["cost"][3]
                for (i, dcline) in ref_data[:dcline]) +
            10 * max_ck1 * sum(slack[i, s] for i = 1:length(buses))
            for s in 1:S)
    )

    # Constraints por s

    @constraint(model, BalanceeqP[i=1:length(buses), s in 1:S],
        sum(p_expr[(s, a...)] for a in ref_data[:bus_arcs][buses[i]]) +
        sum(p_dc[a, s] for a in ref_data[:bus_arcs_dc][buses[i]]) ==
        sum(pg[g, s] for g in ref_data[:bus_gens][buses[i]]) -
        rhs[i, s] +
        slack[i, s]
    )

    for s in 1:S
        for (i, bus) in ref_data[:ref_buses]
            @constraint(model, va[i, s] == 0)
        end



        for (i, branch) in ref_data[:branch]
            f_idx = (i, branch["f_bus"], branch["t_bus"])
            va_fr = va[branch["f_bus"], s]
            va_to = va[branch["t_bus"], s]
            g, b = PowerModels.calc_branch_y(branch)

            @constraint(model, p[f_idx, s] == -b * (va_fr - va_to))
            @constraint(model, va_fr - va_to <= branch["angmax"])
            @constraint(model, va_fr - va_to >= branch["angmin"])
        end

        for (i, dcline) in ref_data[:dcline]
            f_idx = (i, dcline["f_bus"], dcline["t_bus"])
            t_idx = (i, dcline["t_bus"], dcline["f_bus"])
            @constraint(model, (1 - dcline["loss1"]) * p_dc[f_idx, s] + (p_dc[t_idx, s] - dcline["loss0"]) == 0)
        end
    end

    optimize!(model)


    # println("Solver statusDC:    ", termination_status(model))
    # println("Objective value: ", objective_value(model))
    pg_dict = JuMP.value.(pg)
    return pg_dict
end

function ACPMLayerExa(gw::GraphWithRef, Pg_target::AbstractVector{Float64};
    backend=CUDABackend(),
    # backend=nothing, 
    T=Float64)
    data = gw.data
    x = gw.graph.x
    Pg_target_array = Array(Pg_target)
    # println("tamano de Pg", Pg_target_array)
    # Update bus loads from GNN node features (Pd, Qd)
    Pd_full = x[1, :]
    Qd_full = x[2, :]
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bid => i for (i, bid) in enumerate(bus_ids))

    for (load_id, load_data) in data["load"]
        bid = load_data["load_bus"]
        idx = bus_idx_map[bid]
        data["load"][load_id]["pd"] = Pd_full[idx]
        data["load"][load_id]["qd"] = Qd_full[idx]
    end

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    bus_keys = sort(collect(keys(ref[:bus])))
    gen_keys = sort(collect(keys(ref[:gen])))
    branch_keys = sort(collect(keys(ref[:branch])))

    busdict = Dict(k => i for (i, k) in enumerate(bus_keys))
    gendict = Dict(k => i for (i, k) in enumerate(gen_keys))
    branchdict = Dict(k => i for (i, k) in enumerate(branch_keys))
    arcdict = Dict(a => k for (k, a) in enumerate(ref[:arcs]))

    # First collect original generator costs
    raw_cost1 = Dict(k => ref[:gen][k]["cost"][1] for k in gen_keys)
    raw_cost2 = Dict(k => ref[:gen][k]["cost"][2] for k in gen_keys)

    quad_nonzero = filter(!=(0), collect(values(raw_cost1)))
    linear_nonzero = filter(!=(0), collect(values(raw_cost2)))

    min_quad_slack_cost = isempty(quad_nonzero) ? 0.0 : 0.1 * minimum(quad_nonzero)
    min_linear_slack_cost = isempty(linear_nonzero) ? 0.0 : 0.1 * minimum(linear_nonzero)

    # ── Named-tuple arrays for ExaModels ──────────────────────────────────────
    exa_bus = [
        begin
            v = ref[:bus][k]
            bus_loads = [ref[:load][l] for l in ref[:bus_loads][k]]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][k]]
            pd = sum(load["pd"] for load in bus_loads; init=0.0)
            qd = sum(load["qd"] for load in bus_loads; init=0.0)
            gs = sum(shunt["gs"] for shunt in bus_shunts; init=0.0)
            bs = sum(shunt["bs"] for shunt in bus_shunts; init=0.0)
            (i=busdict[k], pd=pd, qd=qd, gs=gs, bs=bs)
        end
        for k in bus_keys
    ]

    exa_gen = [
        begin
            v = ref[:gen][k]

            cost1 = v["cost"][1]
            cost2 = v["cost"][2]

            (
                i=gendict[k],
                valk=k,
                cost1=cost1,
                cost2=cost2,
                cost3=v["cost"][3],

                # fallback slack costs
                slack_cost1=cost1 == 0 ? min_quad_slack_cost : cost1,
                slack_cost2=cost2 == 0 ? min_linear_slack_cost : cost2, bus=busdict[v["gen_bus"]],
                pg_target=Pg_target_array[gendict[k]],
            )
        end
        for k in gen_keys
    ]

    exa_branch = [
        begin
            branch = ref[:branch][k]
            f_idx = arcdict[(k, branch["f_bus"], branch["t_bus"])]
            t_idx = arcdict[(k, branch["t_bus"], branch["f_bus"])]

            g_br, b_br = PowerModels.calc_branch_y(branch)
            tr, ti = PowerModels.calc_branch_t(branch)
            ttm = tr^2 + ti^2
            g_fr, b_fr = branch["g_fr"], branch["b_fr"]
            g_to, b_to = branch["g_to"], branch["b_to"]

            c1b = (-g_br * tr - b_br * ti) / ttm
            c2b = (-b_br * tr + g_br * ti) / ttm
            c3b = (-g_br * tr + b_br * ti) / ttm
            c4b = (-b_br * tr - g_br * ti) / ttm
            c5b = (g_br + g_fr) / ttm
            c6b = (b_br + b_fr) / ttm
            c7b = (g_br + g_to)
            c8b = (b_br + b_to)

            (
                i=branchdict[k],
                j=1,
                f_idx=f_idx,
                t_idx=t_idx,
                f_bus=busdict[branch["f_bus"]],
                t_bus=busdict[branch["t_bus"]],
                c1=c1b, c2=c2b, c3=c3b, c4=c4b,
                c5=c5b, c6=c6b, c7=c7b, c8=c8b,
                rate_a_sq=branch["rate_a"]^2,
            )
        end
        for k in branch_keys
    ]
    # println(exa_gen[1])
    exa_arc = [
        (i=k, rate_a=ref[:branch][l]["rate_a"], bus=busdict[i])
        for (k, (l, i, j)) in enumerate(ref[:arcs])
    ]


    ref_buses = [busdict[i] for (i, _) in ref[:ref_buses]]

    vmin = [ref[:bus][k]["vmin"] for k in bus_keys]
    vmax = [ref[:bus][k]["vmax"] for k in bus_keys]

    pmin = [ref[:gen][k]["pmin"] for k in gen_keys]
    pmax = [ref[:gen][k]["pmax"] for k in gen_keys]
    qmin = [ref[:gen][k]["qmin"] for k in gen_keys]
    qmax = [ref[:gen][k]["qmax"] for k in gen_keys]

    angmin = [ref[:branch][k]["angmin"] for k in branch_keys]
    angmax = [ref[:branch][k]["angmax"] for k in branch_keys]
    rate_a = [ref[:branch][l]["rate_a"] for (k, (l, i, j)) in enumerate(ref[:arcs])]


    nbus = length(exa_bus)
    ngen = length(exa_gen)
    narc = length(exa_arc)
    nbranch = length(exa_branch)

    # Helper: move array to target backend (no-op when backend=nothing)
    # v) = ExaModels.convert_array(v, backend)

    # ── Build ExaCore ─────────────────────────────────────────────────────────
    w = ExaCore(T; backend=backend)
    # w = ExaCore(T; backend=nothing)

    va = variable(w, nbus)

    vm = variable(
        w,
        nbus;
        start=fill!(similar(vmin, Float64), 1.0),
        lvar=vmin,
        uvar=vmax,
    )
    pg = variable(w, ngen; lvar=pmin, uvar=pmax)
    qg = variable(w, ngen; lvar=qmin, uvar=qmax)
    p = variable(w, narc; lvar=-rate_a, uvar=rate_a)
    q = variable(w, narc; lvar=-rate_a, uvar=rate_a)

    # Slack variables (non-negative, no upper bound) for soft pg_target matching
    slack_pos = variable(w, ngen; lvar=zeros(ngen))
    slack_neg = variable(w, ngen; lvar=zeros(ngen))


    # ── Slack penalty costs ─────────────────────────────────────────────────────
    quad_costs = [g.cost1 for g in exa_gen]
    linear_costs = [g.cost2 for g in exa_gen]

    quad_nonzero = filter(!=(0), quad_costs)
    linear_nonzero = filter(!=(0), linear_costs)

    min_quad_slack_cost = isempty(quad_nonzero) ? 0.0 : 0.1 * minimum(quad_nonzero)
    min_linear_slack_cost = isempty(linear_nonzero) ? 0.0 : 0.1 * minimum(linear_nonzero)

    slack_quad_cost = [
        g.cost1 == 0 ? min_quad_slack_cost : g.cost1
        for g in exa_gen
    ]

    slack_linear_cost = [
        g.cost2 == 0 ? min_linear_slack_cost : g.cost2
        for g in exa_gen
    ]

    # Objective: generation cost + quadratic slack penalty (mirrors ACPMLayer)
    # o = objective(w,
    #     g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3
    #     + g.cost1 * (slack_pos[g.i] + slack_neg[g.i])^2
    #     + g.cost2 * (slack_pos[g.i] + slack_neg[g.i])
    #     for g in exa_gen)

    # Objective: generation cost + slack penalty with fallback costs for zero-cost generators
    o = objective(w,
        g.cost1 * pg[g.i]^2 +
        g.cost2 * pg[g.i] +
        g.cost3 +
        g.slack_cost1 * (slack_pos[g.i] + slack_neg[g.i])^2 +
        g.slack_cost2 * (slack_pos[g.i] + slack_neg[g.i])
        for g in exa_gen
    )

    # ── AC-OPF physics constraints (same as OPF_ExaModels.jl) ────────────────
    c2 = constraint(w,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2
        -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
        -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
        for b in exa_branch)

    c1 = constraint(w, va[i] for i in ref_buses)

    c3 = constraint(w,
        q[b.f_idx] + b.c6 * vm[b.f_bus]^2
        + b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
        -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
        for b in exa_branch)

    c4 = constraint(w,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2
        -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
        -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
        for b in exa_branch)

    c5 = constraint(w,
        q[b.t_idx] + b.c8 * vm[b.t_bus]^2
        + b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
        -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
        for b in exa_branch)

    c6 = constraint(w,
        va[b.f_bus] - va[b.t_bus] for b in exa_branch;
        lcon=angmin, ucon=angmax)

    c7 = constraint(w,
        p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in exa_branch;
        lcon=fill(-Inf, nbranch))

    c8 = constraint(w,
        p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in exa_branch;
        lcon=fill(-Inf, nbranch))

    c9 = constraint(w, b.pd + b.gs * vm[b.i]^2 for b in exa_bus)
    c10 = constraint(w, b.qd - b.bs * vm[b.i]^2 for b in exa_bus)

    c11 = constraint!(w, c9, a.bus => p[a.i] for a in exa_arc)
    c12 = constraint!(w, c10, a.bus => q[a.i] for a in exa_arc)
    c13 = constraint!(w, c9, g.bus => -pg[g.i] for g in exa_gen)
    c14 = constraint!(w, c10, g.bus => -qg[g.i] for g in exa_gen)

    # pg_aux LAST — dual = ∂obj/∂Pg_target (gradient signal for backward pass)
    c_pg_aux = constraint(w,
        pg[g.i] - g.pg_target - slack_pos[g.i] + slack_neg[g.i]
        for g in exa_gen)

    m = ExaModel(w)
    # @allowscalar
    # result = madnlp(m, tol=1e-4)
    result = madnlp(m, tol=1e-6, max_iter=3000, print_level=MadNLP.ERROR)
    # println(g.i for g in exa_gen)

    sol = Array(result.solution)
    mults = Array(result.multipliers)

    # Build CPU permutation: sorted-gen-key order → gendict position (no GPU scalar indexing).
    gen_keys_sorted = sort(collect(keys(ref[:gen])))        # CPU Vector{Int}
    perm = [gendict[k] for k in gen_keys_sorted] # CPU Vector{Int}

    # Variable layout: va(nbus) | vm(nbus) | pg(ngen) | qg | p | q | slacks
    # Reorder from exa_gen/dict-iteration order to sorted-gen-key order (matches ACPMLayer).
    pg_val = solution(result, pg)
    duals = multipliers(result, c_pg_aux)
    slack_pos_val = Array(solution(result, slack_pos))
    slack_neg_val = Array(solution(result, slack_neg))

    Cost_slack = sum(
        exa_gen[i].cost1 * (slack_pos_val[i] + slack_neg_val[i])^2 +
        exa_gen[i].cost2 * (slack_pos_val[i] + slack_neg_val[i])
        for i in eachindex(exa_gen)
    )

    Cost_Pg = sum(
        exa_gen[i].cost1 * Pg_target_array[i]^2 +
        exa_gen[i].cost2 * Pg_target_array[i] +
        exa_gen[i].cost3
        for i in eachindex(exa_gen)
    )

    cost_slack_all = result.objective - Cost_Pg
    max_slack = maximum(abs.(vcat(slack_pos_val, slack_neg_val)))

    println("Solver status ACPMLayerExa: ", result.status)
    # println("Objective value: ", result.objective)

    return Dict(
        "flag" => result.status == MadNLP.SOLVE_SUCCEEDED,
        "objective" => result.objective,
        "Pg" => pg_val,
        "Pg_pred_lf" => Pg_target_array,
        "Cost_slack_all" => cost_slack_all,
        "Cost_Pg" => Cost_Pg,
        "pos_slack" => slack_pos_val,
        "neg_slack" => slack_neg_val,
        "max_slack" => max_slack * 100, #[MW],
        "Cost_slack" => Cost_slack,
        "Dual" => Array(duals)
    )
end



function ACPMExa(data, Pd=nothing, Qd=nothing;
    backend=CUDABackend(),
    # backend=nothing,
    T=Float64)
    start = time()
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bid => i for (i, bid) in enumerate(bus_ids))

    if Pd !== nothing && Qd !== nothing
        for (load_id, load_data) in data["load"]
            bus_id = load_data["load_bus"]
            idx = bus_idx_map[bus_id]
            data["load"][load_id]["pd"] = Pd[idx]
            data["load"][load_id]["qd"] = Qd[idx]
        end
    end

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    bus_keys = sort(collect(keys(ref[:bus])))
    gen_keys = sort(collect(keys(ref[:gen])))
    branch_keys = sort(collect(keys(ref[:branch])))

    busdict = Dict(k => i for (i, k) in enumerate(bus_keys))
    gendict = Dict(k => i for (i, k) in enumerate(gen_keys))
    branchdict = Dict(k => i for (i, k) in enumerate(branch_keys))
    arcdict = Dict(a => k for (k, a) in enumerate(ref[:arcs]))

    # ── Named-tuple arrays for ExaModels ──────────────────────────────────────
    exa_bus = [
        begin
            bus_loads = [ref[:load][l] for l in ref[:bus_loads][k]]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][k]]

            pd = sum(load["pd"] for load in bus_loads; init=0.0)
            qd = sum(load["qd"] for load in bus_loads; init=0.0)
            gs = sum(shunt["gs"] for shunt in bus_shunts; init=0.0)
            bs = sum(shunt["bs"] for shunt in bus_shunts; init=0.0)

            (
                i=busdict[k],
                pd=pd,
                qd=qd,
                gs=gs,
                bs=bs
            )
        end
        for k in bus_keys
    ]

    exa_gen = [
        begin
            v = ref[:gen][k]
            (
                i=gendict[k],
                valk=k,
                cost1=v["cost"][1],
                cost2=v["cost"][2],
                cost3=v["cost"][3],
                bus=busdict[v["gen_bus"]]
            )
        end
        for k in gen_keys
    ]

    exa_branch = [
        begin
            branch = ref[:branch][k]

            f_idx = arcdict[(k, branch["f_bus"], branch["t_bus"])]
            t_idx = arcdict[(k, branch["t_bus"], branch["f_bus"])]

            g_br, b_br = PowerModels.calc_branch_y(branch)
            tr, ti = PowerModels.calc_branch_t(branch)
            ttm = tr^2 + ti^2

            g_fr, b_fr = branch["g_fr"], branch["b_fr"]
            g_to, b_to = branch["g_to"], branch["b_to"]

            c1b = (-g_br * tr - b_br * ti) / ttm
            c2b = (-b_br * tr + g_br * ti) / ttm
            c3b = (-g_br * tr + b_br * ti) / ttm
            c4b = (-b_br * tr - g_br * ti) / ttm
            c5b = (g_br + g_fr) / ttm
            c6b = (b_br + b_fr) / ttm
            c7b = (g_br + g_to)
            c8b = (b_br + b_to)

            (
                i=branchdict[k],
                j=1,
                f_idx=f_idx,
                t_idx=t_idx,
                f_bus=busdict[branch["f_bus"]],
                t_bus=busdict[branch["t_bus"]],
                c1=c1b,
                c2=c2b,
                c3=c3b,
                c4=c4b,
                c5=c5b,
                c6=c6b,
                c7=c7b,
                c8=c8b,
                rate_a_sq=branch["rate_a"]^2,
            )
        end
        for k in branch_keys
    ]

    exa_arc = [
        (i=k, rate_a=ref[:branch][l]["rate_a"], bus=busdict[i])
        for (k, (l, i, j)) in enumerate(ref[:arcs])
    ]

    ref_buses = [busdict[i] for (i, _) in ref[:ref_buses]]

    vmin = [ref[:bus][k]["vmin"] for k in bus_keys]
    vmax = [ref[:bus][k]["vmax"] for k in bus_keys]

    pmin = [ref[:gen][k]["pmin"] for k in gen_keys]
    pmax = [ref[:gen][k]["pmax"] for k in gen_keys]
    qmin = [ref[:gen][k]["qmin"] for k in gen_keys]
    qmax = [ref[:gen][k]["qmax"] for k in gen_keys]

    angmin = [ref[:branch][k]["angmin"] for k in branch_keys]
    angmax = [ref[:branch][k]["angmax"] for k in branch_keys]
    rate_a = [ref[:branch][l]["rate_a"] for (k, (l, i, j)) in enumerate(ref[:arcs])]

    nbus = length(exa_bus)
    ngen = length(exa_gen)
    narc = length(exa_arc)
    nbranch = length(exa_branch)

    # ── Build ExaCore ─────────────────────────────────────────────────────────
    w = ExaCore(T; backend=backend)

    va = variable(w, nbus)

    vm = variable(
        w,
        nbus;
        start=fill!(similar(vmin, Float64), 1.0),
        lvar=vmin,
        uvar=vmax,
    )

    pg = variable(w, ngen; lvar=pmin, uvar=pmax)
    qg = variable(w, ngen; lvar=qmin, uvar=qmax)
    p = variable(w, narc; lvar=-rate_a, uvar=rate_a)
    q = variable(w, narc; lvar=-rate_a, uvar=rate_a)

    # Original ACOPF objective
    o = objective(
        w,
        g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3
        for g in exa_gen
    )

    # ── AC-OPF constraints ────────────────────────────────────────────────────
    c1 = constraint(w, va[i] for i in ref_buses)

    c2 = constraint(
        w,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2
        -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
        -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
        for b in exa_branch
    )

    c3 = constraint(
        w,
        q[b.f_idx] + b.c6 * vm[b.f_bus]^2
        + b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
        -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
        for b in exa_branch
    )

    c4 = constraint(
        w,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2
        -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
        -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
        for b in exa_branch
    )

    c5 = constraint(
        w,
        q[b.t_idx] + b.c8 * vm[b.t_bus]^2
        + b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
        -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
        for b in exa_branch
    )

    c6 = constraint(
        w,
        va[b.f_bus] - va[b.t_bus] for b in exa_branch;
        lcon=angmin,
        ucon=angmax,
    )

    c7 = constraint(
        w,
        p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in exa_branch;
        lcon=fill(-Inf, nbranch),
    )

    c8 = constraint(
        w,
        p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in exa_branch;
        lcon=fill(-Inf, nbranch),
    )

    c9 = constraint(w, b.pd + b.gs * vm[b.i]^2 for b in exa_bus)
    c10 = constraint(w, b.qd - b.bs * vm[b.i]^2 for b in exa_bus)

    c11 = constraint!(w, c9, a.bus => p[a.i] for a in exa_arc)
    c12 = constraint!(w, c10, a.bus => q[a.i] for a in exa_arc)
    c13 = constraint!(w, c9, g.bus => -pg[g.i] for g in exa_gen)
    c14 = constraint!(w, c10, g.bus => -qg[g.i] for g in exa_gen)

    m = ExaModel(w)
    result = madnlp(m, tol=1e-6)

    pg_val = Array(solution(result, pg))
    qg_val = Array(solution(result, qg))
    vm_val = Array(solution(result, vm))
    va_val = Array(solution(result, va))

    elapsed = time() - start
    println("Solver status ACPMExa: ", result.status, "  time: ", round(elapsed, digits=2), "s")

    return Dict(
        "flag" => result.status == MadNLP.SOLVE_SUCCEEDED,
        "objective" => result.objective,
        "Pg" => pg_val,
        "Qg" => qg_val,
        "Vm" => vm_val,
        "Va" => va_val,
        "termination_status" => result.status,
        "results" => result
    )
end

function DCPMExa(data, Pd=nothing, Qd=nothing;
    backend=CUDABackend(), T=Float64)
    start = time()

    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # Update loads if provided
    if Pd !== nothing
        for (load_id, load_data) in data["load"]
            bus_id = load_data["load_bus"]
            idx = bus_idx_map[bus_id]
            data["load"][load_id]["pd"] = Pd[idx]
        end
    end

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    bus_keys = sort(collect(keys(ref[:bus])))
    gen_keys = sort(collect(keys(ref[:gen])))
    branch_keys = sort(collect(keys(ref[:branch])))

    busdict = Dict(k => i for (i, k) in enumerate(bus_keys))
    gendict = Dict(k => i for (i, k) in enumerate(gen_keys))
    branchdict = Dict(k => i for (i, k) in enumerate(branch_keys))

    # Buses
    exa_bus = [
        begin
            bus_loads = [ref[:load][l] for l in ref[:bus_loads][k]]
            pd = sum(load["pd"] for load in bus_loads; init=0.0)

            (i=busdict[k], pd=pd)
        end
        for k in bus_keys
    ]

    # Generators
    exa_gen = [
        begin
            v = ref[:gen][k]
            c = v["cost"]

            (
                i=gendict[k],
                bus=busdict[v["gen_bus"]],
                cost1=length(c) >= 3 ? c[1] : 0.0,
                cost2=length(c) >= 2 ? c[end-1] : 0.0,
                cost3=length(c) >= 1 ? c[end] : 0.0,
            )
        end
        for k in gen_keys
    ]

    # Branches (DC model)
    exa_branch = [
        begin
            br = ref[:branch][k]

            tap = br["tap"]
            tap = (tap == 0.0 || isnan(tap)) ? 1.0 : tap

            (
                i=branchdict[k],
                f_bus=busdict[br["f_bus"]],
                t_bus=busdict[br["t_bus"]],
                b=1.0 / (br["br_x"] * tap),   # susceptance
                rate_a=br["rate_a"],
            )
        end
        for k in branch_keys
    ]

    # Reference buses
    ref_buses = [busdict[i] for (i, _) in ref[:ref_buses]]

    # Limits
    pmin = [ref[:gen][k]["pmin"] for k in gen_keys]
    pmax = [ref[:gen][k]["pmax"] for k in gen_keys]

    angmin = [ref[:branch][k]["angmin"] for k in branch_keys]
    angmax = [ref[:branch][k]["angmax"] for k in branch_keys]
    rate_a = [ref[:branch][k]["rate_a"] for k in branch_keys]

    nbus = length(exa_bus)
    ngen = length(exa_gen)
    nbranch = length(exa_branch)

    w = ExaCore(T; backend=backend)

    va = variable(w, nbus)
    pg = variable(w, ngen; lvar=pmin, uvar=pmax)
    pf = variable(w, nbranch; lvar=-rate_a, uvar=rate_a)

    # Objective
    o = objective(
        w,
        g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3
        for g in exa_gen
    )

    # Reference angle
    c_ref = constraint(w, va[i] for i in ref_buses)

    # DC power flow
    c_ohm = constraint(
        w,
        pf[br.i] - br.b * (va[br.f_bus] - va[br.t_bus])
        for br in exa_branch
    )

    # Angle limits
    c_ang = constraint(
        w,
        va[br.f_bus] - va[br.t_bus] for br in exa_branch;
        lcon=angmin,
        ucon=angmax
    )

    # Power balance
    c_pbal = constraint(w, b.pd for b in exa_bus)

    constraint!(w, c_pbal, g.bus => -pg[g.i] for g in exa_gen)
    constraint!(w, c_pbal, br.f_bus => pf[br.i] for br in exa_branch)
    constraint!(w, c_pbal, br.t_bus => -pf[br.i] for br in exa_branch)

    m = ExaModel(w)
    result = madnlp(m, tol=1e-6, print_level=MadNLP.ERROR)
    elapsed = time() - start
    # println("Solver status DCPMExa: ", result.status, "  time: ", round(elapsed, digits=2), "s")

    return Dict(
        "flag" => result.status == MadNLP.SOLVE_SUCCEEDED,
        "objective" => result.objective,
        "Pg" => Array(solution(result, pg)),
        "Va" => Array(solution(result, va)),
        "Pf" => Array(solution(result, pf)),
        "status" => result.status
    )
end

function DCPMLayerExa(gw::GraphWithRef, rhs::AbstractMatrix;
    backend=CUDABackend(), T=Float64)

    start = time()

    data = gw.data

    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    bus_keys = sort(collect(keys(ref[:bus])))
    gen_keys = sort(collect(keys(ref[:gen])))
    branch_keys = sort(collect(keys(ref[:branch])))

    busdict = Dict(k => i for (i, k) in enumerate(bus_keys))
    gendict = Dict(k => i for (i, k) in enumerate(gen_keys))
    branchdict = Dict(k => i for (i, k) in enumerate(branch_keys))


    # ── Named-tuple arrays for ExaModels ──────────────────────────────────────
    exa_bus = [
        begin
            v = ref[:bus][k]
            bus_loads = [ref[:load][l] for l in ref[:bus_loads][k]]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][k]]
            pd = sum(load["pd"] for load in bus_loads; init=0.0)
            qd = sum(load["qd"] for load in bus_loads; init=0.0)
            gs = sum(shunt["gs"] for shunt in bus_shunts; init=0.0)
            bs = sum(shunt["bs"] for shunt in bus_shunts; init=0.0)
            (i=busdict[k], pd=pd, qd=qd, gs=gs, bs=bs, rhs_p=rhs[busdict[k]],)
        end
        for k in bus_keys
    ]

    exa_gen = [
        begin
            v = ref[:gen][k]
            c = v["cost"]

            (
                i=gendict[k],
                bus=busdict[v["gen_bus"]],
                cost1=c[1],
                cost2=c[2],
                cost3=c[3],
            )
        end
        for k in gen_keys
    ]

    exa_branch = [
        begin
            br = ref[:branch][k]

            tap = br["tap"]
            tap = (tap == 0.0 || isnan(tap)) ? 1.0 : tap

            (
                i=branchdict[k],
                f_bus=busdict[br["f_bus"]],
                t_bus=busdict[br["t_bus"]],
                b=1.0 / (br["br_x"] * tap),
                rate_a=br["rate_a"],
                angmin=br["angmin"],
                angmax=br["angmax"],
            )
        end
        for k in branch_keys
    ]

    ref_buses = [busdict[i] for (i, _) in ref[:ref_buses]]

    pmin = [ref[:gen][k]["pmin"] for k in gen_keys]
    pmax = [ref[:gen][k]["pmax"] for k in gen_keys]
    rate_a = [ref[:branch][k]["rate_a"] for k in branch_keys]

    nbus = length(exa_bus)
    ngen = length(exa_gen)
    nbranch = length(exa_branch)

    # ─────────────────────────────────────────────
    # ExaCore
    # ─────────────────────────────────────────────
    w = ExaCore(T; backend=backend)

    va = variable(w, nbus)
    pg = variable(w, ngen; lvar=pmin, uvar=pmax)
    pf = variable(w, nbranch; lvar=-rate_a, uvar=rate_a)

    # Slack (same as your JuMP)
    slack_pos = variable(w, nbus; lvar=zeros(nbus))

    # ─────────────────────────────────────────────
    # Objective
    # ─────────────────────────────────────────────
    max_ck1 = maximum(g.cost2 for g in exa_gen)

    ngen = length(exa_gen)
    nbus = length(exa_bus)

    o = objective(w,
        (g.cost1 * pg[g.i]^2 +
         g.cost2 * pg[g.i] +
         g.cost3) / nbus
        +
        (10 * max_ck1 * slack_pos[b.i]) / ngen
        for g in exa_gen, b in exa_bus
    )


    # ─────────────────────────────────────────────
    # Constraints
    # ─────────────────────────────────────────────

    # Reference angle
    c_ref = constraint(w, va[i] for i in ref_buses)

    # DC flow physics
    c_ohm = constraint(
        w,
        pf[br.i] - br.b * (va[br.f_bus] - va[br.t_bus])
        for br in exa_branch
    )

    # Angle limits
    c_ang = constraint(
        w,
        va[br.f_bus] - va[br.t_bus] for br in exa_branch;
        lcon=[br.angmin for br in exa_branch],
        ucon=[br.angmax for br in exa_branch],
    )

    # Power balance (THIS IS YOUR KEY CONSTRAINT)
    c_pbal = constraint(w, b.rhs_p - slack_pos[b.i] for b in exa_bus)

    constraint!(w, c_pbal, g.bus => -pg[g.i] for g in exa_gen)
    constraint!(w, c_pbal, br.f_bus => pf[br.i] for br in exa_branch)
    constraint!(w, c_pbal, br.t_bus => -pf[br.i] for br in exa_branch)

    # ─────────────────────────────────────────────
    # Solve
    # ─────────────────────────────────────────────
    m = ExaModel(w)
    result = madnlp(m, tol=1e-6)

    elapsed = time() - start
    println("Solver status DCPMLayerExa: ", result.status,
        "  time: ", round(elapsed, digits=2), "s")
    # println("Objective value: ", result.objective)
    return Array(solution(result, pg))
end




end
