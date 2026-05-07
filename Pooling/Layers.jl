using JuMP
using Gurobi
using DiffOpt
using Flux
using Zygote
using MathOptInterface
using Dates


const MOI = MathOptInterface
using ChainRulesCore
const bwd_times_lf = Float64[]
const bwd_times_hf = Float64[]
const fwd_times_lf = Float64[]
const fwd_times_hf = Float64[]

function tunable_layer(instance::Dict{Symbol,Any}, 
    α_fix, β_fix, σ_fix, η_fix;
    model::Model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer))
    
    # Parameter Dictionaries
    sources = instance[:sources]
    blenders = instance[:blenders]
    terminals = instance[:terminals]

    supply_limit = instance[:supply_limit]
    blender_capacity = instance[:blender_capacity]
    demand = instance[:demand]

    properties = instance[:properties]
    prop_limits = instance[:property_limits]

    allowed_source_blender = instance[:allowed_source_blender]
    allowed_source_terminal = instance[:allowed_source_terminal]
    allowed_blender_terminal = instance[:allowed_blender_terminal]
    allowed_ilj = instance[:allowed_ilj]

    f_il = instance[:flow_cost_il]
    f_ij = instance[:flow_cost_ij]
    f_lj = instance[:flow_cost_lj]

    fb_ij = instance[:flow_bound_ij]
    fb_lj = instance[:flow_bound_lj]

    # Variable Definition
    @variable(model, 0 <= q[i_l in allowed_source_blender] <= 1)
    @variable(model, y[l_j in allowed_blender_terminal] >= 0)
    @variable(model, z[i_j in allowed_source_terminal] >= 0)
    @variable(model, x[i_lj in allowed_ilj] >= 0)

   
    # Tunable Parameters
    α_dict_key = Dict()
    for (i,i_l_j) in enumerate(allowed_ilj)
        α_dict_key[i_l_j] = i
    end

    β_dict_key = Dict()
    for (i, l_j) in enumerate(allowed_blender_terminal)
        β_dict_key[l_j] = i
    end

    σ_dict_key = Dict()
    for (i, i_j) in enumerate(allowed_source_terminal)
        σ_dict_key[i_j] = i
    end

    η_dict_key = Dict()
    for (i, i_l) in enumerate(allowed_source_blender)
        η_dict_key[i_l] = i
    end

    @variable(model, α[i_l_j in allowed_ilj] in Parameter(α_fix[α_dict_key[i_l_j]]))
    @variable(model, β[l_j in allowed_blender_terminal] in Parameter(β_fix[β_dict_key[l_j]]))
    @variable(model, σ[i_j in allowed_source_terminal] in Parameter(σ_fix[σ_dict_key[i_j]]))
    @variable(model, η[i_l in allowed_source_blender] in Parameter(η_fix[η_dict_key[i_l]]))

    # Fraction Flow Variable Summation equals 1
    for l in blenders
        @constraint(model, sum(q[i_l] for i_l in allowed_source_blender if i_l[2] == l; init=0.0) == 1)
    end

    # Property Constraints
    for j in terminals
        total_flow = @expression(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
            sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)
        )
        for (prop, prop_dict) in properties
            prop_expr = @expression(model,
                sum(prop_dict[i] * x[(i, l, j)] for (i, l, jt) in allowed_ilj if jt == j && haskey(prop_dict, i); init=0.0) +
                sum(prop_dict[i_j[1]] * z[i_j] for i_j in allowed_source_terminal if i_j[2] == j && haskey(prop_dict, i_j[1]); init=0.0)
            )
            if haskey(prop_limits["upper"], prop) && haskey(prop_limits["upper"][prop], j)
                @constraint(model, prop_expr <= prop_limits["upper"][prop][j] * total_flow + 1e-3)
            end
            if haskey(prop_limits["lower"], prop) && haskey(prop_limits["lower"][prop], j)
                @constraint(model, prop_expr >= prop_limits["lower"][prop][j] * total_flow - 1e-3)
            end
        end
    end

    # Availability Constraint
    for i in sources
        pool_part = sum(x[(i, l, j)] for (ii, l, j) in allowed_ilj if ii == i; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[1] == i; init=0.0)
        @constraint(model, pool_part + direct_part <= supply_limit[i])
    end

    # Pool Capacity
    for l in blenders
        @constraint(model, sum(y[l_j] for l_j in allowed_blender_terminal if l_j[1] == l; init=0.0) <= blender_capacity[l])
    end

    # Demand Constraint
    for j in terminals
        pooled_part = sum(
            x[(i, l, j)] for (i, l, jj) in allowed_ilj if jj == j; init=0.0
        )
        direct_part = sum(
            z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0
        )
        @constraint(model, pooled_part + direct_part <= demand[j])
    end

    # Pool to Terminal Flow Capacity
    for l_j in allowed_blender_terminal
        if haskey(fb_lj, l_j)
            @constraint(model, y[l_j] <= fb_lj[l_j])
        end
    end

    # Source to Terminal Flow Capacity
    for i_j in allowed_source_terminal
        if haskey(fb_ij, i_j)
            @constraint(model, z[i_j] <= fb_ij[i_j])
        end
    end

    # Constraint Tightening
    for l_j in allowed_blender_terminal
        l, j = l_j
        @constraint(model, sum(x[(i, l, j)] for (i, ll, jj) in allowed_ilj if ll == l && jj == j; init=0.0) == y[l_j])
    end

    # Fractional Flow Pool Capacity
    for (i, l) in allowed_source_blender
        @constraint(
            model,
            sum(x[(i, l, j)] for (ii, ll, j) in allowed_ilj if ii == i && ll == l; init=0.0)
            <= blender_capacity[l] * q[(i, l)]
        )
    end

    # McCormick Envelopes
    for (i, l, j) in allowed_ilj
        c_lj = get(fb_lj, (l, j), blender_capacity[l])

        # x >= y + c*q - c
        @constraint(model, x[(i,l,j)] >= y[(l,j)] + c_lj * q[(i,l)] - c_lj)

        # x <= y
        @constraint(model, x[(i,l,j)] <= y[(l,j)])

        # x <= c*q
        @constraint(model, x[(i,l,j)] <= c_lj * q[(i,l)])
    end

    #@variable(model, obj)

    
    quadratic_cost = 0.1
    @objective(model, Min, -sum(
                (get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0))
                * α[(i,l,j)] * x[(i, l, j)]
                for (i, l, j) in allowed_ilj; init=0.0
            )
            - sum(β[(l,j)] * y[(l,j)] for (l,j) in allowed_blender_terminal)
            - sum(f_ij[i_j] * z[i_j] * σ[i_j]
                for i_j in allowed_source_terminal; init=0.0)
            - sum(η[(i,l)] * q[(i,l)] for (i,l) in allowed_source_blender)
            + sum(x[i_l_j]^2 for i_l_j in allowed_ilj)*quadratic_cost
            + sum(y[l_j]^2  for l_j  in allowed_blender_terminal)*quadratic_cost
            + sum(z[i_j]^2  for i_j  in allowed_source_terminal)*quadratic_cost
            + sum(q[i_l]^2 for i_l in allowed_source_blender))*quadratic_cost

    optimize!(model)

    q_val = zeros(length(allowed_source_blender))
    for (i,k) in enumerate(allowed_source_blender)
        q_val[i] = value(model[:q][k])
    end

    return q_val
end

function tunable_layer_logged(instance, α, β, σ, η)
    model = Model(Gurobi.Optimizer)

    q = tunable_layer(
        instance, α, β, σ, η;
        model = model
    )

    tunable_obj = value(model[:obj])

    return q, tunable_obj
end

function ChainRulesCore.rrule(::typeof(tunable_layer), instance::Dict{Symbol,Any}, 
    α_fix, β_fix, σ_fix, η_fix)
    forward_time = time()

    opt = MOI.OptimizerWithAttributes(
        Gurobi.Optimizer,
        "QCPDual" => 1,
        "BarConvTol" => 1e-10,
        "BarQCPConvTol" => 1e-10,
        "FeasibilityTol" => 1e-9,
        "OptimalityTol" => 1e-9
    )
    model = DiffOpt.quadratic_diff_model(opt)

    output = tunable_layer(instance, α_fix, β_fix, σ_fix, η_fix; model = model)

    allowed_ilj = instance[:allowed_ilj]
    allowed_blender_terminal = instance[:allowed_blender_terminal]
    allowed_source_terminal = instance[:allowed_source_terminal]
    allowed_source_blender = instance[:allowed_source_blender]

    α_var = model[:α]
    β_var = model[:β]
    σ_var = model[:σ]
    η_var = model[:η]
    
    α_param_refs = [ParameterRef(α_var[i_l_j]) for i_l_j in allowed_ilj]
    β_param_refs = [ParameterRef(β_var[l_j]) for l_j in allowed_blender_terminal]
    σ_param_refs = [ParameterRef(σ_var[i_j]) for i_j in allowed_source_terminal]
    η_param_refs = [ParameterRef(η_var[i_l]) for i_l in allowed_source_blender]

    fwd_time_ref = time() - forward_time
    push!(fwd_times_lf, fwd_time_ref)
    
    function pullback_tunable_layer(dx)
        backw_time = time()
        q_var = model[:q]

        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), q_var, dx)

        DiffOpt.reverse_differentiate!(model)

        dα = MOI.get.(model, DiffOpt.ReverseConstraintSet(), α_param_refs)
        dβ = MOI.get.(model, DiffOpt.ReverseConstraintSet(), β_param_refs)
        dσ = MOI.get.(model, DiffOpt.ReverseConstraintSet(), σ_param_refs)
        dη = MOI.get.(model, DiffOpt.ReverseConstraintSet(), η_param_refs)

        grad_dα = [g.value for g in dα]
        grad_dβ = [g.value for g in dβ]
        grad_dσ = [g.value for g in dσ]
        grad_dη = [g.value for g in dη]

        bwd_time_ref = time() - backw_time
        push!(bwd_times_lf, bwd_time_ref)
        return NoTangent(), NoTangent(), grad_dα, grad_dβ, grad_dσ, grad_dη
    end

    return output, pullback_tunable_layer
end

function true_model_with_slacks_aggressive_LP(instance:: Dict,
    fixed_q_values;
    model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer))

    if model === nothing
        model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer)
    end

    # Parameter Dictionaries
    sources = instance[:sources]
    blenders = instance[:blenders]
    terminals = instance[:terminals]

    supply_limit = instance[:supply_limit]
    blender_capacity = instance[:blender_capacity]
    demand = instance[:demand]

    properties = instance[:properties]
    prop_limits = instance[:property_limits]

    allowed_source_blender = instance[:allowed_source_blender]
    allowed_source_terminal = instance[:allowed_source_terminal]
    allowed_blender_terminal = instance[:allowed_blender_terminal]
    allowed_ilj = instance[:allowed_ilj]

    f_il = instance[:flow_cost_il]
    f_ij = instance[:flow_cost_ij]
    f_lj = instance[:flow_cost_lj]

    fb_ij = instance[:flow_bound_ij]
    fb_lj = instance[:flow_bound_lj]

    prop_keys = collect(keys(properties))

    # Variables and q is parameter
    q_dict_key = Dict()
    for (i, i_l) in enumerate(allowed_source_blender)
        q_dict_key[i_l] = i
    end

    #@variable(model, q[i_l in allowed_source_blender] in Parameter(fixed_q_values[q_dict_key[i_l]]))
    q = Dict()
    for i_l in allowed_source_blender
        q[i_l] = fixed_q_values[q_dict_key[i_l]]
    end
    
    @variable(model, y[l_j in allowed_blender_terminal] >= 0)
    @variable(model, z[i_j in allowed_source_terminal] >= 0)
    @variable(model, x[i_l_j in allowed_ilj] >= 0)

    # Slacks
    
    @variable(model, 1000>=s_bilinear_plus[i_l_j in allowed_ilj] >= 0)  
    @variable(model, 1000>=s_bilinear_minus[i_l_j in allowed_ilj] >= 0)  
         

    # Auxiliary Flow Variable Substitution
    for (i,l,j) in allowed_ilj
        if(get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)<=0)
            @constraint(model, x[(i,l,j)]  == q[(i,l)] * y[(l,j)])
            @constraint(model, s_bilinear_plus[(i,l,j)] + s_bilinear_minus[(i,l,j)]   == 0)
        else
            @constraint(model, x[(i,l,j)] + s_bilinear_plus[(i,l,j)] - s_bilinear_minus[(i,l,j)] == q[(i,l)] * y[(l,j)])
        end
    end

    # Fraction Flow Variable Summation equals 1
    

    # Property Constraints
    for j in terminals
        total_flow = @expression(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
            sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)
        )
        for (prop, prop_dict) in properties
            prop_expr = @expression(model,
                sum(prop_dict[i] * x[(i, l, j)] for (i, l, jt) in allowed_ilj if jt == j && haskey(prop_dict, i); init=0.0) +
                sum(prop_dict[i_j[1]] * z[i_j] for i_j in allowed_source_terminal if i_j[2] == j && haskey(prop_dict, i_j[1]); init=0.0)
            )
            if haskey(prop_limits["upper"], prop) && haskey(prop_limits["upper"][prop], j)
                @constraint(model, prop_expr <= prop_limits["upper"][prop][j] * total_flow + 1e-3)
            end
            if haskey(prop_limits["lower"], prop) && haskey(prop_limits["lower"][prop], j)
                @constraint(model, prop_expr >= prop_limits["lower"][prop][j] * total_flow - 1e-3)
            end
        end
    end


    # Availability Constraint
    for i in sources
        pool_part = sum(x[(i,l,j)] for (ii,l,j) in allowed_ilj if ii == i; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[1] == i; init=0.0)

        @constraint(model,
            pool_part + direct_part <= supply_limit[i] 
        )
    end

    # Pool Capacity
    for l in blenders
        @constraint(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[1] == l; init=0.0)
            <= blender_capacity[l] 
        )
    end

    # Demand Constraint
    for j in terminals
        pooled_part = sum(x[(i,l,j)] for (i,l,jj) in allowed_ilj if jj == j; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)

        @constraint(model,
            pooled_part + direct_part <= demand[j] 
        )
    end

    # Pool to Terminal Flow Capacity
    for l_j in allowed_blender_terminal
        if haskey(fb_lj, l_j)
            @constraint(model, y[l_j] <= fb_lj[l_j] )
        end
    end

    # Source to Terminal Flow Capacity
    for i_j in allowed_source_terminal
        if haskey(fb_ij, i_j)
            @constraint(model, z[i_j] <= fb_ij[i_j] )
        end
    end

    

    # Fractional Flow Pool Capacity
    for (i, l) in allowed_source_blender
            @constraint(
                model,
                sum(x[(i, l, j)] for (ii, ll, j) in allowed_ilj if ii == i && ll == l; init=0.0)
                <= (blender_capacity[l] * q[(i, l)]) 
            )

    end

    
    slack_coefficient_list = []
    for (i,l,j) in allowed_ilj
        if((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0))>0)
            push!(slack_coefficient_list,get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0))
        end
    end
    slack_coefficient = maximum(slack_coefficient_list)
    
    @expression(model, slack_penalty,
        1.0 * (
            sum(slack_coefficient*(s_bilinear_plus[(i,l,j)] + s_bilinear_minus[(i,l,j)]) for (i,l,j) in allowed_ilj)
        )
    )
    @expression(model, obj_expr,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * x[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0) + 
        slack_penalty
    )

   
    

    @variable(model, obj)

    @constraint(model, obj >= obj_expr)

    @objective(model, Min, obj)

    optimize!(model)
    
    println("Slack: " ,value.(slack_penalty))
    return value.(model[:obj])
end


function true_model_with_slacks_aggressive(instance:: Dict,
    fixed_q_values;
    model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer))

    if model === nothing
        model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer)
    end

    # Parameter Dictionaries
    sources = instance[:sources]
    blenders = instance[:blenders]
    terminals = instance[:terminals]

    supply_limit = instance[:supply_limit]
    blender_capacity = instance[:blender_capacity]
    demand = instance[:demand]

    properties = instance[:properties]
    prop_limits = instance[:property_limits]

    allowed_source_blender = instance[:allowed_source_blender]
    allowed_source_terminal = instance[:allowed_source_terminal]
    allowed_blender_terminal = instance[:allowed_blender_terminal]
    allowed_ilj = instance[:allowed_ilj]

    f_il = instance[:flow_cost_il]
    f_ij = instance[:flow_cost_ij]
    f_lj = instance[:flow_cost_lj]

    fb_ij = instance[:flow_bound_ij]
    fb_lj = instance[:flow_bound_lj]

    prop_keys = collect(keys(properties))

    # Variables and q is parameter
    q_dict_key = Dict()
    for (i, i_l) in enumerate(allowed_source_blender)
        q_dict_key[i_l] = i
    end

    @variable(model, q[i_l in allowed_source_blender] in Parameter(fixed_q_values[q_dict_key[i_l]]))
    
    @variable(model, y[l_j in allowed_blender_terminal] >= 0)
    @variable(model, z[i_j in allowed_source_terminal] >= 0)
    @variable(model, x[i_l_j in allowed_ilj] >= 0)

    # Slacks
    
    @variable(model, 1000>=s_bilinear_plus[i_l_j in allowed_ilj] >= 0)  
    @variable(model, 1000>=s_bilinear_minus[i_l_j in allowed_ilj] >= 0)  
         

    # Auxiliary Flow Variable Substitution
    for (i,l,j) in allowed_ilj
        if(get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)<=0)
            @constraint(model, x[(i,l,j)]  == q[(i,l)] * y[(l,j)])
            @constraint(model, s_bilinear_plus[(i,l,j)] + s_bilinear_minus[(i,l,j)]   == 0)
        else
            @constraint(model, x[(i,l,j)] + s_bilinear_plus[(i,l,j)] - s_bilinear_minus[(i,l,j)] == q[(i,l)] * y[(l,j)])
        end
    end

    # Fraction Flow Variable Summation equals 1
    

    # Property Constraints
    for j in terminals
        total_flow = @expression(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
            sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)
        )
        for (prop, prop_dict) in properties
            prop_expr = @expression(model,
                sum(prop_dict[i] * x[(i, l, j)] for (i, l, jt) in allowed_ilj if jt == j && haskey(prop_dict, i); init=0.0) +
                sum(prop_dict[i_j[1]] * z[i_j] for i_j in allowed_source_terminal if i_j[2] == j && haskey(prop_dict, i_j[1]); init=0.0)
            )
            if haskey(prop_limits["upper"], prop) && haskey(prop_limits["upper"][prop], j)
                @constraint(model, prop_expr <= prop_limits["upper"][prop][j] * total_flow + 1e-3)
            end
            if haskey(prop_limits["lower"], prop) && haskey(prop_limits["lower"][prop], j)
                @constraint(model, prop_expr >= prop_limits["lower"][prop][j] * total_flow - 1e-3)
            end
        end
    end


    # Availability Constraint
    for i in sources
        pool_part = sum(x[(i,l,j)] for (ii,l,j) in allowed_ilj if ii == i; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[1] == i; init=0.0)

        @constraint(model,
            pool_part + direct_part <= supply_limit[i] 
        )
    end

    # Pool Capacity
    for l in blenders
        @constraint(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[1] == l; init=0.0)
            <= blender_capacity[l] 
        )
    end

    # Demand Constraint
    for j in terminals
        pooled_part = sum(x[(i,l,j)] for (i,l,jj) in allowed_ilj if jj == j; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)

        @constraint(model,
            pooled_part + direct_part <= demand[j] 
        )
    end

    # Pool to Terminal Flow Capacity
    for l_j in allowed_blender_terminal
        if haskey(fb_lj, l_j)
            @constraint(model, y[l_j] <= fb_lj[l_j] )
        end
    end

    # Source to Terminal Flow Capacity
    for i_j in allowed_source_terminal
        if haskey(fb_ij, i_j)
            @constraint(model, z[i_j] <= fb_ij[i_j] )
        end
    end

    

    # Fractional Flow Pool Capacity
    for (i, l) in allowed_source_blender
            @constraint(
                model,
                sum(x[(i, l, j)] for (ii, ll, j) in allowed_ilj if ii == i && ll == l; init=0.0)
                <= (blender_capacity[l] * q[(i, l)]) 
            )

    end

    
    slack_coefficient_list = []
    for (i,l,j) in allowed_ilj
        if((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0))>0)
            push!(slack_coefficient_list,get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0))
        end
    end
    slack_coefficient = maximum(slack_coefficient_list)
    
    @expression(model, slack_penalty,
        1.0 * (
            sum(slack_coefficient*(s_bilinear_plus[(i,l,j)] + s_bilinear_minus[(i,l,j)]) for (i,l,j) in allowed_ilj)
        )
    )
    @expression(model, obj_expr,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * x[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0) + 
        slack_penalty
    )

   
    

    @variable(model, obj)

    @constraint(model, obj >= obj_expr)

    @objective(model, Min, obj)

    optimize!(model)
    
    println("Slack: " ,value.(slack_penalty))
    return value.(model[:obj])
end




function ChainRulesCore.rrule(::typeof(true_model_with_slacks_aggressive), instance::Dict, fixed_q_values)
    forward_time = time()
    opt = MOI.OptimizerWithAttributes(
        Gurobi.Optimizer,
        "QCPDual" => 1,
        "BarConvTol" => 1e-10,
        "BarQCPConvTol" => 1e-10,
        "FeasibilityTol" => 1e-9,
        "OptimalityTol" => 1e-9,
        "InfUnbdInfo" => 1,
        "NumericFocus" => 3
    ) 

    model = DiffOpt.nonlinear_diff_model(opt)

    output = true_model_with_slacks_aggressive(instance, fixed_q_values; model = model)

    allowed_source_blender = instance[:allowed_source_blender]

    q_var = model[:q]
    q_param_refs = [ParameterRef(q_var[i_l]) for i_l in allowed_source_blender]
    fwd_time_ref = time() - forward_time
    push!(fwd_times_hf, fwd_time_ref)
    function pullback(dobj)
        backw_time = time()
        obj = model[:obj]

        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), obj, dobj)

        DiffOpt.reverse_differentiate!(model)

        dq = MOI.get.(model, DiffOpt.ReverseConstraintSet(), q_param_refs)

        grad_dq = [g.value for g in dq]
        bwd_time_ref = time() - backw_time
        push!(bwd_times_hf, bwd_time_ref)

        return NoTangent(), NoTangent(), grad_dq
    end

    return output, pullback
end

function true_model_with_slacks_conservative(instance:: Dict,
    fixed_q_values;
    model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer))

    if model === nothing
        model = DiffOpt.quadratic_diff_model(Gurobi.Optimizer)
    end

    # Parameter Dictionaries
    sources = instance[:sources]
    blenders = instance[:blenders]
    terminals = instance[:terminals]

    supply_limit = instance[:supply_limit]
    blender_capacity = instance[:blender_capacity]
    demand = instance[:demand]

    properties = instance[:properties]
    prop_limits = instance[:property_limits]

    allowed_source_blender = instance[:allowed_source_blender]
    allowed_source_terminal = instance[:allowed_source_terminal]
    allowed_blender_terminal = instance[:allowed_blender_terminal]
    allowed_ilj = instance[:allowed_ilj]

    f_il = instance[:flow_cost_il]
    f_ij = instance[:flow_cost_ij]
    f_lj = instance[:flow_cost_lj]

    fb_ij = instance[:flow_bound_ij]
    fb_lj = instance[:flow_bound_lj]

    prop_keys = collect(keys(properties))

    # Variables and q is parameter
    q_dict_key = Dict()
    for (i, i_l) in enumerate(allowed_source_blender)
        q_dict_key[i_l] = i
    end

    @variable(model, q[i_l in allowed_source_blender] in Parameter(fixed_q_values[q_dict_key[i_l]]))
    @variable(model, y[l_j in allowed_blender_terminal] >= 0)
    @variable(model, z[i_j in allowed_source_terminal] >= 0)
    @variable(model, x[i_l_j in allowed_ilj] >= 0)

    
    @variable(model, 1000>=s_bilinear_plus[i_l_j in allowed_ilj] >= 0)  
    @variable(model, 1000>=s_bilinear_minus[i_l_j in allowed_ilj] >= 0)  
   
        

    # Auxiliary Flow Variable Substitution
    for (i,l,j) in allowed_ilj
        if(get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)<=0)
            @constraint(model, x[(i,l,j)]  == q[(i,l)] * y[(l,j)])
            @constraint(model, s_bilinear_plus[(i,l,j)] + s_bilinear_minus[(i,l,j)]   == 0)
        else
            @constraint(model, x[(i,l,j)] + s_bilinear_plus[(i,l,j)] - s_bilinear_minus[(i,l,j)] == q[(i,l)] * y[(l,j)])
        end
    end

    # Fraction Flow Variable Summation equals 1
    

    # Property Constraints
    for j in terminals
        total_flow = @expression(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
            sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)
        )
        for (prop, prop_dict) in properties
            prop_expr = @expression(model,
                sum(prop_dict[i] * x[(i, l, j)] for (i, l, jt) in allowed_ilj if jt == j && haskey(prop_dict, i); init=0.0) +
                sum(prop_dict[i_j[1]] * z[i_j] for i_j in allowed_source_terminal if i_j[2] == j && haskey(prop_dict, i_j[1]); init=0.0)
            )
            if haskey(prop_limits["upper"], prop) && haskey(prop_limits["upper"][prop], j)
                @constraint(model, prop_expr <= prop_limits["upper"][prop][j] * total_flow + 1e-3)
            end
            if haskey(prop_limits["lower"], prop) && haskey(prop_limits["lower"][prop], j)
                @constraint(model, prop_expr >= prop_limits["lower"][prop][j] * total_flow - 1e-3)
            end
        end
    end


    # Availability Constraint
    for i in sources
        pool_part = sum(x[(i,l,j)] for (ii,l,j) in allowed_ilj if ii == i; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[1] == i; init=0.0)

        @constraint(model,
            pool_part + direct_part <= supply_limit[i] 
        )
    end

    # Pool Capacity
    for l in blenders
        @constraint(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[1] == l; init=0.0)
            <= blender_capacity[l] 
        )
    end

    # Demand Constraint
    for j in terminals
        pooled_part = sum(x[(i,l,j)] for (i,l,jj) in allowed_ilj if jj == j; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)

        @constraint(model,
            pooled_part + direct_part <= demand[j] 
        )
    end

    # Pool to Terminal Flow Capacity
    for l_j in allowed_blender_terminal
        if haskey(fb_lj, l_j)
            @constraint(model, y[l_j] <= fb_lj[l_j] )
        end
    end

    # Source to Terminal Flow Capacity
    for i_j in allowed_source_terminal
        if haskey(fb_ij, i_j)
            @constraint(model, z[i_j] <= fb_ij[i_j] )
        end
    end

    

    # Fractional Flow Pool Capacity
    for (i, l) in allowed_source_blender
            @constraint(
                model,
                sum(x[(i, l, j)] for (ii, ll, j) in allowed_ilj if ii == i && ll == l; init=0.0)
                <= (blender_capacity[l] * q[(i, l)]) 
            )
    end


    @expression(model, slack_penalty,
        1.0 * (
            sum(max((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)),0.0)*(s_bilinear_plus[(i,l,j)] + s_bilinear_minus[(i,l,j)]) for (i,l,j) in allowed_ilj)
        ))
    @expression(model, obj_expr,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * x[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0) + 
        slack_penalty
    )

    

    @variable(model, obj)

    @constraint(model, obj >= obj_expr)

    @objective(model, Min, obj)
    optimize!(model)
    println("Slack: " ,value.(slack_penalty))
    return value.(model[:obj])
end



function ChainRulesCore.rrule(::typeof(true_model_with_slacks_conservative), instance::Dict, fixed_q_values)
    forward_time = time()
    opt = MOI.OptimizerWithAttributes(
        Gurobi.Optimizer,
        "QCPDual" => 1,
        "BarConvTol" => 1e-10,
        "BarQCPConvTol" => 1e-10,
        "FeasibilityTol" => 1e-9,
        "OptimalityTol" => 1e-9,
        "InfUnbdInfo" => 1,
        "NumericFocus" => 3
    ) 

    model = DiffOpt.nonlinear_diff_model(opt)

    output = true_model_with_slacks_conservative(instance, fixed_q_values; model = model)

    allowed_source_blender = instance[:allowed_source_blender]

    q_var = model[:q]
    q_param_refs = [ParameterRef(q_var[i_l]) for i_l in allowed_source_blender]
    fwd_time_ref = time() - forward_time
    push!(fwd_times_hf, fwd_time_ref)

    function pullback(dobj)
        backw_time = time()
        obj = model[:obj]

        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), obj, dobj)

        DiffOpt.reverse_differentiate!(model)

        dq = MOI.get.(model, DiffOpt.ReverseConstraintSet(), q_param_refs)

        grad_dq = [g.value for g in dq]
        bwd_time_ref = time() - backw_time
        push!(bwd_times_hf, bwd_time_ref)
        return NoTangent(), NoTangent(), grad_dq
    end

    return output, pullback
end