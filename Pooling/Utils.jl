using Random
using JSON
using JuMP, Ipopt, JSON, Gurobi
import MathOptInterface as MOI
using Dates
function true_model_with_slacks_logged(instance, q)
    HF_time_start = time()
    model = Model(Gurobi.Optimizer)

    # SAME solve used for loss
    loss = true_model_with_slacks_aggressive_LP(instance, q; model=model)
    HF_time  = time() - HF_time_start


    raw_slack_sum =
          sum(value.(model[:s_bilinear_plus])) +
          sum(value.(model[:s_bilinear_minus])) 
    slack_used = value.(model[:slack_penalty])
    #feasibilty_metric =  maximum([maximum(value.(model[:s_bilinear_plus])),maximum(value.(model[:s_bilinear_minus]))])
    feasibilty_metric = mean(collect(Iterators.flatten([
    value.(model[:s_bilinear_plus]),
    value.(model[:s_bilinear_minus])
])))
    all_slack = collect(Iterators.flatten([
    value.(model[:s_bilinear_plus]),
    value.(model[:s_bilinear_minus])
    ]))

    return loss, feasibilty_metric,raw_slack_sum,slack_used, HF_time,all_slack
    
end

function loss_fn_with_logging(m,instance,rel,g,edge_maps,node_id)
    GNN_time_start = time()
    H = node_embeddings(m, rel)
    η, β, σ = decode_edge_coeffs_ordered(m, instance, g, H, edge_maps)
    α = decode_alpha(m, instance, node_id, H)
    GNN_time = time()-GNN_time_start
    LF_time_start = time()
    q̂ = tunable_layer(instance, α, β, σ, η)
    LF_time = time()-LF_time_start
    loss, feasibilty_metric,raw_slack,slack_used,HF_time,all_slack = true_model_with_slacks_logged(instance, q̂)
    @show loss
    return loss, feasibilty_metric,raw_slack,slack_used,[GNN_time,LF_time,HF_time],all_slack
end

function loss_fn_conservative_selfsupervised(m,ref_data)
    H = node_embeddings(m, ref_data.rel)
    η, β, σ = decode_edge_coeffs_ordered(m, ref_data.instance, ref_data.g, H, ref_data.edge_maps)
    α = decode_alpha(m, ref_data.instance, ref_data.node_id, H)
    @show minimum(α), maximum(α)
    @show minimum(β), maximum(β)
    @show minimum(σ), maximum(σ)
    @show minimum(η), maximum(η)

    q̂ = tunable_layer(ref_data.instance, α, β, σ, η)
    #q̂1 = round_ste(q̂)
    loss = true_model_with_slacks_conservative(ref_data.instance, q̂)
    @show loss
    return loss
end


function loss_fn_aggressive_selfsupervised(m,ref_data)
    H = node_embeddings(m, ref_data.rel)
    η, β, σ = decode_edge_coeffs_ordered(m, ref_data.instance, ref_data.g, H, ref_data.edge_maps)
    α = decode_alpha(m, ref_data.instance, ref_data.node_id, H)
    @show minimum(α), maximum(α)
    @show minimum(β), maximum(β)
    @show minimum(σ), maximum(σ)
    @show minimum(η), maximum(η)

    q̂ = tunable_layer(ref_data.instance, α, β, σ, η)
    #q̂1 = round_ste(q̂)
    loss = true_model_with_slacks_aggressive(ref_data.instance, q̂)
    @show loss
    return loss
end


function loss_fn_batch_selfsupervised(m, batch)
    total_loss = 0.0
    
    for b in batch
        # These operations involve the model 'm' and are differentiable 
        H = node_embeddings(m, b.rel)
        η, β, σ = decode_edge_coeffs_ordered(m, b.instance, b.g, H, b.edge_maps)
        α = decode_alpha(m, b.instance, b.node_id, H)

        # The optimization layer [cite: 2, 23, 26]
        q̂ = tunable_layer(b.instance, α, β, σ, η)
        
        #q̂1 = round_ste(q̂)
    
        # The true model loss [cite: 27, 50, 52]
        total_loss += true_model_with_slacks_aggressive(b.instance, q̂)
    end
    
    return total_loss / length(batch)
end


function loss_fn_semisupervised(m,ref_data)
    H = node_embeddings(m, ref_data.rel)
    η, β, σ = decode_edge_coeffs_ordered(m, ref_data.instance, ref_data.g, H, ref_data.edge_maps)
    α = decode_alpha(m, ref_data.instance, ref_data.node_id, H)
    @show minimum(α), maximum(α)
    @show minimum(β), maximum(β)
    @show minimum(σ), maximum(σ)
    @show minimum(η), maximum(η)

    q̂ = tunable_layer(ref_data.instance, α, β, σ, η)
    #q̂1 = round_ste(q̂)
    loss = mean((q̂.-ref_data.true_q).^2)
    @show loss
    return loss
end


function loss_fn_batch_semisupervised(m, batch)
    total_loss = 0.0
    
    for b in batch
        # These operations involve the model 'm' and are differentiable 
        H = node_embeddings(m, b.rel)
        η, β, σ = decode_edge_coeffs_ordered(m, b.instance, b.g, H, b.edge_maps)
        α = decode_alpha(m, b.instance, b.node_id, H)

        # The optimization layer [cite: 2, 23, 26]
        q̂ = tunable_layer(b.instance, α, β, σ, η)
        
        #q̂1 = round_ste(q̂)
    
        # The true model loss [cite: 27, 50, 52]
        total_loss +=  mean((q̂.-b.true_q).^2)
    end
    
    return total_loss / length(batch)
end


function loss_fn_warm_start(m,ref_data)
    H = node_embeddings(m, ref_data.rel)
    η, β, σ = decode_edge_coeffs_ordered(m, ref_data.instance, ref_data.g, H, ref_data.edge_maps)
    α = decode_alpha(m, ref_data.instance, ref_data.node_id, H)
    loss = loss = mean(vcat(
    (α .- 1).^2,
    (σ .- 1).^2,
    (η .- 1).^2,
    (β .- 1).^2
    ))
    @show loss
    return loss
end




function parse_instance(filepath::String)
    data = JSON.parsefile(filepath)

    sources = Symbol.(unique([c["name"] for c in data["components"]]))
    blenders = Symbol.(keys(data["pool_size"]))
    terminals = Symbol.(unique([p["name"] for p in data["products"]]))

    supply_limit = Dict(Symbol(c["name"]) => c["upper"] for c in data["components"])
    blender_capacity = Dict(Symbol(pool) => cap for (pool, cap) in data["pool_size"])
    demand = Dict(Symbol(p["name"]) => p["upper"] for p in data["products"])

    flow_cost_il = Dict((Symbol(e["component"]), Symbol(e["pool"])) => e["cost"] for e in data["component_to_pool_fraction"])
    flow_cost_ij = Dict((Symbol(e["component"]), Symbol(e["product"])) => e["cost"] for e in data["component_to_product_bound"])
    flow_cost_lj = Dict((Symbol(e["pool"]), Symbol(e["product"])) => e["cost"] for e in data["pool_to_product_bound"])

    flow_bound_ij = Dict((Symbol(b["component"]), Symbol(b["product"])) => b["bound"] for b in data["component_to_product_bound"])
    flow_bound_lj = Dict((Symbol(b["pool"]), Symbol(b["product"])) => b["bound"] for b in data["pool_to_product_bound"])

    properties = Dict{String, Dict{Symbol, Float64}}()
    for c in data["components"]
        for (prop, val) in c["quality"]
            if !haskey(properties, prop)
                properties[prop] = Dict{Symbol, Float64}()
            end
            properties[prop][Symbol(c["name"])] = val
        end
    end

    prop_limits = Dict("upper" => Dict{String, Dict{Symbol, Float64}}(), "lower" => Dict{String, Dict{Symbol, Float64}}())
    for p in data["products"]
        j = Symbol(p["name"])
        if haskey(p, "quality_upper") && p["quality_upper"] !== nothing
            for (prop, val) in p["quality_upper"]
                prop_limits["upper"][prop] = get(prop_limits["upper"], prop, Dict{Symbol, Float64}())
                prop_limits["upper"][prop][j] = val
            end
        end
        if haskey(p, "quality_lower") && p["quality_lower"] !== nothing
            for (prop, val) in p["quality_lower"]
                prop_limits["lower"][prop] = get(prop_limits["lower"], prop, Dict{Symbol, Float64}())
                prop_limits["lower"][prop][j] = val
            end
        end
    end

    allowed_source_blender = [(Symbol(e["component"]), Symbol(e["pool"])) for e in data["component_to_pool_fraction"]]
    allowed_source_terminal = [(Symbol(e["component"]), Symbol(e["product"])) for e in data["component_to_product_bound"]]
    allowed_blender_terminal = [(Symbol(e["pool"]), Symbol(e["product"])) for e in data["pool_to_product_bound"]]

    allowed_ilj = [(i, l, j) for (i, l) in allowed_source_blender for (lp, j) in allowed_blender_terminal if lp == l]

    return Dict(
        :sources => sources,
        :blenders => blenders,
        :terminals => terminals,
        :supply_limit => supply_limit,
        :blender_capacity => blender_capacity,
        :demand => demand,
        :properties => properties,
        :property_limits => prop_limits,
        :allowed_source_blender => allowed_source_blender,
        :allowed_source_terminal => allowed_source_terminal,
        :allowed_blender_terminal => allowed_blender_terminal,
        :allowed_ilj => allowed_ilj,
        :flow_cost_il => flow_cost_il,
        :flow_cost_ij => flow_cost_ij,
        :flow_cost_lj => flow_cost_lj,
        :flow_bound_ij => flow_bound_ij,
        :flow_bound_lj => flow_bound_lj
    )
end



# --- helpers ---
deepcopy_json(x) = JSON.parse(JSON.json(x))  # robust deep copy for Dict/Vector/primitive JSON-like structures

# perturb a number by ±frac (multiplicative), optionally rounding
function perturb_val(rng::AbstractRNG, x::Real; frac::Real=0.20, digits::Int=2, minval::Real=0.0)
    s = 1.0 + rand(rng, (-frac):0.0001:(frac))  # fine grid so it's "continuous enough"
    y = float(x) * s
    y = max(y, minval)
    return round(y; digits=digits)
end

# build name->upper dicts for components/products, and pool->cap
function build_lookup_tables(inst::Dict)
    comp_upper = Dict{String,Float64}()
    for c in inst["components"]
        comp_upper[c["name"]] = float(c["upper"])
    end

    prod_upper = Dict{String,Float64}()
    for p in inst["products"]
        prod_upper[p["name"]] = float(p["upper"])
    end

    pool_cap = Dict{String,Float64}()
    for (k,v) in inst["pool_size"]
        pool_cap[string(k)] = float(v)
    end

    return comp_upper, prod_upper, pool_cap
end

# after perturbing supply/demand/poolcap, update bounds that depend on them
function repair_bounds!(inst::Dict)
    comp_upper, prod_upper, pool_cap = build_lookup_tables(inst)

    # component_to_product_bound bound <= min(supply_i, demand_j)
    for e in inst["component_to_product_bound"]
        i = e["component"]
        j = e["product"]
        newb = min(comp_upper[i], prod_upper[j])
        e["bound"] = round(newb; digits=2)
    end

    # pool_to_product_bound bound <= min(poolcap_l, demand_j)
    for e in inst["pool_to_product_bound"]
        l = e["pool"]
        j = e["product"]
        newb = min(pool_cap[l], prod_upper[j])
        e["bound"] = round(newb; digits=2)
    end

    return inst
end

"""
    perturb_instance(base_inst; frac=0.20, seed=1, digits=2, min_supply=0.0, min_demand=0.0, min_poolcap=0.0)

Create one perturbed instance in the SAME JSON Dict format.
Topology + qualities/quality bounds are kept identical.
"""
function perturb_instance(base_inst::Dict,perturb_supplies,perturb_capacities,perturb_demands,perturb_costs;
    frac::Real=0.20,
    seed::Int=1,
    digits::Int=2,
    min_supply::Real=0.0,
    min_demand::Real=0.0,
    min_poolcap::Real=0.0
)
    rng = MersenneTwister(seed)
    inst = deepcopy_json(base_inst)

    # 1) perturb supplies (components[*]["upper"])\
    if(perturb_supplies == 1)
        for c in inst["components"]
            c["upper"] = perturb_val(rng, c["upper"]; frac=frac, digits=digits, minval=min_supply)
            # KEEP c["quality"] unchanged
        end
    end

    if(perturb_capacities == 1)

        # 2) perturb pool capacities (pool_size)
        for (k, v) in collect(inst["pool_size"])  # collect to avoid mutation-during-iterate issues
            inst["pool_size"][k] = perturb_val(rng, v; frac=frac, digits=digits, minval=min_poolcap)
        end
    end
    # 3) perturb demands (products[*]["upper"])
    if(perturb_demands == 1)
        for p in inst["products"]
            p["upper"] = perturb_val(rng, p["upper"]; frac=frac, digits=digits, minval=min_demand)
            # KEEP p["quality_upper"], p["quality_lower"] unchanged
        end
    end

    # 4) perturb arc costs (keep topology exactly; only edit "cost")
    if(perturb_costs == 1)
        for e in inst["component_to_pool_fraction"]
            e["cost"] = perturb_val(rng, e["cost"]; frac=frac, digits=digits, minval=-Inf)
        end
        for e in inst["component_to_product_bound"]
            e["cost"] = perturb_val(rng, e["cost"]; frac=frac, digits=digits, minval=-Inf)
        end
        for e in inst["pool_to_product_bound"]
            e["cost"] = perturb_val(rng, e["cost"]; frac=frac, digits=digits, minval=-Inf)
        end

    end

    # 5) repair bounds so they remain consistent with updated supply/demand/poolcap
    repair_bounds!(inst)

    return inst
end

"""
    generate_variations(base_inst, n; frac=0.20, seed=1)

Return Vector{Dict} of n perturbed instances.
"""
function generate_variations(base_inst::Dict, n1,n2,perturb_supplies,perturb_capacities,perturn_demands,perturb_costs; frac::Real=0.20, seed::Int=1, digits::Int=2)
    out = Vector{Dict}(undef, n2-n1+2)
    for k in n1:n2
        out[k - n1 + 1] = perturb_instance(base_inst,perturb_supplies,perturb_capacities,perturn_demands,perturb_costs; frac=frac, seed=seed + k, digits=digits)
    end
    return out
end

"""
    save_variations(base_path, out_dir, n; frac=0.20, seed=1, prefix="instance_var")

Load one base instance JSON from base_path and write n perturbed instances to out_dir.
"""
function save_variations(base_path::String, out_dir::String, n1::Int,n2::Int,perturb_supplies,perturb_capacities,perturn_demands,perturb_costs;
    frac::Real=0.20, seed::Int=1, prefix::String="instance_var", digits::Int=2
)
    mkpath(out_dir)
    base_inst = JSON.parsefile(base_path)
    vars = generate_variations(base_inst, n1,n2,perturb_supplies,perturb_capacities,perturn_demands,perturb_costs; frac=frac, seed=seed, digits=digits)

    for k in n1:n2
        path = joinpath(out_dir, "$(prefix)_$(k).json")
        open(path, "w") do io
            JSON.print(io, vars[k - n1 + 1], 4)
        end
    end
    return nothing
end


function generate_variations_1(base_path, perturb_supplies,perturb_capacities,perturn_demands,perturb_costs; frac::Real=0.20,seed::Int=1, digits::Int=2)
    base_inst = JSON.parsefile(base_path)
    out = perturb_instance(base_inst,perturb_supplies,perturb_capacities,perturn_demands,perturb_costs; frac=frac,seed = seed,digits=digits)
    return out
end


function true_model_q(instance::Dict)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "TimeLimit", 500.0) 
    #set_optimizer_attribute(model, "TimeLimit", 2000.0) 
    set_optimizer_attribute(model, "MIPGap", 0.005)  

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

    # 1i
    @variable(model, 0 <= q[i_l in allowed_source_blender] <= 1)
    @variable(model, y[l_j in allowed_blender_terminal] >= 0)
    @variable(model, z[i_j in allowed_source_terminal] >= 0)
    @variable(model, v[i_lj in allowed_ilj] >= 0)

    #1a
    @expression(model, flow_cost,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * v[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0)
    )
    @objective(model, Min, flow_cost)

    #1f 
    for i in sources
        pool_part = sum(v[(i, l, j)] for (ii, l, j) in allowed_ilj if ii == i; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[1] == i; init=0.0)
        @constraint(model, pool_part + direct_part <= supply_limit[i])
    end

    # 1f
    for l in blenders
        @constraint(model, sum(y[l_j] for l_j in allowed_blender_terminal if l_j[1] == l; init=0.0) <= blender_capacity[l])
    end

    # # 1g
    # for j in terminals
    #     @constraint(model,
    #         sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
    #         sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0) <= demand[j]
    #     )
    # end

    # 1g
    for j in terminals
        pooled_part = sum(
            v[(i, l, j)] for (i, l, jj) in allowed_ilj if jj == j; init=0.0
        )
        direct_part = sum(
            z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0
        )
        @constraint(model, pooled_part + direct_part <= demand[j])
    end

    # 1d, 1e
    for j in terminals
        total_flow = @expression(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
            sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)
        )
        for (prop, prop_dict) in properties
            prop_expr = @expression(model,
                sum(prop_dict[i] * v[(i, l, j)] for (i, l, jt) in allowed_ilj if jt == j && haskey(prop_dict, i); init=0.0) +
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

    #1j
    for l_j in allowed_blender_terminal
        l, j = l_j
        @constraint(model, sum(v[(i, l, j)] for (i, ll, jj) in allowed_ilj if ll == l && jj == j; init=0.0) == y[l_j])
    end

    #1c
    for l in blenders
        @constraint(model, sum(q[i_l] for i_l in allowed_source_blender if i_l[2] == l; init=0.0) == 1)
    end

    # v=q*y (1b)
    for (i, l, j) in allowed_ilj
        @constraint(model, v[(i, l, j)] == q[(i, l)] * y[(l, j)])
    end

    # 1h
    for l_j in allowed_blender_terminal
        if haskey(fb_lj, l_j)
            @constraint(model, y[l_j] <= fb_lj[l_j])
        end
    end

    # 1h
    for i_j in allowed_source_terminal
        if haskey(fb_ij, i_j)
            @constraint(model, z[i_j] <= fb_ij[i_j])
        end
    end

    # 1k
    for (i, l) in allowed_source_blender
        @constraint(
            model,
            sum(v[(i, l, j)] for (ii, ll, j) in allowed_ilj if ii == i && ll == l; init=0.0)
            <= blender_capacity[l] * q[(i, l)]
        )
    end

    optimize!(model)
    
    q_true= []
    for (i, i_l) in enumerate(allowed_source_blender)
        push!(q_true,value(q[i_l]))
    end

    return q_true,objective_value(model), objective_bound(model)
    
end


function true_model(instance::Dict)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "TimeLimit", 1000.0) 
    #set_optimizer_attribute(model, "TimeLimit", 500.0) 
    set_optimizer_attribute(model, "MIPGap", 0.005)  

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

    # 1i
    @variable(model, 0 <= q[i_l in allowed_source_blender] <= 1)
    @variable(model, y[l_j in allowed_blender_terminal] >= 0)
    @variable(model, z[i_j in allowed_source_terminal] >= 0)
    @variable(model, v[i_lj in allowed_ilj] >= 0)

    #1a
    @expression(model, flow_cost,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * v[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0)
    )
    @objective(model, Min, flow_cost)

    #1f 
    for i in sources
        pool_part = sum(v[(i, l, j)] for (ii, l, j) in allowed_ilj if ii == i; init=0.0)
        direct_part = sum(z[i_j] for i_j in allowed_source_terminal if i_j[1] == i; init=0.0)
        @constraint(model, pool_part + direct_part <= supply_limit[i])
    end

    # 1f
    for l in blenders
        @constraint(model, sum(y[l_j] for l_j in allowed_blender_terminal if l_j[1] == l; init=0.0) <= blender_capacity[l])
    end

    # # 1g
    # for j in terminals
    #     @constraint(model,
    #         sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
    #         sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0) <= demand[j]
    #     )
    # end

    # 1g
    for j in terminals
        pooled_part = sum(
            v[(i, l, j)] for (i, l, jj) in allowed_ilj if jj == j; init=0.0
        )
        direct_part = sum(
            z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0
        )
        @constraint(model, pooled_part + direct_part <= demand[j])
    end

    # 1d, 1e
    for j in terminals
        total_flow = @expression(model,
            sum(y[l_j] for l_j in allowed_blender_terminal if l_j[2] == j; init=0.0) +
            sum(z[i_j] for i_j in allowed_source_terminal if i_j[2] == j; init=0.0)
        )
        for (prop, prop_dict) in properties
            prop_expr = @expression(model,
                sum(prop_dict[i] * v[(i, l, j)] for (i, l, jt) in allowed_ilj if jt == j && haskey(prop_dict, i); init=0.0) +
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

    #1j
    for l_j in allowed_blender_terminal
        l, j = l_j
        @constraint(model, sum(v[(i, l, j)] for (i, ll, jj) in allowed_ilj if ll == l && jj == j; init=0.0) == y[l_j])
    end

    #1c
    for l in blenders
        @constraint(model, sum(q[i_l] for i_l in allowed_source_blender if i_l[2] == l; init=0.0) == 1)
    end

    # v=q*y (1b)
    for (i, l, j) in allowed_ilj
        @constraint(model, v[(i, l, j)] == q[(i, l)] * y[(l, j)])
    end

    # 1h
    for l_j in allowed_blender_terminal
        if haskey(fb_lj, l_j)
            @constraint(model, y[l_j] <= fb_lj[l_j])
        end
    end

    # 1h
    for i_j in allowed_source_terminal
        if haskey(fb_ij, i_j)
            @constraint(model, z[i_j] <= fb_ij[i_j])
        end
    end

    # 1k
    for (i, l) in allowed_source_blender
        @constraint(
            model,
            sum(v[(i, l, j)] for (ii, ll, j) in allowed_ilj if ii == i && ll == l; init=0.0)
            <= blender_capacity[l] * q[(i, l)]
        )
    end

    optimize!(model)
    quadratic_cost = 1
    y_vec = []
    q_true_n = []
    for (i,i_l_j) in enumerate(allowed_ilj)
        push!(y_vec,2*quadratic_cost*value(v[i_l_j]))
    end
    for (i, l_j) in enumerate(allowed_blender_terminal)
        push!(y_vec,2*quadratic_cost*value(y[l_j]))
    end
    for (i, i_j) in enumerate(allowed_source_terminal)
        push!(y_vec,2*quadratic_cost*value(z[i_j]))
    end
    for (i, i_l) in enumerate(allowed_source_blender)
        push!(y_vec,2*quadratic_cost*value(q[i_l]))
        push!(q_true_n,value(q[i_l]))
    end

    return objective_bound(model)
    
end

function mccormick_model_bound(instance::Dict)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)

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

    # Objective
    @expression(model, flow_cost,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * x[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0)
    )
    
    @objective(model, Min, flow_cost)

    optimize!(model)
    q_mck = []
    for (i, i_l) in enumerate(allowed_source_blender)
        push!(q_mck,value(q[i_l]))
    end
    return objective_value(model)
    #return value.(model[:q])
end

function mccormick_model_bound_q(instance::Dict)
    LF_time_start = time()
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)

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

    # Objective
    @expression(model, flow_cost,
        -sum(f_ij[i_j] * z[i_j] for i_j in allowed_source_terminal; init=0.0) +
        -sum((get(f_il, (i, l), 0.0) + get(f_lj, (l, j), 0.0)) * x[(i, l, j)] for (i, l, j) in allowed_ilj; init=0.0)
    )
    
    @objective(model, Min, flow_cost)

    optimize!(model)
    q_mck = []
    for (i, i_l) in enumerate(allowed_source_blender)
        push!(q_mck,value(q[i_l]))
    end
    LF_time = time()-LF_time_start
    return objective_value(model),q_mck,LF_time
    #return value.(model[:q])
end