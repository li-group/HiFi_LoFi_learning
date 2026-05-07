using Random
using JSON
using BSON
include("Utils.jl")
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
n_tuning = 25
base_instance_list = ["instance_1_1","instance_2_1","instance_3_1","instance_4_1","instance_5_1"]

for base_instance in base_instance_list
    for i in 1:n_tuning
        if(i==1)
            cp("instances_final/"*base_instance*".json", "instances_gen_vary_train/"*base_instance*"_$i.json"; force=true)
        else
            instance_vary = generate_variations_1("instances_final/"*base_instance*".json",1,0,1,0; frac=0.10,seed = i)
            path =  "instances_gen_vary_train/"*base_instance*"_$i.json"
            open(path, "w") do io
                JSON.print(io, instance_vary, 4)
            end
        end
    end
end
base_instance_list = ["instance_1_1","instance_2_1","instance_3_1","instance_4_1","instance_5_1"]
for base_instance in base_instance_list
    path = "instances_final/"*base_instance*".json"
    inst = parse_instance(path)
    true_q,obj_model,obj_bound = true_model_q(inst)
    print(obj_bound)
    println(obj_model)
    open("results/Base_samples_variations.csv", "a") do io
        write(io, "$base_instance,$obj_model,$obj_bound,\n")
    end
    BSON.@save "instances_gen_vary_train/"*base_instance*"_q.bson" true_q
    for i in 1:n_tuning
        path = "instances_gen_vary_train/"*base_instance*"_$i.json"
        inst = parse_instance(path)
        true_q,obj_model,obj_bound = true_model_q(inst)

        open("results/Base_samples_variations.csv", "a") do io
            write(io, "$base_instance,$i,$obj_model,$obj_bound,\n")
        end
        BSON.@save "instances_gen_vary_train/"*base_instance*"_$i"*"_q.bson" true_q
    end
end
