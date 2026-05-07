using Random,JSON

rand_uniform(rng, a, b) = round(rand(rng) * (b - a) + a,digits = 2)
# --- 1) fixed_topology returns ONLY the arc sets (topology) ---
function fixed_topology(ns::Int, nl::Int, nj::Int;
    rng::AbstractRNG = Random.default_rng(),
    p_il::Float64 = 0.35,
    p_lj::Float64 = 0.42,
    p_ij::Float64 = 0.04,
    min_pool_in_frac::Float64 = 0.40
)
    I = [Symbol("S$(k)") for k in 1:ns]
    L = [Symbol("L$(k)") for k in 1:nl]
    J = [Symbol("T$(k)") for k in 1:nj]

    il = Set{Tuple{Symbol,Symbol}}()
    for i in I, l in L
        rand(rng) < p_il && push!(il, (i,l))
    end

    # enforce each pool has >= ceil(min_pool_in_frac*|I|) incoming
    min_in = max(1, ceil(Int, min_pool_in_frac * ns))
    for l in L
        incoming = [i for (i,ll) in il if ll == l]
        if length(incoming) < min_in
            need = min_in - length(incoming)
            candidates = setdiff(I, incoming)
            shuffle!(rng, candidates)
            for k in 1:min(need, length(candidates))
                push!(il, (candidates[k], l))
            end
        end
    end

    lj = Set{Tuple{Symbol,Symbol}}()
    for l in L
        has_out = false
        for j in J
            if rand(rng) < p_lj
                push!(lj, (l,j)); has_out = true
            end
        end
        if !has_out
            push!(lj, (l, J[rand(rng, 1:nj)]))
        end
    end

    ij = Set{Tuple{Symbol,Symbol}}()
    for i in I, j in J
        rand(rng) < p_ij && push!(ij, (i,j))
    end

    return il, ij, lj
end

# --- 2) generate_instance now samples f_out, f_in each time, and assigns costs ---
function generate_instance(ns::Int, nl::Int, nj::Int, nk::Int,
    il::Set{Tuple{Symbol,Symbol}},
    ij::Set{Tuple{Symbol,Symbol}},
    lj::Set{Tuple{Symbol,Symbol}};
    rng::AbstractRNG = Random.default_rng()
)
    I = [Symbol("S$(k)") for k in 1:ns]
    L = [Symbol("L$(k)") for k in 1:nl]
    J = [Symbol("T$(k)") for k in 1:nj]
    props = ["P$(k)" for k in 1:nk]

    # --- build adjacency for reachability (needs topology) ---
    adj = Dict{Symbol,Vector{Symbol}}()
    for u in vcat(I, L, J)
        adj[u] = Symbol[]
    end
    for (i,l) in il; push!(adj[i], l); end
    for (l,j) in lj; push!(adj[l], j); end
    for (i,j) in ij; push!(adj[i], j); end

    function reachable_outputs(i::Symbol)
        seen = Set{Symbol}([i])
        q = [i]
        outs = Set{Symbol}()
        while !isempty(q)
            u = popfirst!(q)
            for v in adj[u]
                if !(v in seen)
                    push!(seen, v)
                    push!(q, v)
                end
                if startswith(String(v), "T")
                    push!(outs, v)
                end
            end
        end
        return collect(outs)
    end

    # --- sample f_out, f_in PER INSTANCE (Appendix B logic) ---
    f_out = Dict{Symbol,Float64}(j => rand(rng, 15:100) for j in J)

    f_in = Dict{Symbol,Float64}()
    for i in I
        Ji = reachable_outputs(i)
        if isempty(Ji)
            f_in[i] = rand(rng, 15:100)
            continue
        end
        Ji_sorted = sort(Ji, by=j -> f_out[j])
        m = floor(Int, 0.3 * length(Ji_sorted))

        Jminus = Symbol[]
        Jplus  = Symbol[]
        for (idx, j) in enumerate(Ji_sorted)
            if idx <= m && rand(rng) < 0.25
                push!(Jminus, j)
            else
                push!(Jplus, j)
            end
        end

        fmin = isempty(Jminus) ? minimum(f_out[j] for j in Ji_sorted) : maximum(f_out[j] for j in Jminus)
        cands = [f_out[j] for j in Jplus if f_out[j] > fmin]
        fmax  = isempty(cands) ? maximum(f_out[j] for j in Ji_sorted) : minimum(cands)
        fmax < fmin && (fmax = fmin)

        f_in[i] = rand(rng, Int(round(fmin)):Int(round(fmax)))
    end

    # --- capacities (still per instance) ---
    supply = Dict{Symbol,Float64}()
    small_set = rand(rng, 10:60)
    for i in I
        supply[i] = (rand(rng) < 0.15) ? small_set[rand(rng, 1:length(small_set))] : rand(rng, 60:180)
    end
    poolcap = Dict(l => rand(rng, 50:140) for l in L)
    demand  = Dict(j => rand(rng, 70:220) for j in J)

    # --- build edge dict arrays with PER-INSTANCE costs ---
    component_to_pool_fraction = [
        Dict("component" => string(i),
             "pool"      => string(l),
             "fraction"  => 1,
             "cost"      => -f_in[i])
        for (i,l) in collect(il)
    ]

    component_to_product_bound = [
        Dict("component" => string(i),
             "product"   => string(j),
             "cost"      => (-f_in[i] + f_out[j]),
             "bound"     => min(supply[i], demand[j]))
        for (i,j) in collect(ij)
    ]

    pool_to_product_bound = [
        Dict("pool"    => string(l),
             "product" => string(j),
             "cost"    => f_out[j],
             "bound"   => min(poolcap[l], demand[j]))
        for (l,j) in collect(lj)
    ]

    # --- qualities + property limits (your existing code can stay as-is) ---
    source_quality = Dict{String,Dict{Symbol,Float64}}()
    for p in props
        source_quality[p] = Dict(i => rand_uniform(rng, 8.0,75.0) for i in I)
    end

    property_limits = Dict(
        "upper" => Dict{String,Dict{Symbol,Float64}}(),
        "lower" => Dict{String,Dict{Symbol,Float64}}()
    )

    # reachable_sources computation (unchanged, but now uses the arrays above)
    reachable_sources = Dict{Symbol,Vector{Symbol}}()
    for j in J
        reachable_sources[j] = Symbol[]
        for (ii,jj) in [(Symbol(e["component"]), Symbol(e["product"])) for e in component_to_product_bound]
            jj == j && push!(reachable_sources[j], ii)
        end
        for (l,jj) in [(Symbol(e["pool"]), Symbol(e["product"])) for e in pool_to_product_bound]
            if jj == j
                for (ii,ll) in [(Symbol(e2["component"]), Symbol(e2["pool"])) for e2 in component_to_pool_fraction]
                    ll == l && push!(reachable_sources[j], ii)
                end
            end
        end
        reachable_sources[j] = unique(reachable_sources[j])
    end

    for p in props
        property_limits["upper"][p] = Dict{Symbol,Float64}()
        property_limits["lower"][p] = Dict{Symbol,Float64}()
        for j in J
            qs = [source_quality[p][i] for i in reachable_sources[j]]
            qmin, qmax = minimum(qs), maximum(qs)
            #a = rand(rng, qmin:0.5:(qmin + 0.4*(qmax - qmin)))
            #b = rand(rng, a:0.5:qmax)
            #=a = rand_uniform(rng, qmin, qmin + 0.5*(qmax - qmin))
            b = rand_uniform(rng, a, qmax)
            qs = [source_quality[p][i] for i in reachable_sources[j]]=#
            Δ = qmax - qmin
            a = rand_uniform(rng, qmin, qmin + 0.5*Δ)

            min_width = 0.17*Δ          # pick 0.25–0.45 depending on how strong you want McCormick
            b_lo = min(a + min_width, qmax)
            b = rand_uniform(rng, b_lo, qmax)

    

            property_limits["lower"][p][j] = round(a; digits=2)
            property_limits["upper"][p][j] = round(b; digits=2)

            end
        end

    components = [
        Dict("name"    => string(i),
             "price"   => 0.0,
             "upper"   => supply[i],
             "quality" => Dict(p => source_quality[p][i] for p in props))
        for i in I
    ]

    products = [
        Dict("name"          => string(j),
             "price"         => 0.0,
             "upper"         => demand[j],
             "quality_upper" => Dict(p => property_limits["upper"][p][j] for p in props),
             "quality_lower" => Dict(p => property_limits["lower"][p][j] for p in props))
        for j in J
    ]

    return Dict(
        "components" => components,
        "pool_size"  => Dict(string(l) => poolcap[l] for l in L),
        "products"   => products,
        "component_to_pool_fraction" => component_to_pool_fraction,
        "component_to_product_bound" => component_to_product_bound,
        "pool_to_product_bound"      => pool_to_product_bound
    )
end

# --- 3) save_instances: topology fixed once, costs vary each instance ---
function save_instances(n::Int, dir::String,type::Int; ns=40, nl=30, nj=50, nk=14, seed=1)
    mkpath(dir)
    rng_topo = MersenneTwister(seed)
    il, ij, lj = fixed_topology(ns, nl, nj; rng=rng_topo)  # FIXED topology

    for k in 1:n
        rng_inst = MersenneTwister(seed + k)               # different randomness per instance
        inst = generate_instance(ns, nl, nj, nk, il, ij, lj; rng=rng_inst)
        path = joinpath(dir, "instance_$type"*"_"*"$(k).json")
        open(path, "w") do io
            JSON.print(io, inst, 4)
        end
    end
end

save_instances(1, "instances_final",1;ns=25, nl=18, nj=25, nk=8)
save_instances(1, "instances_final",2;ns=25, nl=22, nj=30, nk=10)
save_instances(1, "instances_final",3;ns=30, nl=22, nj=35, nk=10)
save_instances(1, "instances_final",4;ns=40, nl=30, nj=45, nk=10)
save_instances(1, "instances_final",5;ns=40, nl=30, nj=50, nk=14)



