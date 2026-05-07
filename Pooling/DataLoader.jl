using Flux
using GraphNeuralNetworks
using JSON
using Statistics

using Graphs
# ============================================================
# Normalization helpers (unchanged)
# ============================================================
function zscore_rows(A::AbstractMatrix{<:Real}; eps=1f-8)
    Af = Float32.(A)
    μ = mean(Af, dims=2)
    σ = std(Af, dims=2)
    σ = σ .+ eps
    return (Af .- μ) ./ σ
end

function zscore_selected_rows(A::AbstractMatrix{<:Real}, rows::Vector{Int}; eps=1f-8)
    Af = Float32.(A)
    out = copy(Af)
    μ = mean(Af[rows, :], dims=2)
    σ = std(Af[rows, :], dims=2) .+ eps
    out[rows, :] .= (Af[rows, :] .- μ) ./ σ
    return out
end

# ============================================================
# Fixed normalizer: fit once, apply everywhere
# ============================================================
struct TripartiteNormalizer
    μ_src::Matrix{Float32}  # size (F,1) where F=4
    σ_src::Matrix{Float32}
    μ_bl::Matrix{Float32}
    σ_bl::Matrix{Float32}
    μ_trm::Matrix{Float32}
    σ_trm::Matrix{Float32}
    μ_e::Matrix{Float32}    # size (2,1) for edge rows [1,2]
    σ_e::Matrix{Float32}
end

# apply z-score with provided stats
zapply(A::AbstractMatrix{<:Real}, μ::AbstractMatrix, σ::AbstractMatrix) =
    (Float32.(A) .- μ) ./ (σ .+ 1f-8)

# fit stats for one group: μ,σ are (F,1) across columns
function fit_group_stats(A::AbstractMatrix{<:Real})
    Af = Float32.(A)
    μ = mean(Af, dims=2)
    σ = std(Af,  dims=2)
    return μ, σ
end

# Fit normalizer from already-built X and Eattr (recommended)
function fit_normalizer_from_features(X::AbstractMatrix, src_range, bl_range, trm_range,
                                      Eattr::AbstractMatrix)
    μs, σs = fit_group_stats(X[:, src_range])
    μb, σb = fit_group_stats(X[:, bl_range])
    μt, σt = fit_group_stats(X[:, trm_range])

    # only rows 1 and 2 of edge features (cost, bound)
    μe, σe = fit_group_stats(Eattr[[1,2], :])

    return TripartiteNormalizer(μs, σs, μb, σb, μt, σt, μe, σe)
end

# ============================================================
# Utilities: indexing + node features (unchanged)
# ============================================================
"""
Create a unified node index for tripartite sets:
sources -> 1:ns, blenders -> ns+1:ns+nl, terminals -> ns+nl+1:ns+nl+nj
Returns:
  node_id::Dict{Symbol,Int}, and ranges for each type.
"""
function build_node_index(instance::Dict)
    sources  = instance[:sources]
    blenders = instance[:blenders]
    terms    = instance[:terminals]

    ns, nl, nj = length(sources), length(blenders), length(terms)

    node_id = Dict{Symbol,Int}()

    # sources
    for (k,i) in enumerate(sources)
        node_id[i] = k
    end
    # blenders
    for (k,l) in enumerate(blenders)
        node_id[l] = ns + k
    end
    # terminals
    for (k,j) in enumerate(terms)
        node_id[j] = ns + nl + k
    end

    src_range = 1:ns
    bl_range  = (ns+1):(ns+nl)
    trm_range = (ns+nl+1):(ns+nl+nj)

    return node_id, src_range, bl_range, trm_range
end

"""
Node features:
- one-hot type (3 dims)
- a capacity/limit scalar (1 dim) depending on node type:
    source: supply_limit[i]
    blender: blender_capacity[l]
    terminal: demand[j]
Total node feature dim = 4
"""
function make_node_features(instance::Dict, node_id::Dict{Symbol,Int}, src_range, bl_range, trm_range)
    N = length(node_id)
    X = zeros(Float32, 4, N)

    # type one-hot: [is_source, is_blender, is_terminal]
    for v in src_range
        X[1, v] = 1.0
    end
    for v in bl_range
        X[2, v] = 1.0
    end
    for v in trm_range
        X[3, v] = 1.0
    end

    # limit scalar
    supply_limit     = instance[:supply_limit]
    blender_capacity = instance[:blender_capacity]
    demand           = instance[:demand]

    # We need reverse lookup by symbol for these ranges; easiest: iterate symbols
    for (sym, v) in node_id
        if v in src_range
            X[4, v] = Float32(get(supply_limit, sym, 0.0))
        elseif v in bl_range
            X[4, v] = Float32(get(blender_capacity, sym, 0.0))
        else
            X[4, v] = Float32(get(demand, sym, 0.0))
        end
    end

    return X
end

# ============================================================
# Build tripartite graph (unchanged), but ALSO return per-relation graphs
# ============================================================
"""
Edge types:
1 = I-L
2 = L-J
3 = I-J

Edge features:
- cost (1)
- bound (1)
- edge-type one-hot (3)
Total edge feature dim = 5
"""
function build_tripartite_graph(instance::Dict;
    normalizer::Union{Nothing,TripartiteNormalizer}=nothing,
    return_features::Bool=false)
    node_id, src_range, bl_range, trm_range = build_node_index(instance)
    X = make_node_features(instance, node_id, src_range, bl_range, trm_range)

    IL = instance[:allowed_source_blender]
    LJ = instance[:allowed_blender_terminal]
    IJ = instance[:allowed_source_terminal]

    f_il = instance[:flow_cost_il]
    f_lj = instance[:flow_cost_lj]
    f_ij = instance[:flow_cost_ij]

    fb_ij = get(instance, :flow_bound_ij, Dict{Tuple{Symbol,Symbol},Float64}())
    fb_lj = get(instance, :flow_bound_lj, Dict{Tuple{Symbol,Symbol},Float64}())

    srcs = Int[]
    dsts = Int[]
    etype = Int[]
    Efeat = Float32[]

    edge_id_il = Dict{Tuple{Symbol,Symbol},Int}()
    edge_id_lj = Dict{Tuple{Symbol,Symbol},Int}()
    edge_id_ij = Dict{Tuple{Symbol,Symbol},Int}()

    function push_edge(u::Int, v::Int, t::Int, cost::Float32, bound::Float32)
        onehot = (t == 1 ? (1.0f0,0.0f0,0.0f0) :
                  t == 2 ? (0.0f0,1.0f0,0.0f0) :
                           (0.0f0,0.0f0,1.0f0))
        feats = (cost, bound, onehot...)
        push!(srcs, u); push!(dsts, v); push!(etype, t)
        append!(Efeat, feats)
        return length(srcs)
    end

    # I-L
    for (i,l) in IL
        u, v = node_id[i], node_id[l]
        c = Float32(get(f_il, (i,l), 0.0))
        b = 0.0f0
        e_fwd = push_edge(u, v, 1, c, b)
        push_edge(v, u, 1, c, b)  # reverse
        edge_id_il[(i,l)] = e_fwd
    end

    # L-J
    for (l,j) in LJ
        u, v = node_id[l], node_id[j]
        c = Float32(get(f_lj, (l,j), 0.0))
        b = Float32(get(fb_lj, (l,j), 0.0))
        e_fwd = push_edge(u, v, 2, c, b)
        push_edge(v, u, 2, c, b)
        edge_id_lj[(l,j)] = e_fwd
    end

    # I-J
    for (i,j) in IJ
        u, v = node_id[i], node_id[j]
        c = Float32(get(f_ij, (i,j), 0.0))
        b = Float32(get(fb_ij, (i,j), 0.0))
        e_fwd = push_edge(u, v, 3, c, b)
        push_edge(v, u, 3, c, b)
        edge_id_ij[(i,j)] = e_fwd
    end

    E = length(srcs)
    Eattr = reshape(Efeat, 5, E)


    # -----------------------------
    # NORMALIZATION (fixed or per-instance)
    # -----------------------------
    Xn = Float32.(X)

    if normalizer === nothing
        # OLD behavior (instance-wise)
        Xn[:, src_range] .= zscore_rows(Xn[:, src_range])
        Xn[:, bl_range]  .= zscore_rows(Xn[:, bl_range])
        Xn[:, trm_range] .= zscore_rows(Xn[:, trm_range])
        En = zscore_selected_rows(Eattr, [1,2])
    else
        # NEW behavior (fixed stats from reference instance)
        Xn[:, src_range] .= zapply(Xn[:, src_range], normalizer.μ_src, normalizer.σ_src)
        Xn[:, bl_range]  .= zapply(Xn[:, bl_range],  normalizer.μ_bl,  normalizer.σ_bl)
        Xn[:, trm_range] .= zapply(Xn[:, trm_range], normalizer.μ_trm, normalizer.σ_trm)

        En = copy(Float32.(Eattr))
        En[[1,2], :] .= zapply(En[[1,2], :], normalizer.μ_e, normalizer.σ_e)
    end

    g = GNNGraph(srcs, dsts; ndata=(x=Xn,), edata=(e=En, etype=etype,))

    edge_maps = (edge_id_il=edge_id_il, edge_id_lj=edge_id_lj, edge_id_ij=edge_id_ij)
    ranges = (src_range, bl_range, trm_range)

    if return_features
        return g, node_id, ranges, edge_maps, X, Eattr
    else
        return g, node_id, ranges, edge_maps
    end
end

# ============================================================
# NEW: build per-edge-type subgraphs (this is the "hetero wiring")
# ============================================================
"""
Create three relation-specific graphs (same node set, different edge sets):
- gIL: only edges with etype==1
- gLJ: only edges with etype==2
- gIJ: only edges with etype==3

We keep node features identical; we filter edge_index and edge_attr.
"""
function split_relation_graphs(g::GNNGraph)
    src, dst = edge_index(g)
    Efeat = g.edata.e
    etype = g.edata.etype
    X = g.ndata.x
    #N = GraphNeuralNetworks.nv(g)
    N =  Graphs.nv(g)

    function make_rel_graph(t::Int)
        idx = findall(==(t), etype)
        src_t = src[idx]
        dst_t = dst[idx]
        E_t = Efeat[:, idx]
        # ndata x reused; each graph has same nodes 1..N
        #return GNNGraph(src_t, dst_t, N; ndata=(x=X,), edata=(e=E_t,))
        return GNNGraph(src_t, dst_t; num_nodes=N, ndata=(x=X,), edata=(e=E_t,))
    end

    return (gIL = make_rel_graph(1),
            gLJ = make_rel_graph(2),
            gIJ = make_rel_graph(3))
end
