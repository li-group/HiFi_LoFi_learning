
using Flux
using GraphNeuralNetworks
using JSON
using Statistics
# ============================================================
# Heterogeneous version of your PoolingGNN
# - SAME input features (4-dim nodes, 5-dim edges)
# - SAME output format (no constraints; can be +/-)
# - SAME residual structure + embed_out + decoders
# - ONLY change: each relation has its OWN GATv2Conv weights per layer
# ============================================================
struct HeteroPoolingGNN
    embed_in::Dense

    # layer 1: separate weights per relation
    conv1_IL
    conv1_LJ
    conv1_IJ

    # layer 2
    conv2_IL
    conv2_LJ
    conv2_IJ

    # layer 3
    conv3_IL
    conv3_LJ
    conv3_IJ

    # residual projections (keep exactly as your original dims)
    res1::Dense
    res2::Dense
    res3::Dense

    embed_out::Dense

    # decoders (unchanged)
    dec_eta::Chain
    dec_beta::Chain
    dec_sigma::Chain
    dec_alpha::Chain
end

Flux.@functor HeteroPoolingGNN

function HeteroPoolingGNN(; hdim=64, edge_dim=5, heads=4)
    embed_in = Dense(4, hdim)

    # ---- conv layer 1 (output = heads*hdim) ----
    conv1_IL = GATv2Conv((hdim, edge_dim) => hdim; add_self_loops=false, bias=false, heads=heads)
    conv1_LJ = GATv2Conv((hdim, edge_dim) => hdim; add_self_loops=false, bias=false, heads=heads)
    conv1_IJ = GATv2Conv((hdim, edge_dim) => hdim; add_self_loops=false, bias=false, heads=heads)

    # ---- conv layer 2 (output = heads*(hdim/2)) ----
    conv2_IL = GATv2Conv((heads*hdim, edge_dim) => Int(hdim ÷ 2); add_self_loops=false, bias=false, heads=heads)
    conv2_LJ = GATv2Conv((heads*hdim, edge_dim) => Int(hdim ÷ 2); add_self_loops=false, bias=false, heads=heads)
    conv2_IJ = GATv2Conv((heads*hdim, edge_dim) => Int(hdim ÷ 2); add_self_loops=false, bias=false, heads=heads)

    # ---- conv layer 3 (output = heads*(hdim/4)) ----
    conv3_IL = GATv2Conv((heads*Int(hdim ÷ 2), edge_dim) => Int(hdim ÷ 4); add_self_loops=false, bias=false, heads=heads)
    conv3_LJ = GATv2Conv((heads*Int(hdim ÷ 2), edge_dim) => Int(hdim ÷ 4); add_self_loops=false, bias=false, heads=heads)
    conv3_IJ = GATv2Conv((heads*Int(hdim ÷ 2), edge_dim) => Int(hdim ÷ 4); add_self_loops=false, bias=false, heads=heads)

    d1 = heads * hdim
    d2 = heads * Int(hdim ÷ 2)
    d3 = heads * Int(hdim ÷ 4)

    # residuals unchanged (same dims as your original)
    res1 = Dense(hdim, d1, bias=false)
    res2 = Dense(d1, d2, bias=false)
    res3 = Dense(d2, d3, bias=false)

    embed_out = Dense(d3, hdim)

    # decoders unchanged (same as your final version: identity output)
    ed_in = 2hdim + edge_dim

    dec_eta = Chain(Dense(ed_in, 128, relu),
                    Dense(128, 1, identity))
    dec_beta = Chain(Dense(ed_in, 128, relu),
                     Dense(128, 1, identity))
    dec_sigma = Chain(Dense(ed_in, 128, relu),
                      Dense(128, 1, identity))

    dec_alpha = Chain(Dense(3hdim, 128, relu),
                      Dense(128, 1, identity))

    return HeteroPoolingGNN(embed_in,
                            conv1_IL, conv1_LJ, conv1_IJ,
                            conv2_IL, conv2_LJ, conv2_IJ,
                            conv3_IL, conv3_LJ, conv3_IJ,
                            res1, res2, res3,
                            embed_out,
                            dec_eta, dec_beta, dec_sigma, dec_alpha)
end

# ============================================================
# Node embeddings: SAME shape/output; hetero message passing inside
# ============================================================
"""
Forward pass: returns node embeddings H (hdim x N)
No softplus. No LayerNorm. Residuals only.
Hetero behavior:
- messages from IL/LJ/IJ each use different weights (per layer)
- we SUM the three relation outputs at each layer
"""
function node_embeddings(m::HeteroPoolingGNN, rel)

    # split the original graph into relation-specific graphs (same nodes)
    #rel = split_relation_graphs(g)
    gIL, gLJ, gIJ = rel.gIL, rel.gLJ, rel.gIJ
    X = gIL.ndata.x  # same as original

    E_IL = gIL.edata.e
    E_LJ = gLJ.edata.e
    E_IJ = gIJ.edata.e

    H0 = m.embed_in(X)  # (hdim, N)

    # ---- layer 1 ----
    Z1 = m.conv1_IL(gIL, H0, E_IL) .+
         m.conv1_LJ(gLJ, H0, E_LJ) .+
         m.conv1_IJ(gIJ, H0, E_IJ)                     # (d1, N)
    H1 = Z1 .+ m.res1(H0)

    # ---- layer 2 ----
    Z2 = m.conv2_IL(gIL, H1, E_IL) .+
         m.conv2_LJ(gLJ, H1, E_LJ) .+
         m.conv2_IJ(gIJ, H1, E_IJ)                     # (d2, N)
    H2 = Z2 .+ m.res2(H1)

    # ---- layer 3 ----
    Z3 = m.conv3_IL(gIL, H2, E_IL) .+
         m.conv3_LJ(gLJ, H2, E_LJ) .+
         m.conv3_IJ(gIJ, H2, E_IJ)                     # (d3, N)
    H3 = Z3 .+ m.res3(H2)

    H  = m.embed_out(H3)                                # (hdim, N)
    return H
end

# ============================================================
# Decoders: UNCHANGED from your original (keep same ordering)
# ============================================================
"""
Decode edge coefficients on the EXISTING edges in g.
Returns η, β, σ in the SAME ORDER as instance[:allowed_*] lists.
(Uses edge_maps.edge_id_* for forward edge indices.)
"""
function decode_edge_coeffs_ordered(m::HeteroPoolingGNN, instance::Dict, g::GNNGraph, H, edge_maps)
    src, dst = edge_index(g)
    Efeat = g.edata.e

    IL = instance[:allowed_source_blender]
    η = [
        m.dec_eta(vcat(H[:, src[edge_maps.edge_id_il[key]]],
                       H[:, dst[edge_maps.edge_id_il[key]]],
                       Efeat[:, edge_maps.edge_id_il[key]]))[1]
        for key in IL
    ]

    LJ = instance[:allowed_blender_terminal]
    β = [
        m.dec_beta(vcat(H[:, src[edge_maps.edge_id_lj[key]]],
                        H[:, dst[edge_maps.edge_id_lj[key]]],
                        Efeat[:, edge_maps.edge_id_lj[key]]))[1]
        for key in LJ
    ]

    IJ = instance[:allowed_source_terminal]
    σ = [
        m.dec_sigma(vcat(H[:, src[edge_maps.edge_id_ij[key]]],
                         H[:, dst[edge_maps.edge_id_ij[key]]],
                         Efeat[:, edge_maps.edge_id_ij[key]]))[1]
        for key in IJ
    ]

    return η, β, σ
end

"""
Decode α for feasible triplets in instance[:allowed_ilj]
Returns α_hat in the same order as allowed_ilj.
"""
function decode_alpha(m::HeteroPoolingGNN, instance::Dict, node_id::Dict{Symbol,Int}, H)
    triplets = instance[:allowed_ilj]  # (i,l,j)
    return [
        m.dec_alpha(vcat(H[:, node_id[i]], H[:, node_id[l]], H[:, node_id[j]]))[1]
        for (i,l,j) in triplets
    ]
end

