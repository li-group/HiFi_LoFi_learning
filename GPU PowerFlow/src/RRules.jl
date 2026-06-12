module RRules

using ChainRulesCore
using JuMP
using DiffOpt
using PowerModels
using Gurobi
using Ipopt
using GraphNeuralNetworks

import ..Layers: DCPMLayer, ACPMLayer, DCPMLayerBatch, ACPMLayerExa, ACPM_PROJECTION, fwd_times_dc, fwd_times_ac, bwd_times_dc, bwd_times_ac, grad_log_dc, silent_gurobi_optimizer


function ChainRulesCore.rrule(
    ::typeof(ACPMLayerExa),
    gw,
    Pg_probe::AbstractVector
)
    start_fwd = time()
    results = ACPMLayerExa(gw, Pg_probe)
    push!(fwd_times_ac, time() - start_fwd)
    obj = results["objective"]
    μP_all = results["Dual"]

    function pullback_ACPExa(_)
        start_bwd = time()
        dPg_probe = μP_all
        push!(bwd_times_ac, time() - start_bwd)
        return (NoTangent(), NoTangent(), -dPg_probe)
    end

    return obj, pullback_ACPExa
end

function ChainRulesCore.rrule(::typeof(DCPMLayer), gw, rhs::AbstractMatrix)
    model = DiffOpt.quadratic_diff_model(silent_gurobi_optimizer)
    start_fwd = time()
    pg_values = DCPMLayer(gw, rhs; model=model)
    push!(fwd_times_dc, time() - start_fwd)
    pg = model[:pg]

    function pullback_DCPM(Δobj)
        backw_time = time()
        balanceeqP = model[:BalanceeqP]
        # println("Reverse pass DCPMLayer: Δobj = ", Δobj)
        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), pg, Δobj)
        DiffOpt.reverse_differentiate!(model)

        # println("pass reverse")

        grad_β = JuMP.constant.(MOI.get.(model, DiffOpt.ReverseConstraintFunction(), balanceeqP))


        # println("yupi")
        bwd_time_ref = time() - backw_time
        push!(bwd_times_dc, bwd_time_ref)

        return (NoTangent(), NoTangent(), grad_β)
    end

    return pg_values, pullback_DCPM
end

function ChainRulesCore.rrule(
    ::typeof(ACPMLayer),
    gw,
    Pg_probe::AbstractVector{Float64},
    ac_init
)
    start_fwd = time()
    results = ACPMLayer(gw, Pg_probe, ac_init)
    push!(fwd_times_ac, time() - start_fwd)
    obj = results["objective"]
    μP_all = results["Dual"]

    function pullback_ACP(_)
        start_bwd = time()
        dPg_probe = μP_all
        backw_time = time() - start_bwd
        push!(bwd_times_ac, backw_time)
        return (NoTangent(), NoTangent(), dPg_probe, NoTangent())
    end

    return obj, pullback_ACP
end

function ChainRulesCore.rrule(::typeof(DCPMLayerBatch), data, rhs)
    model = direct_model(DiffOpt.diff_optimizer(silent_gurobi_optimizer))
    start_fwd = time()
    pg_values = DCPMLayerBatch(data, rhs; model=model)
    push!(fwd_times_dc, time() - start_fwd)

    function pullback_DCPM(Δobj)
        pg = model[:pg]
        balanceeqP = model[:BalanceeqP]
        # println(pg)
        backw_time = time()
        # println("type og Δobj: ", typeof(Δobj))
        # println("size of Δobj: ", size(Δobj))

        # println("type of pg: ", typeof(pg))
        # println("size of pg: ", size(pg))

        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), pg, Δobj)
        # DiffOpt.set_reverse_variable.(model, pg, Δobj)
        DiffOpt.reverse_differentiate!(model)
        # println("pass reverse")
        bwd_time_ref = time() - backw_time
        push!(bwd_times_dc, bwd_time_ref)

        dbeta = JuMP.constant.(MOI.get.(model, DiffOpt.ReverseConstraintFunction(), balanceeqP))
        return (NoTangent(), NoTangent(), dbeta)
    end

    return pg_values, pullback_DCPM
end

end
