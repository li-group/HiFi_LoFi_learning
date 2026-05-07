module Layers

export DCPMLayer, DCPMLayerTruncado, ACPMLayer, fwd_times_dc, fwd_times_ac, bwd_times_lf, bwd_times_hf, fwd_times_lf, fwd_times_hf, ACPM, DCPM, DCPMLayerBatch, feasibility, top_congested_lines,
    LRWP, LRWPLayer_GM, WPLayer_GM, WP, Feasibility_GM, LRWPLayer_GM_truncated, WPLayer_GM_wt_objslack, LRDWP, CRDWP

#export DCPMLayer, DCPMLayerTruncado, ACPMLayer_v5,



using JuMP
using PowerModels
using DiffOpt
using MadNLP
using GasModels
using Gurobi
using Ipopt
using ChainRulesCore
using MathOptInterface
using PolyhedralRelaxations
using InfrastructureModels
using ForwardDiff
using GasModels
using ..DataLoader: GraphWithRef
# using ..Utils: fwd_times_dc, bwd_times_dc, fwd_times_ac, bwd_times_ac

const fwd_times_dc = Float64[]
const fwd_times_ac = Float64[]
const bwd_times_lf = Float64[]
const bwd_times_hf = Float64[]
const fwd_times_lf = Float64[]
const fwd_times_hf = Float64[]

const MOI = MathOptInterface

###############################################################
# Function to generate data LF
###############################################################
function LRWP(
    data, delivery_samples;
    # data;
    model=Model(Gurobi.Optimizer)
)


    # println("delivery samples ", delivery_samples)

    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    delivery_ids = sort(parse.(Int, collect(keys(data["delivery"]))))

    # println("delivery ids ", delivery_ids)
    for (i, id) in enumerate(delivery_ids)
        data["delivery"][string(id)]["withdrawal_nominal"] = delivery_samples[i]
        # println("Delivery ID: ", id, " Withdrawal Nominal: ", data["delivery"][string(id)]["withdrawal_nominal"])
    end



    # println("delivery en LRWP",delivery_samples)

    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    compressors = sort(collect(keys(ref[:compressor])))
    junctions = sort(collect(keys(ref[:junction])))
    loss_resistors = sort(collect(keys(ref[:loss_resistor])))
    pipes = sort(collect(keys(ref[:pipe])))
    receipts = sort(collect(keys(ref[:receipt])))
    regulators = sort(collect(keys(ref[:regulator])))
    resistors = sort(collect(keys(ref[:resistor])))
    short_pipes = sort(collect(keys(ref[:short_pipe])))
    transfers = sort(collect(keys(ref[:transfer])))
    valves = sort(collect(keys(ref[:valve])))
    deliveries = sort(collect(keys(ref[:delivery])))
    storages = sort(collect(keys(ref[:storage])))

    # println(ref[:delivery][23])

    # ------------------------------------------------------------------ #
    # 2. Variables (todas guardadas en model[:...])
    # ------------------------------------------------------------------ #

    ##############################
    # 1. Conjuntos de índices
    ##############################

    # junctions = sort(collect(keys(ref[:junction])))
    # compressors = haskey(ref, :compressor) ? sort(collect(keys(ref[:compressor]))) : Int[]
    # loss_resistors = haskey(ref, :loss_resistor) ? sort(collect(keys(ref[:loss_resistor]))) : Int[]
    # pipes = haskey(ref, :pipe) ? sort(collect(keys(ref[:pipe]))) : Int[]
    # receipts = haskey(ref, :receipt) ? sort(collect(keys(ref[:receipt]))) : Int[]
    # regulators = haskey(ref, :regulator) ? sort(collect(keys(ref[:regulator]))) : Int[]
    # resistors = haskey(ref, :resistor) ? sort(collect(keys(ref[:resistor]))) : Int[]
    # short_pipes = haskey(ref, :short_pipe) ? sort(collect(keys(ref[:short_pipe]))) : Int[]
    # transfers = haskey(ref, :transfer) ? sort(collect(keys(ref[:transfer]))) : Int[]
    # valves = haskey(ref, :valve) ? sort(collect(keys(ref[:valve]))) : Int[]
    # storages = haskey(ref, :storage) ? sort(collect(keys(ref[:storage]))) : Int[]
    # deliveries = haskey(ref, :delivery) ? sort(collect(keys(ref[:delivery]))) : Int[]

    ##############################
    # 2. Variables de flujo
    ##############################

    #! Define variable y_compressor



    # Flujo en compresores
    model[:f_compressor] = @variable(model,
        [k in compressors],
        lower_bound = ref[:compressor][k]["flow_min"],
        upper_bound = ref[:compressor][k]["flow_max"],
        start = get(ref[:compressor][k], "f_start", 0.0),
        base_name = "f_comp",
    )

    # Flujo en resistores
    model[:f_resistor] = @variable(model,
        [k in resistors],
        lower_bound = ref[:resistor][k]["flow_min"],
        upper_bound = ref[:resistor][k]["flow_max"],
        start = get(ref[:resistor][k], "f_start", 1e-6),
        base_name = "f_res",
    )

    # Flujo en loss_resistors
    model[:f_loss_resistor] = @variable(model,
        [k in loss_resistors],
        lower_bound = ref[:loss_resistor][k]["flow_min"],
        upper_bound = ref[:loss_resistor][k]["flow_max"],
        start = get(ref[:loss_resistor][k], "f_start", 1e-6),
        base_name = "f_lres",
    )

    # Flujo en pipes
    model[:f_pipe] = @variable(model,
        [k in pipes],
        lower_bound = ref[:pipe][k]["flow_min"],
        upper_bound = ref[:pipe][k]["flow_max"],
        start = get(ref[:pipe][k], "f_start", 0.0),
        base_name = "f_pipe",
    )

    # Flujo en short_pipes
    model[:f_short_pipe] = @variable(model,
        [k in short_pipes],
        lower_bound = ref[:short_pipe][k]["flow_min"],
        upper_bound = ref[:short_pipe][k]["flow_max"],
        # GasModels usualmente no pone start aquí
        base_name = "f_spipe",
    )

    # Flujo en válvulas
    lb_valve = Dict(k => ref[:valve][k]["flow_min"] for k in valves)
    ub_valve = Dict(k => ref[:valve][k]["flow_max"] for k in valves)

    model[:f_valve] = @variable(model,
        [k in valves],
        lower_bound = lb_valve[k],
        upper_bound = ub_valve[k],
        base_name = "f_valve",
    )

    # Flujo en reguladores
    lb_reg = Dict(k => ref[:regulator][k]["flow_min"] for k in regulators)
    ub_reg = Dict(k => ref[:regulator][k]["flow_max"] for k in regulators)

    model[:f_regulator] = @variable(model,
        [k in regulators],
        lower_bound = lb_reg[k],
        upper_bound = ub_reg[k],
        base_name = "f_reg",
    )

    ##############################
    # 3. Inyección / retiro / transfer
    ##############################

    # Transferencias (mgc.transfer)
    model[:ft] = @variable(model,
        [k in transfers],
        lower_bound = ref[:transfer][k]["withdrawal_min"],
        upper_bound = ref[:transfer][k]["withdrawal_max"],
        start = get(ref[:transfer][k], "ft_start",
            ref[:transfer][k]["withdrawal_max"]),
        base_name = "ft",
    )

    # Receipts (suministros)
    model[:fg] = @variable(model,
        [k in receipts],
        lower_bound = ref[:receipt][k]["injection_min"],
        upper_bound = ref[:receipt][k]["injection_max"],
        start = get(ref[:receipt][k], "fg_start",
            ref[:receipt][k]["injection_max"]),
        base_name = "fg",
    )

    # Deliveries dispatchable (cargas)
    dispatchable_deliveries = [
        k for k in deliveries
        if ref[:delivery][k]["is_dispatchable"] == 1
    ]

    model[:fl] = @variable(model,
        [k in dispatchable_deliveries],
        lower_bound = ref[:delivery][k]["withdrawal_min"],
        upper_bound = ref[:delivery][k]["withdrawal_max"],
        start = get(ref[:delivery][k], "fl_start",
            ref[:delivery][k]["withdrawal_max"]),
        base_name = "fl",
    )

    ##############################
    # 4. Storage (si existe)
    ##############################

    if !isempty(storages)
        model[:well_head_flow] = @variable(model,
            [s in storages],
            lower_bound = ref[:storage][s]["injection_min"],
            upper_bound = ref[:storage][s]["injection_max"],
            start = get(ref[:storage][s], "flow_start", 0.0),
            base_name = "f_storage",
        )
    end

    ##############################
    # 5. Presiones
    ##############################

    # Presión al cuadrado en todas las junctions
    model[:psqr] = @variable(model,
        [i in junctions],
        lower_bound = ref[:junction][i]["p_min"]^2,
        upper_bound = ref[:junction][i]["p_max"]^2,
        start = get(ref[:junction][i], "p_start",
            ref[:junction][i]["p_min"])^2,
        base_name = "psqr",
    )

    # Presión no al cuadrado solo donde hay resistor / loss_resistor
    prs_junc = unique(vcat(
        [ref[:resistor][k]["fr_junction"] for k in resistors],
        [ref[:resistor][k]["to_junction"] for k in resistors],
        [ref[:loss_resistor][k]["fr_junction"] for k in loss_resistors],
        [ref[:loss_resistor][k]["to_junction"] for k in loss_resistors],
    )
    )

    model[:p_nonsq] = @variable(model,
        [i in prs_junc],
        lower_bound = ref[:junction][i]["p_min"],
        upper_bound = ref[:junction][i]["p_max"],
        start = get(ref[:junction][i], "p_start",
            ref[:junction][i]["p_max"]),
        base_name = "p_nsq",
    )

    ##############################
    # 6. Ratios de compresor (rsqr)
    ##############################

    model[:rsqr] = @variable(model,
        [k in compressors],
        lower_bound = ref[:compressor][k]["c_ratio_min"]^2,
        upper_bound = ref[:compressor][k]["c_ratio_max"]^2,
        start = get(ref[:compressor][k], "ratio_start", 1.0),
        base_name = "rsqr",
    )

    ##############################
    # 7. Variable específica LRWP: f |f|
    ##############################

    model[:fmf_l_pipe] = @variable(model,
        [k in pipes],
        base_name = "fmf_l",
    )

    ##############################
    # 8. Binarias (válvulas / reguladores)
    ##############################

    vstart_valve = Dict(k => get(ref[:valve][k], "v_start", 1) for k in valves)
    model[:v_valve] = @variable(model,
        [k in valves],
        Bin,
        start = vstart_valve[k],
        base_name = "v_valve",
    )

    vstart_reg = Dict(k => get(ref[:regulator][k], "v_start", 1) for k in regulators)
    model[:v_regulator] = @variable(model,
        [k in regulators],
        Bin,
        start = vstart_reg[k],
        base_name = "v_reg",
    )


    # constraint_mass_flow_balance
    for i in junctions
        # -----------------------------
        # RHS: constants only
        # -----------------------------
        const_inj = haskey(ref[:nondispatchable_receipts_in_junction], i) ?
                    sum(ref[:receipt][r]["injection_nominal"]
                        for r in ref[:nondispatchable_receipts_in_junction][i]; init=0.0) : 0.0

        const_with = haskey(ref[:nondispatchable_deliveries_in_junction], i) ?
                     sum(ref[:delivery][d]["withdrawal_nominal"]
                         for d in ref[:nondispatchable_deliveries_in_junction][i]; init=0.0) : 0.0

        const_trans = haskey(ref[:nondispatchable_transfers_in_junction], i) ?
                      sum(ref[:transfer][t]["withdrawal_nominal"]
                          for t in ref[:nondispatchable_transfers_in_junction][i]; init=0.0) : 0.0

        rhs = const_inj - const_with - const_trans


        # -----------------------------
        # Variable dispatch term (same as your lhs variable part)
        # D(i) = fg - fl - ft - well_head_flow
        # -----------------------------
        disp =
            sum(model[:fg][r] for r in get(ref[:dispatchable_receipts_in_junction], i, []); init=0.0) -
            sum(model[:fl][d] for d in get(ref[:dispatchable_deliveries_in_junction], i, []); init=0.0) -
            sum(model[:ft][t] for t in get(ref[:dispatchable_transfers_in_junction], i, []); init=0.0) -
            sum(model[:well_head_flow][s] for s in get(ref[:storages_in_junction], i, []); init=0.0)


        # -----------------------------
        # Variable network term (EXACTLY your original rhs expression)
        # N(i)
        # -----------------------------
        net =
            sum(model[:f_pipe][p] for p in get(ref[:pipes_fr], i, []); init=0.0) -
            sum(model[:f_pipe][p] for p in get(ref[:pipes_to], i, []); init=0.0) +
            sum(model[:f_compressor][c] for c in get(ref[:compressors_fr], i, []); init=0.0) -
            sum(model[:f_compressor][c] for c in get(ref[:compressors_to], i, []); init=0.0) +
            sum(model[:f_resistor][r] for r in get(ref[:resistors_fr], i, []); init=0.0) -
            sum(model[:f_resistor][r] for r in get(ref[:resistors_to], i, []); init=0.0) +
            sum(model[:f_loss_resistor][lr] for lr in get(ref[:loss_resistors_fr], i, []); init=0.0) -
            sum(model[:f_loss_resistor][lr] for lr in get(ref[:loss_resistors_to], i, []); init=0.0) +
            sum(model[:f_short_pipe][sp] for sp in get(ref[:short_pipes_fr], i, []); init=0.0) -
            sum(model[:f_short_pipe][sp] for sp in get(ref[:short_pipes_to], i, []); init=0.0) +
            sum(model[:f_valve][v] for v in get(ref[:valves_fr], i, []); init=0.0) -
            sum(model[:f_valve][v] for v in get(ref[:valves_to], i, []); init=0.0) +
            sum(model[:f_regulator][rg] for rg in get(ref[:regulators_fr], i, []); init=0.0) -
            sum(model[:f_regulator][rg] for rg in get(ref[:regulators_to], i, []); init=0.0)


        # -----------------------------
        # All variables on LHS, constants on RHS:
        # net - disp == rhs
        # This is EXACTLY equivalent to: rhs + disp == net
        # -----------------------------
        lhs = net - disp

        @constraint(model, lhs == rhs)
    end





    # constraint_pressure
    for (i, d) in ref[:junction]
        if get(d, "junction_type", 0) == 1 && haskey(d, "p_nominal")
            @constraint(model, model[:psqr][i] == d["p_nominal"]^2)
        end
    end



    # constraint_pipe_pressure
    for (k, pipe) in ref[:pipe]

        i = pipe["fr_junction"]
        j = pipe["to_junction"]

        i_junc = ref[:junction][i]
        j_junc = ref[:junction][j]

        # === 1. Compute pressure-drop bounds exactly like GasModels ===
        pd_min, pd_max = GasModels._calc_pipe_pd_bounds_sqr(pipe, i_junc, j_junc)

        # === 2. Add constraints exactly as GasModels does ===
        @constraint(model, pd_min <= model[:psqr][i] - model[:psqr][j])
        @constraint(model, model[:psqr][i] - model[:psqr][j] <= pd_max)
    end

    # constraint_pipe_mass_flow
    for (k, pipe) in ref[:pipe]
        f = model[:f_pipe][k]

        f_min = pipe["flow_min"]
        f_max = pipe["flow_max"]

        lb = JuMP.has_lower_bound(f) ? max(JuMP.lower_bound(f), f_min) : f_min
        ub = JuMP.has_upper_bound(f) ? min(JuMP.upper_bound(f), f_max) : f_max

        JuMP.set_lower_bound(f, lb)
        JuMP.set_upper_bound(f, ub)
    end


    # constraint_pipe_weymouth
    for k in pipes
        pipe = ref[:pipe][k]
        i = pipe["fr_junction"]
        j = pipe["to_junction"]
        θ = pipe["theta"]
        D = pipe["diameter"]

        # === 1. GasModels: skip pipes with diameter == 0 ===
        if D == 0.0
            continue  # GasModels does not add constraints here
        end

        # === 2. Horizontal pipe? (|θ| ≤ 5°) ===
        if abs(rad2deg(θ)) ≤ 5
            # compute resistance w exactly like GasModels
            w = GasModels._calc_pipe_resistance(
                pipe,
                data["base_length"],
                data["base_pressure"],
                data["base_flow"],
                data["sound_speed"],
            )

            fmin = pipe["flow_min"]
            fmax = pipe["flow_max"]

            pi = model[:psqr][i]
            pj = model[:psqr][j]
            f = model[:f_pipe][k]
            fmf_l = model[:fmf_l_pipe][k]

            # === Special case: fmin == fmax (fixed flow) ===
            if fmin == fmax
                @constraint(model, w * (pi - pj) == fmin)

                # === Normal LRWP case ===
            else
                # Weymouth linearized form
                @constraint(model, w * (pi - pj) == fmf_l)

                # --- Partitioning exactly like GasModels ---
                partition = (fmin < 0 && fmax > 0) ? [fmin, 0.0, fmax] : [fmin, fmax]
                # println("number of partition", partition)

                # Univariate function f(a) = a*|a|
                ffun(a) = a * abs(a)
                f_dash(x) = ForwardDiff.derivative(ffun, x)

                # Build data for PolyhedralRelaxations
                ufd = PolyhedralRelaxations.UnivariateFunctionData(
                    ffun,
                    f_dash,
                    partition,
                    NaN,
                    1e-6,
                    1e-6,
                    0,
                    length(partition),
                )

                # Vertices of convex relaxation
                vertices = PolyhedralRelaxations._get_lp_relaxation_vertices(ufd)
                N = length(vertices)

                # λ-variables for convex combination
                λ = @variable(model, [1:N], lower_bound = 0.0, upper_bound = 1.0,
                    base_name = "lambda_pipe_$k")

                # Sum λ_j = 1
                @constraint(model, sum(λ[j] for j in 1:N) == 1)

                # f_pipe approximation
                @constraint(model, f == sum(λ[j] * vertices[j][1] for j in 1:N))

                # fmf_l_pipe approximation
                @constraint(model, fmf_l == sum(λ[j] * vertices[j][2] for j in 1:N))
            end

            # === 3. Inclined pipe (|θ| > 5°) ===
        else
            # GasModels uses "inclined pipe pressure drop":
            # constraint_inclined_pipe_pressure_drop(gm, ...)

            # You MUST compute the two resistances r1,r2:
            r1, r2 = GasModels._calc_inclined_pipe_resistance(
                pipe,
                data["base_length"],
                data["base_pressure"],
                data["base_flow"],
                data["sound_speed"],
            )

            # And pressure-drop bounds (not strictly required)
            # but included for full equivalence:
            # (you may skip if not needed)
            # pd_bounds = GasModels._calc_inclined_pipe_pd_bounds_sqr(...)

            # TODO: Implement inclined-pipe LRWP constraint
            # For now we mimic GasModels structure with a placeholder:
            @warn "Inclined pipe k=$k detected (|theta| > 5°). Must implement inclined LRWP constraint."

            # Minimal placeholder to avoid crashing:
            @constraint(model, model[:psqr][i] - model[:psqr][j] == 0)
        end
    end


    # constraint_loss_resistor_mass_flow
    for (k, loss_resistor) in ref[:loss_resistor]
        f = model[:f_loss_resistor][k]

        f_min = loss_resistor["flow_min"]
        f_max = loss_resistor["flow_max"]

        lb = JuMP.has_lower_bound(f) ? max(JuMP.lower_bound(f), f_min) : f_min
        ub = JuMP.has_upper_bound(f) ? min(JuMP.upper_bound(f), f_max) : f_max

        JuMP.set_lower_bound(f, lb)
        JuMP.set_upper_bound(f, ub)
    end



    # constraint_short_pipe_pressure
    for (k, short_pipe) in ref[:short_pipe]
        i = short_pipe["fr_junction"]
        j = short_pipe["to_junction"]

        @constraint(model, model[:psqr][i] == model[:psqr][j])
    end


    # constraint_compressor_ratios
    for (k, compressor) in ref[:compressor]

        # === Node indices ===
        i = compressor["fr_junction"]
        j = compressor["to_junction"]

        # === Variables ===
        pi = model[:psqr][i]
        pj = model[:psqr][j]
        f = model[:f_compressor][k]

        # === Ratios and directionality ===
        min_ratio = compressor["c_ratio_min"]
        max_ratio = compressor["c_ratio_max"]
        type = get(compressor, "directionality", 0)

        # === Junction limits ===
        i_pmax = ref[:junction][i]["p_max"]
        j_pmax = ref[:junction][j]["p_max"]
        i_pmin = ref[:junction][i]["p_min"]
        j_pmin = ref[:junction][j]["p_min"]

        # ======================================================================================
        # TYPE 0 : BIDIRECTIONAL COMPRESSION
        # ======================================================================================
        if type == 0

            if (min_ratio <= 1.0 && max_ratio >= 1)
                # === pk auxiliary pressure level ===
                pk = @variable(model,
                    lower_bound = min(JuMP.lower_bound(pi), JuMP.lower_bound(pj)),
                    upper_bound = max(JuMP.upper_bound(pi), JuMP.upper_bound(pj))
                )

                # === pressure differences ===
                pik = @variable(model,
                    lower_bound = JuMP.lower_bound(pi) - JuMP.upper_bound(pk),
                    upper_bound = JuMP.upper_bound(pi) - JuMP.lower_bound(pk)
                )

                pjk = @variable(model,
                    lower_bound = JuMP.lower_bound(pj) - JuMP.upper_bound(pk),
                    upper_bound = JuMP.upper_bound(pj) - JuMP.lower_bound(pk)
                )

                fpik = @variable(model)
                fpjk = @variable(model)

                @constraint(model, pk - max_ratio^2 * pi <= 0)
                @constraint(model, min_ratio^2 * pi - pk <= 0)
                @constraint(model, fpik <= 0)

                @constraint(model, pk - max_ratio^2 * pj <= 0)
                @constraint(model, min_ratio^2 * pj - pk <= 0)
                @constraint(model, -fpjk <= 0)

                @constraint(model, pjk == pj - pk)
                @constraint(model, pik == pi - pk)

                relaxation_product(model, f, pik, fpik)
                relaxation_product(model, f, pjk, fpjk)

            else
                # === Disjunction with binary y ===
                y = get_compressor_y(gm, n, k)

                @constraint(model, pj - max_ratio^2 * pi <= (1 - y) * (j_pmax^2))
                @constraint(model, min_ratio^2 * pi - pj <= (1 - y) * (min_ratio^2 * i_pmax^2))
                @constraint(model, pi - max_ratio^2 * pj <= y * (i_pmax^2))
                @constraint(model, min_ratio^2 * pj - pi <= y * (min_ratio^2 * j_pmax^2))
            end


            # ======================================================================================
            # TYPE 1 : ONE-DIRECTIONAL (i → j)
            # ======================================================================================
        elseif type == 1

            @constraint(model, pj - max_ratio^2 * pi <= 0)
            @constraint(model, min_ratio^2 * pi - pj <= 0)


            # ======================================================================================
            # TYPE 2 : SEMI-DIRECTIONAL
            # ======================================================================================
        else

            if min_ratio == 1
                pij = @variable(model,
                    lower_bound = JuMP.lower_bound(pi) - JuMP.upper_bound(pj),
                    upper_bound = JuMP.upper_bound(pi) - JuMP.lower_bound(pj)
                )
                fpij = @variable(model)

                @constraint(model, pj - max_ratio^2 * pi <= 0)
                @constraint(model, min_ratio^2 * pi - pj <= 0)
                @constraint(model, fpij <= 0)
                @constraint(model, pij == pi - pj)

                relaxation_product(model, f, pij, fpij)

            else
                # === DISJUNCTION (flow can reverse) ===
                y = get_compressor_y(gm, n, k)

                @constraint(model, pj - max_ratio^2 * pi <= (1 - y) * (j_pmax^2))
                @constraint(model, min_ratio^2 * pi - pj <= (1 - y) * (i_pmax^2))
                @constraint(model, pi - pj <= y * (i_pmax^2))
                @constraint(model, pj - pi <= y * (j_pmax^2))
            end
        end
    end



    # constraint_compressor_ratio_value  
    for (k, compressor) in ref[:compressor]

        # === Indices ===
        i = compressor["fr_junction"]
        j = compressor["to_junction"]

        # === Variables ===
        pi = model[:psqr][i]
        pj = model[:psqr][j]
        r = model[:rsqr][k]

        # === Pressure bounds ===
        i_pmax = ref[:junction][i]["p_max"]
        j_pmax = ref[:junction][j]["p_max"]

        # === Directionality and ratios ===
        type = get(compressor, "directionality", 0)
        max_ratio = compressor["c_ratio_max"]
        min_ratio = compressor["c_ratio_min"]

        if type == 0
            # === Bidirectional compression (needs binary y) ===

            # → Use GasModels helper to ensure SAME y across all constraints
            y = @variable(model, binary = true)


            # Auxiliary variables
            rpi = @variable(model)
            rpj = @variable(model)

            # McCormick relaxations
            relaxation_product(model, pi, r, rpi)
            relaxation_product(model, pj, r, rpj)

            # GasModels-compliant inequalities
            @constraint(model, rpi <= pj + (1 - y) * i_pmax^2 * max_ratio^2)
            @constraint(model, rpi >= pj - (1 - y) * j_pmax^2)
            @constraint(model, rpj <= pi + y * j_pmax^2 * max_ratio^2)
            @constraint(model, rpj >= pi - y * i_pmax^2)

        else
            # → Unidirectional or semi-directional: simple product
            relaxation_product(model, pi, r, pj)
        end
    end




    # constraint_on_off_valve_mass_flow
    for (k, valve) in ref[:valve]
        # Retrieve the JuMP variables for this valve
        f = f_valve[k]      # mass flow through the valve
        v = v_valve[k]      # binary/opening variable (1 = open, 0 = closed)

        # Extract flow limits from valve data
        f_min = valve["flow_min"]
        f_max = valve["flow_max"]

        # Add the on/off flow constraints directly to the model
        @constraint(model, f_min * v <= f)
        @constraint(model, f <= f_max * v)
    end

    # constraint_on_off_valve_pressure
    for (k, valve) in ref[:valve]
        i = valve["fr_junction"]
        j = valve["to_junction"]

        pi = model[:psqr][i]
        pj = model[:psqr][j]
        v = model[:v_valve][k]

        i_pmax = ref[:junction][i]["p_max"]
        j_pmax = ref[:junction][j]["p_max"]

        @constraint(model, pj - ((1 - v) * j_pmax^2) <= pi)
        @constraint(model, pi <= pj + ((1 - v) * i_pmax^2))
    end

    # constraint_on_off_regulator_mass_flow
    for (k, regulator) in ref[:regulator]
        f = model[:f_regulator][k]
        v = model[:v_regulator][k]

        f_min = regulator["flow_min"]
        f_max = regulator["flow_max"]

        @constraint(model, f_min * v <= f)
        @constraint(model, f <= f_max * v)
    end


    # constraint_on_off_regulator_pressure
    for (k, reg) in ref[:regulator]
        # === Indices and parameters ===
        i = reg["fr_junction"]
        j = reg["to_junction"]

        # If your data uses "r_ratio_min/max" instead of "reduction_factor_*",
        # you can swap these field names back to "r_ratio_min"/"r_ratio_max".
        min_ratio = reg["reduction_factor_min"]
        max_ratio = reg["reduction_factor_max"]

        f_min = reg["flow_min"]

        # Junction pressure limits (in base units, not squared yet)
        i_pmin = ref[:junction][i]["p_min"]
        i_pmax = ref[:junction][i]["p_max"]
        j_pmin = ref[:junction][j]["p_min"]
        j_pmax = ref[:junction][j]["p_max"]

        # === Variables ===
        psqr_i = model[:psqr][i]
        psqr_j = model[:psqr][j]
        v_reg = model[:v_regulator][k]
        f_reg = model[:f_regulator][k]

        if max_ratio == 1
            # === Case 1: Fixed-ratio regulation (max_ratio = 1) ===
            pij = @variable(
                model,
                lower_bound = JuMP.lower_bound(psqr_i) - JuMP.upper_bound(psqr_j),
                upper_bound = JuMP.upper_bound(psqr_i) - JuMP.lower_bound(psqr_j),
            )
            fpij = @variable(model)

            @constraint(model,
                psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2
            )
            @constraint(model,
                min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2)
            )
            @constraint(model, fpij >= 0)
            @constraint(model, pij == psqr_i - psqr_j)

            # Bilinear relaxation product f_reg * (psqr_i - psqr_j)
            relaxation_product(model, f_reg, pij, fpij)

        elseif f_min >= 0
            # === Case 2: Forward flow only (f_min ≥ 0) ===
            @constraint(model,
                psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2
            )
            @constraint(model,
                min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2)
            )

        else
            # === Case 3: Disjunctive on/off behavior (flow can reverse) ===
            y = @variable(model, binary = true)

            @constraint(model,
                psqr_j - (max_ratio^2 * psqr_i) <= (2 - y - v_reg) * j_pmax^2
            )
            @constraint(model,
                (min_ratio^2 * psqr_i) - psqr_j <= (2 - y - v_reg) * i_pmax^2
            )
            @constraint(model,
                psqr_j - psqr_i <= (1 + y - v_reg) * j_pmax^2
            )
            @constraint(model,
                psqr_i - psqr_j <= (1 + y - v_reg) * i_pmax^2
            )
        end
    end






    # ------------------------------------------------------------------ #
    # min_supply_demand_costs
    # ------------------------------------------------------------------ #
    gamma = get_specific_heat_capacity_ratio(data)
    m = ((gamma - 1) / gamma) / 2
    T = get_temperature(data)
    G = get_gas_specific_gravity(data)
    work = ((286.76 / G) * T * (gamma / (gamma - 1)))
    baseF = get_base_flow(data)
    econW = get_economic_weighting(data)

    load_prices = Dict(k => get(ref[:delivery][k], "bid_price", 1.0)
                       for k in dispatchable_deliveries)
    prod_prices = Dict(k => get(ref[:receipt][k], "offer_price", 1.0)
                       for k in receipts if ref[:receipt][k]["is_dispatchable"] == 1)
    transf_prices = Dict(k => ref[:transfer][k]["withdrawal_min"] ≥ 0 ?
                              get(ref[:transfer][k], "bid_price", 1.0) :
                              -get(ref[:transfer][k], "offer_price", 1.0)
                         for k in transfers
                         if ref[:transfer][k]["is_dispatchable"] == 1)


    @variable(model, z)

    @constraint(model, z >=
                       (
        -sum(load_prices[k] * model[:fl][k] for k in keys(load_prices)) -
        sum(transf_prices[k] * model[:ft][k] for k in keys(transf_prices)) +
        sum(prod_prices[k] * model[:fg][k] for k in keys(prod_prices))
    )
    )

    @objective(model, Min, z)

    set_silent(model)


    # ------------------------------------------------------------------ #
    # 5. Optimiza y retorna
    # ------------------------------------------------------------------ #
    optimize!(model)

    return Dict(
        "flag" => (JuMP.termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(model),
        "fg" => JuMP.value.(model[:fg]),
        "ft" => JuMP.value.(model[:ft]),
        "model" => model
    )
end

###############################################################
# Function to generate data HF
###############################################################
function WP(
    data, delivery_samples;
    solver=Ipopt.Optimizer
)

    ###############
    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    delivery_ids = sort(parse.(Int, collect(keys(data["delivery"]))))

    for (i, id) in enumerate(delivery_ids)
        data["delivery"][string(id)]["withdrawal_nominal"] = delivery_samples[i]
    end

    spec = GasModels.instantiate_model(
        data,
        WPGasModel,
        GasModels.build_ogf_comp_power_unc
    )

    model = spec.model
    set_silent(model)

    # =============================
    # REFS (same style)
    # =============================
    junctions_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:junction])))
    receipt_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:receipt])))
    transfer_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:transfer])))

    valve_ref = haskey(spec.ref[:it][:gm][:nw][0], :valve) ?
                sort(collect(keys(spec.ref[:it][:gm][:nw][0][:valve]))) : Int[]

    regulator_ref = haskey(spec.ref[:it][:gm][:nw][0], :regulator) ?
                    sort(collect(keys(spec.ref[:it][:gm][:nw][0][:regulator]))) : Int[]

    resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :resistor) ?
                   sort(collect(keys(spec.ref[:it][:gm][:nw][0][:resistor]))) : Int[]

    compressor_ref = haskey(spec.ref[:it][:gm][:nw][0], :compressor) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:compressor]))) : Int[]

    pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :pipe) ?
               sort(collect(keys(spec.ref[:it][:gm][:nw][0][:pipe]))) : Int[]

    short_pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :short_pipe) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:short_pipe]))) : Int[]

    loss_resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :loss_resistor) ?
                        sort(collect(keys(spec.ref[:it][:gm][:nw][0][:loss_resistor]))) : Int[]

    # =============================
    # VARS (same style)
    # =============================
    fg_vars = spec.var[:it][:gm][:nw][0][:fg]
    ft_vars = spec.var[:it][:gm][:nw][0][:ft]

    f_pipe_vars = spec.var[:it][:gm][:nw][0][:f_pipe]
    f_short_pipe_vars = spec.var[:it][:gm][:nw][0][:f_short_pipe]
    f_compressor_vars = spec.var[:it][:gm][:nw][0][:f_compressor]
    f_resistor_vars = spec.var[:it][:gm][:nw][0][:f_resistor]
    f_loss_resistor_vars = spec.var[:it][:gm][:nw][0][:f_loss_resistor]
    f_valve_vars = spec.var[:it][:gm][:nw][0][:f_valve]
    f_regulator_vars = spec.var[:it][:gm][:nw][0][:f_regulator]
    fl_vars = spec.var[:it][:gm][:nw][0][:fl]

    p_vars = spec.var[:it][:gm][:nw][0][:p]
    psqr_vars = spec.var[:it][:gm][:nw][0][:psqr]
    rsqr_vars = spec.var[:it][:gm][:nw][0][:rsqr]

    v_valve_vars = spec.var[:it][:gm][:nw][0][:v_valve]
    v_regulator_vars = spec.var[:it][:gm][:nw][0][:v_regulator]

    well_head_flow_vars = spec.var[:it][:gm][:nw][0][:well_head_flow]

    JuMP.set_optimizer(model, () -> solver())
    optimize!(model)

    # =============================
    # EXTRACTION (EXACT SAME PATTERN)
    # =============================
    fg_vec = [JuMP.value(fg_vars[k]) for k in receipt_ref]
    ft_vec = [JuMP.value(ft_vars[k]) for k in transfer_ref]

    f_pipe_vec = [JuMP.value(f_pipe_vars[k]) for k in pipe_ref]
    f_short_pipe_vec = [JuMP.value(f_short_pipe_vars[k]) for k in short_pipe_ref]
    f_compressor_vec = [JuMP.value(f_compressor_vars[k]) for k in compressor_ref]
    f_resistor_vec = [JuMP.value(f_resistor_vars[k]) for k in resistor_ref]
    f_loss_resistor_vec = [JuMP.value(f_loss_resistor_vars[k]) for k in loss_resistor_ref]
    f_valve_vec = [JuMP.value(f_valve_vars[k]) for k in valve_ref]
    f_regulator_vec = [JuMP.value(f_regulator_vars[k]) for k in regulator_ref]
    fl_vec = [JuMP.value(ft_vars[k]) for k in transfer_ref]


    psqr_vec = [JuMP.value(psqr_vars[k]) for k in junctions_ref]
    rsqr_vec = [JuMP.value(rsqr_vars[k]) for k in compressor_ref]

    v_valve_vec = [JuMP.value(v_valve_vars[k]) for k in valve_ref]
    v_regulator_vec = [JuMP.value(v_regulator_vars[k]) for k in regulator_ref]



    println("Objective value: ", JuMP.objective_value(model))
    println("termination_status: ", JuMP.termination_status(model))

    return Dict(
        "flag" => (JuMP.termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(model), "fg" => fg_vec,
        "ft" => ft_vec,
        "f_pipe" => f_pipe_vec,
        "f_short_pipe" => f_short_pipe_vec,
        "f_compressor" => f_compressor_vec,
        "f_resistor" => f_resistor_vec,
        "f_loss_resistor" => f_loss_resistor_vec,
        "f_valve" => f_valve_vec,
        "f_regulator" => f_regulator_vec,
        "fl" => fl_vec,

        # "p" => p_vec,
        "psqr" => psqr_vec,
        "rsqr" => rsqr_vec, "v_valve" => v_valve_vec,
        "v_regulator" => v_regulator_vec,
        # "well_head_flow" => well_head_flow_vec,

        "model" => model
    )
end


function CRDWP(
    data, delivery_samples;
    solver=Gurobi.Optimizer
)

    ###############
    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    delivery_ids = sort(parse.(Int, collect(keys(data["delivery"]))))

    for (i, id) in enumerate(delivery_ids)
        data["delivery"][string(id)]["withdrawal_nominal"] = delivery_samples[i]
    end

    spec = GasModels.instantiate_model(
        data,
        CRDWPGasModel,
        GasModels.build_ogf_comp_power_unc
    )

    model = spec.model
    set_silent(model)


    junctions_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:junction])))
    receipt_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:receipt])))
    transfer_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:transfer])))

    valve_ref = haskey(spec.ref[:it][:gm][:nw][0], :valve) ?
                sort(collect(keys(spec.ref[:it][:gm][:nw][0][:valve]))) : Int[]

    regulator_ref = haskey(spec.ref[:it][:gm][:nw][0], :regulator) ?
                    sort(collect(keys(spec.ref[:it][:gm][:nw][0][:regulator]))) : Int[]

    resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :resistor) ?
                   sort(collect(keys(spec.ref[:it][:gm][:nw][0][:resistor]))) : Int[]

    compressor_ref = haskey(spec.ref[:it][:gm][:nw][0], :compressor) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:compressor]))) : Int[]

    pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :pipe) ?
               sort(collect(keys(spec.ref[:it][:gm][:nw][0][:pipe]))) : Int[]

    short_pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :short_pipe) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:short_pipe]))) : Int[]

    loss_resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :loss_resistor) ?
                        sort(collect(keys(spec.ref[:it][:gm][:nw][0][:loss_resistor]))) : Int[]

    # =============================
    # VARS (same style)
    # =============================
    fg_vars = spec.var[:it][:gm][:nw][0][:fg]
    ft_vars = spec.var[:it][:gm][:nw][0][:ft]

    f_pipe_vars = spec.var[:it][:gm][:nw][0][:f_pipe]
    f_short_pipe_vars = spec.var[:it][:gm][:nw][0][:f_short_pipe]
    f_compressor_vars = spec.var[:it][:gm][:nw][0][:f_compressor]
    f_resistor_vars = spec.var[:it][:gm][:nw][0][:f_resistor]
    f_loss_resistor_vars = spec.var[:it][:gm][:nw][0][:f_loss_resistor]
    f_valve_vars = spec.var[:it][:gm][:nw][0][:f_valve]
    f_regulator_vars = spec.var[:it][:gm][:nw][0][:f_regulator]
    fl_vars = spec.var[:it][:gm][:nw][0][:fl]

    p_vars = spec.var[:it][:gm][:nw][0][:p]
    psqr_vars = spec.var[:it][:gm][:nw][0][:psqr]
    rsqr_vars = spec.var[:it][:gm][:nw][0][:rsqr]

    v_valve_vars = spec.var[:it][:gm][:nw][0][:v_valve]
    v_regulator_vars = spec.var[:it][:gm][:nw][0][:v_regulator]

    well_head_flow_vars = spec.var[:it][:gm][:nw][0][:well_head_flow]

    JuMP.set_optimizer(model, () -> solver())
    optimize!(model)

    # =============================
    # EXTRACTION (EXACT SAME PATTERN)
    # =============================
    fg_vec = [JuMP.value(fg_vars[k]) for k in receipt_ref]
    ft_vec = [JuMP.value(ft_vars[k]) for k in transfer_ref]

    f_pipe_vec = [JuMP.value(f_pipe_vars[k]) for k in pipe_ref]
    f_short_pipe_vec = [JuMP.value(f_short_pipe_vars[k]) for k in short_pipe_ref]
    f_compressor_vec = [JuMP.value(f_compressor_vars[k]) for k in compressor_ref]
    f_resistor_vec = [JuMP.value(f_resistor_vars[k]) for k in resistor_ref]
    f_loss_resistor_vec = [JuMP.value(f_loss_resistor_vars[k]) for k in loss_resistor_ref]
    f_valve_vec = [JuMP.value(f_valve_vars[k]) for k in valve_ref]
    f_regulator_vec = [JuMP.value(f_regulator_vars[k]) for k in regulator_ref]
    fl_vec = [JuMP.value(ft_vars[k]) for k in transfer_ref]


    psqr_vec = [JuMP.value(psqr_vars[k]) for k in junctions_ref]
    rsqr_vec = [JuMP.value(rsqr_vars[k]) for k in compressor_ref]

    v_valve_vec = [JuMP.value(v_valve_vars[k]) for k in valve_ref]
    v_regulator_vec = [JuMP.value(v_regulator_vars[k]) for k in regulator_ref]



    println("Objective value: ", JuMP.objective_value(model))
    println("termination_status: ", JuMP.termination_status(model))

    return Dict(
        "flag" => (JuMP.termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(model), "fg" => fg_vec,
        "ft" => ft_vec,
        "f_pipe" => f_pipe_vec,
        "f_short_pipe" => f_short_pipe_vec,
        "f_compressor" => f_compressor_vec,
        "f_resistor" => f_resistor_vec,
        "f_loss_resistor" => f_loss_resistor_vec,
        "f_valve" => f_valve_vec,
        "f_regulator" => f_regulator_vec,
        "fl" => fl_vec,

        # "p" => p_vec,
        "psqr" => psqr_vec,
        "rsqr" => rsqr_vec, "v_valve" => v_valve_vec,
        "v_regulator" => v_regulator_vec,
        # "well_head_flow" => well_head_flow_vec,

        "model" => model
    )

end


function LRDWP(
    data, delivery_samples;
    solver=Gurobi.Optimizer
)

    ###############
    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    delivery_ids = sort(parse.(Int, collect(keys(data["delivery"]))))

    for (i, id) in enumerate(delivery_ids)
        data["delivery"][string(id)]["withdrawal_nominal"] = delivery_samples[i]
    end

    spec = GasModels.instantiate_model(
        data,
        LRDWPGasModel,
        GasModels.build_ogf_comp_power_unc
    )

    model = spec.model
    set_silent(model)


    junctions_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:junction])))
    receipt_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:receipt])))
    transfer_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:transfer])))

    valve_ref = haskey(spec.ref[:it][:gm][:nw][0], :valve) ?
                sort(collect(keys(spec.ref[:it][:gm][:nw][0][:valve]))) : Int[]

    regulator_ref = haskey(spec.ref[:it][:gm][:nw][0], :regulator) ?
                    sort(collect(keys(spec.ref[:it][:gm][:nw][0][:regulator]))) : Int[]

    resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :resistor) ?
                   sort(collect(keys(spec.ref[:it][:gm][:nw][0][:resistor]))) : Int[]

    compressor_ref = haskey(spec.ref[:it][:gm][:nw][0], :compressor) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:compressor]))) : Int[]

    pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :pipe) ?
               sort(collect(keys(spec.ref[:it][:gm][:nw][0][:pipe]))) : Int[]

    short_pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :short_pipe) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:short_pipe]))) : Int[]

    loss_resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :loss_resistor) ?
                        sort(collect(keys(spec.ref[:it][:gm][:nw][0][:loss_resistor]))) : Int[]

    # =============================
    # VARS (same style)
    # =============================
    fg_vars = spec.var[:it][:gm][:nw][0][:fg]
    ft_vars = spec.var[:it][:gm][:nw][0][:ft]

    f_pipe_vars = spec.var[:it][:gm][:nw][0][:f_pipe]
    f_short_pipe_vars = spec.var[:it][:gm][:nw][0][:f_short_pipe]
    f_compressor_vars = spec.var[:it][:gm][:nw][0][:f_compressor]
    f_resistor_vars = spec.var[:it][:gm][:nw][0][:f_resistor]
    f_loss_resistor_vars = spec.var[:it][:gm][:nw][0][:f_loss_resistor]
    f_valve_vars = spec.var[:it][:gm][:nw][0][:f_valve]
    f_regulator_vars = spec.var[:it][:gm][:nw][0][:f_regulator]
    fl_vars = spec.var[:it][:gm][:nw][0][:fl]

    p_vars = spec.var[:it][:gm][:nw][0][:p]
    psqr_vars = spec.var[:it][:gm][:nw][0][:psqr]
    rsqr_vars = spec.var[:it][:gm][:nw][0][:rsqr]

    v_valve_vars = spec.var[:it][:gm][:nw][0][:v_valve]
    v_regulator_vars = spec.var[:it][:gm][:nw][0][:v_regulator]

    well_head_flow_vars = spec.var[:it][:gm][:nw][0][:well_head_flow]

    JuMP.set_optimizer(model, () -> solver())
    optimize!(model)

    # =============================
    # EXTRACTION (EXACT SAME PATTERN)
    # =============================
    fg_vec = [JuMP.value(fg_vars[k]) for k in receipt_ref]
    ft_vec = [JuMP.value(ft_vars[k]) for k in transfer_ref]

    f_pipe_vec = [JuMP.value(f_pipe_vars[k]) for k in pipe_ref]
    f_short_pipe_vec = [JuMP.value(f_short_pipe_vars[k]) for k in short_pipe_ref]
    f_compressor_vec = [JuMP.value(f_compressor_vars[k]) for k in compressor_ref]
    f_resistor_vec = [JuMP.value(f_resistor_vars[k]) for k in resistor_ref]
    f_loss_resistor_vec = [JuMP.value(f_loss_resistor_vars[k]) for k in loss_resistor_ref]
    f_valve_vec = [JuMP.value(f_valve_vars[k]) for k in valve_ref]
    f_regulator_vec = [JuMP.value(f_regulator_vars[k]) for k in regulator_ref]
    fl_vec = [JuMP.value(ft_vars[k]) for k in transfer_ref]


    psqr_vec = [JuMP.value(psqr_vars[k]) for k in junctions_ref]
    rsqr_vec = [JuMP.value(rsqr_vars[k]) for k in compressor_ref]

    v_valve_vec = [JuMP.value(v_valve_vars[k]) for k in valve_ref]
    v_regulator_vec = [JuMP.value(v_regulator_vars[k]) for k in regulator_ref]



    println("Objective value: ", JuMP.objective_value(model))
    println("termination_status: ", JuMP.termination_status(model))

    return Dict(
        "flag" => (JuMP.termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(model), "fg" => fg_vec,
        "ft" => ft_vec,
        "f_pipe" => f_pipe_vec,
        "f_short_pipe" => f_short_pipe_vec,
        "f_compressor" => f_compressor_vec,
        "f_resistor" => f_resistor_vec,
        "f_loss_resistor" => f_loss_resistor_vec,
        "f_valve" => f_valve_vec,
        "f_regulator" => f_regulator_vec,
        "fl" => fl_vec,

        # "p" => p_vec,
        "psqr" => psqr_vec,
        "rsqr" => rsqr_vec, "v_valve" => v_valve_vec,
        "v_regulator" => v_regulator_vec,
        # "well_head_flow" => well_head_flow_vec,

        "model" => model
    )

end


# Funcion OptLayer for LF 
function LRWPLayer_GM(
    gw::GraphWithRef,
    β_param::Matrix,
    α_param::Matrix;
    model=DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
)

    data = gw.data
    # withdrawal_per_junction = gw.graph.x[1, :]   # size = N_junctions
    # nbatch = gw.graph.num_graphs

    #! re check something wrong because if you analize the idx it gives you something strrange (the order is wrong)



    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    compressors = sort(collect(keys(ref[:compressor])))
    junctions = sort(collect(keys(ref[:junction])))
    loss_resistors = sort(collect(keys(ref[:loss_resistor])))
    pipes = sort(collect(keys(ref[:pipe])))
    receipts = sort(collect(keys(ref[:receipt])))
    regulators = sort(collect(keys(ref[:regulator])))
    resistors = sort(collect(keys(ref[:resistor])))
    short_pipes = sort(collect(keys(ref[:short_pipe])))
    transfers = sort(collect(keys(ref[:transfer])))
    valves = sort(collect(keys(ref[:valve])))


    delivery_data = gw.graph.x[1, :]
    nbatch = gw.graph.num_graphs


    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    for (load_id, load_data) in data["delivery"]
        bus_id = load_data["junction_id"]
        idx = bus_idx_map[bus_id]
        load_data["withdrawal_nominal"] = delivery_data[idx]
    end

    # ------------------------------------------------------------------ #
    # 2. Variables (todas guardadas en model[:...])
    # ------------------------------------------------------------------ #

    pipes = sort(collect(keys(ref[:pipe])))
    pipe_idx_map = Dict(p => idx for (idx, p) in enumerate(pipes))
    junction_idx_map = Dict(p => idx for (idx, p) in enumerate(junctions))

    # println("rhs size = ", size(β_param))
    # println("nbatch = ", nbatch)
    # @variable(model, β[p in pipes, b in 1:nbatch] in Parameter(rhs[b, pipe_idx_map[p]]))


    #! Use for training Ipopt
    @variable(model, β[p in junctions, b in 1:nbatch] in Parameter(β_param'[junction_idx_map[p], b]))
    @variable(model, α[p in pipes, b in 1:nbatch] in Parameter(α_param'[pipe_idx_map[p], b]))


    #! Use for predict with Gurobi
    # β = [β_param'[junction_idx_map[p], b] for p in junctions, b in 1:nbatch]
    # α = [α_param'[pipe_idx_map[p], b] for p in pipes, b in 1:nbatch]



    model[:f_compressor] = @variable(model,
        [k in compressors, b in 1:nbatch],
        lower_bound = ref[:compressor][k]["flow_min"],
        upper_bound = ref[:compressor][k]["flow_max"],
        start = get(ref[:compressor][k], "f_start", 0.0),
        base_name = "f_comp",
    )

    model[:f_resistor] = @variable(model,
        [k in resistors, b in 1:nbatch],
        lower_bound = ref[:resistor][k]["flow_min"],
        upper_bound = ref[:resistor][k]["flow_max"],
        start = get(ref[:resistor][k], "f_start", 1e-6),
        base_name = "f_res",
    )

    model[:f_loss_resistor] = @variable(model,
        [k in loss_resistors, b in 1:nbatch],
        lower_bound = ref[:loss_resistor][k]["flow_min"],
        upper_bound = ref[:loss_resistor][k]["flow_max"],
        start = get(ref[:loss_resistor][k], "f_start", 1e-6),
        base_name = "f_lres",
    )

    model[:f_pipe] = @variable(model,
        [k in pipes, b in 1:nbatch],
        lower_bound = ref[:pipe][k]["flow_min"],
        upper_bound = ref[:pipe][k]["flow_max"],
        start = get(ref[:pipe][k], "f_start", 0.0),
        base_name = "f_pipe",
    )

    model[:f_short_pipe] = @variable(model,
        [k in short_pipes, b in 1:nbatch],
        lower_bound = ref[:short_pipe][k]["flow_min"],
        upper_bound = ref[:short_pipe][k]["flow_max"],
        start = get(ref[:short_pipe][k], "f_start", 0.0),
        base_name = "f_spipe",
    )

    model[:f_valve] = @variable(model,
        [k in valves, b in 1:nbatch],
        lower_bound = min.(ref[:valve][k]["flow_min"] for k in valves),
        upper_bound = max.(ref[:valve][k]["flow_max"] for k in valves),
        start = get.(Ref(ref[:valve]), valves, Ref("f_start"), 0.0),
        base_name = "f_valve",
    )

    model[:f_regulator] = @variable(model,
        [k in regulators, b in 1:nbatch],
        lower_bound = min.(ref[:regulator][k]["flow_min"] for k in regulators),
        upper_bound = max.(ref[:regulator][k]["flow_max"] for k in regulators),
        start = get.(Ref(ref[:regulator]), regulators, Ref("f_start"), 0.0),
        base_name = "f_reg",
    )

    ## b) Inyecciones / retiros
    model[:ft] = @variable(model,
        [k in transfers, b in 1:nbatch],
        lower_bound = ref[:transfer][k]["withdrawal_min"],
        upper_bound = ref[:transfer][k]["withdrawal_max"],
        start = get(ref[:transfer][k], "ft_start",
            ref[:transfer][k]["withdrawal_max"]),
        base_name = "ft",
    )

    model[:fg] = @variable(model,
        [k in receipts, b in 1:nbatch],
        lower_bound = ref[:receipt][k]["injection_min"],
        upper_bound = ref[:receipt][k]["injection_max"],
        start = get(ref[:receipt][k], "fg_start",
            ref[:receipt][k]["injection_max"]),
        base_name = "fg",
    )

    dispatchable_deliveries =
        [k for k in keys(ref[:delivery]) if ref[:delivery][k]["is_dispatchable"] == 1]

    model[:fl] = @variable(model,
        [k in dispatchable_deliveries],
        lower_bound = ref[:delivery][k]["withdrawal_min"],
        upper_bound = ref[:delivery][k]["withdrawal_max"],
        start = get(ref[:delivery][k], "fl_start",
            ref[:delivery][k]["withdrawal_max"]),
        base_name = "fl",
    )

    ## c) Presiones
    model[:psqr] = @variable(model,
        [i in junctions, b in 1:nbatch],
        lower_bound = ref[:junction][i]["p_min"]^2,
        upper_bound = ref[:junction][i]["p_max"]^2,
        start = get(ref[:junction][i], "p_start",
            ref[:junction][i]["p_min"])^2,
        base_name = "psqr",
    )

    # no-square pressure donde sólo hay resistores / loss_resistores
    prs_junc = unique(vcat(
        [ref[:resistor][k]["fr_junction"] for k in resistors],
        [ref[:resistor][k]["to_junction"] for k in resistors],
        [ref[:loss_resistor][k]["fr_junction"] for k in loss_resistors],
        [ref[:loss_resistor][k]["to_junction"] for k in loss_resistors],
    ))
    model[:p_nonsq] = @variable(model,
        [i in prs_junc, b in 1:nbatch],
        lower_bound = ref[:junction][i]["p_min"],
        upper_bound = ref[:junction][i]["p_max"],
        start = get(ref[:junction][i], "p_start",
            ref[:junction][i]["p_max"]),
        base_name = "p_nsq",
    )

    ## d) Ratios de compresor
    model[:rsqr] = @variable(model,
        [k in compressors, b in 1:nbatch],
        lower_bound = ref[:compressor][k]["c_ratio_min"]^2,
        upper_bound = ref[:compressor][k]["c_ratio_max"]^2,
        start = get(ref[:compressor][k], "ratio_start", 1.0),
        base_name = "rsqr",
    )

    ## e) Auxiliar f |f|
    model[:fmf_l_pipe] = @variable(model, [k in pipes, b in 1:nbatch], base_name = "fmf_l")
    # model[:subts_psqr] = @variable(model, [k in pipes], base_name = "subts_psqr")

    ## f) Binarias de válvulas / reguladores
    model[:v_valve] = @variable(model, [k in valves], Bin,
        start = get.(Ref(ref[:valve]), valves, Ref("v_start"), 1))
    model[:v_regulator] = @variable(model, [k in regulators], Bin,
        start = get.(Ref(ref[:regulator]), regulators,
            Ref("v_start"), 1))

    # @variable(model, slack_pos[j in junctions, b in 1:nbatch] >= 0)
    # @variable(model, slack_neg[j in junctions, b in 1:nbatch] >= 0)


    # ------------------------------------------------------------------ #
    # 3. Restricciones clave
    # ------------------------------------------------------------------ #


    # constraint_mass_flow_balance
    for b in 1:nbatch
        for i in junctions

            # -----------------------------
            # RHS: constants only
            # -----------------------------
            const_inj = haskey(ref[:nondispatchable_receipts_in_junction], i) ?
                        sum(ref[:receipt][r]["injection_nominal"]
                            for r in ref[:nondispatchable_receipts_in_junction][i]; init=0.0) : 0.0

            const_with = haskey(ref[:nondispatchable_deliveries_in_junction], i) ?
                         sum(ref[:delivery][d]["withdrawal_nominal"] #* β[d, b]
                             for d in ref[:nondispatchable_deliveries_in_junction][i]; init=0.0) : 0.0

            const_trans = haskey(ref[:nondispatchable_transfers_in_junction], i) ?
                          sum(ref[:transfer][t]["withdrawal_nominal"]
                              for t in ref[:nondispatchable_transfers_in_junction][i]; init=0.0) : 0.0

            rhs = const_inj - const_with - const_trans
            # NOTE: β is intentionally NOT used here


            # -----------------------------
            # Dispatchable term D(i,b)
            # -----------------------------
            disp =
                sum(model[:fg][r, b] for r in get(ref[:dispatchable_receipts_in_junction], i, []); init=0.0) -
                sum(model[:fl][d, b] for d in get(ref[:dispatchable_deliveries_in_junction], i, []); init=0.0) -
                sum(model[:ft][t, b] for t in get(ref[:dispatchable_transfers_in_junction], i, []); init=0.0) -
                sum(model[:well_head_flow][s, b] for s in get(ref[:storages_in_junction], i, []); init=0.0)


            # -----------------------------
            # Network term N(i,b)
            # EXACT SAME SIGNS AS LRWP
            # -----------------------------
            net =
                sum(model[:f_pipe][p, b] for p in get(ref[:pipes_fr], i, []); init=0.0) -
                sum(model[:f_pipe][p, b] for p in get(ref[:pipes_to], i, []); init=0.0) +
                sum(model[:f_compressor][c, b] for c in get(ref[:compressors_fr], i, []); init=0.0) -
                sum(model[:f_compressor][c, b] for c in get(ref[:compressors_to], i, []); init=0.0) +
                sum(model[:f_resistor][r, b] for r in get(ref[:resistors_fr], i, []); init=0.0) -
                sum(model[:f_resistor][r, b] for r in get(ref[:resistors_to], i, []); init=0.0) +
                sum(model[:f_loss_resistor][lr, b] for lr in get(ref[:loss_resistors_fr], i, []); init=0.0) -
                sum(model[:f_loss_resistor][lr, b] for lr in get(ref[:loss_resistors_to], i, []); init=0.0) +
                sum(model[:f_short_pipe][sp, b] for sp in get(ref[:short_pipes_fr], i, []); init=0.0) -
                sum(model[:f_short_pipe][sp, b] for sp in get(ref[:short_pipes_to], i, []); init=0.0) +
                sum(model[:f_valve][v, b] for v in get(ref[:valves_fr], i, []); init=0.0) -
                sum(model[:f_valve][v, b] for v in get(ref[:valves_to], i, []); init=0.0) +
                sum(model[:f_regulator][rg, b] for rg in get(ref[:regulators_fr], i, []); init=0.0) -
                sum(model[:f_regulator][rg, b] for rg in get(ref[:regulators_to], i, []); init=0.0)


            # -----------------------------
            # FINAL BALANCE (unchanged)
            # -----------------------------
            @constraint(model, net - disp == rhs)
        end
    end


    # constraint_pressure
    for b in 1:nbatch, (i, d) in ref[:junction]
        if get(d, "junction_type", 0) == 1 && haskey(d, "p_nominal")
            @constraint(model, model[:psqr][i, b] == d["p_nominal"]^2)
        end
    end


    # constraint_pipe_pressure 
    for b in 1:nbatch
        for (k, pipe) in ref[:pipe]
            i = pipe["fr_junction"]
            j = pipe["to_junction"]

            i_junction = ref[:junction][i]
            j_junction = ref[:junction][j]

            pd_min, pd_max = GasModels._calc_pipe_pd_bounds_sqr(pipe, i_junction, j_junction)

            @constraint(model, pd_min <= model[:psqr][i, b] - model[:psqr][j, b])
            @constraint(model, model[:psqr][i, b] - model[:psqr][j, b] <= pd_max)
        end
    end

    # constraint_pipe_mass_flow
    for b in 1:nbatch
        for (k, pipe) in ref[:pipe]
            f = model[:f_pipe][k, b]

            f_min = pipe["flow_min"]
            f_max = pipe["flow_max"]

            lb = JuMP.has_lower_bound(f) ? max(JuMP.lower_bound(f), f_min) : f_min
            ub = JuMP.has_upper_bound(f) ? min(JuMP.upper_bound(f), f_max) : f_max

            JuMP.set_lower_bound(f, lb)
            JuMP.set_upper_bound(f, ub)
        end
    end



    # Storage dictionary for the main LRWP pipe constraint (with beta)
    con_fmf_pipe = Dict{Tuple{Int,Int},JuMP.ConstraintRef}()

    # constraint_pipe_weymouth
    for b in 1:nbatch
        # constraint_pipe_weymouth
        for k in pipes
            pipe = ref[:pipe][k]
            i = pipe["fr_junction"]
            j = pipe["to_junction"]
            θ = pipe["theta"]
            D = pipe["diameter"]

            # === 1. GasModels: skip pipes with diameter == 0 ===
            if D == 0.0
                continue  # GasModels does not add constraints here
            end

            # === 2. Horizontal pipe? (|θ| ≤ 5°) ===
            if abs(rad2deg(θ)) ≤ 5
                # compute resistance w exactly like GasModels
                w = GasModels._calc_pipe_resistance(
                    pipe,
                    data["base_length"],
                    data["base_pressure"],
                    data["base_flow"],
                    data["sound_speed"],
                )

                fmin = pipe["flow_min"]
                fmax = pipe["flow_max"]

                pi = model[:psqr][i, b]
                pj = model[:psqr][j, b]
                f = model[:f_pipe][k, b]
                fmf_l = model[:fmf_l_pipe][k, b]

                # === Special case: fmin == fmax (fixed flow) ===
                if fmin == fmax
                    @constraint(model, w * (pi - pj) == fmin)

                    # === Normal LRWP case ===
                else
                    # Weymouth linearized form
                    @constraint(model, w * (pi - pj) == fmf_l)

                    # --- Partitioning exactly like GasModels ---
                    partition = (fmin < 0 && fmax > 0) ? [fmin, 0.0, fmax] : [fmin, fmax]
                    # println("number of partition", partition)

                    # Univariate function f(a) = a*|a|
                    ffun(a) = a * abs(a)
                    f_dash(x) = ForwardDiff.derivative(ffun, x)

                    # Build data for PolyhedralRelaxations
                    ufd = PolyhedralRelaxations.UnivariateFunctionData(
                        ffun,
                        f_dash,
                        partition,
                        NaN,
                        1e-6,
                        1e-6,
                        0,
                        length(partition),
                    )

                    # Vertices of convex relaxation
                    vertices = PolyhedralRelaxations._get_lp_relaxation_vertices(ufd)
                    N = length(vertices)

                    # λ-variables for convex combination
                    λ = @variable(model, [1:N], lower_bound = 0.0, upper_bound = 1.0,
                        base_name = "lambda_pipe_$k")

                    # Sum λ_j = 1
                    @constraint(model, sum(λ[j] for j in 1:N) == 1)

                    # f_pipe approximation
                    @constraint(model, f == sum(λ[j] * vertices[j][1] for j in 1:N))

                    # fmf_l_pipe approximation
                    @constraint(model, fmf_l == sum(λ[j] * vertices[j][2] for j in 1:N))
                end

                # === 3. Inclined pipe (|θ| > 5°) ===
            else
                # GasModels uses "inclined pipe pressure drop":
                # constraint_inclined_pipe_pressure_drop(gm, ...)

                # You MUST compute the two resistances r1,r2:
                r1, r2 = GasModels._calc_inclined_pipe_resistance(
                    pipe,
                    data["base_length"],
                    data["base_pressure"],
                    data["base_flow"],
                    data["sound_speed"],
                )

                # And pressure-drop bounds (not strictly required)
                # but included for full equivalence:
                # (you may skip if not needed)
                # pd_bounds = GasModels._calc_inclined_pipe_pd_bounds_sqr(...)

                # TODO: Implement inclined-pipe LRWP constraint
                # For now we mimic GasModels structure with a placeholder:
                @warn "Inclined pipe k=$k detected (|theta| > 5°). Must implement inclined LRWP constraint."

                # Minimal placeholder to avoid crashing:
                @constraint(model, model[:psqr][i, b] - model[:psqr][j, b] == 0)
            end
        end
    end






    # constraint_resistor_mass_flow
    for b in 1:nbatch
        for (k, pipe) in ref[:pipe]
            f = model[:f_pipe][k, b]

            f_min = pipe["flow_min"]
            f_max = pipe["flow_max"]

            lb = JuMP.has_lower_bound(f) ? max(JuMP.lower_bound(f), f_min) : f_min
            ub = JuMP.has_upper_bound(f) ? min(JuMP.upper_bound(f), f_max) : f_max

            JuMP.set_lower_bound(f, lb)
            JuMP.set_upper_bound(f, ub)
        end
    end

    # constraint_short_pipe_pressure 
    for b in 1:nbatch
        for (k, short_pipe) in ref[:short_pipe]
            i = short_pipe["fr_junction"]
            j = short_pipe["to_junction"]

            @constraint(model, model[:psqr][i, b] == model[:psqr][j, b])
        end
    end

    # constraint_compressor_ratios
    for b in 1:nbatch
        for (k, compressor) in ref[:compressor]

            # === Node indices ===
            i = compressor["fr_junction"]
            j = compressor["to_junction"]

            # === Variables ===
            pi = model[:psqr][i, b]
            pj = model[:psqr][j, b]
            f = model[:f_compressor][k, b]

            # === Ratios and directionality ===
            min_ratio = compressor["c_ratio_min"]
            max_ratio = compressor["c_ratio_max"]
            type = get(compressor, "directionality", 0)

            # === Junction limits ===
            i_pmax = ref[:junction][i]["p_max"]
            j_pmax = ref[:junction][j]["p_max"]
            i_pmin = ref[:junction][i]["p_min"]
            j_pmin = ref[:junction][j]["p_min"]

            # ======================================================================================
            # TYPE 0 : BIDIRECTIONAL COMPRESSION
            # ======================================================================================
            if type == 0

                if (min_ratio <= 1.0 && max_ratio >= 1)
                    # === pk auxiliary pressure level ===
                    pk = @variable(model,
                        lower_bound = min(JuMP.lower_bound(pi), JuMP.lower_bound(pj)),
                        upper_bound = max(JuMP.upper_bound(pi), JuMP.upper_bound(pj))
                    )

                    # === pressure differences ===
                    pik = @variable(model,
                        lower_bound = JuMP.lower_bound(pi) - JuMP.upper_bound(pk),
                        upper_bound = JuMP.upper_bound(pi) - JuMP.lower_bound(pk)
                    )

                    pjk = @variable(model,
                        lower_bound = JuMP.lower_bound(pj) - JuMP.upper_bound(pk),
                        upper_bound = JuMP.upper_bound(pj) - JuMP.lower_bound(pk)
                    )

                    fpik = @variable(model)
                    fpjk = @variable(model)

                    @constraint(model, pk - max_ratio^2 * pi <= 0)
                    @constraint(model, min_ratio^2 * pi - pk <= 0)
                    @constraint(model, fpik <= 0)

                    @constraint(model, pk - max_ratio^2 * pj <= 0)
                    @constraint(model, min_ratio^2 * pj - pk <= 0)
                    @constraint(model, -fpjk <= 0)

                    @constraint(model, pjk == pj - pk)
                    @constraint(model, pik == pi - pk)

                    relaxation_product(model, f, pik, fpik)
                    relaxation_product(model, f, pjk, fpjk)

                else
                    # === Disjunction with binary y ===
                    y = get_compressor_y(gm, n, k)

                    @constraint(model, pj - max_ratio^2 * pi <= (1 - y) * (j_pmax^2))
                    @constraint(model, min_ratio^2 * pi - pj <= (1 - y) * (min_ratio^2 * i_pmax^2))
                    @constraint(model, pi - max_ratio^2 * pj <= y * (i_pmax^2))
                    @constraint(model, min_ratio^2 * pj - pi <= y * (min_ratio^2 * j_pmax^2))
                end


                # ======================================================================================
                # TYPE 1 : ONE-DIRECTIONAL (i → j)
                # ======================================================================================
            elseif type == 1

                @constraint(model, pj - max_ratio^2 * pi <= 0)
                @constraint(model, min_ratio^2 * pi - pj <= 0)


                # ======================================================================================
                # TYPE 2 : SEMI-DIRECTIONAL
                # ======================================================================================
            else

                if min_ratio == 1
                    pij = @variable(model,
                        lower_bound = JuMP.lower_bound(pi) - JuMP.upper_bound(pj),
                        upper_bound = JuMP.upper_bound(pi) - JuMP.lower_bound(pj)
                    )
                    fpij = @variable(model)

                    @constraint(model, pj - max_ratio^2 * pi <= 0)
                    @constraint(model, min_ratio^2 * pi - pj <= 0)
                    @constraint(model, fpij <= 0)
                    @constraint(model, pij == pi - pj)

                    relaxation_product(model, f, pij, fpij)

                else
                    # === DISJUNCTION (flow can reverse) ===
                    y = get_compressor_y(gm, n, k)

                    @constraint(model, pj - max_ratio^2 * pi <= (1 - y) * (j_pmax^2))
                    @constraint(model, min_ratio^2 * pi - pj <= (1 - y) * (i_pmax^2))
                    @constraint(model, pi - pj <= y * (i_pmax^2))
                    @constraint(model, pj - pi <= y * (j_pmax^2))
                end
            end
        end
    end

    # constraint_on_off_regulator_pressure
    for b in 1:nbatch
        for (k, reg) in ref[:regulator]
            i = reg["fr_junction"]
            j = reg["to_junction"]
            min_ratio = reg["r_ratio_min"]
            max_ratio = reg["r_ratio_max"]
            f_min = get(reg, "f_min", 0.0)

            psqr_i = model[:psqr][i, b]
            psqr_j = model[:psqr][j, b]
            v_reg = model[:v_regulator][k, b]
            f_reg = model[:f_regulator][k, b]

            i_pmax = upper_bound(psqr_i)
            j_pmax = upper_bound(psqr_j)
            i_pmin = lower_bound(psqr_i)
            j_pmin = lower_bound(psqr_j)

            if max_ratio == 1
                # === Case 1: Fixed-ratio regulation (max_ratio = 1) ===
                pij = @variable(
                    model,
                    upper_bound = upper_bound(psqr_i) - lower_bound(psqr_j),
                    lower_bound = lower_bound(psqr_i) - upper_bound(psqr_j),
                    base_name = "pij_reg_$(k)_b$(b)"
                )
                fpij = @variable(model, base_name = "fpij_reg_$(k)_b$(b)")

                @constraint(model, psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2)
                @constraint(model, min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2))
                @constraint(model, fpij >= 0)
                @constraint(model, pij == psqr_i - psqr_j)

                # Bilinear relaxation
                relaxation_product(model, f_reg, pij, fpij)

            elseif f_min >= 0
                # === Case 2: Forward flow only (f_min ≥ 0) ===
                @constraint(model, psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2)
                @constraint(model, min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2))

            else
                # === Case 3: Disjunctive on/off (flow can reverse) ===
                y = @variable(model, binary = true, base_name = "y_reg_$(k)_b$(b)")

                @constraint(model, psqr_j - (max_ratio^2 * psqr_i) <= (2 - y - v_reg) * j_pmax^2)
                @constraint(model, (min_ratio^2 * psqr_i) - psqr_j <= (2 - y - v_reg) * i_pmax^2)
                @constraint(model, psqr_j - psqr_i <= (1 + y - v_reg) * j_pmax^2)
                @constraint(model, psqr_i - psqr_j <= (1 + y - v_reg) * i_pmax^2)
            end
        end
    end





    # constraint_compressor_ratio_value  
    for b in 1:nbatch
        for (k, compressor) in ref[:compressor]

            # === Indices ===
            i = compressor["fr_junction"]
            j = compressor["to_junction"]

            # === Variables ===
            pi = model[:psqr][i, b]
            pj = model[:psqr][j, b]
            r = model[:rsqr][k, b]

            # === Pressure bounds ===
            i_pmax = ref[:junction][i]["p_max"]
            j_pmax = ref[:junction][j]["p_max"]

            # === Directionality and ratios ===
            type = get(compressor, "directionality", 0)
            max_ratio = compressor["c_ratio_max"]
            min_ratio = compressor["c_ratio_min"]

            if type == 0
                # === Bidirectional compression (needs binary y) ===

                # → Use GasModels helper to ensure SAME y across all constraints
                y = @variable(model, binary = true)


                # Auxiliary variables
                rpi = @variable(model)
                rpj = @variable(model)

                # McCormick relaxations
                relaxation_product(model, pi, r, rpi)
                relaxation_product(model, pj, r, rpj)

                # GasModels-compliant inequalities
                @constraint(model, rpi <= pj + (1 - y) * i_pmax^2 * max_ratio^2)
                @constraint(model, rpi >= pj - (1 - y) * j_pmax^2)
                @constraint(model, rpj <= pi + y * j_pmax^2 * max_ratio^2)
                @constraint(model, rpj >= pi - y * i_pmax^2)

            else
                # → Unidirectional or semi-directional: simple product
                relaxation_product(model, pi, r, pj)
            end
        end


        # constraint_on_off_valve_mass_flow
        for b in 1:nbatch
            for (k, valve) in ref[:valve]
                # Retrieve the JuMP variables for this valve and batch
                f = model[:f_valve][k, b]      # mass flow through the valve
                v = model[:v_valve][k, b]      # binary/opening variable (1 = open, 0 = closed)

                # Extract flow limits from valve data
                f_min = valve["flow_min"]
                f_max = valve["flow_max"]

                # Add the on/off flow constraints for this batch
                @constraint(model, f_min * v <= f)
                @constraint(model, f <= f_max * v)
            end
        end

        # constraint_on_off_valve_pressure 
        for b in 1:nbatch
            for (k, valve) in ref[:valve]
                i = valve["fr_junction"]
                j = valve["to_junction"]

                pi = model[:psqr][i, b]
                pj = model[:psqr][j, b]
                v = model[:v_valve][k, b]

                i_pmax = ref[:junction][i]["p_max"]
                j_pmax = ref[:junction][j]["p_max"]

                @constraint(model, pj - ((1 - v) * j_pmax^2) <= pi)
                @constraint(model, pi <= pj + ((1 - v) * i_pmax^2))
            end
        end

        # constraint_on_off_regulator_mass_flow 
        for b in 1:nbatch
            for (k, regulator) in ref[:regulator]
                f = model[:f_regulator][k, b]
                v = model[:v_regulator][k, b]

                f_min = regulator["flow_min"]
                f_max = regulator["flow_max"]

                @constraint(model, f_min * v <= f)
                @constraint(model, f <= f_max * v)
            end
        end

        # constraint_on_off_regulator_pressure
        for (k, reg) in ref[:regulator]
            # === Indices and parameters ===
            i = reg["fr_junction"]
            j = reg["to_junction"]

            # If your data uses "r_ratio_min/max" instead of "reduction_factor_*",
            # you can swap these field names back to "r_ratio_min"/"r_ratio_max".
            min_ratio = reg["reduction_factor_min"]
            max_ratio = reg["reduction_factor_max"]

            f_min = reg["flow_min"]

            # Junction pressure limits (in base units, not squared yet)
            i_pmin = ref[:junction][i]["p_min"]
            i_pmax = ref[:junction][i]["p_max"]
            j_pmin = ref[:junction][j]["p_min"]
            j_pmax = ref[:junction][j]["p_max"]

            # === Variables ===
            psqr_i = model[:psqr][i]
            psqr_j = model[:psqr][j]
            v_reg = model[:v_regulator][k]
            f_reg = model[:f_regulator][k]

            if max_ratio == 1
                # === Case 1: Fixed-ratio regulation (max_ratio = 1) ===
                pij = @variable(
                    model,
                    lower_bound = JuMP.lower_bound(psqr_i) - JuMP.upper_bound(psqr_j),
                    upper_bound = JuMP.upper_bound(psqr_i) - JuMP.lower_bound(psqr_j),
                )
                fpij = @variable(model)

                @constraint(model,
                    psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2
                )
                @constraint(model,
                    min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2)
                )
                @constraint(model, fpij >= 0)
                @constraint(model, pij == psqr_i - psqr_j)

                # Bilinear relaxation product f_reg * (psqr_i - psqr_j)
                relaxation_product(model, f_reg, pij, fpij)

            elseif f_min >= 0
                # === Case 2: Forward flow only (f_min ≥ 0) ===
                @constraint(model,
                    psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2
                )
                @constraint(model,
                    min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2)
                )

            else
                # === Case 3: Disjunctive on/off behavior (flow can reverse) ===
                y = @variable(model, binary = true)

                @constraint(model,
                    psqr_j - (max_ratio^2 * psqr_i) <= (2 - y - v_reg) * j_pmax^2
                )
                @constraint(model,
                    (min_ratio^2 * psqr_i) - psqr_j <= (2 - y - v_reg) * i_pmax^2
                )
                @constraint(model,
                    psqr_j - psqr_i <= (1 + y - v_reg) * j_pmax^2
                )
                @constraint(model,
                    psqr_i - psqr_j <= (1 + y - v_reg) * i_pmax^2
                )
            end
        end
    end
    # ------------------------------------------------------------------ #
    # 4. Objetivo económico (simplificado)
    # ------------------------------------------------------------------ #
    gamma = get_specific_heat_capacity_ratio(data)
    m = ((gamma - 1) / gamma) / 2
    T = get_temperature(data)
    G = get_gas_specific_gravity(data)
    work = ((286.76 / G) * T * (gamma / (gamma - 1)))
    baseF = get_base_flow(data)
    econW = get_economic_weighting(data)

    load_prices = Dict(k => get(ref[:delivery][k], "bid_price", 1.0)
                       for k in dispatchable_deliveries)
    prod_prices = Dict(k => get(ref[:receipt][k], "offer_price", 1.0)
                       for k in receipts if ref[:receipt][k]["is_dispatchable"] == 1)
    transf_prices = Dict(k => ref[:transfer][k]["withdrawal_min"] ≥ 0 ?
                              get(ref[:transfer][k], "bid_price", 1.0) :
                              -get(ref[:transfer][k], "offer_price", 1.0)
                         for k in transfers
                         if ref[:transfer][k]["is_dispatchable"] == 1)

    # println(dispatchable_deliveries)

    # Objective variable per batch
    @objective(
        model,
        Min,
        sum(1 / nbatch * (
                -sum(load_prices[k] * model[:fl][k, b] for k in keys(load_prices)) -
                sum(transf_prices[k] * model[:ft][k, b] for k in keys(transf_prices)) +
                sum(prod_prices[k] * model[:fg][k, b] for k in keys(prod_prices))
            )
            +
            # #! Use for training Ipopt
            sum(β[p, b] * model[:psqr][p, b]^2 for p in junctions, b in 1:nbatch)
            +
            sum(α[p, b] * model[:f_pipe][p, b]^2 for p in pipes, b in 1:nbatch)
            #  #! Use for inference Gurobi
            # sum(β[junction_idx_map[p], b] * model[:psqr][p, b]^2 for p in junctions)
            # +
            # sum(α[pipe_idx_map[p], b] * model[:f_pipe][p, b]^2 for p in pipes)


            for b in 1:nbatch)
    )

    model[:con_fmf_pipe] = con_fmf_pipe
    set_silent(model)

    # ------------------------------------------------------------------ #
    # 5. Optimiza y retorna
    # ------------------------------------------------------------------ #
    optimize!(model)
    # dual_value = JuMP.dual.(con_fmf_pipe)


    println("value of LF", objective_value(model))
    println("Status: ", termination_status(model))
    # println("value slack LF", 1e6 * sum(value.(model[:slack_pos][j, b]) for j in junctions, b in 1:nbatch))
    # return value.(model[:fg]),value.(model[:ft])
    return value.(model[:fg])
end

###############################################################
#! Funcion OptLayer for HF 
###############################################################
function WPLayer_GM(
    gw::GraphWithRef, fg_target, val_init;
    solver=Ipopt.Optimizer
)

    ###############
    data = gw.data
    delivery_data = gw.graph.x[1, :]
    nbatch = gw.graph.num_graphs

    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    for (load_id, load_data) in data["delivery"]
        bus_id = load_data["junction_id"]
        idx = bus_idx_map[bus_id]
        load_data["withdrawal_nominal"] = delivery_data[idx]
        # println(idx)
    end

    spec = GasModels.instantiate_model(
        data,
        WPGasModel,
        GasModels.build_ogf_comp_power_unc
    )

    model = spec.model
    set_silent(model)

    ############################
    # REFS (same as WP)
    ############################
    junctions_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:junction])))
    receipt_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:receipt])))
    transfer_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:transfer])))

    compressor_ref = haskey(spec.ref[:it][:gm][:nw][0], :compressor) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:compressor]))) : Int[]

    pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :pipe) ?
               sort(collect(keys(spec.ref[:it][:gm][:nw][0][:pipe]))) : Int[]

    short_pipe_ref = haskey(spec.ref[:it][:gm][:nw][0], :short_pipe) ?
                     sort(collect(keys(spec.ref[:it][:gm][:nw][0][:short_pipe]))) : Int[]

    resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :resistor) ?
                   sort(collect(keys(spec.ref[:it][:gm][:nw][0][:resistor]))) : Int[]

    loss_resistor_ref = haskey(spec.ref[:it][:gm][:nw][0], :loss_resistor) ?
                        sort(collect(keys(spec.ref[:it][:gm][:nw][0][:loss_resistor]))) : Int[]

    valve_ref = haskey(spec.ref[:it][:gm][:nw][0], :valve) ?
                sort(collect(keys(spec.ref[:it][:gm][:nw][0][:valve]))) : Int[]

    regulator_ref = haskey(spec.ref[:it][:gm][:nw][0], :regulator) ?
                    sort(collect(keys(spec.ref[:it][:gm][:nw][0][:regulator]))) : Int[]

    ############################
    # VARS
    ############################
    fg_vars = spec.var[:it][:gm][:nw][0][:fg]
    ft_vars = spec.var[:it][:gm][:nw][0][:ft]

    f_pipe_vars = spec.var[:it][:gm][:nw][0][:f_pipe]
    f_short_pipe_vars = spec.var[:it][:gm][:nw][0][:f_short_pipe]
    f_compressor_vars = spec.var[:it][:gm][:nw][0][:f_compressor]
    f_resistor_vars = spec.var[:it][:gm][:nw][0][:f_resistor]
    f_loss_resistor_vars = spec.var[:it][:gm][:nw][0][:f_loss_resistor]
    f_valve_vars = spec.var[:it][:gm][:nw][0][:f_valve]
    f_regulator_vars = spec.var[:it][:gm][:nw][0][:f_regulator]
    fl_vars = spec.var[:it][:gm][:nw][0][:fl]

    p_vars = spec.var[:it][:gm][:nw][0][:p]
    psqr_vars = spec.var[:it][:gm][:nw][0][:psqr]
    rsqr_vars = spec.var[:it][:gm][:nw][0][:rsqr]

    v_valve_vars = spec.var[:it][:gm][:nw][0][:v_valve]
    v_regulator_vars = spec.var[:it][:gm][:nw][0][:v_regulator]

    well_head_flow_vars = spec.var[:it][:gm][:nw][0][:well_head_flow]

    ############################
    # WARM START (batch = 1)
    ############################
    b = 1

    if val_init !== nothing

        # fg, ft
        for (i, k) in enumerate(receipt_ref)
            JuMP.set_start_value(fg_vars[k, b], val_init["fg"][i])
        end
        for (i, k) in enumerate(transfer_ref)
            JuMP.set_start_value(ft_vars[k, b], val_init["ft"][i])
        end

        # pipes & components
        for (i, k) in enumerate(pipe_ref)
            JuMP.set_start_value(f_pipe_vars[k, b], val_init["f_pipe"][i])
        end
        for (i, k) in enumerate(short_pipe_ref)
            JuMP.set_start_value(f_short_pipe_vars[k, b], val_init["f_short_pipe"][i])
        end
        for (i, k) in enumerate(compressor_ref)
            JuMP.set_start_value(f_compressor_vars[k, b], val_init["f_compressor"][i])
            JuMP.set_start_value(rsqr_vars[k, b], val_init["rsqr"][i])
        end
        for (i, k) in enumerate(resistor_ref)
            JuMP.set_start_value(f_resistor_vars[k, b], val_init["f_resistor"][i])
        end
        for (i, k) in enumerate(loss_resistor_ref)
            JuMP.set_start_value(f_loss_resistor_vars[k, b], val_init["f_loss_resistor"][i])
        end

        # valves / regulators
        for (i, k) in enumerate(valve_ref)
            JuMP.set_start_value(f_valve_vars[k, b], val_init["f_valve"][i])
            JuMP.set_start_value(v_valve_vars[k], val_init["v_valve"][i])
        end
        for (i, k) in enumerate(regulator_ref)
            JuMP.set_start_value(f_regulator_vars[k, b], val_init["f_regulator"][i])
            JuMP.set_start_value(v_regulator_vars[k], val_init["v_regulator"][i])
        end

        # pressures (skip junction 0)

    end

    ############################
    # ORIGINAL MODEL LOGIC
    ############################
    @variables(model, begin
        slack_fg1[i in receipt_ref] >= 0
        slack_fg2[i in receipt_ref] >= 0
    end)

    receipt_index_map = Dict(k => idx for (idx, k) in enumerate(receipt_ref))
    # println("receipt_index_map", receipt_index_map)
    fg_target_array = Array(fg_target)

    @constraint(model, aux_fg[k in receipt_ref],
        fg_vars[k, b] ==
        fg_target_array[receipt_index_map[k]]
        +
        slack_fg1[k] -
        slack_fg2[k]
    )

    deliveries = spec.ref[:it][:gm][:nw][0][:delivery]
    receipts = spec.ref[:it][:gm][:nw][0][:receipt]
    transfers = spec.ref[:it][:gm][:nw][0][:transfer]

    load_prices = Dict(k => get(deliveries[k], "bid_price", 1.0)
                       for k in keys(deliveries))

    prod_prices = Dict(
        k => get(receipts[k], "offer_price", 1.0)
        for k in receipt_ref
        if receipts[k]["is_dispatchable"] == 1
    )

    transf_prices = Dict(
        k => (transfers[k]["withdrawal_min"] ≥ 0 ?
              get(transfers[k], "bid_price", 1.0) :
              -get(transfers[k], "offer_price", 1.0))
        for k in transfer_ref
        if transfers[k]["is_dispatchable"] == 1
    )
    max_prod_prices = maximum(prod_prices[k] for k in keys(prod_prices))
    @objective(model, Min,
        # max_prod_prices * sum((slack_fg1[k] + slack_fg2[k]) for k in receipt_ref) +
        # max_prod_prices * sum((slack_fg1[k]^2 + slack_fg2[k]^2) for k in receipt_ref) +
        #! check the constant coef
        sum(prod_prices[k] * (slack_fg2[k] + slack_fg1[k])^2 for k in keys(prod_prices))
        # +
        # sum( prod_prices[k] *(slack_fg2[k]^2 + slack_fg1[k]^2) for k in keys(prod_prices))
        # +
        # sum(prod_prices[k] * (slack_fg2[k] + slack_fg1[k]) for k in keys(prod_prices))
        -
        sum(transf_prices[k] * ft_vars[k, b] for k in keys(transf_prices)) +
        sum(prod_prices[k] * fg_vars[k, b] for k in keys(prod_prices))
    )

    JuMP.set_optimizer(model, () -> solver())
    optimize!(model)
    # println("Value of sum slacks", sum(JuMP.value.(slack_fg1[k]) + JuMP.value.(slack_fg2[k]) for k in receipt_ref))
    λ_fg = JuMP.dual.(model[:aux_fg])


    println("objective HF", JuMP.objective_value(model))


    cost_hf = (#- sum(transf_prices[k] * JuMP.value.(ft_vars[k, b]) for k in keys(transf_prices)) 
        +sum(prod_prices[k] * JuMP.value.(fg_vars[k, b]) for k in keys(prod_prices)))


    cost_lf = (#- sum(transf_prices[k] * JuMP.value.(ft_vars[k, b]) for k in keys(transf_prices)) 
        +sum(prod_prices[k] * fg_target_array[receipt_index_map[k]] for k in keys(prod_prices)))

    slack_fg2_value = Array(JuMP.value.(slack_fg2))
    slack_fg1_value = Array(JuMP.value.(slack_fg1))

    max_slack_value = maximum(abs.(vcat(slack_fg2_value, slack_fg1_value))) * data["base_flow"] #[ks/s]
    # println("Cost of HF", cost_hf)

    # println("Cost of LF", cost_lf)

    sum_slack = sum(JuMP.value.(slack_fg1[k]) + JuMP.value.(slack_fg2[k])

                    for k in receipt_ref)


    # println("sum duals", λ_fg)
    return Dict(
        "flag" => (JuMP.termination_status(model)),
        "objective" => JuMP.objective_value(model),
        "cost_hf" => cost_hf,
        "cost_lf" => cost_lf,
        "fg" => [JuMP.value(fg_vars[k]) for k in receipt_ref],
        "ft" => Array(JuMP.value.(ft_vars[:, b])),
        "slack_fg1" => [JuMP.value(slack_fg1[k]) for k in receipt_ref],
        "slack_fg2" => [JuMP.value(slack_fg2[k]) for k in receipt_ref],
        "fg_target" => fg_target_array,
        "dual" => sum(λ_fg),
        "sum_slack" => sum_slack,
        "max_slack" => max_slack_value,
        "model" => model
    )
end
###############################################################
# Function to calculate the distance of feasibility
###############################################################
function Feasibility_GM(
    gw::GraphWithRef,
    fg_target,
    solver=Ipopt.Optimizer
)

    data = gw.data
    delivery_data = gw.graph.x[1, :]

    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    for (load_id, load_data) in data["delivery"]
        bus_id = load_data["junction_id"]
        idx = bus_idx_map[bus_id]
        load_data["withdrawal_nominal"] = delivery_data[idx]
    end

    spec = GasModels.instantiate_model(
        data,
        WPGasModel,
        GasModels.build_ogf_comp_power_unc
    )

    model = spec.model
    set_silent(model)

    receipt_ref = sort(collect(keys(spec.ref[:it][:gm][:nw][0][:receipt])))
    fg_vars = spec.var[:it][:gm][:nw][0][:fg]

    receipts = spec.ref[:it][:gm][:nw][0][:receipt]
    prod_prices = Dict(
        k => get(receipts[k], "offer_price", 1.0)
        for k in receipt_ref
        if receipts[k]["is_dispatchable"] == 1
    )

    b = 1
    fg_target_array = vec(Array(fg_target))
    receipt_index_map = Dict(k => idx for (idx, k) in enumerate(receipt_ref))

    @objective(model, Min,
        1 / length(fg_target_array) *
        sum((fg_vars[k, b] - fg_target_array[receipt_index_map[k]])^2 for k in keys(prod_prices))
    )

    JuMP.set_optimizer(model, () -> solver())

    optimize!(model)
    println("Objective value: ", JuMP.objective_value(model))
    println("termination_status: ", JuMP.termination_status(model))

    return JuMP.objective_value(model)
end

# Extra if need just LF
###############################################################
function LRWPLayer_GM_truncated(
    gw::GraphWithRef,
    rhs::Matrix;
    model=direct_model(DiffOpt.diff_optimizer(Gurobi.Optimizer))
)

    data = gw.data
    # withdrawal_per_junction = gw.graph.x[1, :]   # size = N_junctions
    # nbatch = gw.graph.num_graphs

    #! re check something wrong because if you analize the idx it gives you something strrange (the order is wrong)



    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    compressors = sort(collect(keys(ref[:compressor])))
    junctions = sort(collect(keys(ref[:junction])))
    loss_resistors = sort(collect(keys(ref[:loss_resistor])))
    pipes = sort(collect(keys(ref[:pipe])))
    receipts = sort(collect(keys(ref[:receipt])))
    regulators = sort(collect(keys(ref[:regulator])))
    resistors = sort(collect(keys(ref[:resistor])))
    short_pipes = sort(collect(keys(ref[:short_pipe])))
    transfers = sort(collect(keys(ref[:transfer])))
    valves = sort(collect(keys(ref[:valve])))


    delivery_data = gw.graph.x[1, :]
    nbatch = gw.graph.num_graphs


    bus_ids = sort(parse.(Int, keys(data["junction"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    for (load_id, load_data) in data["delivery"]
        bus_id = load_data["junction_id"]
        idx = bus_idx_map[bus_id]
        load_data["withdrawal_nominal"] = delivery_data[idx]
    end

    # ------------------------------------------------------------------ #
    # 2. Variables (todas guardadas en model[:...])
    # ------------------------------------------------------------------ #

    pipes = sort(collect(keys(ref[:pipe])))        # e.g. [0,2,5,7,10]
    Npipe = length(pipes)

    pipe_idx_map = Dict(p => idx for (idx, p) in enumerate(pipes))
    junction_idx_map = Dict(p => idx for (idx, p) in enumerate(junctions))

    # println("rhs size = ", size(rhs))

    @variable(model, β[p in pipes, b in 1:nbatch] in Parameter(rhs[b, pipe_idx_map[p]]))
    # @variable(model, β[p in junctions, b in 1:nbatch] in Parameter(rhs[junction_idx_map[p],b]))
    # @variable(model, slack[i in junctions] >= 0)


    model[:f_compressor] = @variable(model,
        [k in compressors, b in 1:nbatch],
        lower_bound = ref[:compressor][k]["flow_min"],
        upper_bound = ref[:compressor][k]["flow_max"],
        start = get(ref[:compressor][k], "f_start", 0.0),
        base_name = "f_comp",
    )

    model[:f_resistor] = @variable(model,
        [k in resistors, b in 1:nbatch],
        lower_bound = ref[:resistor][k]["flow_min"],
        upper_bound = ref[:resistor][k]["flow_max"],
        start = get(ref[:resistor][k], "f_start", 1e-6),
        base_name = "f_res",
    )

    model[:f_loss_resistor] = @variable(model,
        [k in loss_resistors, b in 1:nbatch],
        lower_bound = ref[:loss_resistor][k]["flow_min"],
        upper_bound = ref[:loss_resistor][k]["flow_max"],
        start = get(ref[:loss_resistor][k], "f_start", 1e-6),
        base_name = "f_lres",
    )

    model[:f_pipe] = @variable(model,
        [k in pipes, b in 1:nbatch],
        lower_bound = ref[:pipe][k]["flow_min"],
        upper_bound = ref[:pipe][k]["flow_max"],
        start = get(ref[:pipe][k], "f_start", 0.0),
        base_name = "f_pipe",
    )

    model[:f_short_pipe] = @variable(model,
        [k in short_pipes, b in 1:nbatch],
        lower_bound = ref[:short_pipe][k]["flow_min"],
        upper_bound = ref[:short_pipe][k]["flow_max"],
        start = get(ref[:short_pipe][k], "f_start", 0.0),
        base_name = "f_spipe",
    )

    model[:f_valve] = @variable(model,
        [k in valves, b in 1:nbatch],
        lower_bound = min.(ref[:valve][k]["flow_min"] for k in valves),
        upper_bound = max.(ref[:valve][k]["flow_max"] for k in valves),
        start = get.(Ref(ref[:valve]), valves, Ref("f_start"), 0.0),
        base_name = "f_valve",
    )

    model[:f_regulator] = @variable(model,
        [k in regulators, b in 1:nbatch],
        lower_bound = min.(ref[:regulator][k]["flow_min"] for k in regulators),
        upper_bound = max.(ref[:regulator][k]["flow_max"] for k in regulators),
        start = get.(Ref(ref[:regulator]), regulators, Ref("f_start"), 0.0),
        base_name = "f_reg",
    )

    ## b) Inyecciones / retiros
    model[:ft] = @variable(model,
        [k in transfers, b in 1:nbatch],
        lower_bound = ref[:transfer][k]["withdrawal_min"],
        upper_bound = ref[:transfer][k]["withdrawal_max"],
        start = get(ref[:transfer][k], "ft_start",
            ref[:transfer][k]["withdrawal_max"]),
        base_name = "ft",
    )

    model[:fg] = @variable(model,
        [k in receipts, b in 1:nbatch],
        lower_bound = ref[:receipt][k]["injection_min"],
        upper_bound = ref[:receipt][k]["injection_max"],
        start = get(ref[:receipt][k], "fg_start",
            ref[:receipt][k]["injection_max"]),
        base_name = "fg",
    )

    dispatchable_deliveries =
        [k for k in keys(ref[:delivery]) if ref[:delivery][k]["is_dispatchable"] == 1]

    model[:fl] = @variable(model,
        [k in dispatchable_deliveries],
        lower_bound = ref[:delivery][k]["withdrawal_min"],
        upper_bound = ref[:delivery][k]["withdrawal_max"],
        start = get(ref[:delivery][k], "fl_start",
            ref[:delivery][k]["withdrawal_max"]),
        base_name = "fl",
    )

    ## c) Presiones
    model[:psqr] = @variable(model,
        [i in junctions, b in 1:nbatch],
        lower_bound = ref[:junction][i]["p_min"]^2,
        upper_bound = ref[:junction][i]["p_max"]^2,
        start = get(ref[:junction][i], "p_start",
            ref[:junction][i]["p_min"])^2,
        base_name = "psqr",
    )

    # no-square pressure donde sólo hay resistores / loss_resistores
    prs_junc = unique(vcat(
        [ref[:resistor][k]["fr_junction"] for k in resistors],
        [ref[:resistor][k]["to_junction"] for k in resistors],
        [ref[:loss_resistor][k]["fr_junction"] for k in loss_resistors],
        [ref[:loss_resistor][k]["to_junction"] for k in loss_resistors],
    ))
    model[:p_nonsq] = @variable(model,
        [i in prs_junc, b in 1:nbatch],
        lower_bound = ref[:junction][i]["p_min"],
        upper_bound = ref[:junction][i]["p_max"],
        start = get(ref[:junction][i], "p_start",
            ref[:junction][i]["p_max"]),
        base_name = "p_nsq",
    )

    ## d) Ratios de compresor
    model[:rsqr] = @variable(model,
        [k in compressors, b in 1:nbatch],
        lower_bound = ref[:compressor][k]["c_ratio_min"]^2,
        upper_bound = ref[:compressor][k]["c_ratio_max"]^2,
        start = get(ref[:compressor][k], "ratio_start", 1.0),
        base_name = "rsqr",
    )

    ## e) Auxiliar f |f|
    model[:fmf_l_pipe] = @variable(model, [k in pipes, b in 1:nbatch], base_name = "fmf_l")
    # model[:subts_psqr] = @variable(model, [k in pipes], base_name = "subts_psqr")

    ## f) Binarias de válvulas / reguladores
    model[:v_valve] = @variable(model, [k in valves], Bin,
        start = get.(Ref(ref[:valve]), valves, Ref("v_start"), 1))
    model[:v_regulator] = @variable(model, [k in regulators], Bin,
        start = get.(Ref(ref[:regulator]), regulators,
            Ref("v_start"), 1))

    @variable(model, slack_pos[j in junctions, b in 1:nbatch] >= 0)
    @variable(model, slack_neg[j in junctions, b in 1:nbatch] >= 0)


    # ------------------------------------------------------------------ #
    # 3. Restricciones clave
    # ------------------------------------------------------------------ #


    # constraint_mass_flow_balance
    for b in 1:nbatch
        for i in junctions

            # === Constants for nondispatchable injections/withdrawals ===
            const_inj = haskey(ref[:nondispatchable_receipts_in_junction], i) ?
                        sum(ref[:receipt][r]["injection_nominal"] for r in ref[:nondispatchable_receipts_in_junction][i]; init=0.0) : 0.0

            const_with = haskey(ref[:nondispatchable_deliveries_in_junction], i) ?
                         sum(ref[:delivery][d]["withdrawal_nominal"] for d in ref[:nondispatchable_deliveries_in_junction][i]; init=0.0) : 0.0

            # const_with = haskey(ref[:nondispatchable_deliveries_in_junction], i) ?
            #              sum(ref[:delivery][d]["withdrawal_nominal"] for d in ref[:nondispatchable_deliveries_in_junction][i]; init=0.0) : 0.0


            const_trans = haskey(ref[:nondispatchable_transfers_in_junction], i) ?
                          sum(ref[:transfer][t]["withdrawal_nominal"] for t in ref[:nondispatchable_transfers_in_junction][i]; init=0.0) : 0.0

            # === Left-hand side (injections - withdrawals) ===
            lhs = const_inj - (const_with + const_trans)
            lhs += sum(model[:fg][r, b] for r in get(ref[:dispatchable_receipts_in_junction], i, []); init=0.0)
            lhs -= sum(model[:fl][d, b] for d in get(ref[:dispatchable_deliveries_in_junction], i, []); init=0.0)
            lhs -= sum(model[:ft][t, b] for t in get(ref[:dispatchable_transfers_in_junction], i, []); init=0.0)
            lhs -= sum(model[:well_head_flow][s, b] for s in get(ref[:storages_in_junction], i, []); init=0.0)

            # === Right-hand side (flows in and out of components) ===
            rhs =
                sum(model[:f_pipe][p, b] for p in get(ref[:pipes_fr], i, []); init=0.0) -
                sum(model[:f_pipe][p, b] for p in get(ref[:pipes_to], i, []); init=0.0) +
                sum(model[:f_compressor][c, b] for c in get(ref[:compressors_fr], i, []); init=0.0) -
                sum(model[:f_compressor][c, b] for c in get(ref[:compressors_to], i, []); init=0.0) +
                sum(model[:f_resistor][r, b] for r in get(ref[:resistors_fr], i, []); init=0.0) -
                sum(model[:f_resistor][r, b] for r in get(ref[:resistors_to], i, []); init=0.0) +
                sum(model[:f_loss_resistor][lr, b] for lr in get(ref[:loss_resistors_fr], i, []); init=0.0) -
                sum(model[:f_loss_resistor][lr, b] for lr in get(ref[:loss_resistors_to], i, []); init=0.0) +
                sum(model[:f_short_pipe][sp, b] for sp in get(ref[:short_pipes_fr], i, []); init=0.0) -
                sum(model[:f_short_pipe][sp, b] for sp in get(ref[:short_pipes_to], i, []); init=0.0) +
                sum(model[:f_valve][v, b] for v in get(ref[:valves_fr], i, []); init=0.0) -
                sum(model[:f_valve][v, b] for v in get(ref[:valves_to], i, []); init=0.0) +
                sum(model[:f_regulator][rg, b] for rg in get(ref[:regulators_fr], i, []); init=0.0) -
                sum(model[:f_regulator][rg, b] for rg in get(ref[:regulators_to], i, []); init=0.0)

            # === Slack balance per junction and batch ===
            @constraint(model, lhs - rhs == -slack_neg[i, b])
        end
    end

    # constraint_pressure
    for b in 1:nbatch, (i, d) in ref[:junction]
        if get(d, "junction_type", 0) == 1 && haskey(d, "p_nominal")
            @constraint(model, model[:psqr][i, b] == d["p_nominal"]^2)
        end
    end


    # constraint_pipe_pressure 
    for b in 1:nbatch
        for (k, pipe) in ref[:pipe]
            i = pipe["fr_junction"]
            j = pipe["to_junction"]

            i_junction = ref[:junction][i]
            j_junction = ref[:junction][j]

            pd_min, pd_max = GasModels._calc_pipe_pd_bounds_sqr(pipe, i_junction, j_junction)

            @constraint(model, pd_min <= model[:psqr][i, b] - model[:psqr][j, b])
            @constraint(model, model[:psqr][i, b] - model[:psqr][j, b] <= pd_max)
        end
    end

    # constraint_pipe_mass_flow
    for b in 1:nbatch
        for (k, pipe) in ref[:pipe]
            f = model[:f_pipe][k, b]

            f_min = pipe["flow_min"]
            f_max = pipe["flow_max"]

            lb = JuMP.has_lower_bound(f) ? max(JuMP.lower_bound(f), f_min) : f_min
            ub = JuMP.has_upper_bound(f) ? min(JuMP.upper_bound(f), f_max) : f_max

            JuMP.set_lower_bound(f, lb)
            JuMP.set_upper_bound(f, ub)
        end
    end



    # Storage dictionary for the main LRWP pipe constraint (with beta)
    con_fmf_pipe = Dict{Tuple{Int,Int},JuMP.ConstraintRef}()

    # constraint_pipe_weymouth
    for b in 1:nbatch
        for k in pipes

            pipe = ref[:pipe][k]
            i = pipe["fr_junction"]
            j = pipe["to_junction"]

            θ = pipe["theta"]
            D = pipe["diameter"]

            # Skip diameter == 0 pipes (GasModels behavior)
            if D == 0.0
                continue
            end

            # Variables for this batch
            pi = model[:psqr][i, b]
            pj = model[:psqr][j, b]
            f = model[:f_pipe][k, b]
            fmf_l = model[:fmf_l_pipe][k, b]

            # Your beta term (assumed already defined as β[k,b])
            βkb = β[k, b]

            # Flow limits
            fmin = pipe["flow_min"]
            fmax = pipe["flow_max"]

            # === CASE 1: horizontal pipe (|theta| ≤ 5°) ===
            if abs(rad2deg(θ)) ≤ 5

                # GasModels pipe resistance
                w = GasModels._calc_pipe_resistance(
                    pipe,
                    data["base_length"],
                    data["base_pressure"],
                    data["base_flow"],
                    data["sound_speed"],
                )

                # Link pressure drop and f*|f| surrogate, including beta:
                #   β[k,b] * w * (pi - pj) = fmf_l
                con_fmf_pipe[(k, b)] = @constraint(
                    model,
                    βkb * w * (pi - pj) == fmf_l
                    # w * (pi - pj) == fmf_l
                )

                # === Polyhedral relaxation of fmf_l ≈ f*|f| ===

                # Partition like before / GasModels style
                partition = (fmin < 0 && fmax > 0) ? [fmin, 0.0, fmax] : [fmin, fmax]

                # f(a) = a*|a|
                ffun(a) = a * abs(a)
                f_dash(x) = ForwardDiff.derivative(ffun, x)

                # Build univariate function data
                ufd = PolyhedralRelaxations.UnivariateFunctionData(
                    ffun,
                    f_dash,
                    partition,
                    NaN,
                    1e-6,
                    1e-6,
                    0,
                    length(partition),
                )

                # Get relaxation vertices (x_j, f(x_j))
                vertices = PolyhedralRelaxations._get_lp_relaxation_vertices(ufd)
                N = length(vertices)

                # Lambda variables for convex combination (one set per (k,b))
                λ = @variable(
                    model,
                    [1:N],
                    lower_bound = 0.0,
                    upper_bound = 1.0,
                    base_name = "lambda_$(k)_b$(b)",
                )

                # Convex combination constraints
                @constraint(model, sum(λ[j] for j in 1:N) == 1)

                # f_pipe[k,b] = Σ λ_j * x_j
                @constraint(
                    model,
                    f == sum(λ[j] * vertices[j][1] for j in 1:N)
                )

                # fmf_l_pipe[k,b] = Σ λ_j * f(x_j)
                @constraint(
                    model,
                    fmf_l == sum(λ[j] * vertices[j][2] for j in 1:N)
                )

            else
                # === CASE 2: inclined pipe (|theta| > 5°) ===
                r1, r2 = GasModels._calc_inclined_pipe_resistance(
                    pipe,
                    data["base_length"],
                    data["base_pressure"],
                    data["base_flow"],
                    data["sound_speed"],
                )

                @warn "Inclined pipe k=$k, batch=$b detected. LRWP inclined-pipe constraint not implemented; using placeholder."

                # Minimal placeholder to keep model feasible; no beta here, just like your
                # original "psqr[i] - psqr[j] == 0" when w == 0.
                con_fmf_pipe[(k, b)] = @constraint(model, pi - pj == 0)
            end
        end
    end






    # constraint_resistor_mass_flow
    for b in 1:nbatch
        for (k, pipe) in ref[:pipe]
            f = model[:f_pipe][k, b]

            f_min = pipe["flow_min"]
            f_max = pipe["flow_max"]

            lb = JuMP.has_lower_bound(f) ? max(JuMP.lower_bound(f), f_min) : f_min
            ub = JuMP.has_upper_bound(f) ? min(JuMP.upper_bound(f), f_max) : f_max

            JuMP.set_lower_bound(f, lb)
            JuMP.set_upper_bound(f, ub)
        end
    end

    # constraint_short_pipe_pressure 
    for b in 1:nbatch
        for (k, short_pipe) in ref[:short_pipe]
            i = short_pipe["fr_junction"]
            j = short_pipe["to_junction"]

            @constraint(model, model[:psqr][i, b] == model[:psqr][j, b])
        end
    end

    # constraint_compressor_ratios
    for b in 1:nbatch
        for (k, compressor) in ref[:compressor]

            # === Node indices ===
            i = compressor["fr_junction"]
            j = compressor["to_junction"]

            # === Variables ===
            pi = model[:psqr][i, b]
            pj = model[:psqr][j, b]
            f = model[:f_compressor][k, b]

            # === Ratios and directionality ===
            min_ratio = compressor["c_ratio_min"]
            max_ratio = compressor["c_ratio_max"]
            type = get(compressor, "directionality", 0)

            # === Junction limits ===
            i_pmax = ref[:junction][i]["p_max"]
            j_pmax = ref[:junction][j]["p_max"]
            i_pmin = ref[:junction][i]["p_min"]
            j_pmin = ref[:junction][j]["p_min"]

            # ======================================================================================
            # TYPE 0 : BIDIRECTIONAL COMPRESSION
            # ======================================================================================
            if type == 0

                if (min_ratio <= 1.0 && max_ratio >= 1)
                    # === pk auxiliary pressure level ===
                    pk = @variable(model,
                        lower_bound = min(JuMP.lower_bound(pi), JuMP.lower_bound(pj)),
                        upper_bound = max(JuMP.upper_bound(pi), JuMP.upper_bound(pj))
                    )

                    # === pressure differences ===
                    pik = @variable(model,
                        lower_bound = JuMP.lower_bound(pi) - JuMP.upper_bound(pk),
                        upper_bound = JuMP.upper_bound(pi) - JuMP.lower_bound(pk)
                    )

                    pjk = @variable(model,
                        lower_bound = JuMP.lower_bound(pj) - JuMP.upper_bound(pk),
                        upper_bound = JuMP.upper_bound(pj) - JuMP.lower_bound(pk)
                    )

                    fpik = @variable(model)
                    fpjk = @variable(model)

                    @constraint(model, pk - max_ratio^2 * pi <= 0)
                    @constraint(model, min_ratio^2 * pi - pk <= 0)
                    @constraint(model, fpik <= 0)

                    @constraint(model, pk - max_ratio^2 * pj <= 0)
                    @constraint(model, min_ratio^2 * pj - pk <= 0)
                    @constraint(model, -fpjk <= 0)

                    @constraint(model, pjk == pj - pk)
                    @constraint(model, pik == pi - pk)

                    relaxation_product(model, f, pik, fpik)
                    relaxation_product(model, f, pjk, fpjk)

                else
                    # === Disjunction with binary y ===
                    y = get_compressor_y(gm, n, k)

                    @constraint(model, pj - max_ratio^2 * pi <= (1 - y) * (j_pmax^2))
                    @constraint(model, min_ratio^2 * pi - pj <= (1 - y) * (min_ratio^2 * i_pmax^2))
                    @constraint(model, pi - max_ratio^2 * pj <= y * (i_pmax^2))
                    @constraint(model, min_ratio^2 * pj - pi <= y * (min_ratio^2 * j_pmax^2))
                end


                # ======================================================================================
                # TYPE 1 : ONE-DIRECTIONAL (i → j)
                # ======================================================================================
            elseif type == 1

                @constraint(model, pj - max_ratio^2 * pi <= 0)
                @constraint(model, min_ratio^2 * pi - pj <= 0)


                # ======================================================================================
                # TYPE 2 : SEMI-DIRECTIONAL
                # ======================================================================================
            else

                if min_ratio == 1
                    pij = @variable(model,
                        lower_bound = JuMP.lower_bound(pi) - JuMP.upper_bound(pj),
                        upper_bound = JuMP.upper_bound(pi) - JuMP.lower_bound(pj)
                    )
                    fpij = @variable(model)

                    @constraint(model, pj - max_ratio^2 * pi <= 0)
                    @constraint(model, min_ratio^2 * pi - pj <= 0)
                    @constraint(model, fpij <= 0)
                    @constraint(model, pij == pi - pj)

                    relaxation_product(model, f, pij, fpij)

                else
                    # === DISJUNCTION (flow can reverse) ===
                    y = get_compressor_y(gm, n, k)

                    @constraint(model, pj - max_ratio^2 * pi <= (1 - y) * (j_pmax^2))
                    @constraint(model, min_ratio^2 * pi - pj <= (1 - y) * (i_pmax^2))
                    @constraint(model, pi - pj <= y * (i_pmax^2))
                    @constraint(model, pj - pi <= y * (j_pmax^2))
                end
            end
        end
    end

    # constraint_on_off_regulator_pressure
    for b in 1:nbatch
        for (k, reg) in ref[:regulator]
            i = reg["fr_junction"]
            j = reg["to_junction"]
            min_ratio = reg["r_ratio_min"]
            max_ratio = reg["r_ratio_max"]
            f_min = get(reg, "f_min", 0.0)

            psqr_i = model[:psqr][i, b]
            psqr_j = model[:psqr][j, b]
            v_reg = model[:v_regulator][k, b]
            f_reg = model[:f_regulator][k, b]

            i_pmax = upper_bound(psqr_i)
            j_pmax = upper_bound(psqr_j)
            i_pmin = lower_bound(psqr_i)
            j_pmin = lower_bound(psqr_j)

            if max_ratio == 1
                # === Case 1: Fixed-ratio regulation (max_ratio = 1) ===
                pij = @variable(
                    model,
                    upper_bound = upper_bound(psqr_i) - lower_bound(psqr_j),
                    lower_bound = lower_bound(psqr_i) - upper_bound(psqr_j),
                    base_name = "pij_reg_$(k)_b$(b)"
                )
                fpij = @variable(model, base_name = "fpij_reg_$(k)_b$(b)")

                @constraint(model, psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2)
                @constraint(model, min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2))
                @constraint(model, fpij >= 0)
                @constraint(model, pij == psqr_i - psqr_j)

                # Bilinear relaxation
                relaxation_product(model, f_reg, pij, fpij)

            elseif f_min >= 0
                # === Case 2: Forward flow only (f_min ≥ 0) ===
                @constraint(model, psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2)
                @constraint(model, min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2))

            else
                # === Case 3: Disjunctive on/off (flow can reverse) ===
                y = @variable(model, binary = true, base_name = "y_reg_$(k)_b$(b)")

                @constraint(model, psqr_j - (max_ratio^2 * psqr_i) <= (2 - y - v_reg) * j_pmax^2)
                @constraint(model, (min_ratio^2 * psqr_i) - psqr_j <= (2 - y - v_reg) * i_pmax^2)
                @constraint(model, psqr_j - psqr_i <= (1 + y - v_reg) * j_pmax^2)
                @constraint(model, psqr_i - psqr_j <= (1 + y - v_reg) * i_pmax^2)
            end
        end
    end





    # constraint_compressor_ratio_value  
    for b in 1:nbatch
        for (k, compressor) in ref[:compressor]

            # === Indices ===
            i = compressor["fr_junction"]
            j = compressor["to_junction"]

            # === Variables ===
            pi = model[:psqr][i, b]
            pj = model[:psqr][j, b]
            r = model[:rsqr][k, b]

            # === Pressure bounds ===
            i_pmax = ref[:junction][i]["p_max"]
            j_pmax = ref[:junction][j]["p_max"]

            # === Directionality and ratios ===
            type = get(compressor, "directionality", 0)
            max_ratio = compressor["c_ratio_max"]
            min_ratio = compressor["c_ratio_min"]

            if type == 0
                # === Bidirectional compression (needs binary y) ===

                # → Use GasModels helper to ensure SAME y across all constraints
                y = get_compressor_y(gm, n, k)

                # Auxiliary variables
                rpi = @variable(model)
                rpj = @variable(model)

                # McCormick relaxations
                relaxation_product(model, pi, r, rpi)
                relaxation_product(model, pj, r, rpj)

                # GasModels-compliant inequalities
                @constraint(model, rpi <= pj + (1 - y) * i_pmax^2 * max_ratio^2)
                @constraint(model, rpi >= pj - (1 - y) * j_pmax^2)
                @constraint(model, rpj <= pi + y * j_pmax^2 * max_ratio^2)
                @constraint(model, rpj >= pi - y * i_pmax^2)

            else
                # → Unidirectional or semi-directional: simple product
                relaxation_product(model, pi, r, pj)
            end
        end


        # constraint_on_off_valve_mass_flow
        for b in 1:nbatch
            for (k, valve) in ref[:valve]
                # Retrieve the JuMP variables for this valve and batch
                f = model[:f_valve][k, b]      # mass flow through the valve
                v = model[:v_valve][k, b]      # binary/opening variable (1 = open, 0 = closed)

                # Extract flow limits from valve data
                f_min = valve["flow_min"]
                f_max = valve["flow_max"]

                # Add the on/off flow constraints for this batch
                @constraint(model, f_min * v <= f)
                @constraint(model, f <= f_max * v)
            end
        end

        # constraint_on_off_valve_pressure 
        for b in 1:nbatch
            for (k, valve) in ref[:valve]
                i = valve["fr_junction"]
                j = valve["to_junction"]

                pi = model[:psqr][i, b]
                pj = model[:psqr][j, b]
                v = model[:v_valve][k, b]

                i_pmax = ref[:junction][i]["p_max"]
                j_pmax = ref[:junction][j]["p_max"]

                @constraint(model, pj - ((1 - v) * j_pmax^2) <= pi)
                @constraint(model, pi <= pj + ((1 - v) * i_pmax^2))
            end
        end

        # constraint_on_off_regulator_mass_flow 
        for b in 1:nbatch
            for (k, regulator) in ref[:regulator]
                f = model[:f_regulator][k, b]
                v = model[:v_regulator][k, b]

                f_min = regulator["flow_min"]
                f_max = regulator["flow_max"]

                @constraint(model, f_min * v <= f)
                @constraint(model, f <= f_max * v)
            end
        end

        # constraint_on_off_regulator_pressure
        for (k, reg) in ref[:regulator]
            # === Indices and parameters ===
            i = reg["fr_junction"]
            j = reg["to_junction"]

            # If your data uses "r_ratio_min/max" instead of "reduction_factor_*",
            # you can swap these field names back to "r_ratio_min"/"r_ratio_max".
            min_ratio = reg["reduction_factor_min"]
            max_ratio = reg["reduction_factor_max"]

            f_min = reg["flow_min"]

            # Junction pressure limits (in base units, not squared yet)
            i_pmin = ref[:junction][i]["p_min"]
            i_pmax = ref[:junction][i]["p_max"]
            j_pmin = ref[:junction][j]["p_min"]
            j_pmax = ref[:junction][j]["p_max"]

            # === Variables ===
            psqr_i = model[:psqr][i]
            psqr_j = model[:psqr][j]
            v_reg = model[:v_regulator][k]
            f_reg = model[:f_regulator][k]

            if max_ratio == 1
                # === Case 1: Fixed-ratio regulation (max_ratio = 1) ===
                pij = @variable(
                    model,
                    lower_bound = JuMP.lower_bound(psqr_i) - JuMP.upper_bound(psqr_j),
                    upper_bound = JuMP.upper_bound(psqr_i) - JuMP.lower_bound(psqr_j),
                )
                fpij = @variable(model)

                @constraint(model,
                    psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2
                )
                @constraint(model,
                    min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2)
                )
                @constraint(model, fpij >= 0)
                @constraint(model, pij == psqr_i - psqr_j)

                # Bilinear relaxation product f_reg * (psqr_i - psqr_j)
                relaxation_product(model, f_reg, pij, fpij)

            elseif f_min >= 0
                # === Case 2: Forward flow only (f_min ≥ 0) ===
                @constraint(model,
                    psqr_j - max_ratio^2 * psqr_i <= (1 - v_reg) * j_pmax^2
                )
                @constraint(model,
                    min_ratio^2 * psqr_i - psqr_j <= (1 - v_reg) * (min_ratio * i_pmax^2)
                )

            else
                # === Case 3: Disjunctive on/off behavior (flow can reverse) ===
                y = @variable(model, binary = true)

                @constraint(model,
                    psqr_j - (max_ratio^2 * psqr_i) <= (2 - y - v_reg) * j_pmax^2
                )
                @constraint(model,
                    (min_ratio^2 * psqr_i) - psqr_j <= (2 - y - v_reg) * i_pmax^2
                )
                @constraint(model,
                    psqr_j - psqr_i <= (1 + y - v_reg) * j_pmax^2
                )
                @constraint(model,
                    psqr_i - psqr_j <= (1 + y - v_reg) * i_pmax^2
                )
            end
        end
    end
    # ------------------------------------------------------------------ #
    # 4. Objetivo económico (simplificado)
    # ------------------------------------------------------------------ #
    gamma = get_specific_heat_capacity_ratio(data)
    m = ((gamma - 1) / gamma) / 2
    T = get_temperature(data)
    G = get_gas_specific_gravity(data)
    work = ((286.76 / G) * T * (gamma / (gamma - 1)))
    baseF = get_base_flow(data)
    econW = get_economic_weighting(data)

    load_prices = Dict(k => get(ref[:delivery][k], "bid_price", 1.0)
                       for k in dispatchable_deliveries)
    prod_prices = Dict(k => get(ref[:receipt][k], "offer_price", 1.0)
                       for k in receipts if ref[:receipt][k]["is_dispatchable"] == 1)
    transf_prices = Dict(k => ref[:transfer][k]["withdrawal_min"] ≥ 0 ?
                              get(ref[:transfer][k], "bid_price", 1.0) :
                              -get(ref[:transfer][k], "offer_price", 1.0)
                         for k in transfers
                         if ref[:transfer][k]["is_dispatchable"] == 1)

    # println(dispatchable_deliveries)

    # Objective variable per batch
    @variable(model, z[b in 1:nbatch])

    max_prod_prices = maximum(prod_prices[k] for k in keys(prod_prices))
    # Batched objective constraint
    for b in 1:nbatch
        @constraint(model, z[b] >= 1 / nbatch * (
            # -sum(load_prices[k] * model[:fl][k, b] for k in keys(load_prices)) -
            -sum(transf_prices[k] * model[:ft][k, b] for k in keys(transf_prices)) +
            sum(prod_prices[k] * model[:fg][k, b] for k in keys(prod_prices))
            +
            1e6 / length(junctions) * sum(model[:slack_neg][j, b] for j in junctions)
        ))
    end

    # Optional: single aggregated objective (sum or average over batches)
    @objective(model, Min, sum(z[b] for b in 1:nbatch))

    model[:con_fmf_pipe] = con_fmf_pipe
    set_silent(model)

    # ------------------------------------------------------------------ #
    # 5. Optimiza y retorna
    # ------------------------------------------------------------------ #
    optimize!(model)
    # dual_value = JuMP.dual.(con_fmf_pipe)


    println("value of LF", objective_value(model))
    println("Status: ", termination_status(model))
    println("value of slack LF", sum(value.(model[:slack_neg])))
    # return value.(model[:fg]),value.(model[:ft])
    return value.(model[:fg]), objective_value(model)
end


###############################################################
# rrule for LRWPLayer_GM
###############################################################

function ChainRulesCore.rrule(::typeof(LRWPLayer_GM), gw, β::Matrix, α::Matrix)
    forward_time = time()
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    val = LRWPLayer_GM(gw, β, α; model=model)

    refs_β = [ParameterRef(model[:β][i]) for i in keys(model[:β])]
    refs_α = [ParameterRef(model[:α][i]) for i in keys(model[:α])]

    fg_values = [JuMP.value(model[:fg][i]) for i in keys(model[:fg])]
    fg = model[:fg]

    fwd_time_ref = time() - forward_time
    push!(fwd_times_lf, fwd_time_ref)

    function pullback_LRWP(Δobj)
        backw_time = time()
        DiffOpt.empty_input_sensitivities!(model)
        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), fg, Δobj)
        DiffOpt.reverse_differentiate!(model)

        grads_β = MOI.get.(model, DiffOpt.ReverseConstraintSet(), refs_β)
        grads_α = MOI.get.(model, DiffOpt.ReverseConstraintSet(), refs_α)

        values_β = [p.value for p in grads_β]
        values_α = [p.value for p in grads_α]

        bwd_time_ref = time() - backw_time
        push!(bwd_times_lf, bwd_time_ref)

        return (NoTangent(), NoTangent(), values_β, values_α)
    end

    return fg_values, pullback_LRWP
end


###############################################################
# rrule for WPLayer_GM
###############################################################

function ChainRulesCore.rrule(::typeof(WPLayer_GM), gw, fg, val_init)
    forward_time = time()
    val = WPLayer_GM(gw, fg, val_init)
    fwd_time_ref = time() - forward_time
    push!(fwd_times_hf, fwd_time_ref)

    obj = val["objective"]
    model = val["model"]

    function pullback_WP(_)
        backw_time = time()
        λ_fg = JuMP.dual.(model[:aux_fg])
        bwd_time_ref = time() - backw_time
        push!(bwd_times_hf, bwd_time_ref)
        return (NoTangent(), NoTangent(), Array(λ_fg), NoTangent())
    end

    return obj, pullback_WP
end


end
