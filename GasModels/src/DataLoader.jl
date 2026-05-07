module DataLoader

export GraphWithRef,
    data_graph_powermodels, data_graph_gasmodels,
    load_system_powermodels, load_generation_data, process_system_powermodels, process_system_powermodels_beta, load_beta_values,
    process_system_from_data_contingencies, process_system_gasmodels, load_system_gasmodels, load_GM_data

using PowerModels
using MAT
using GraphNeuralNetworks
using LinearAlgebra
using BSON
using BSON: @load
using GasModels

struct GraphWithRef
    graph
    data::Dict{String,Any}
end


function _build_gas_feature_matrices(
    data::Dict{String,Any},
    delivery::AbstractMatrix
)
    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    junctions = sort(collect(keys(ref[:junction])))
    junction_map = Dict(old => new for (new, old) in enumerate(junctions))
    delivery_ids = sort(collect(keys(ref[:delivery])))

    nsamples = size(delivery, 1)
    njunctions = length(junctions)
    withdrawal_full = zeros(Float64, nsamples, njunctions)

    for (k, del_id) in enumerate(delivery_ids)
        junction_id = ref[:delivery][del_id]["junction_id"]
        withdrawal_full[:, junction_map[junction_id]] .= delivery[:, k]
    end

    x = withdrawal_full'

    edge_features = Float64[]
    num_edges = length(ref[:pipe])

    for (_, pipe) in ref[:pipe]
        push!(edge_features,
            pipe["diameter"],
            pipe["length"],
            pipe["friction_factor"]
        )
    end

    e = reshape(edge_features, :, num_edges)

    return x, e
end


function _build_gas_graphs(
    data::Dict{String,Any},
    delivery::AbstractMatrix,
    y,
    from_numgraph::Int,
    to_numgraph::Int,
    all_graphs::Vector{GNNGraph}
)
    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    junctions = sort(collect(keys(ref[:junction])))
    pmin = [ref[:junction][j]["p_min"] for j in junctions]
    pmax = [ref[:junction][j]["p_max"] for j in junctions]
    junction_types = [ref[:junction][j]["junction_type"] for j in junctions]

    x, e = _build_gas_feature_matrices(data, delivery)
    println(size(x))
    junction_map = Dict(old => new for (new, old) in enumerate(junctions))
    source = Int[]
    target = Int[]

    for (_, pipe) in ref[:pipe]
        push!(source, junction_map[pipe["fr_junction"]])
        push!(target, junction_map[pipe["to_junction"]])
    end

    edge_index = GNNGraph(source, target)
    println(size(pmax))
    println(size(pmin))
    println(size(junction_types))
    for s in from_numgraph:to_numgraph
        x_ts = Matrix(hcat(x[:,s], pmax, pmin, junction_types))'
        e_ts = Matrix(e)
        y_ts = y

        g_ts = GNNGraph(
            edge_index;
            ndata=(; x=x_ts),
            edata=(; e=e_ts),
            gdata=(; y=y_ts)
        )

        push!(all_graphs, g_ts)
    end

    return all_graphs
end


function data_graph_gasmodels(
    data::Dict{String,Any},
    x::AbstractMatrix,
    e::AbstractMatrix,
    y,
    from_numgraph::Int,
    to_numgraph::Int,
    all_graphs::Vector{GNNGraph}
)
    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    junctions = sort(collect(keys(ref[:junction])))
    pmin = [ref[:junction][j]["p_min"] for j in junctions]
    pmax = [ref[:junction][j]["p_max"] for j in junctions]
    junction_types = [ref[:junction][j]["junction_type"] for j in junctions]
    junction_map = Dict(old => new for (new, old) in enumerate(junctions))
    source = Int[]
    target = Int[]

    for (_, pipe) in ref[:pipe]
        push!(source, junction_map[pipe["fr_junction"]])
        push!(target, junction_map[pipe["to_junction"]])
    end

    edge_index = GNNGraph(source, target)

    for s in from_numgraph:to_numgraph
        x_ts = Matrix(hcat(x[:, s], pmax, pmin, junction_types))'
        e_ts = Matrix(e)

        g_ts = GNNGraph(
            edge_index;
            ndata=(; x=x_ts),
            edata=(; e=e_ts),
            gdata=(; y=y)
        )

        push!(all_graphs, g_ts)
    end

    return all_graphs
end


function data_graph_powermodels(
    data::Dict{String,Any},
    x::AbstractMatrix,
    e::AbstractMatrix,
    y::AbstractMatrix,
    from_numgraph::Int,
    to_numgraph::Int,
    all_graphs::Vector{GNNGraph}
)
    buses = sort(parse.(Int, keys(data["bus"])))
    all_branches = sort(parse.(Int, keys(data["branch"])))
    active_branches = filter(bid -> data["branch"][string(bid)]["br_status"] == 1, all_branches)
    nbus = length(buses)

    Vmax = [data["bus"][string(bus)]["vmax"] for bus in buses]
    Vmin = [data["bus"][string(bus)]["vmin"] for bus in buses]

    Gs = Float64[]
    Bs = Float64[]
    shunt_per_bus = Dict{Int,Tuple{Float64,Float64}}()
    if haskey(data, "shunt")
        for (_, shunt) in data["shunt"]
            shunt_per_bus[shunt["shunt_bus"]] = (shunt["gs"], shunt["bs"])
        end
    end

    for bus in buses
        if haskey(shunt_per_bus, bus)
            gs, bs = shunt_per_bus[bus]
            push!(Gs, gs)
            push!(Bs, bs)
        else
            push!(Gs, 0.0)
            push!(Bs, 0.0)
        end
    end

    xP = x[1:nbus, :]
    xQ = x[nbus+1:2*nbus, :]
    bus_to_index = Dict(buses[i] => i for i in 1:nbus)

    sources = [bus_to_index[data["branch"][string(bid)]["f_bus"]] for bid in active_branches]
    targets = [bus_to_index[data["branch"][string(bid)]["t_bus"]] for bid in active_branches]
    edge_index = GNNGraph(sources, targets)

    for s in from_numgraph:to_numgraph
        x_ts = Matrix(hcat(xP[:, s], xQ[:, s], Vmax, Vmin, Bs, Gs))'
        e_ts = Matrix(e)
        y_ts = y[:, s]

        g_ts = GNNGraph(
            edge_index;
            ndata=(; x=x_ts),
            edata=(; e=e_ts),
            gdata=(; y=y_ts)
        )
        push!(all_graphs, g_ts)
    end

    return all_graphs
end


function load_GM_data(path_data::String, system_name::String; use_test_data::Bool=false)
    file_name = use_test_data ? "data_test_$(system_name).bson" : "data_$(system_name).bson"
    file_path = joinpath(path_data, file_name)
    println(file_path)
    println(pwd())
    results = BSON.load(file_path)
    results_key = use_test_data ? :results_test : :results
    results_data = results[results_key]

    fg_HF = results_data["fg_HF"]
    fg_LF = results_data["fg_LRWP"]
    # ft_HF = results_data["ft_HF"]
    # ft_LF = results_data["ft_LF"]
    delivery = results_data["delivery"]

    return fg_HF, fg_LF, delivery
end

###############################################################
# Function: process_system_gasmodels
#
# Purpose:
#   Prepare a GasModels system for GNN training.
#
#   This function:
#     • Loads a GasModels network
#     • Builds node features (withdrawals per junction)
#     • Builds edge features (pipe parameters)
#     • Defines targets (HF generator flows)
#     • Constructs graph topology
#
# Output:
#     data               -> original GasModels dictionary
#     x                  -> node feature matrix (Nj × Nsamples)
#     edge_feat_matrix   -> edge feature matrix (Fe × E)
#     y                  -> training targets (e.g. fg_HF)
#     g                  -> graph topology (GNNGraph)
#
###############################################################

function process_system_gasmodels(
    system_name::String,
    path_data::String;
    from_numgraph::Int=1,
    to_numgraph::Union{Nothing,Int}=nothing,
    all_graphs::Vector{GNNGraph}=GNNGraph[],
    return_components::Bool=false,
    use_test_data::Bool=false
)

    println("Processing system: ", system_name)

    ###########################################################
    # 1) Load GasModels network
    ###########################################################

    file_name = joinpath(path_data, "$(system_name).m")
    println(file_name)

    data = GasModels.parse_file(file_name)

    # Structured reference (clean access to network components)
    ref = GasModels.build_ref(data)[:it][:gm][:nw][0]

    ###########################################################
    # 2) Load previously generated LF/HF data
    #
    # fg_HF  -> High-fidelity flows (target)
    # fg_LF  -> Low-fidelity flows
    # delivery -> withdrawal samples
    ###########################################################

    fg_HF, fg_LF, delivery = load_GM_data(path_data, system_name; use_test_data=use_test_data)

    y = fg_HF
    last_sample = isnothing(to_numgraph) ? size(delivery, 1) : to_numgraph
    graphs = _build_gas_graphs(data, delivery, y, from_numgraph, last_sample, all_graphs)

    if return_components
        x, edge_feat_matrix = _build_gas_feature_matrices(data, delivery)
        return data, x, edge_feat_matrix, y, graphs
    end

    return data, graphs


end


#! Different codes (experiments with powermodels)

"""    load_system_powermodels(file_name::String)
    Load a power system model from a MAT file.
    - `file_name`: The path to the MAT file containing the power system model.
    Returns a dictionary containing the parsed data.
"""

function load_system_powermodels(file_name::String)
    data = PowerModels.parse_file(file_name)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    return data
end


"""    load_system_gasmodels(file_name::String)
    Load a gas system model from a MAT file.
    - `file_name`: The path to the MAT file containing the gas system model.
    Returns a dictionary containing the parsed data.
"""

function load_system_gasmodels(file_name::String)
    data = GasModels.parse_file(file_name)
    return data
end


"""    load_generation_data(path_data::String, system_name::String)
    Loads generation and demand data from a BSON file.
    - `path_data`: The path to the data files.
    - `system_name`: The name of the system to load.
    Returns a tuple containing the Pg_AC (AC dispatch) and load_power (demand).
"""

function load_generation_data(path_data::String, system_name::String; use_test_data::Bool=false)
    file_name = use_test_data ? "data_test_$(system_name).bson" : "data_$(system_name).bson"
    file_path = joinpath(path_data, file_name)
    results = BSON.load(file_path)

    Pg_AC = results[:results]["DispatchAC"]
    load_power = results[:results]["Demand"]

    return Pg_AC, load_power
end




"""   load_beta_values(path_data::String, system_name::String)
    Loads beta values from a BSON file.
    - `path_data`: The path to the data files.
    - `system_name`: The name of the system to load.
    Returns a tuple containing the pg_values, beta_values, and demand_values.
"""

function load_beta_values(path_data::String, system_name::String)
    bson_file = joinpath(path_data, "full_supervised_data_$(system_name).bson")
    results = Dict{String,Any}()
    @load bson_file results  # Carga el diccionario "results"

    beta_values = results["beta_values"]
    pg_values = results["pg_values"]
    demand_values = results["demand_values"]

    return pg_values, beta_values, demand_values
end


"""
    process_system_powermodels(system_name::String, path_data::String)
    Processes a power system model and returns the graph representation.
    - `system_name`: The name of the system to process.
    - `path_data`: The path to the data files.
    Returns a tuple containing the data dictionary, node features, edge features,
    target values, and edge index.
"""

function process_system_powermodels(system_name::String, path_data::String; use_test_data::Bool=false)
    println("Processing system: ", system_name)

    file_name = joinpath(path_data, "$(system_name).m")
    data = load_system_powermodels(file_name)

    DispatchAC, Demand = load_generation_data(path_data, system_name; use_test_data=use_test_data)

    buses = sort(parse.(Int, keys(data["bus"])))
    # Only keep branches with br_status == 1
    branches_all = sort(parse.(Int, keys(data["branch"])))
    active_branches = filter(bid -> data["branch"][string(bid)]["br_status"] == 1,
        branches_all)

    # node features
    x = Demand[:, 1:2*length(buses)]'

    # edge features: now sized to active_branches
    e = zeros(5, length(active_branches))
    for (i, branch_id) in enumerate(active_branches)
        branch = data["branch"][string(branch_id)]
        r = branch["br_r"]
        x_ = branch["br_x"]
        b = get(branch, "b_fr", 0.0) * 2
        rateA = get(branch, "rate_a", 100.0)
        tap = branch["tap"] == 0 ? 1.0 : branch["tap"]
        e[:, i] .= (r, x_, b, rateA, tap)
    end

    y = DispatchAC'

    # build edge index only over active branches
    source = Int[]
    target = Int[]
    for branch_id in active_branches
        branch = data["branch"][string(branch_id)]
        push!(source, branch["f_bus"])
        push!(target, branch["t_bus"])
    end
    edge_index = GNNGraph(source, target)

    return data, x, e, y, edge_index
end



"""
    process_system_powermodels_beta(system_name::String, path_data::String)
    Processes a power system model with beta values and returns the graph representation.
    - `system_name`: The name of the system to process.
    - `path_data`: The path to the data files.
    Returns a tuple containing the data dictionary, node features, edge features,
    target values (beta values), edge index, and DispatchAC.
"""

function process_system_powermodels_beta(system_name::String, path_data::String)
    println("Processing system: ", system_name)

    file_name = joinpath(path_data, "$(system_name).m")
    data = load_system_powermodels(file_name)

    DispatchAC, beta_values, Demand = load_beta_values(path_data, system_name)
    #pg_values, beta_values, demand_values
    buses = sort(parse.(Int, keys(data["bus"])))
    branches = sort(parse.(Int, keys(data["branch"])))
    gens = sort(parse.(Int, keys(data["gen"])))

    x = Demand[:, 1:2*length(buses)]'

    e = zeros(5, length(branches))
    for (i, branch_id) in enumerate(branches)
        branch = data["branch"][string(branch_id)]
        r = branch["br_r"]
        x_ = branch["br_x"]
        b = get(branch, "b_fr", 0.0) .* 2
        rateA = get(branch, "rate_a", 100.0)
        tap = branch["tap"] == 0 ? 1.0 : branch["tap"]
        e[:, i] = [r, x_, b, rateA, tap]
    end

    y = beta_values

    source = Int[]
    target = Int[]
    for branch_id in branches
        branch = data["branch"][string(branch_id)]
        push!(source, branch["f_bus"])
        push!(target, branch["t_bus"])
    end
    edge_index = GNNGraph(source, target)
    # println(DispatchAC)
    return data, x, e, y, edge_index, DispatchAC
end


"""
    process_system_from_data_contingencies(data::Dict, DispatchAC, Demand)
    Processes a power system model from a data dictionary and returns the graph representation.
    - `data`: A dictionary containing the system data.
    - `DispatchAC`: The AC dispatch data.
    - `Demand`: The demand data.
    Returns a tuple containing node features, edge features, target values, and edge index deleting a specific line.
"""

function process_system_from_data_contingencies(data::Dict, DispatchAC, Demand)
    buses = sort(parse.(Int, keys(data["bus"])))
    branches_all = sort(parse.(Int, keys(data["branch"])))
    active_branches = filter(bid -> data["branch"][string(bid)]["br_status"] == 1,
        branches_all)

    x = Demand[:, 1:2*length(buses)]'

    e = zeros(5, length(active_branches))
    for (i, branch_id) in enumerate(active_branches)
        branch = data["branch"][string(branch_id)]
        r = branch["br_r"]
        x_ = branch["br_x"]
        b = get(branch, "b_fr", 0.0) .* 2
        rateA = get(branch, "rate_a", 100.0)
        tap = branch["tap"] == 0 ? 1.0 : branch["tap"]
        e[:, i] .= [r, x_, b, rateA, tap]
    end

    y = DispatchAC'

    source = Int[]
    target = Int[]
    for branch_id in active_branches
        branch = data["branch"][string(branch_id)]
        push!(source, branch["f_bus"])
        push!(target, branch["t_bus"])
    end
    edge_index = GNNGraph(source, target)

    return x, e, y, edge_index
end








end
