

module DataLoader

export GraphWithRef,
    data_graph_powermodels,
    dataset_bson_path,
    load_system_powermodels,
    load_generation_data,
    process_system_powermodels,
    process_system_powermodels_beta,
    load_beta_values,
    process_system_from_data_contingencies

using BSON
using BSON: @load
using GraphNeuralNetworks
using MAT
using PowerModels

struct GraphWithRef
    graph
    data::Dict{String,Any}
end

function load_system_powermodels(file_name::String)
    data = PowerModels.parse_file(file_name)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    return data
end

function dataset_bson_path(path_data::String, system_name::String; use_test_data::Bool=false)
    prefix = use_test_data ? "data_test_" : "data_"
    return joinpath(path_data, "$(prefix)$(system_name).bson")
end

function load_generation_data(path_data::String, system_name::String; use_test_data::Bool=false)
    results = BSON.load(dataset_bson_path(path_data, system_name; use_test_data=use_test_data))
    pg_ac = results[:results]["DispatchAC"]
    demand = results[:results]["Demand"]
    return pg_ac, demand
end

function load_beta_values(path_data::String, system_name::String)
    bson_file = joinpath(path_data, "full_supervised_data_$(system_name).bson")
    results = Dict{String,Any}()
    @load bson_file results

    beta_values = results["beta_values"]
    pg_values = results["pg_values"]
    demand_values = results["demand_values"]

    return pg_values, beta_values, demand_values
end

function active_branches(data::Dict{String,Any})
    branches_all = sort(parse.(Int, keys(data["branch"])))
    return filter(bid -> data["branch"][string(bid)]["br_status"] == 1, branches_all)
end

function branch_features(data::Dict{String,Any}, branch_ids)
    e = zeros(5, length(branch_ids))
    for (i, branch_id) in enumerate(branch_ids)
        branch = data["branch"][string(branch_id)]
        r = branch["br_r"]
        x_ = branch["br_x"]
        b = get(branch, "b_fr", 0.0) * 2
        rateA = get(branch, "rate_a", 100.0)
        tap = branch["tap"] == 0 ? 1.0 : branch["tap"]
        e[:, i] .= (r, x_, b, rateA, tap)
    end
    return e
end

function edge_index_from_branches(data::Dict{String,Any}, branch_ids)
    source = Int[]
    target = Int[]
    for branch_id in branch_ids
        branch = data["branch"][string(branch_id)]
        push!(source, branch["f_bus"])
        push!(target, branch["t_bus"])
    end
    return GNNGraph(source, target)
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
    branch_ids = active_branches(data)
    nbus = length(buses)

    vmax = [data["bus"][string(bus)]["vmax"] for bus in buses]
    vmin = [data["bus"][string(bus)]["vmin"] for bus in buses]

    gs = Float64[]
    bs = Float64[]
    shunt_per_bus = Dict{Int,Tuple{Float64,Float64}}()
    if haskey(data, "shunt")
        for (_, shunt) in data["shunt"]
            shunt_per_bus[shunt["shunt_bus"]] = (shunt["gs"], shunt["bs"])
        end
    end

    for bus in buses
        if haskey(shunt_per_bus, bus)
            gs_i, bs_i = shunt_per_bus[bus]
            push!(gs, gs_i)
            push!(bs, bs_i)
        else
            push!(gs, 0.0)
            push!(bs, 0.0)
        end
    end

    xP = x[1:nbus, :]
    xQ = x[nbus+1:2*nbus, :]
    bus_to_index = Dict(buses[i] => i for i in 1:nbus)
    sources = [bus_to_index[data["branch"][string(bid)]["f_bus"]] for bid in branch_ids]
    targets = [bus_to_index[data["branch"][string(bid)]["t_bus"]] for bid in branch_ids]
    edge_index = GNNGraph(sources, targets)

    for s in from_numgraph:to_numgraph
        x_ts = Matrix(hcat(xP[:, s], xQ[:, s], vmax, vmin, bs, gs))'
        y_ts = y[:, s]
        g_ts = GNNGraph(
            edge_index;
            ndata=(; x=x_ts),
            edata=(; e=Matrix(e)),
            gdata=(; y=y_ts)
        )
        push!(all_graphs, g_ts)
    end

    return all_graphs
end

function process_system_powermodels(system_name::String, path_data::String; use_test_data::Bool=false)
    println("Processing system: ", system_name)

    file_name = joinpath(path_data, "$(system_name).m")
    data = load_system_powermodels(file_name)

    dispatch_ac, demand = load_generation_data(path_data, system_name; use_test_data=use_test_data)
    buses = sort(parse.(Int, keys(data["bus"])))
    branch_ids = active_branches(data)

    x = demand[:, 1:2*length(buses)]'
    e = branch_features(data, branch_ids)
    y = dispatch_ac'
    edge_index = edge_index_from_branches(data, branch_ids)

    return data, x, e, y, edge_index
end

function process_system_powermodels_beta(system_name::String, path_data::String)
    println("Processing system: ", system_name)

    file_name = joinpath(path_data, "$(system_name).m")
    data = load_system_powermodels(file_name)

    dispatch_ac, beta_values, demand = load_beta_values(path_data, system_name)
    buses = sort(parse.(Int, keys(data["bus"])))
    branch_ids = sort(parse.(Int, keys(data["branch"])))

    x = demand[:, 1:2*length(buses)]'
    e = branch_features(data, branch_ids)
    y = beta_values
    edge_index = edge_index_from_branches(data, branch_ids)

    return data, x, e, y, edge_index, dispatch_ac
end

function process_system_from_data_contingencies(data::Dict, dispatch_ac, demand)
    buses = sort(parse.(Int, keys(data["bus"])))
    branch_ids = active_branches(data)

    x = demand[:, 1:2*length(buses)]'
    e = branch_features(data, branch_ids)
    y = dispatch_ac'
    edge_index = edge_index_from_branches(data, branch_ids)

    return x, e, y, edge_index
end

end
