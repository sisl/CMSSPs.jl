## Define DREAMR types (based on US, UA)
@enum DREAMR_MODETYPE FLIGHT=1 RIDE=2
@enum HOP_ACTION HOPON=1 HOPOFF=2 STAY=3

## Mode Info
const DREAMR_MODES = [FLIGHT, RIDE]
const DREAMR_GOAL_MODE = FLIGHT
const DREAMR_MODE_ACTIONS = [HOPON, HOPOFF, STAY]

const HOP_INFLATION_PARAMETER = 2.0

mutable struct DREAMRModeAction
    mode_action::HOP_ACTION
    car_id::String
end

Base.isequal(dma1::DREAMRModeAction, dma2::DREAMRModeAction) = isequal(dma1.mode_action, dma2.mode_action) && isequal(dma1.car_id, dma2.car_id)

DREAMRModeAction(mode_action::HOP_ACTION) = DREAMRModeAction(mode_action, "")

## Car route stuff
struct DREAMRVertexMetadata
    car_id::String
    vertex_id::String
end

Base.zero(DREAMRVertexMetadata) = DREAMRVertexMetadata("", "")

struct DREAMRBookkeeping
    active_car_to_idx_ranges::Dict{String, Vector{Vector{Int64}}} # Maps car name to range of vertices that represent hopon
    route_vert_id_to_idx_pair::Dict{String, Vector{Int64}} # Maps a specific route ID (car-id+"-"+vertex_id) to the indices of hopon and hopoff versions
    goal_vertex_idx::Int64
end
DREAMRBookkeeping() = DREAMRBookkeeping(Dict{String, Vector{Vector{Int64}}}(), Dict{String, Vector{Int64}}(), 0)



## Context
mutable struct DREAMRContextSet
    curr_epoch::Int64
    epochs_dict::Dict
    car_id::String
end

function get_dreamr_episode_context(ep_dict::Dict)
    return DREAMRContextSet(0, ep_dict, "")
end


const DREAMRStateType{US} = CMSSPState{DREAMR_MODETYPE, US} where {US <: UAVState}
const DREAMRActionType{UA} = CMSSPAction{DREAMRModeAction, UA} where {UA <: UAVAction}
const DREAMRCMSSPType{US,UA} = CMSSP{DREAMR_MODETYPE, US, DREAMRModeAction, UA, Parameters, DREAMRContextSet} where {US <: UAVState, UA <: UAVAction}
const DREAMRCFMDPType{US,UA} = ModalFinHorMDP{DREAMR_MODETYPE, US, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRUFMDPType{US,UA} = ModalInfHorMDP{DREAMR_MODETYPE, US, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRHopoffMDPType = ModalFinHorMDP{DREAMR_MODETYPE, Nothing, Nothing, Parameters}
const DREAMROpenLoopVertex{US} = OpenLoopVertex{DREAMR_MODETYPE, US, DREAMRModeAction, DREAMRVertexMetadata} where {US <: UAVState}
const DREAMRGraphTracker{US,UA,RNG} = GraphTracker{DREAMR_MODETYPE, US, DREAMRModeAction, DREAMRVertexMetadata, DREAMRBookkeeping, RNG} where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}
const DREAMRModePair = Tuple{DREAMR_MODETYPE, DREAMR_MODETYPE}
const DREAMRSolverType{US,UA,RNG} = HHPCSolver{DREAMR_MODETYPE, US, DREAMRModeAction, DREAMRVertexMetadata, DREAMRBookkeeping, RNG} where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}


# This is only really needed for baseline
function get_dreamr_actions(::Type{UA}, params::Parameters) where {UA <: UAVAction}

    actions = Vector{DREAMRActionType{UA}}(undef, 0)
    idx = 1
    for dma in DREAMR_MODE_ACTIONS
        push!(actions, DREAMRActionType{UA}(DREAMRModeAction(dma), idx))
        idx += 1
    end

    uav_dynamics_actions = get_uav_dynamics_actions(UA, params)
    for uavda in uav_dynamics_actions
        push!(actions, DREAMRActionType{UA}(uavda, idx))
        idx += 1
    end

    return actions
end

function get_dreamr_mdp()

    T = zeros(2, 3, 2)
    # Flight + hopon = ride
    T[1, 1, 2] = 1.0
    # Ride + hopoff = flight
    T[2, 2, 1] = 1.0
    # Ride + STAY = Ride
    T[2, 3, 2] = 1.0

    R = zeros(2, 3)

    # Undiscounted
    return TabularMDP(T, R, 1.0)
end

function create_dreamr_cmssp(::Type{US}, ::Type{UA}, context_set::DREAMRContextSet,
                             goal_point::Point, params::Parameters) where {US <: UAVState, UA <: UAVAction}

    actions = get_dreamr_actions(UA, params)
    switch_mdp = get_dreamr_mdp()
    goal_state = DREAMRStateType{US}(FLIGHT, get_state_at_rest(US, goal_point))

    return DREAMRCMSSPType{US,UA}(actions, DREAMR_MODES, switch_mdp, goal_state, params, context_set)
end


function POMDPs.actions(cmssp::DREAMRCMSSPType, state::DREAMRStateType)

    if cmssp.curr_context_set.curr_epoch >= length(collect(keys(cmssp.curr_context_set.epochs_dict)))
        return cmssp.actions[4:end]
    end

    if state.mode == FLIGHT
        return [cmssp.actions[1]; cmssp.actions[4:end]]
    else
        return cmssp.actions[2:3]
    end
end

function set_dreamr_goal!(cmssp::DREAMRCMSSPType{US,UA}, goal_point::Point) where {US <: UAVState, UA <: UAVAction}
    cmssp.goal_state = DREAMRStateType{US}(FLIGHT, get_state_at_rest(US, goal_point))
end

function POMDPs.isterminal(cmssp::DREAMRCMSSPType, state::DREAMRStateType)

    curr_point = get_position(state.continuous)
    curr_speed = get_speed(state.continuous)
    
    goal_point = get_position(cmssp.goal_state.continuous)

    dist = point_dist(curr_point, goal_point)

    params = cmssp.params

    return (state.mode == FLIGHT) && (dist < 2.0*params.time_params.MDP_TIMESTEP*params.scale_params.HOP_DISTANCE_THRESHOLD) &&
            (curr_speed < params.scale_params.XYDOT_HOP_THRESH)
end


function POMDPs.isterminal(mdp::Union{DREAMRCFMDPType{US,UA},DREAMRUFMDPType{US,UA}}, relative_state::US) where {US <: UAVState, UA <: UAVAction}
    
    curr_pos_norm = point_norm(get_position(relative_state))
    curr_speed = get_speed(relative_state)

    return (curr_pos_norm < mdp.params.time_params.MDP_TIMESTEP*mdp.params.scale_params.HOP_DISTANCE_THRESHOLD) &&
        (curr_speed < mdp.params.scale_params.XYDOT_HOP_THRESH)
end


function HHPC.get_relative_state(mdp::Union{DREAMRCFMDPType{US,UA},DREAMRUFMDPType{US,UA}}, source::US, target::US) where {US <: UAVState, UA <: UAVAction}
    return rel_state(source, target)
end


## Conversion for multirotor - can add others as needed
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::MultiRotorUAVState,
                          mdp::Union{DREAMRCFMDPType{MultiRotorUAVState, MultiRotorUAVAction},DREAMRUFMDPType{MultiRotorUAVState,MultiRotorUAVAction}})
    v = [s.x, s.y, s.xdot, s.ydot]
end

function POMDPs.convert_s(::Type{MultiRotorUAVState}, v::AbstractVector{Float64},
                          mdp::Union{DREAMRCFMDPType{MultiRotorUAVState, MultiRotorUAVAction},DREAMRUFMDPType{MultiRotorUAVState,MultiRotorUAVAction}})
    s = MultiRotorUAVState(v[1], v[2], v[3], v[4])
end

function POMDPs.reward(mdp::DREAMRCFMDPType{US,UA}, state::US, cont_action::UA, statep::US) where {US <: UAVState, UA <: UAVAction}
    
    cost = mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP
    cost += dynamics_cost(mdp.params, state, statep)
    
    return -cost
end

function POMDPs.reward(mdp::DREAMRUFMDPType{US,UA}, state::US, cont_action::UA, statep::US) where {US <: UAVState, UA <: UAVAction}
    
    cost = mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP
    cost += dynamics_cost(mdp.params, state, statep)

    if isterminal(mdp, statep)
        cost -= mdp.terminal_reward
    end
    
    return -cost
end


# FH version of generate_sr
function POMDPs.generate_sr(mdp::Union{DREAMRCFMDPType{US,UA},DREAMRUFMDPType{US,UA}}, state::US, cont_action::UA, 
                            rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}
    
    statep = next_state(mdp.params, state, cont_action, rng)
    reward = POMDPs.reward(mdp, state, cont_action, statep)
    
    return (statep, reward)
end


function get_cf_mdp(::Type{US}, ::Type{UA},
                    params::Parameters, beta::Float64) where {US <:UAVState, UA <: UAVAction}

    uav_dynamics_actions = get_uav_dynamics_actions(UA, params)

    return DREAMRCFMDPType{US,UA}(FLIGHT, params, uav_dynamics_actions, beta, params.time_params.HORIZON_LIM)
end


function get_uf_mdp(::Type{US}, ::Type{UA}, params::Parameters) where {US <:UAVState, UA <: UAVAction}

    uav_dynamics_actions = get_uav_dynamics_actions(UA, params)

    return DREAMRUFMDPType{US,UA}(FLIGHT, uav_dynamics_actions, params, params.cost_params.FLIGHT_REACH_REWARD) # beta = 1 and horizon = 0
end



function HHPC.expected_reward(mdp::Union{DREAMRCFMDPType{US,UA},DREAMRUFMDPType{US,UA}}, 
                              state::US, cont_action::UA, rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}
    
    params = mdp.params
    avg_rew = 0.0
    
    for i = 1:params.scale_params.MC_GENERATIVE_NUMSAMPLES
        (_, r) = generate_sr(mdp, state, cont_action, rng)
        avg_rew += r
    end
    
    avg_rew /= params.scale_params.MC_GENERATIVE_NUMSAMPLES
    return avg_rew
end


function HHPC.generate_goal_vertex_set!(cmssp::DREAMRCMSSPType{US,UA}, vertex::DREAMROpenLoopVertex{US},
                                        graph_tracker::DREAMRGraphTracker,
                                        rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    # If current goal vertex idx is 0, add goal and send it as nbr
    if graph_tracker.curr_goal_idx == 0
        graph_tracker.curr_goal_idx = num_vertices(graph_tracker.curr_graph)+1 # Since it will be added
        goal_vert = DREAMROpenLoopVertex{US}(cmssp.goal_state, cmssp.mode_actions[1])
        return [goal_vert], []
    else
        # Just return current goal vertex idx
        return [], [graph_tracker.curr_goal_idx]
    end
end



function get_arrival_time_distribution(curr_timestep::Int64, mean_time::Float64, params::Parameters, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    # If very close to time, collapse around estimate
    if mean_time / params.time_params.MDP_TIMESTEP - curr_timestep > 2.0
        time_dist = Normal(mean_time, params.time_params.CAR_TIME_STD)
    else
        time_dist = Normal(mean_time, params.scale_params.EPSILON)
    end

    temp_tp_dict = Dict{Int64,Float64}()

    for j = 1:params.time_params.MC_TIME_NUMSAMPLES
        tval = rand(rng, time_dist) / params.time_params.MDP_TIMESTEP

        low = convert(Int64, floor(tval))
        high = convert(Int64, ceil(tval))
        low_wt = tval - floor(tval)

        cum_low_wt = get(temp_tp_dict, low+1, 0.0)
        temp_tp_dict[low+1] = cum_low_wt + low_wt
        cum_high_wt = get(temp_tp_dict, high+1, 0.0)
        temp_tp_dict[high+1] = cum_high_wt + 1.0 - low_wt
    end

    vals = Vector{Int64}(undef, 0)
    probs = Vector{Float64}(undef, 0)

    for (tval, prob) in temp_tp_dict
        push!(vals, tval)
        push!(probs, prob/params.time_params.MC_TIME_NUMSAMPLES)
    end

    return TPDistribution(vals, probs)
end


function HHPC.generate_bridge_vertex_set!(cmssp::DREAMRCMSSPType, vertex::DREAMROpenLoopVertex,
                                         mode_pair::DREAMRModePair,
                                         graph_tracker::DREAMRGraphTracker,
                                         action::DREAMRModeAction,
                                         rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # NOTE - Will never actually generate anything; only get neighbour idxs
    context_set = cmssp.curr_context_set
    params = cmssp.params

    nbrs_to_add = Vector{Int64}(undef, 0)

    if vertex.state.mode == FLIGHT # If it is a drone or post-hopoff vertex

        @assert mode_pair == (FLIGHT, RIDE)

        # Iterate over cars
        for (car_id, ranges) in graph_tracker.bookkeeping.active_car_to_idx_ranges

            # If this vertex just hopped off a car, don't hop on to another!
            # if car_id == vertex.metadata.car_id
            #     @warn "Hopoff searching for hopon"
            #     @show vertex
            #     readline()
            #     continue
            # end

            hopon_vert_range = ranges[1]

            for next_idx in hopon_vert_range[1]:hopon_vert_range[2]

                # Check if max drone speed not enough
                next_vert = graph_tracker.curr_graph.vertices[next_idx]
                state_dist = point_dist(get_position(vertex.state.continuous), get_position(next_vert.state.continuous))

                if params.scale_params.MAX_DRONE_SPEED*(mean(next_vert.tp) - mean(vertex.tp)) > 
                    state_dist*params.scale_params.VALID_FLIGHT_EDGE_DIST_RATIO
                    push!(nbrs_to_add, next_idx)
                end
            end
        end
    else
        # It is a car route vertex
        # Only add hopoff vertices for that car
        @assert mode_pair == (RIDE, FLIGHT)
        @assert vertex.metadata.car_id != ""

        hopoff_vert_range = graph_tracker.bookkeeping.active_car_to_idx_ranges[vertex.metadata.car_id][2]

        # Loop through range and add ones that come after current
        for hopoff_idx = hopoff_vert_range[1]:hopoff_vert_range[2]

            next_vert = graph_tracker.curr_graph.vertices[hopoff_idx]

            # Remember vertex is a hopon vertex, so can only add hopoff ones; compare by tp
            if mean(next_vert.tp) > mean(vertex.tp)
                push!(nbrs_to_add, hopoff_idx)
            end
        end
        # if context_set.car_id != ""
        #     @warn "Popping start ride vertex"
        #     @show vertex
        #     @show nbrs_to_add
        #     readline()
        # end
    end

    return ([], nbrs_to_add)
end


function HHPC.generate_next_valid_modes(cmssp::DREAMRCMSSPType, mode::DREAMR_MODETYPE)

    if mode == FLIGHT
        return [(DREAMRModeAction(HOPON), RIDE)]
    else
        return [(DREAMRModeAction(HOPOFF), FLIGHT)]
    end
end


function HHPC.update_vertices_with_context!(cmssp::DREAMRCMSSPType{US,UA}, graph_tracker::DREAMRGraphTracker,
                                            curr_timestep::Int64) where {US <: UAVState, UA <: UAVAction}

    # Iterate over cars, add if new or update if old
    context_set = cmssp.curr_context_set

    curr_epoch_dict = context_set.epochs_dict[string(context_set.curr_epoch)]
    epoch_cars = curr_epoch_dict["car-info"]
    params = cmssp.params

    keys_to_delete = Vector{String}(undef, 0)

    for (car_id, car_info) in epoch_cars

        # Update car route if it exists
        if haskey(graph_tracker.bookkeeping.active_car_to_idx_ranges, car_id)

            route_info = car_info["route"]

            if route_info != nothing

                # Update times of all future vertices
                sorted_route = sort(collect(route_info), by=x->parse(Float64, x[1]))

                for (id, timept) in sorted_route

                    timeval = timept[2]
                    tp_dist = get_arrival_time_distribution(curr_timestep, timeval, params, graph_tracker.rng)

                    # Update the hopon and hopoff vert tps
                    vert_id = string(car_id, "-", id)
                    graph_tracker.curr_graph.vertices[graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id][1]].tp = tp_dist
                    graph_tracker.curr_graph.vertices[graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id][2]].tp = tp_dist
                end

                # If first waypoint of route updated, change BOTH hopon and hopoff first idxs
                first_route_idx_hopon = graph_tracker.bookkeeping.route_vert_id_to_idx_pair[string(car_id,"-",sorted_route[1][1])][1]
                first_route_idx_hopoff = graph_tracker.bookkeeping.route_vert_id_to_idx_pair[string(car_id,"-",sorted_route[1][1])][2]
                
                if graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][1][1] != first_route_idx_hopon
                    # @info "Car ",car_id," has updated its next route point"
                    graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][1][1] = first_route_idx_hopon
                    graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][2][1] = first_route_idx_hopoff
                end
            else
                push!(keys_to_delete, car_id)
            end
        else

            # New car
            route_info = car_info["route"]

            if route_info != nothing

                graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id] = [[0.,0.], [0.,0.]]

                # Add two variants - one for flight to hopon and one for hopon to hopoff
                prev_last_route_idx = num_vertices(graph_tracker.curr_graph)
                last_route_idx = prev_last_route_idx

                temp_vertices = Vector{DREAMROpenLoopVertex{US}}(undef, 0)
                temp_vert_ids = Vector{String}(undef, 0)

                # First add hopon variants
                for (id, timept) in sort(collect(route_info), by=x->parse(Float64, x[1]))

                    last_route_idx += 1

                    # Extract time and point respectively
                    timeval = timept[2]
                    waypt = Point(timept[1][1], timept[1][2])

                    # Get the ETA distribution
                    tp_dist = get_arrival_time_distribution(curr_timestep, timeval, params, graph_tracker.rng)

                    # Set pre and post bridge state to waypoint position
                    pre_bridge_state = DREAMRStateType{US}(FLIGHT, get_state_at_rest(US, waypt))
                    post_bridge_state = DREAMRStateType{US}(RIDE, get_state_at_rest(US, waypt))

                    metadata = DREAMRVertexMetadata(car_id, id)

                    new_vertex = DREAMROpenLoopVertex{US}(post_bridge_state, pre_bridge_state, DREAMRModeAction(HOPON), tp_dist, metadata)
                    add_vertex!(graph_tracker.curr_graph, new_vertex)

                    push!(temp_vertices, new_vertex)

                    # Now update bookkeeping vert_id_to_idx
                    vert_id = string(car_id, "-", id)
                    graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id] = [last_route_idx, 0]
                
                    push!(temp_vert_ids, vert_id)
                end

                # Set hopon range
                graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][1] = [prev_last_route_idx+1, last_route_idx]

                # NOW DO SAME FOR HOPOFF VERSIONS
                prev_last_route_idx = last_route_idx

                for (i, tmpver) in enumerate(temp_vertices)

                    last_route_idx += 1

                    # can copy data from tmpver, just change mode and action
                    cont_state = tmpver.pre_bridge_state.continuous

                    pre_bridge_state = DREAMRStateType{US}(RIDE, cont_state)
                    post_bridge_state = DREAMRStateType{US}(FLIGHT, cont_state)

                    new_vertex = DREAMROpenLoopVertex{US}(post_bridge_state, pre_bridge_state, DREAMRModeAction(HOPOFF), tmpver.tp, tmpver.metadata)
                    add_vertex!(graph_tracker.curr_graph, new_vertex)

                    # Update hopoff pair member
                    vert_id = temp_vert_ids[i]
                    graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id][2] = last_route_idx
                end

                # Set hopoff range
                graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][2] = [prev_last_route_idx+1, last_route_idx]

                # @info "Car", car_id, "has been added!"
            end
        end
    end

    # @show graph_tracker.bookkeeping.active_car_to_idx_ranges["car-115"]
    # @show keys_to_delete

    # Delete inactive keys
    for kd in keys_to_delete
        delete!(graph_tracker.bookkeeping.active_car_to_idx_ranges, kd)
    end
end


function HHPC.update_next_target!(cmssp::DREAMRCMSSPType, solver::DREAMRSolverType, curr_timestep::Int64)

    context_set = cmssp.curr_context_set
    next_target = solver.graph_tracker.curr_graph.vertices[solver.graph_tracker.curr_soln_path_idxs[2]]

    car_id = next_target.metadata.car_id
    vertex_id = next_target.metadata.vertex_id

    car_route_info = context_set.epochs_dict[string(context_set.curr_epoch)]["car-info"][car_id]["route"]

    if haskey(car_route_info, vertex_id) == false
        return false # Invalid update as car has disappeared
    end
    
    timeval = car_route_info[vertex_id][2]
    next_target.tp = get_arrival_time_distribution(curr_timestep, timeval, cmssp.params, solver.graph_tracker.rng)

    return true
end


function HHPC.simulate_cmssp!(cmssp::DREAMRCMSSPType{US,UA}, state::DREAMRStateType{US}, a::Union{Nothing,DREAMRActionType{UA}}, t::Int64,
                              rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    # Update context
    timeout = false
    cmssp.curr_context_set.curr_epoch += 1

    context_set = cmssp.curr_context_set
    params = cmssp.params

    if haskey(context_set.epochs_dict, string(context_set.curr_epoch))
        curr_epoch_dict = context_set.epochs_dict[string(context_set.curr_epoch)]
        epoch_cars = curr_epoch_dict["car-info"]
    else
        # No more car actions after 360
        timeout = true
        return (state, 0.0, false, timeout)
    end

    reward = -params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP
    failed_mode_switch = false

    # FIRST CHECK FOR nothing (abort) - just return same state
    if a == nothing
        return (state, reward, failed_mode_switch, timeout)
    end

    curr_mode = state.mode
    curr_cont_state = state.continuous


    if typeof(a.action) <: UAVAction

        @assert curr_mode == FLIGHT

        new_uav_state = next_state(params, curr_cont_state, a.action, rng)
        reward += -dynamics_cost(params, curr_cont_state, new_uav_state)
        next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)

    else
        if a.action.mode_action == HOPON

            @assert curr_mode == FLIGHT

            reward += -cmssp.modeswitch_mdp.R[1, 1]

            hopon_car_id = a.action.car_id

            # Hop on to specific car
            if hopon_car_id != ""

                # Check against car_position
                car_pos = Point(epoch_cars[hopon_car_id]["pos"][1], epoch_cars[hopon_car_id]["pos"][2])

                uav_pos = get_position(curr_cont_state)
                uav_speed = get_speed(curr_cont_state)

                car_uav_dist = point_dist(car_pos, uav_pos)

                if car_uav_dist < 2.0*params.scale_params.HOP_DISTANCE_THRESHOLD*params.time_params.MDP_TIMESTEP &&
                    uav_speed < params.scale_params.XYDOT_HOP_THRESH

                    @warn "Successful hop on to", hopon_car_id, " at epoch ", context_set.curr_epoch
                    new_uav_state = get_state_at_rest(US, car_pos)
                    next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
                    context_set.car_id = hopon_car_id
                else

                    @warn "Failed to hop on to car!"
                    @debug car_pos
                    failed_mode_switch = true
                    next_dreamr_state = state
                    context_set.car_id = ""
                end
            else
                # Undirected - search over all and update if possible
                next_dreamr_state = state
                hop_success = false
                min_car_uav_dist = Inf

                for (car_id, car_info) in epoch_cars

                    if car_info["route"] != nothing

                        car_pos = Point(car_info["pos"][1], car_info["pos"][2])

                        uav_pos = get_position(curr_cont_state)
                        uav_speed = get_speed(curr_cont_state)

                        car_uav_dist = point_dist(car_pos, uav_pos)

                        if car_uav_dist < min_car_uav_dist
                            min_car_uav_dist = car_uav_dist
                        end

                        if car_uav_dist < HOP_INFLATION_PARAMETER*params.scale_params.HOP_DISTANCE_THRESHOLD*params.time_params.MDP_TIMESTEP &&
                                uav_speed < params.scale_params.XYDOT_HOP_THRESH

                            new_uav_state = get_state_at_rest(US, car_pos)
                            next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
                            context_set.car_id = car_id

                            hop_success = true
                            @info "Successful hop on to", car_id
                            break
                        end
                    end
                end

                if hop_success == false
                    # @warn "Failed to hop on to car!"
                    reward += -params.cost_params.INVALID_HOP_PENALTY
                end
            end
        elseif a.action.mode_action == HOPOFF
            
            @assert curr_mode == RIDE

            # Set drone to car's position
            reward += -cmssp.modeswitch_mdp.R[2, 2]
            
            hopoff_car_id = context_set.car_id
            car_pos = Point(epoch_cars[hopoff_car_id]["pos"][1], epoch_cars[hopoff_car_id]["pos"][2])
            new_uav_state = get_state_at_rest(US, car_pos)
            
            next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)
            context_set.car_id = ""
        else
            reward += -cmssp.modeswitch_mdp.R[2, 3]
            @assert curr_mode == RIDE

            car_pos = Point(epoch_cars[context_set.car_id]["pos"][1], epoch_cars[context_set.car_id]["pos"][2])
            new_uav_state = get_state_at_rest(US, car_pos)

            next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
        end
    end

    return (next_dreamr_state, reward, failed_mode_switch, timeout)
end


function HHPC.display_context_future(context_set::DREAMRContextSet, future_epoch::Int64)
    println(context_set.epochs_dict[future_epoch])
end

# Return the vertex bridging action and metadata car id
function HHPC.get_bridging_action(vertex::DREAMROpenLoopVertex)
    return DREAMRModeAction(vertex.bridging_action.mode_action, vertex.metadata.car_id)
end


function get_ride_mdp(params::Parameters)
    return DREAMRHopoffMDPType(RIDE, params, Vector{Nothing}(undef, 0))
end

# Create a fake deterministic policy for hopoff - override methods
struct DREAMRDeterministicPolicy <: Policy
    mdp::Union{MDP,POMDP}
end

DREAMRDeterministicPolicy(params::Parameters) = DREAMRDeterministicPolicy(get_ride_mdp(params))

HHPC.get_mdp(ddp::DREAMRDeterministicPolicy) = ddp.mdp

function HHPC.get_best_intramodal_action(policy::DREAMRDeterministicPolicy, curr_timestep::Int64,
                                         tp_dist::TPDistribution, curr_state::US,
                                         target_state::US) where {US <: UAVState}
    
    # Just blindly return STAY - HOPOFF will be returned based on time
    return ModalAction(DREAMRModeAction(STAY), 0)
end



function HHPC.horizon_weighted_value(policy::DREAMRDeterministicPolicy, curr_timestep::Int64,
                                    tp_dist::TPDistribution, curr_state::US,
                                    target_state::US) where {US <: UAVState}
    
    # Just the cost corresponding to time difference
    # Return -Inf for minvalue because will never abort
    params = policy.mdp.params
    return -params.cost_params.TIME_COEFFICIENT*(mean(tp_dist) - curr_timestep), -Inf
end


# Based on the current context set, get the position of a car
# k timesteps into the future. Return nothing if car route inactive by then
function get_future_projected_car_position(car_route_info::Dict, curr_time::Float64, future_time::Float64)

    car_pos = Point(car_route_info["pos"][1], car_route_info["pos"][2])
    time_val = curr_time

    route_info = car_route_info["route"]
    sorted_route = sort(collect(route_info), by=x->parse(Float64, x[1]))
    
    next_car_pos = Point()
    next_time_val = Inf   

    for (id, timept) in sorted_route

        timeval = timept[2]
        if timeval > future_time
            next_time_val = timeval
            next_car_pos = Point(timept[1][1], timept[1][2])
        end
    end

    if next_time_val == Inf
        return nothing
    end

    # Now interpolate car position
    frac = (future_time - curr_time)/(next_time_val - curr_time)
    estimated_point = interpolate(car_pos, next_car_pos, frac)

    return estimated_point
end   


# Define specific CMSSP type for MCTS (with root depth)
struct DREAMRMCTSState{US <: UAVState}
    cmssp_state::DREAMRStateType{US}
    car_id::String
    depth_root::Int64
end

function DREAMRMCTSState(cmssp_state::DREAMRStateType)
    return DREAMRMCTSState(cmssp_state, "", 0)
end


mutable struct DREAMRMCTSType{US <: UAVState, UA <: UAVAction} <: POMDPs.MDP{DREAMRMCTSState{US}, DREAMRActionType{UA}} 
    cmssp::DREAMRCMSSPType{US,UA}
end

POMDPs.actions(dmcts::DREAMRMCTSType) = dmcts.cmssp.actions
POMDPs.actions(dmcts::DREAMRMCTSType, state::DREAMRMCTSState) = actions(dmcts.cmssp, state.cmssp_state)
POMDPs.n_actions(dmcts::DREAMRMCTSType) = length(dmcts.cmssp.actions)
POMDPs.discount(dmcts::DREAMRMCTSType) = 1.0 # SSP - Undiscounted
POMDPs.actionindex(dmcts::DREAMRMCTSType, a::DREAMRActionType) = a.action_idx

function POMDPs.isterminal(dmcts::DREAMRMCTSType, state::DREAMRMCTSState)
    return isterminal(dmcts.cmssp, state.cmssp_state)
end

# Requirement for MCTS-DPW
# NOTE - This uses the 
function POMDPs.generate_sr(dmcts::DREAMRMCTSType{US,UA}, state::DREAMRMCTSState{US},
                            a::DREAMRActionType{UA}, rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    cmssp = dmcts.cmssp
    params = cmssp.params
    cmssp_state = state.cmssp_state
    curr_cont_state = cmssp_state.continuous

    reward = -params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP
    
    if typeof(a.action) <: UAVAction # Just simulate control

        new_uav_state = next_state(params, curr_cont_state, a.action, rng)
        reward += -dynamics_cost(params, curr_cont_state, new_uav_state)
        next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)
        next_dmcts_state = DREAMRMCTSState(next_dreamr_state, "", state.depth_root+1)
    
    else
        
        # Is a mode-switch action, propagate context and see what to do
        context_set = cmssp.curr_context_set
        curr_time = context_set.curr_epoch*params.time_params.MDP_TIMESTEP
        future_time = curr_time + (state.depth_root+1)*params.time_params.MDP_TIMESTEP

        curr_epoch_dict = context_set.epochs_dict[string(context_set.curr_epoch)]
        epoch_cars = curr_epoch_dict["car-info"]
        
        # Loop through all current cars and evaluate
        # DOESN'T MATTER if currently it has an ID
        if a.action.mode_action == HOPON

            # If nothing else, copy current state
            next_dmcts_state = DREAMRMCTSState(cmssp_state, "", state.depth_root+1)
            hop_success = false

            reward += -cmssp.modeswitch_mdp.R[1, 1]

            for (car_id, car_info) in epoch_cars

                if car_info["route"] != nothing

                    car_pos = get_future_projected_car_position(car_info, curr_time, future_time)

                    if car_pos != nothing # Not if car not valid at that future timestep

                        uav_pos = get_position(curr_cont_state)
                        uav_speed = get_speed(curr_cont_state)

                        car_uav_dist = point_dist(car_pos, uav_pos)

                        if car_uav_dist < HOP_INFLATION_PARAMETER*params.scale_params.HOP_DISTANCE_THRESHOLD*params.time_params.MDP_TIMESTEP &&
                                uav_speed < params.scale_params.XYDOT_HOP_THRESH

                            new_uav_state = get_state_at_rest(US, car_pos)
                            next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
                            
                            next_dmcts_state = DREAMRMCTSState(next_dreamr_state, car_id, state.depth_root+1)

                            hop_success = true

                            @info "Successful hop to ", car_id," in sim!"

                            break
                        end
                    end
                end
            end

            # if hop_success == false
            #     reward += -params.cost_params.INVALID_HOP_PENALTY
            # end
        elseif a.action.mode_action == HOPOFF

            @debug "HOPOFF in SIM!"
            reward += -cmssp.modeswitch_mdp.R[2, 2]

            hopoff_car_id = state.car_id
            car_pos = Point(epoch_cars[hopoff_car_id]["pos"][1], epoch_cars[hopoff_car_id]["pos"][2])
            new_uav_state = get_state_at_rest(US, car_pos)
            next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)
            
            next_dmcts_state = DREAMRMCTSState(next_dreamr_state, "", state.depth_root+1)

        else
            reward += -cmssp.modeswitch_mdp.R[2, 3]
            car_info = epoch_cars[state.car_id]
            @debug "STAY in SIM!"
            
            car_pos = get_future_projected_car_position(car_info, curr_time, future_time)
            if car_pos != nothing
                new_uav_state = get_state_at_rest(US, car_pos)
                next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
                next_dmcts_state = DREAMRMCTSState(next_dreamr_state, state.car_id, state.depth_root+1)
            else
                # Car has stopped, must get off
                @debug "STAY cannot be simulated as car has ended by this time"
                reward += -params.cost_params.INVALID_HOP_PENALTY
                car_pos = Point(car_info["pos"][1], car_info["pos"][2])

                new_uav_state = get_state_at_rest(US, car_pos)
                next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)
                next_dmcts_state = DREAMRMCTSState(next_dreamr_state, "", state.depth_root+1)
            end
        end
    end

    if isterminal(dmcts, next_dmcts_state)
        reward += params.cost_params.FLIGHT_REACH_REWARD
    end

    return (next_dmcts_state, reward)
end 


function estimate_value_dreamr(dmcts::DREAMRMCTSType, state::DREAMRMCTSState, depth::Int64)

    goal_pos = get_position(dmcts.cmssp.goal_state.continuous)
    curr_pos = get_position(state.cmssp_state.continuous)

    value = -0.75*dmcts.cmssp.params.cost_params.FLIGHT_COEFFICIENT*point_dist(curr_pos, goal_pos)

    return value
end

function estimate_value_dreamr(policy::P, dmcts::DREAMRMCTSType, state::DREAMRMCTSState{US}, depth::Int64) where {P <: Policy, US <: UAVState}

    if state.cmssp_state.mode == FLIGHT
        relative_state = rel_state(state.cmssp_state.continuous, dmcts.cmssp.goal_state.continuous)
        value = POMDPs.value(policy, relative_state) - dmcts.cmssp.params.cost_params.FLIGHT_REACH_REWARD
        return value
    else
        # How to compute value of ride state?
        # Look at all car waypoints and compute their value to goal
        # and also consider time to get there
        params = dmcts.cmssp.params
        context_set = dmcts.cmssp.curr_context_set

        curr_time = context_set.curr_epoch*params.time_params.MDP_TIMESTEP
        future_time = curr_time + (state.depth_root+1)*params.time_params.MDP_TIMESTEP

        curr_epoch_dict = context_set.epochs_dict[string(context_set.curr_epoch)]
        epoch_cars = curr_epoch_dict["car-info"]
        # @show state
        car_info = epoch_cars[state.car_id]

        best_value = -Inf
        
        # Now iterate over car waypoints and find their best value
        route_info = car_info["route"]
        sorted_route = sort(collect(route_info), by=x->parse(Float64, x[1]))

        for (id, timept) in sorted_route

            timeval = timept[2]
            if timeval >= future_time # Only choose points in future

                hopoff_state = get_state_at_rest(US, Point(timept[1][1], timept[1][2]))
                relative_state = rel_state(hopoff_state, dmcts.cmssp.goal_state.continuous)

                val = POMDPs.value(policy, relative_state) - params.cost_params.FLIGHT_REACH_REWARD -
                      params.cost_params.TIME_COEFFICIENT*(timeval - future_time)

                best_value = (val > best_value) ? val : best_value
            end
        end

        # @show best_value
        if best_value == -Inf
            relative_state = rel_state(state.cmssp_state.continuous, dmcts.cmssp.goal_state.continuous)
            best_value = POMDPs.value(policy, relative_state) - dmcts.cmssp.params.cost_params.FLIGHT_REACH_REWARD
        end

        return best_value
    end
end


function init_q_dreamr(policy::P, dmcts::DREAMRMCTSType, state::DREAMRMCTSState{US}, a::DREAMRActionType) where {P <: Policy, US <: UAVState}

    params = dmcts.cmssp.params
    curr_cont_state = state.cmssp_state.continuous

    val = -params.time_params.MDP_TIMESTEP*params.cost_params.TIME_COEFFICIENT

    # If control action, use flight policy action value function for goal state
    if typeof(a.action) <: UAVAction

        new_uav_state = next_state(params, curr_cont_state, a.action)
        
        val += -dynamics_cost(params, curr_cont_state, new_uav_state)       
        
        relative_state = rel_state(new_uav_state, dmcts.cmssp.goal_state.continuous)
        val += POMDPs.value(policy, relative_state) - dmcts.cmssp.params.cost_params.FLIGHT_REACH_REWARD
        
    end
    
    return val
    
    # else
    #     context_set = dmcts.cmssp.curr_context_set
    #     curr_epoch_dict = context_set.epochs_dict[string(context_set.curr_epoch)]
    #     epoch_cars = curr_epoch_dict["car-info"]

    #     hop_success = false
        
    #     # If hopon and fail, use flight policy + penalty for missed hop
    #     if a.action.mode_action == HOPON

    #         next_dmcts_state = DREAMRMCTSState(state.cmssp_state, "", state.depth_root+1)

    #         val += -dmcts.cmssp.modeswitch_mdp.R[1, 1]

    #         # val += params.cost_params.INVALID_HOP_PENALTY/2.0

    #         for (car_id, car_info) in epoch_cars

    #             if car_info["route"] != nothing

    #                 car_pos = Point(car_info["pos"][1], car_info["pos"][2])

    #                 uav_pos = get_position(curr_cont_state)
    #                 uav_speed = get_speed(curr_cont_state)

    #                 car_uav_dist = point_dist(car_pos, uav_pos)

    #                 if car_uav_dist < HOP_INFLATION_PARAMETER*params.scale_params.HOP_DISTANCE_THRESHOLD*params.time_params.MDP_TIMESTEP &&
    #                         uav_speed < params.scale_params.XYDOT_HOP_THRESH

    #                     new_uav_state = get_state_at_rest(US, car_pos)
    #                     next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
                        
    #                     next_dmcts_state = DREAMRMCTSState(next_dreamr_state, car_id, state.depth_root+1)

    #                     hop_success = true

    #                     @show next_dmcts_state
    #                     val += estimate_value_dreamr(policy, dmcts, next_dmcts_state, 0)

    #                     @debug "Successful hop to ", car_id," in sim!"
    #                     break
    #                 end
    #             end
    #         end
    #     elseif a.action.mode_action == HOPOFF

    #         val += -dmcts.cmssp.modeswitch_mdp.R[2, 2]

    #         hopoff_car_id = state.car_id


    #         car_pos = Point(epoch_cars[hopoff_car_id]["pos"][1], epoch_cars[hopoff_car_id]["pos"][2])
    #         new_uav_state = get_state_at_rest(US, car_pos)
    #         next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)
            
    #         next_dmcts_state = DREAMRMCTSState(next_dreamr_state, "", state.depth_root+1)
    #         val += estimate_value_dreamr(policy, dmcts, next_dmcts_state, 0)

    #     else
    #         val += -dmcts.cmssp.modeswitch_mdp.R[2, 3]

    #         car_pos = Point(epoch_cars[state.car_id]["pos"][1], epoch_cars[state.car_id]["pos"][2])

    #         new_uav_state = get_state_at_rest(US, car_pos)

    #         next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
    #         next_dmcts_state = DREAMRMCTSState(next_dreamr_state, state.car_id, state.depth_root+1)
    #         val += estimate_value_dreamr(policy, dmcts, next_dmcts_state, 0)
    #     end

    #     return val
    # end
end