## Define DREAMR types (based on US, UA)
@enum DREAMR_MODETYPE FLIGHT=1 RIDE=2
@enum HOP_ACTION HOPON=1 HOPOFF=2 STAY=3

struct DREAMRModeAction
    mode_action::HOP_ACTION
    car_id::String
end

DREAMRModeAction(mode_action::HOP_ACTION) = DREAMRModeAction(mode_action, "")

const DREAMRStateType{US} = CMSSPState{DREAMR_MODETYPE, US} where {US <: UAVState}
const DREAMRActionType{UA} = CMSSPAction{DREAMRModeAction, UA} where {UA <: UAVAction}
const DREAMRCMSSPType{US, UA} = CMSSP{DREAMR_MODETYPE, US, DREAMRModeAction, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRModalMDPType{US, UA} = ModalMDP{DREAMR_MODETYPE, US, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMROpenLoopVertex{US} = OpenLoopVertex{DREAMR_MODETYPE, US, DREAMRModeAction, DREAMRVertexMetadata} where {US <: UAVState}
const DREAMRGraphTracker{US, UA} = GraphTracker{DREAMR_MODETYPE, US, DREAMRModeAction, UA, DREAMRVertexMetadata, DREAMRBookkeeping, RNG} where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}
const DREAMRModePair = Tuple{DREAMR_MODETYPE, DREAMR_MODETYPE}

## Context
struct DREAMRContextSet
    curr_epoch::Int64
    curr_epoch_dict::Dict
    curr_car_id::String
end

## Mode Info
const DREAMR_MODES = [FLIGHT, RIDE]
const DREAMR_GOAL_MODE = FLIGHT
const DREAMR_MODE_ACTIONS = [HOPON, HOPOFF, STAY]

const DREAMRSolverType{US, UA} = HHPCSolver{DREAMR_MODETYPE, US, DREAMRModeAction, UA, DREAMRContextSet, Parameters} where {US <: UAVState, UA <: UAVAction}


## Car route stuff
struct DREAMRVertexMetadata
    car_id::String
    vertex_id::String
end

Base.zero(DREAMRVertexMetadata) = DREAMRVertexMetadata("", "")

struct DREAMRBookkeeping
    active_car_to_idx_ranges::Dict{String, MVector{2, MVector{2, Int64}}} # Maps car name to range of vertices that represent hopon
    route_vert_id_to_idx_pair::Dict{String, MVector{2, Int64}} # Maps a specific route ID (car-id+"-"+vertex_id) to the indices of hopon and hopoff versions
    goal_vertex_idx::Int64
end

DREAMRBookkeeping() = DREAMRBookkeeping(Dict{String, MVector{2, MVector{2, Int64}}}, Dict{String, MVector{2, Int64}}(), 0)


# This is only really needed for baseline
function get_dreamr_actions{UA}(dynamics::UDM) where {UA <: UAVAction, UDM <: UAVDynamicsModel}

    actions = Vector{DREAMRActionType{UA}}(undef, 0)
    idx = 1
    for dma in DREAMR_MODE_ACTIONS
        push!(actions, DREAMRActionType(DREAMRModeAction(dma), idx))
        idx += 1
    end

    uav_dynamics_actions = get_uav_dynamics_actions(dynamics)
    for uavda in uav_dynamics_actions
        push!(actions, DREAMRActionType(uavda, idx))
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

function create_dreamr_cmssp{US,UA}(dynamics::UDM, goal_point::DREAMRStateType{US}, params::Parameters) where {US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel}

    actions = get_dreamr_actions(dynamics)
    switch_mdp = get_dreamr_mdp()

    return DREAMRCMSSPType{US,UA}(actions, DREAMR_MODES, switch_mdp, goal_point, params)
end

function POMDPs.isterminal(mdp::DREAMRModalMDPType{US,UA}, relative_state::US) where {US <: UAVState, UA <: UAVAction}
    
    curr_pos_norm = point_norm(get_position(relative_state))
    curr_speed = get_speed(relative_state)

    return (curr_pos_norm < mdp.params.time_params.MDP_TIMESTEP*mdp.params.scale_params.HOP_DISTANCE_THRESHOLD) &&
        (curr_speed < mdp.params.scale_params.XYDOT_HOP_THRESH)
end

# Needs to be bound at top-level
function isterminal(cmssp::DREAMRCMSSPType{US,UA}, state::DREAMRStateType{US},
                    goal_point::Point) where {US <: UAVState, UA <: UAVAction}
    
    curr_point = get_position(state)
    curr_speed = get_speed(relative_state)
    dist = point_dist(curr_point, goal_point)

    return (state.mode == FLIGHT) && (dist < mdp.params.time_params.MDP_TIMESTEP*mdp.params.scale_params.HOP_DISTANCE_THRESHOLD) &&
            (curr_speed < mdp.params.scale_params.XYDOT_HOP_THRESH)
end

function HHPC.get_relative_state(mdp::DREAMRModalMDPType{US,UA}, source::US, target::US) where {US <: UAVState, UA <: UAVAction}
    return rel_state(source, target)
end


## Conversion for multirotor - can add others as needed
function POMDPs.convert_s(::Type{V}, s::MultiRotorUAVState, mdp::DREAMRModalMDPType{MultiRotorUAVState, MultiRotorUAVAction})
    v = SVector{4,Float64}(s.x, s.y, s.xdot, s.ydot)
end

function POMDPs.convert_s(::Type{MultiRotorUAVState}, v::AbstractVector{Float64}, mdp::DREAMRModalMDPType{MultiRotorUAVState, MultiRotorUAVAction})
    s = MultiRotorUAVState(v[1], v[2], v[3], v[4])
end

function POMDPs.reward(mdp::DREAMRModalMDPType{US, UA}, state::US, cont_action::UA, statep::US) where {US <: UAVState, UA <: UAVAction}
    
    cost = mdp.energy_time_alpha*mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP
    cost += dynamics_cost(mdp.params, state, statep)
    
    return -cost
end

# NEEDS TO BE BOUND
function generate_sr(mdp::DREAMRModalMDPType{US,UA}, state::US, cont_action::UA, dynamics::UDM, 
                     rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel. RNG <: AbstractRNG}
    
    statep = next_state(dynamics, state, cont_action, rng)
    reward = reward(mdp, state, cont_action, statep)
    
    return (statep, reward)
end


function HHPC.expected_reward(mdp::DREAMRModalMDPType{US,UA}, 
                              state::US, cont_action::UA, rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}
    
    params = mdp.params
    avg_rew = 0.0
    
    for i = 1:params.scale_params.MC_GENERATIVE_NUMSAMPLES
        (_, r) = next_state_reward(mdp,state, cont_action, rng)
        avg_rew += r
    end
    
    avg_rew /= params.num_generative_samples
    return avg_rew
end


function HHPC.generate_goal_sample_set!(cmssp::DREAMRCMSSPType{US,UA}, vertex::DREAMROpenLoopVertex{US},
                                       context_set::DREAMRContextSet, graph_tracker::DREAMRGraphTracker{US, UA},
                                       rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    # If current goal vertex idx is 0, add goal and send it as nbr
    if graph_tracker.goal_vertex_idx == 0
        graph_tracker.goal_vertex_idx = num_vertices(graph_tracker.curr_graph)+1 # Since it will be added
        return [DREAMRStateType(FLIGHT, cmssp.goal_point)], []
    else
        # Just return current goal vertex idx
        return [], [graph_tracker.goal_vertex_idx]
    end
end



function get_arrival_time_distribution(mean_time::Float64, params::Parameters, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    # Returns a TPDist
    time_dist = Normal(mean_time, params.time_params.CAR_TIME_STD)
    temp_tp_dict = Dict{Int64, Float64}()


    for j = 1:params.time_params.MC_TIME_NUMSAMPLES
        tval = (mean_time + rand(rng, noise_model)) / params.time_params.MDP_TIMESTEP

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


function HHPC.generate_bridge_sample_set!(cmssp::DREAMRCMSSPType{US,UA}, vertex::DREAMROpenLoopVertex,
                                         mode_pair::DREAMRModePair,
                                         context_set::DREAMRContextSet, graph_tracker::DREAMRGraphTracker{US, UA},
                                         rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    # NOTE - Will never actually generate anything; only get neighbour idxs
    params = cmssp.params

    nbrs_to_add = Vector{Int64}(undef, 0)

    if vertex.state.mode == FLIGHT # If it is a drone or post-hopoff vertex

        # Iterate over cars
        for (car_id, ranges) in graph_tracker.bookkeeping.active_car_to_idx_ranges

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
    end

    return ([], nbrs_to_add)
end


function HHPC.generate_next_valid_modes(cmssp::DREAMRCMSSPType{US, UA}, mode::DREAMR_MODETYPE, context_set::DREAMRContextSet)

    if mode == FLIGHT
        return [(DREAMRModeAction(HOPON), RIDE)]
    else
        return [(DREAMRModeAction(HOPOFF), FLIGHT)]
    end
end


function HHPC.update_vertices_with_context!(cmssp::DREAMRCMSSPType{US, UA}, graph_tracker::DREAMRGraphTracker{US, UA},
                                            switch::DREAMRModePair, context_set::DREAMRContextSet) where {US <: UAVState, UA <: UAVAction}

    # Iterate over cars, add if new or update if old
    epoch_cars = epoch["car-info"]

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
                    tp_dist = get_arrival_time_distribution(timeval, params, graph_tracker.rng)

                    # Update the hopon and hopoff vert tps
                    vert_id = string(car_id, "-", id)
                    graph_tracker.vertices[graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id][1]].tp = tp_dist
                    graph_tracker.vertices[graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id][2]].tp = tp_dist
                end

                # If first waypoint of route updated, change BOTH hopon and hopoff first idxs
                first_route_idx_hopon = graph_tracker.bookkeeping.route_vert_id_to_idx_pair[string(car_id,"-",sorted_route[1][1])][1]
                first_route_idx_hopoff = graph_tracker.bookkeeping.route_vert_id_to_idx_pair[string(car_id,"-",sorted_route[1][1])][2]
                
                if graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][1][1] != first_route_idx_hopon
                    @info "Car ",car_id," has updated its next route point"
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
                    tp_dist = get_arrival_time_distribution(timeval, params, rng)

                    # Set pre and post bridge state to waypoint position
                    pre_bridge_state = DREAMRStateType{US}(FLIGHT, get_state_at_rest(US, waypt))
                    post_bridge_state = DREAMRStateType{US}(RIDE, get_state_at_rest(US, waypt))

                    metadata = DREAMRVertexMetadata(car_id, id)

                    new_vertex = DREAMROpenLoopVertex{US}(post_bridge_state, pre_bridge_state, HOPON, tp_dist, metadata)
                    add_vertex!(graph_tracker.curr_graph, new_vertex)

                    push!(temp_vertices, new_vertex)

                    # Now update bookkeeping vert_id_to_idx
                    vert_id = string(car_id, "-", id)
                    graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id] = [last_route_idx, 0]
                
                    push(temp_vert_ids, vert_id)
                end

                # Set hopon range
                graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][1] = [prev_last_route_idx+1, last_route_idx]

                # NOW DO SAME FOR HOPOFF VERSIONS
                prev_last_route_idx = last_route_idx

                for (i, tmpver) in enumerate(temp_vertices)

                    last_route_idx += 1

                    # can copy data from tmpver, just change mode and action
                    state = tmpver.pre_bridge_state.continuous

                    pre_bridge_state = DREAMRStateType{US}(RIDE, state)
                    post_bridge_state = DREAMRStateType{US}(FLIGHT, state)

                    new_vertex = DREAMROpenLoopVertex{US}(post_bridge_state, pre_bridge_state, HOPOFF, tmpver.tp, tmpver.metadata)
                    add_vertex!(graph_tracker.curr_graph, new_vertex)

                    # Update hopoff pair member
                    vert_id = temp_vert_ids[i]
                    graph_tracker.bookkeeping.route_vert_id_to_idx_pair[vert_id][2] = last_route_idx
                end

                # Set hopoff range
                graph_tracker.bookkeeping.active_car_to_idx_ranges[car_id][2] = [prev_last_route_idx+1, last_route_idx]

                @info "Car", car_id, "has been added!"
            end
        end
    end

    # Delete inactive keys
    for kd in keys_to_delete
        delete!(graph_tracker.bookkeeping.active_car_to_idx_ranges, kd)
    end
end

# Needs to be BOUND
function HHPC.update_context_set!(cmssp::DREAMRCMSSPType, context_set::DREAMRContextSet, 
                             episode_dict::Dict, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Update epoch number
    context_set.curr_epoch+=1

    # Update context set
    context_set.curr_epoch_dict = episode_dict[context_set.curr_epoch]
end


function HHPC.simulate_cmssp!(cmssp::DREAMRCMSSPType{US, UA}, state::DREAMRStateType{US}, a::DREAMRActionType{UA}, t::Int64,
                             context_set::DREAMRContextSet, dynamics::UDM, rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel, RNG <: AbstractRNG}

    params = cmssp.params
    epoch_cars = context_set.curr_epoch_dict["car-info"]

    curr_mode = state.mode
    curr_cont_state = state.continuous

    reward = -params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP
    failed_mode_switch = false

    if typeof(a.action) <: UAVAction

        @assert curr_mode == FLIGHT

        new_uav_state = next_state(dynamics, curr_cont_state, a.action, rng)
        reward += -dynamics_cost(dynamics, curr_cont_state, new_uav_state)
        next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)

    else
        if a.action.mode_action == HOPON

            @assert curr_mode == FLIGHT

            reward += -cmssp.modeswitch_mdp.R[1, 1]

            hopon_car_id = a.action.car_id

            # Check against car_position
            car_pos = Point(epoch_cars[hopon_car_id]["pos"][1], epoch_cars[hopon_car_id]["pos"][2])

            uav_pos = get_position(curr_cont_state)
            uav_speed = get_speed(curr_cont_state)

            car_uav_dist = point_dist(car_pos, uav_pos)

            if car_uav_dist < 2.0*params.scale_params.HOP_DISTANCE_THRESHOLD*params.time_params.MDP_TIMESTEP &&
                uav_speed < params.scale_params.XYDOT_HOP_THRESH

                @info "Successful hop on to", hopon_car_id, " at epoch ", context_set.curr_epoch
                new_uav_state = get_state_at_rest(US, car_pos)
                next_dreamr_state = DREAMRStateType{US}(RIDE, car_pos)
                context_set.curr_car_id = hopon_car_id
            else

                @warn "Failed to hop on to car!"
                println("Position of car ",hopon_car_id,": ",car_pos)
                failed_mode_switch = true
                next_dreamr_state = state
                context_set.curr_car_id = ""
            end
        elseif a.action.mode_action == HOPOFF
            
            @assert curr_mode == RIDE

            # Set drone to car's position
            reward += -cmssp.modeswitch_mdp.R[2, 2]
            
            hopoff_car_id = context_set.curr_car_id
            car_pos = Point(epoch_cars[hopoff_car_id]["pos"][1], epoch_cars[hopoff_car_id]["pos"][2])
            new_uav_state = get_state_at_rest(US, car_pos)
            
            next_dreamr_state = DREAMRStateType{US}(FLIGHT, new_uav_state)
            context_set.curr_car_id = ""
        else
            @assert curr_mode == RIDE

            curr_car_pos = Point(epoch_cars[context_set.curr_car_id]["pos"][1], epoch_cars[context_set.curr_car_id]["pos"][2])
            new_uav_state = get_state_at_rest(US, car_pos)

            next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
        end
    end

    return (next_dreamr_state, reward, failed_mode_switch)
end

# function HHPC.get_bridging_action(cmssp::DREAMRCMSSPType{US, UA}, state::DREAMRStateType{US},
#                                   next_target::DREAMROpenLoopVertex{US}) where {US <: UAVState, UA <: UAVAction}

#     return next_target.bridging_action
# end

# TODO : Create a fake deterministic policy for hopoff - in edge weight and get_best_intramodal_action