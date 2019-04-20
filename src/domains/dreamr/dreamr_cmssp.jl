## Define DREAMR types (based on US, UA)
@enum DREAMR_MODETYPE FLIGHT=1 RIDE=2
@enum HOP_ACTION HOPON=1 HOPOFF=2 STAY=3

## Mode Info
const DREAMR_MODES = [FLIGHT, RIDE]
const DREAMR_GOAL_MODE = FLIGHT
const DREAMR_MODE_ACTIONS = [HOPON, HOPOFF, STAY]


struct DREAMRModeAction
    mode_action::HOP_ACTION
    car_id::String
end

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
    curr_car_id::String
end

function get_dreamr_episode_context(ep_dict::Dict)
    return DREAMRContextSet(0, ep_dict, "")
end


const DREAMRStateType{US} = CMSSPState{DREAMR_MODETYPE, US} where {US <: UAVState}
const DREAMRActionType{UA} = CMSSPAction{DREAMRModeAction, UA} where {UA <: UAVAction}
const DREAMRCMSSPType{US,UA} = CMSSP{DREAMR_MODETYPE, US, DREAMRModeAction, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRCFMDPType{US,UA} = ModalFinHorMDP{DREAMR_MODETYPE, US, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRUFMDPType{US,UA} = ModalInfHorMDP{DREAMR_MODETYPE, US, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRHopoffMDPType = ModalFinHorMDP{DREAMR_MODETYPE, Nothing, Nothing, Parameters}
const DREAMROpenLoopVertex{US} = OpenLoopVertex{DREAMR_MODETYPE, US, DREAMRModeAction, DREAMRVertexMetadata} where {US <: UAVState}
const DREAMRGraphTracker{US,UA,RNG} = GraphTracker{DREAMR_MODETYPE, US, DREAMRModeAction, DREAMRVertexMetadata, DREAMRBookkeeping, RNG} where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}
const DREAMRModePair = Tuple{DREAMR_MODETYPE, DREAMR_MODETYPE}
const DREAMRSolverType{US,UA,RNG} = HHPCSolver{DREAMR_MODETYPE, US, DREAMRModeAction, UA, DREAMRContextSet, Parameters, DREAMRVertexMetadata, DREAMRBookkeeping, RNG} where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}


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

function create_dreamr_cmssp(::Type{US}, ::Type{UA},
                             goal_point::Point, params::Parameters) where {US <: UAVState, UA <: UAVAction}

    actions = get_dreamr_actions(UA, params)
    switch_mdp = get_dreamr_mdp()
    goal_state = DREAMRStateType{US}(FLIGHT, get_state_at_rest(US, goal_point))

    return DREAMRCMSSPType{US,UA}(actions, DREAMR_MODES, switch_mdp, goal_state, params)
end

function set_dreamr_goal!(cmssp::DREAMRCMSSPType{US,UA}, goal_point::Point) where {US <: UAVState, UA <: UAVAction}
    cmssp.goal_state = DREAMRStateType{US}(FLIGHT, get_state_at_rest(US, goal_point))
end

function POMDPs.isterminal(cmssp::DREAMRCMSSPType{US,UA}, state::DREAMRStateType{US}) where {US <: UAVState, UA <: UAVAction}
    
    curr_point = get_position(state.continuous)
    curr_speed = get_speed(state.continuous)
    
    goal_point = get_position(cmssp.goal_state.continuous)

    dist = point_dist(curr_point, goal_point)

    params = cmssp.params

    return (state.mode == FLIGHT) && (dist < params.time_params.MDP_TIMESTEP*params.scale_params.HOP_DISTANCE_THRESHOLD) &&
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
                                       context_set::DREAMRContextSet, graph_tracker::DREAMRGraphTracker,
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



function get_arrival_time_distribution(mean_time::Float64, params::Parameters, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    # Returns a TPDist
    time_dist = Normal(mean_time, params.time_params.CAR_TIME_STD/2.0)
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
                                         context_set::DREAMRContextSet, graph_tracker::DREAMRGraphTracker,
                                         rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # NOTE - Will never actually generate anything; only get neighbour idxs
    params = cmssp.params

    nbrs_to_add = Vector{Int64}(undef, 0)

    if vertex.state.mode == FLIGHT # If it is a drone or post-hopoff vertex

        @assert mode_pair == (FLIGHT, RIDE)

        # Iterate over cars
        for (car_id, ranges) in graph_tracker.bookkeeping.active_car_to_idx_ranges

            hopon_vert_range = ranges[1]

            for next_idx in hopon_vert_range[1]:hopon_vert_range[2]

                # Check if max drone speed not enough
                if next_idx == 0
                    @info "NEXT_IDX is 0"
                    @show car_id
                    @show ranges
                    readline()
                end
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
        # @show vertex
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
    end

    return ([], nbrs_to_add)
end


function HHPC.generate_next_valid_modes(cmssp::DREAMRCMSSPType, mode::DREAMR_MODETYPE, context_set::DREAMRContextSet)

    if mode == FLIGHT
        return [(DREAMRModeAction(HOPON), RIDE)]
    else
        return [(DREAMRModeAction(HOPOFF), FLIGHT)]
    end
end


function HHPC.update_vertices_with_context!(cmssp::DREAMRCMSSPType{US,UA}, graph_tracker::DREAMRGraphTracker,
                                            context_set::DREAMRContextSet) where {US <: UAVState, UA <: UAVAction}

    # Iterate over cars, add if new or update if old
    @show context_set.curr_epoch
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
                    tp_dist = get_arrival_time_distribution(timeval, params, graph_tracker.rng)

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

            # if car_id == "car-115"
            #     println("CAR 115!")
            #     @show first_route_idx_hopon, first_route_idx_hopoff
            #     readline()
            # end
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
                    tp_dist = get_arrival_time_distribution(timeval, params, graph_tracker.rng)

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

function HHPC.update_context_set!(cmssp::DREAMRCMSSPType, context_set::DREAMRContextSet,
                                  rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Update epoch number
    context_set.curr_epoch+=1

end

function HHPC.update_next_target!(cmssp::DREAMRCMSSPType, solver::DREAMRSolverType, context_set::DREAMRContextSet)

    next_target = solver.graph_tracker.curr_graph.vertices[solver.graph_tracker.curr_soln_path_idxs[2]]
    # @show next_target
    if is_inf_hor(next_target.tp) == false
        car_id = next_target.metadata.car_id
        vertex_id = next_target.metadata.vertex_id

        car_route_info = context_set.epochs_dict[string(context_set.curr_epoch)]["car-info"][car_id]["route"]

        timeval = car_route_info[vertex_id][2]

        next_target.tp = get_arrival_time_distribution(timeval, cmssp.params, solver.graph_tracker.rng)
    end
    # @show next_target
    # readline()
end


function HHPC.simulate_cmssp!(cmssp::DREAMRCMSSPType{US,UA}, state::DREAMRStateType{US}, a::DREAMRActionType{UA}, t::Int64,
                             context_set::DREAMRContextSet, rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    params = cmssp.params

    curr_epoch_dict = context_set.epochs_dict[string(context_set.curr_epoch)]
    epoch_cars = curr_epoch_dict["car-info"]

    curr_mode = state.mode
    curr_cont_state = state.continuous

    reward = -params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP
    failed_mode_switch = false

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

            # Check against car_position
            car_pos = Point(epoch_cars[hopon_car_id]["pos"][1], epoch_cars[hopon_car_id]["pos"][2])

            uav_pos = get_position(curr_cont_state)
            uav_speed = get_speed(curr_cont_state)

            car_uav_dist = point_dist(car_pos, uav_pos)

            if car_uav_dist < 3.0*params.scale_params.HOP_DISTANCE_THRESHOLD*params.time_params.MDP_TIMESTEP &&
                uav_speed < params.scale_params.XYDOT_HOP_THRESH

                @info "Successful hop on to", hopon_car_id, " at epoch ", context_set.curr_epoch
                new_uav_state = get_state_at_rest(US, car_pos)
                next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
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

            car_pos = Point(epoch_cars[context_set.curr_car_id]["pos"][1], epoch_cars[context_set.curr_car_id]["pos"][2])
            new_uav_state = get_state_at_rest(US, car_pos)

            next_dreamr_state = DREAMRStateType{US}(RIDE, new_uav_state)
        end
    end

    return (next_dreamr_state, reward, failed_mode_switch)
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