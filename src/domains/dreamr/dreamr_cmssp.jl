## Epoch dict format
## Top-level



## Define DREAMR types (based on US, UA)
@enum DREAMR_MODETYPE FLIGHT=1 RIDE=2
@enum HOP_ACTION HOPON=1 HOPOFF=2

const DREAMRStateType{US} = CMSSPState{DREAMR_MODE, US} where {US <: UAVState}
const DREAMRActionType{UA} = CMSSPAction{HOP_ACTION, UA} where {UA <: UAVAction}
const DREAMRCMSSPType{US, UA} = CMSSP{DREAMR_MODE, US, HOP_ACTION, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMRModalMDPType{US, UA} = ModalMDP{DREAMR_MODE, US, UA, Parameters} where {US <: UAVState, UA <: UAVAction}
const DREAMROpenLoopVertex{US} = OpenLoopVertex{DREAMR_MODE, US, HOP_ACTION} where {US <: UAVState}
const DREAMRModePair = Tuple{DREAMR_MODE, DREAMR_MODE}

## Context
struct DREAMRContextSet
    curr_epoch::Int64
    curr_epoch_dict::Dict
end

## Mode Info
const DREAMR_MODES = [FLIGHT, RIDE]
const DREAMR_GOAL_MODE = FLIGHT
const DREAMR_MODE_ACTIONS = [HOPON, STAY, HOPOFF]

const DREAMRSolverType{US, UA} = HHPCSolver{DREAMR_MODE, US, HOP_ACTION, UA, DREAMRContextSet, Parameters} where {US <: UAVState, UA <: UAVAction}


## Car route stuff
struct RouteVertexMetadata
    car_id::String
    vertex_id::String
end



# This is only really needed for baseline
function get_dreamr_actions{UA}(dynamics::UDM) where {UA <: UAVAction, UDM <: UAVDynamicsModel}

    actions = Vector{DREAMRActionType{UA}}(undef, 0)
    idx = 1
    for dma in DREAMR_MODE_ACTIONS
        push!(actions, DREAMRActionType(dma,idx))
        idx += 1
    end

    uav_dynamics_actions = get_uav_dynamics_actions(dynamics)
    for uavda in uav_dynamics_actions
        push!(actions, DREAMRActionType(uavda,idx))
        idx += 1
    end

    return actions
end

function get_dreamr_mdp()

    T = zeros(2,2,2)
    # Flight + hopon = ride
    T[1,1,2] = 1.0
    # Ride + hopoff = flight
    T[2,2,1] = 1.0

    R = zeros(2,2)

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


function HHPC.generate_goal_sample_set(cmssp::DREAMRCMSSPType{US,UA}, popped_cont::US,
                                       num_samples::Int64, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    goal_samples = [DREAMRStateType(FLIGHT, cmssp.goal_point)]
end

function get_arrival_time_distribution(time, noise_model, samples)
    # Returns a TPDist


function HHPC.generate_bridge_sample_set(cmssp::DREAMRCMSSPType{US,UA}, state::US,
                                         mode_pair::DREAMRModePair, num_samples::Int64,
                                         context_set::DREAMRContextSet, 
                                         rng::RNG=Random.GLOBAL_RNG) where {US <: UAVState, UA <: UAVAction, RNG <: AbstractRNG}

    # If RIDE to FLIGHT, then only add subsequent route points of car


    # TODO : How do we incorporate new cars?????
    # If FLIGHT to RIDE, then add all possible future route vertices (edge weight filter will take care of rest)
    # Do check for max-speed-filter
    # Remember to sample around time and divide by MDP timestep


end


function HHPC.update_vertices_with_context!()

    # Another biggie - loop through vertices and 