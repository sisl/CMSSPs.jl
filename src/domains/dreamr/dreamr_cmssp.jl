## Define DREAMR types (based on US, UA)
@enum DREAMR_MODETYPE FLIGHT=1 RIDE=2
@enum HOP_ACTION HOPON=1 STAY=2 HOPOFF=3

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

    T = zeros(2,3,2)
    # Flight + hopon = ride
    T[1,1,2] = 1.0
    # Ride + stay = ride
    T[2,2,2] = 1.0
    # Ride + hopoff = flight
    T[2,3,1] = 1.0

    R = zeros(2,3)

    # Undiscounted
    return TabularMDP(T, R, 1.0)
end

function create_dreamr_cmssp(dynamics::UDM, params::Parameters) where {UDM <: UAVDynamicsModel}

    actions = get_dreamr_actions(dynamics)
    switch_mdp = get_dreamr_mdp()

    return DREAMRCMSSPType(actions, DREAMR_MODES, switch_mdp, params)
end