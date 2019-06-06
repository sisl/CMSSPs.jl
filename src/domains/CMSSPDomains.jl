module CMSSPDomains

# Stdlib Requirements
using Logging
using Printf
using Random
using LinearAlgebra
using Distributions

# Non-stdlib essentials
using StaticArrays

# POMDP Stuff
using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using Graphs

# For reading params
using TOML

# For continuum world
using ContinuumWorld

# CMSSPs submodule
using CMSSPs.CMSSPModel
using CMSSPs.HHPC


# Continuum exports
export
    GridsContinuumParams,
    GridsContinuumCMSSPType,
    GridsContinuumMDPType,
    GridsContinuumSolverType,
    GridsContinuumContextSet,
    continuum_parse_params,
    create_continuum_cmssp,
    set_start_context_set!,
    generate_start_state


# DREAMR exports
export
    Point,
    Parameters,
    parse_params,
    UAVDynamicsModel,
    UAVState,
    UAVAction,
    MultiRotorUAVState,
    MultiRotorUAVAction,
    MultiRotorUAVDynamicsModel,
    generate_start_state,
    get_state_at_rest,
    HopOnOffSingleCarSimulator,
    reset_sim,
    step_sim,
    DREAMR_MODETYPE,
    DREAMRModeAction,
    DREAMRContextSet,
    FLIGHT, RIDE,
    HOP_ACTION,
    HOPON, HOPOFF, STAY,
    DREAMRStateType,
    DREAMRActionType,
    DREAMRBookkeeping,
    DREAMRVertexMetadata,
    DREAMRSolverType,
    get_dreamr_actions,
    get_dreamr_mdp,
    set_dreamr_goal!,
    create_dreamr_cmssp,
    get_dreamr_episode_context,
    get_cf_mdp,
    get_uf_mdp,
    get_ride_mdp,
    get_arrival_time_distribution,
    DREAMRDeterministicPolicy,
    DREAMRMCTSType,
    DREAMRMCTSState,
    estimate_value_dreamr,
    init_q_dreamr


# include("toy_2d.jl")
include("grids_continuum/grids_continuum.jl")

include("dreamr/dreamr_types.jl")
include("dreamr/dreamr_params.jl")
include("dreamr/dreamr_dynamics.jl")
include("dreamr/dreamr_cf_simulator.jl")
include("dreamr/dreamr_cmssp.jl")

end # End of submodule