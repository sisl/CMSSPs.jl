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

# For reading params
using TOML

# CMSSPs submodule
using CMSSPs.CMSSPModel
using CMSSPs.HHPC


# Toy2D exports
# export
#     Toy2DContState,
#     Toy2DContAction,
#     Toy2DParameters,
#     Toy2DStateType,
#     Toy2DActionType,
#     Toy2DCMSSPType,
#     Toy2DModalMDPType,
#     Toy2DContextType,
#     Toy2DContextSet,
#     Toy2DSolverType,
#     Toy2DOpenLoopVertex,
#     TOY2D_MODES,
#     TOY2D_GOAL_MODE,
#     create_toy2d_cmssp,
#     sample_toy2d,
#     toy2d_parse_params,
#     generate_start_state,
#     generate_start_context_set


# DREAMR exports
export
    Parameters,
    parse_params,
    UAVDynamicsModel,
    UAVState,
    UAVAction,
    MultiRotorUAVState,
    MultiRotorUAVAction,
    MultiRotorUAVDynamicsModel,
    DREAMR_MODETYPE,
    HOP_ACTION,
    get_dreamr_actions,
    get_dreamr_mdp,
    create_dreamr_cmssp,
    get_dreamr_episode_context,
    get_cf_mdp,
    get_uf_mdp,
    get_ride_mdp,
    DREAMRDeterministicPolicy


# include("toy_2d.jl")
include("dreamr/dreamr_types.jl")
include("dreamr/dreamr_params.jl")
include("dreamr/dreamr_dynamics.jl")
include("dreamr/dreamr_cmssp.jl")

end # End of submodule