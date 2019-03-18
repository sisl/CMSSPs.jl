module CMSSPModel

# Stdlib Requirements
using Logging
using Printf
using Random
using Statistics
using LinearAlgebra

# Non-stdlib essentials
using StaticArrays

# POMDP Stuff
using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using LocalApproximationValueIteration


# Requirements
# POMDPs.isterminal
# POMDPs.generate_sr # For continuous regions
# POMDPs.reward # For cost - decomposed
 

export
    CMSSPState,
    CMSSPAction,
    CMSSP,
    modetype,
    continuoustype,
    modeactiontype,
    controlactiontype,
    get_modeswitch_actions,
    get_control_actions,
    set_modeswitch_mdp!,
    mode_index,
    TPDistribution,
    BridgeSample,
    OpenLoopVertex

# Toy2D exports
export
    Toy2DContState,
    Toy2DContAction,
    Toy2DParameters,
    Toy2DStateType,
    Toy2DActionType,
    Toy2DCMSSPType,
    Toy2DModalMDPType,
    Toy2DContextType,
    create_toy2d_cmssp,
    isterminal


# General required exports
export
    get_relative_state,
    startstate_context,
    update_vertices_with_context!,
    generate_goal_sample_set,
    generate_next_valid_modes,
    generate_bridge_sample_set,
    simulate


include("cmssp_base.jl")
include("toy_2d.jl")

end # End of submodule