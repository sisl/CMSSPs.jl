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

# CMSSPs submodule
using CMSSPs.CMSSPModel
using CMSSPs.HHPC


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
    TOY2D_MODES,
    TOY2D_GOAL_CENTRE,
    TOY2D_GOAL_MODE,
    create_toy2d_cmssp,
    isterminal,
    sample_toy2d


# General required exports
export
    get_relative_state,
    expected_reward,
    startstate_context,
    update_vertices_with_context!,
    generate_goal_sample_set,
    generate_next_valid_modes,
    generate_bridge_sample_set,
    simulate


include("toy_2d.jl")

end # End of submodule