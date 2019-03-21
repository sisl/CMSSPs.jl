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
export
    Toy2DContState,
    Toy2DContAction,
    Toy2DParameters,
    Toy2DStateType,
    Toy2DActionType,
    Toy2DCMSSPType,
    Toy2DModalMDPType,
    Toy2DContextType,
    Toy2DContextSet,
    Toy2DSolverType,
    Toy2DOpenLoopVertex,
    TOY2D_MODES,
    TOY2D_GOAL_MODE,
    create_toy2d_cmssp,
    sample_toy2d,
    toy2d_parse_params,
    generate_start_state,
    generate_start_context_set
    


# General required exports
export
    # Local
    expected_reward,
    get_relative_state,
    # Global
    update_vertices_with_context!,
    generate_goal_sample_set,
    generate_next_valid_modes,
    generate_bridge_sample_set,
    # HHPC
    simulate_cmssp,
    update_contextset!,
    display_context_future


include("toy_2d.jl")

end # End of submodule