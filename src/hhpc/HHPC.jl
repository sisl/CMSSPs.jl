module HHPC

# Stdlib Requirements
using Logging
using Printf
using Random
using Statistics

# Non-stdlib essentials
using StaticArrays

# Open-loop requirements
using Graphs

# POMDP Stuff
using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using LocalFunctionApproximation
using LocalApproximationValueIteration

# CMSSPs submodule
using CMSSPs.CMSSPModel


# Open-loop exports
# Requirements:
# update_vertices_with_context!(cmssp, range_subvector, context_set)
# generate_goal_sample_set(vis.cmssp, popped_cont, vis.graph_tracker.num_samples)
# generate_next_valid_modes(vis.cmssp, vis.context_set, popped_mode)
# generate_bridge_sample_set(vis.cmssp, vis.context_set, popped_cont, (popped_mode, nvm), vis.graph_tracker.num_samples)
export
    GraphTracker,
    open_loop_plan!,
    update_graph_tracker!


# Closed-loop exports
export
    ModalHorizonPolicy,
    ModalState,
    ModalAction,
    ModalStateAugmented,
    ModalMDP,
    set_horizon_limit!,
    compute_terminalcost_localapprox!,
    finite_horizon_VI_localapprox!,
    compute_min_value_per_horizon_localapprox!,
    horizon_weighted_value,
    horizon_weighted_actionvalue,
    get_best_intramodal_action

# HHPC framework
# Requirements
# ensure that mode-switch MDP available?
export
    HHPCSolver,
    solve

# Utils
export
    save_localapproxvi_policy_to_jld2,
    load_localapproxvi_policy_from_jld2,
    load_modal_horizon_policy_localapprox,
    save_modal_horizon_policy_localapprox



## Requirements
# reward(s,a) - as expected method

include("global_layer.jl")
include("local_layer.jl")
include("hhpc_framework.jl")
include("utils.jl")

end