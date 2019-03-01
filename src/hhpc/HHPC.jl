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


## Required to implement
# generate_next_valid_modes
# generate_bridge_sample_set
# generate_goal_sample_set
# update_vertices_with_context
# mode switches as tabular MDP
# heuristic

## Requirements
# reward(s,a) - as expected method

include("global_layer.jl")
include("local_layer.jl")

end