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

# For saving and loading 
using JLD2
using FileIO

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
    save_modal_horizon_policy_localapprox,
    log2space_symmetric,
    polyspace_symmetric

export
    expected_reward,
    get_relative_state


function expected_reward end
function get_relative_state end

include("global_layer.jl")
include("local_layer.jl")
include("hhpc_framework.jl")
include("utils.jl")

end