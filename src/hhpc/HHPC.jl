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
    metadatatype,
    bookkeepingtype,
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
    infinite_horizon_VI_localapprox,
    compute_min_value_per_horizon_localapprox!,
    horizon_weighted_value,
    horizon_weighted_actionvalue,
    get_best_intramodal_action,
    get_best_intramodal_action_infhor

# HHPC framework
# Requirements
# ensure that mode-switch MDP available?
export
    HHPCSolver,
    set_start_state_context_set!,
    set_open_loop_samples

# Utils
export
    save_localapproxvi_policy_to_jld2,
    load_localapproxvi_policy_from_jld2,
    load_modal_horizon_policy_localapprox,
    save_modal_horizon_policy_localapprox,
    log2space_symmetric,
    polyspace_symmetric

# General functions that need to be implemented
export
    # Local
    expected_reward,
    get_relative_state,
    # Global
    update_vertices_with_context!,
    generate_goal_sample_set!,
    generate_next_valid_modes,
    generate_bridge_sample_set!,
    # HHPC
    simulate_cmssp,
    update_context_set!,
    display_context_future

# Local reqs
function get_relative_state end
function expected_reward end
# Global reqs
function update_vertices_with_context! end
function generate_goal_sample_set end
function generate_next_valid_modes end
function generate_bridge_sample_set end
# HHPC reqs
function simulate_cmssp end
function update_context_set! end
function display_context_future end


include("global_layer.jl")
include("local_layer.jl")
include("hhpc_framework.jl")
include("utils.jl")

end