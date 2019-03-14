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

include("cmssp_base.jl")

end # End of submodule