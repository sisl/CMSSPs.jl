"""
The CMSSP state space is a cartesian product of a discrete enumeration set and
a continuous space.

Attributes:
    - `mode::D` A discrete value for the mode of the system.
    - `continuous::C` A continuous value for the intra-modal state of the system.
"""
struct CMSSPState{D,C}
    mode::D
    continuous::C
end

"""
A CMSSP action is either a discrete mode-switching action or a continuous control action.

Attributes:
    - `action::Union{AD,AC}` Either a discrete mode switching action or a continuous control action.
"""
struct CMSSPAction{AD,AC}
    action::Union{AD,AC}
end


# mutable struct CMSSPModalPolicy{D,S,AS} <: Policy
# end

## Q - DO WE NEED IT AS AN ABSTRACT MDP?
struct CMSSP{D,S,AD,AS} <: POMDPs.MDP{CMSSPState{D,S}, CMSSPAction{AD,AS}}
    actions::Vector{CMSSPAction{AD,AS}}
end

# POMDPs overrides
POMDPs.actions(mdp::CMSSP) = mdp.actions
POMDPs.n_actions(mdp::CMSSP) = length(mdp.actions)
POMDPs.discount(mdp::CMSSP) = 1.0 # SSP - Undiscounted