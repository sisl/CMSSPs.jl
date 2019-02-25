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
    action_idx::Int64
end


# mutable struct CMSSPModalPolicy{D,S,AS} <: Policy
# end

## Q - DO WE NEED IT AS AN ABSTRACT MDP?
struct CMSSP{D,C,AD,AC} <: POMDPs.MDP{CMSSPState{D,C}, CMSSPAction{AD,AC}}
    actions::Vector{CMSSPAction{AD,AC}}
    start_state::CMSSPState{D,C}
    goal_mode::D
end

# POMDPs overrides
POMDPs.actions(mdp::CMSSP) = mdp.actions
POMDPs.n_actions(mdp::CMSSP) = length(mdp.actions)
POMDPs.discount(mdp::CMSSP) = 1.0 # SSP - Undiscounted
POMDPs.actionindex(mdp::CMSSP, a::CMSSPAction) = a.action_idx


const TPDistribution = SparseCat{Vector{Int64},Vector{Float64}}
const VKey = MVector{2,Float64}

"""
Return type for bridge sample
"""
struct BridgeSample{C}
    pre_bridge_state::C
    post_bridge_state::C
    tp::TPDistribution
end


"""
Vertex type for open-loop layer
"""
mutable struct OpenLoopVertex{D,C}
    state::CMSSPState{D,C}
    pre_bridge_state::CMSSPState{D,C}
    tp::TPDistribution
end

function OpenLoopVertex(state::CMSSPState{D,C}) where {D,C}
    return OpenLoopVertex{D,C}(state, state, TPDistribution([-1],[1.0]))
end

function OpenLoopVertex(pre_mode::D, post_mode::D, bridge_sample::BridgeSample{C}) where {D,C}
    state = CMSSPState{D,C}(post_mode, bridge_sample.post_bridge_state)
    pre_bridge_state = CMSSPState{D,C}(pre_mode, bridge_sample.pre_bridge_state)
    return OpenLoopVertex{D,C}(state, pre_bridge_state, bridge_sample.tp)
end

## User needs to implement
# POMDPs.isterminal
# POMDPs.generate_sr # For continuous
# POMDPs.transition # For discrete
# POMDPs.reward # For cost - decomposed