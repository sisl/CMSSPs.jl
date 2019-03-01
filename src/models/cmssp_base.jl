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


# TODO : Require discrete stuff as a Tabular MDP and map from d to integer and ad to integer (actionindex)

## Q - DO WE NEED IT AS AN ABSTRACT MDP?
mutable struct CMSSP{D,C,AD,AC} <: POMDPs.MDP{CMSSPState{D,C}, CMSSPAction{AD,AC}}
    actions::Vector{CMSSPAction{AD,AC}}
    modes::Vector{D}
    mode_actions::Vector{AD}
    modeswitch_mdp::TabularMDP
    control_actions::Vector{AC}
end

function CMSSP{D,C,AD,AC}(actions::Vector{CMSSPAction{AD,AC}}, modes::Vector{D},
                          switch_mdp=TabularMDP(Array{Float64,3}(undef,0,0,0), Matrix{Float64}(undef,0,0), 1.0)) where {D,C,AD,AC}
    return CMSSP{D,C,AD,AC}(actions, modes,  
                 get_modeswitch_actions(actions), switch_mdp, 
                 get_control_actions(actions))
end

# POMDPs overrides
POMDPs.actions(cmssp::CMSSP) = cmssp.actions
POMDPs.n_actions(cmssp::CMSSP) = length(cmssp.actions)
POMDPs.discount(cmssp::CMSSP) = 1.0 # SSP - Undiscounted
POMDPs.actionindex(cmssp::CMSSP, a::CMSSPAction) = a.action_idx


const TPDistribution = SparseCat{Vector{Int64},Vector{Float64}}
const VKey = MVector{2,Float64}

"""
Returns a vector of the mode-switch CMSSP actions
"""
function get_modeswitch_actions(actions::Vector{CMSSPAction{AD,AC}}) where {AD,AC}
    mode_actions = Vector{AD}(undef,0)
    for a in actions
        if typeof(a.action) <: AD
            push!(mode_actions,a.action)
        end
    end
    return mode_actions
end

"""
Returns a vector of only the control actions
"""
function get_control_actions(actions::Vector{CMSSPAction{AD,AC}}) where {AD,AC}
    control_actions = Vector{AC}(undef,0)
    for a in actions
        if typeof(a.action) <: AC
            push!(control_actions,a.action)
        end
    end
    return control_actions
end

function set_modeswitch_mdp!(cmssp::CMSSP, mdp::TabularMDP)
    cmssp.modeswitch_mdp = mdp
end

function mode_index(cmssp::CMSSP{D,C,AD,AC}, mode::D) where {D,C,AD,AC}
    idx = findfirst(isequal(mode), cmssp.modes)
    @assert idx != Nothing "Mode not present in set of modes"
    return idx
end

function mode_actionindex(cmssp::CMSSP{D,C,AD,AC}, mode_action::AD) where {D,C,AD,AC}
    idx = findfirst(isequal(mode_action), cmssp.mode_actions)
    @assert idx != Nothing "Mode-switch action not present"
    return idx
end

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
mutable struct OpenLoopVertex{D,C,AD}
    state::CMSSPState{D,C}
    pre_bridge_state::CMSSPState{D,C}
    bridging_action::AD
    tp::TPDistribution
end

function OpenLoopVertex{D,C,AD}(state::CMSSPState{D,C}, action::AD) where {D,C,AD}
    return OpenLoopVertex{D,C,AD}(state, state, action, TPDistribution([-1],[1.0]))
end

function OpenLoopVertex{D,C,AD}(pre_mode::D, post_mode::D, bridge_sample::BridgeSample{C}, action::AD) where {D,C,AD}
    state = CMSSPState(post_mode, bridge_sample.post_bridge_state)
    pre_bridge_state = CMSSPState(pre_mode, bridge_sample.pre_bridge_state)
    return OpenLoopVertex{D,C,AD}(state, pre_bridge_state, action, bridge_sample.tp)
end

## User needs to implement
# POMDPs.isterminal
# POMDPs.generate_sr # For continuous
# POMDPs.transition # For discrete
# POMDPs.reward # For cost - decomposed