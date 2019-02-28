# This implements the functionality required by the local layer
# For now, we're doing LocalApproximationVI, so the user has to 
# provide the interpolation object and define the necessary convert functions
# struct ModalPolicy
#     in_horizon_policy::LocalApproximationValueIterationPolicy
#     out_horizon_policy::LocalApproximationValueIterationPolicy
# end

# Dict{D,ModalPolicy}

# function finitehorizonlocalVI{D,C}(cmssp, mode, interp,...)
# function get_intramodal_action(cmssp,mode)
# function compute_penalty
# function convert_s (use convert_s for underlying state)
# struct CMSSPAugmented (for horizon)
# function incorporate_closedloop_interrupt
"""
The Modal Policy has an in-horizon and an out-horizon policy. If the context has no temporal nature,
the in-horizon-policy is null and only out-horizon policy is used.
"""
struct ModalPolicy{C,AC}
    in_horizon_policy::Policy
    out_horizon_policy::Policy
end

"""
For a modal policy, the state is a pair of the current and target.
Expects a convert_s function (specify requirements in @POMDP_require)
"""
struct ModalState{C}
    curr_state::C
    target_state::C
end

struct ModalAction{AC}
    action::AC
    action_idx::Int64
end


"""
Define modal MDP type which will be used with approximate VI
(Take horizon as a function argument)
"""
mutable struct ModalMDP{C,AC} <: POMDPs.MDP{ModalStateAugmented{C},ModalAction{AC}}
    actions::Vector{ModalAction{AC}}
    beta_threshold::Float64
    horizon_limit::Int64
    max_cost_per_horizon::Vector{Float64}
    terminal_cost_penalty::Float64
    terminal_costs_set::Bool
end

function ModalMDP{C,AC}(actions::Vector{ModalAction{AC}}, beta::Float64, horizon_limit::Int64) where {C,AC}
    return ModalMDP{C,AC}(actions, beta, horizon_limit, Inf, Vector{Float64}(undef,0), Inf, false)
end

function ModalMDP{C,AC}(actions::Vector{ModalAction{AC}}, beta::Float64) where {C,AC}
    return ModalMDP{C,AC}(actions, beta, 0)
end

function set_horizon_limit!(mdp::ModalMDP{C,AC}, h::Int64) where {C,AC}
    mdp.horizon_limit = h
end


POMDPs.actions(mdp::ModalMDP) = mdp.actions
POMDPs.n_actions(mdp::ModalMDP) = length(mdp.actions)
POMDPs.discount(mdp::ModalMDP) = 1.0 # SSP - Undiscounted
POMDPs.actionindex(mdp::ModalMDP, a::ModalAction) = a.action_idx


"""
CMSSP State augmented with horizon value. Needed for LocalApproximationVI
"""
struct ModalStateAugmented{C}
    state::ModalState{C}
    horizon::Int64
end

function ModalStateAugmented{C}(state::ModalState{C}) where {C}
    return ModalStateAugmented{C}(state, 0)
end


"""
Convert augmented state to a vector by calling the underlying convert_s function
for non-augmented state
"""
function POMDPs.convert_s(::Type{V}, s::ModalStateAugmented{C}, 
                          mdp::ModalMDP{C,AC}) where {C,AC,V <: AbstractVector{Float64}}
    v = convert_s(Vector{Float64}, s.state, mdp)
    push!(v, convert(Float64, s.horizon))
    return v
end

"""
Convert horizon-augmented vector to CMSSPStateAugmented
"""
function POMDPs.convert_s(::Type{ModalStateAugmented}, v::AbstractVector{Float64},
                          mdp::ModalMDP{C,AC}) where {C,AC}
    state = convert_s(ModalState, v, mdp)
    horizon = convert(Int64,v[end])
    s = ModalStateAugmented{C}(state,horizon)
    return s
end


function POMDPs.generate_sr(mdp::ModalMDP{C,AC}, s::ModalStateAugmented{C}, a::ModalAction{AC}, 
                            rng::RNG=Random.GLOBAL_RNG) where {C,AC,RNG <: AbstractRNG}

    # Get next state and underlying cost for dynamics
    (sp,r) = generate_sr(mdp,s.state,a,rng)
    cost = -1.0*r

    # If at end of horizon, increment with terminal cost penalty
    if s.horizon == 1
        if mdp.terminal_costs_set
            if POMDPs.isterminal(mdp, s.state) == false
                cost += mdp.terminal_cost_penalty
            end
        end
    end

    return ModalStateAugmented(sp, s.horizon-1), -cost
end

function POMDPs.isterminal(mdp::ModalMDP{C,AC}, s::ModalStateAugmented{C}) where {C,AC}
    return s.horizon == 0
end


"""
Compute the terminal cost penalty for the modal region. (\phi_CF) from paper
This version uses a local function approximator
"""
function compute_terminalcost_localapprox(mdp::ModalMDP{C,AC}, cmssp::CMSSP{D,C,AD,AC}, mode::D,
                                          lfa::LFA) where {D,C,AD,AC,LFA <: LocalFunctionApproximator}
    max_contr_cost = Inf
    max_switch_cost = Inf

    # First iterate over continuous actions
    for action in actions(mdp)
        for point in get_all_interpolating_points(lfa)
            state = convert_s(ModalState{C}, point, mdp)
            cost = -1.0*reward(state,action)
            if cost > max_contr_cost
                max_contr_cost = cost
            end
        end
    end

    mode_idx = mode_index(cmssp, d)

    # now iterate over discrete actions and get out-of-mode costs
    for (ac_idx,action) in enumerate(cmssp.mode_actions)
        for (next_mode_idx,_mode) in enumerate(cmssp.modes)
            if cmssp.modeswitch_mdp.T[next_mode_idx, ac_idx, mode_idx] > 0.0
                cost = -1.0*cmssp.modeswitch_mdp.R[mode_idx, ac_idx]
                if cost > max_switch_cost
                    max_switch_cost = cost
                end
            end
        end
    end

    mdp.terminal_cost_penalty = max_contr_cost + max_switch_cost
end

"""
Remember - cannot set values of terminal states - must incorporate in reward function!
"""
function finite_horizon_VI_localapprox(mdp::ModalMDP{C,AC}, lfa::LFA,
                                       is_mdp_generative::Bool, n_generative_samples::Int64=0,
                                       rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # IMP - For out-of-horizon, can just run for that many iterations and it should be fine
    # Setup an out-horizon-approximator with just 1 dimension
    mdp.terminal_costs_set = false
    outhor_lfa = finite_horizon_extension(lfa, 0:1:mdp.horizon_limit)
    outhor_solver = LocalApproximationValueIterationSolver(outhor_lfa, max_iterations = 1,
                                                           verbose=true, rng, is_mdp_generative=is_mdp_generative,
                                                           n_generative_samples=n_generative_samples)
    out_horizon_policy_augmented = solve(outhor_solver, mdp)

    # Create the true out-horizon LFA that has just one time axis
    outhor_lfa_true = finite_horizon_extension(lfa, mdp.horizon_limit+1:1:mdp.horizon_limit+1)
    all_interp_values = get_all_interpolating_values(out_horizon_policy_augmented.interp)
    all_interp_states = get_all_interpolating_states(out_horizon_policy_augmented.interp)

    outhor_interp_values = Vector{Float64}(undef,0)
    # copy over value for horizon = K
    for (v,s) in zip(all_interp_values, all_interp_states)
        if s[end] == mdp.horizon_limit
            push!(outhor_interp_values, v)
        end
    end

    # Check that there are indeed that many points as true out-horizon-LFA
    @assert length(outhor_interp_values) == n_interpolating_points(outhor_lfa_true)
    out_horizon_policy = LocalApproximationValueIterationPolicy(outhor_lfa_true, ordered_actions(mdp),
                                                                    mdp, is_mdp_generative, n_generative_samples, rng)

    ## OUT-HORIZON DONE!

    ## Now do for in-horizon
    mdp.terminal_costs_set = true
    inhor_lfa = finite_horizon_extension(lfa, 0:1:mdp.horizon_limit)
    inhor_solver = LocalApproximationValueIterationSolver(inhor_lfa, max_iterations = 1,
                                                           verbose=true, rng, is_mdp_generative=is_mdp_generative,
                                                           n_generative_samples=n_generative_samples)
    in_horizon_policy = solve(inhor_solver, mdp)

    return ModalPolicy(in_horizon_policy, out_horizon_policy)
end


function compute_max_cost_per_horizon end


# Incorporate interrupt logic here
function get_intramodal_action end 
