"""
The Modal Policy has an in-horizon and an out-horizon policy. If the context has no temporal nature,
the in-horizon-policy is null and only out-horizon policy is used.
"""
struct ModalHorizonPolicy{P <: Policy}
    in_horizon_policy::P
    out_horizon_policy::P
end

"""
CMSSP State augmented with horizon value. Needed for LocalApproximationVI
"""
struct ModalStateAugmented{C}
    relative_state::C
    horizon::Int64
end

struct ModalAction{AC}
    action::AC
    action_idx::Int64
end

function ModalStateAugmented(state::C) where {C}
    return ModalStateAugmented(state, 0)
end

function ModalStateAugmented{C}(state::C) where {C}
    return ModalStateAugmented{C}(state, 0)
end


"""
Define modal MDP type which will be used with approximate VI to get local layer policy

Attributes:
    - `mode::D` The specific mode that defines the smaller regional MDP
    - `actions::Vector{ModalAction{AC}}` The control actions available in this region
    - `beta_threshold::Float64` The interrupt threshold for this MDP
    - `horizon_limit::Int64` The horizon up to which the value function is computed
    - `min_value_per_horizon::Vector{Float64}` The worst cost or min value from each horizon value
    - `terminal_cost_penalty::Float64` The phi_CF value
    - `terminal_costs_set::Bool` Flag used for out-horizon vs in-horizon VI
"""
mutable struct ModalMDP{D,C,AC,P} <: POMDPs.MDP{ModalStateAugmented{C},ModalAction{AC}}
    mode::D
    actions::Vector{ModalAction{AC}}
    beta_threshold::Float64
    horizon_limit::Int64
    min_value_per_horizon::Vector{Float64}
    terminal_cost_penalty::Float64
    terminal_costs_set::Bool
    params::P
end


function ModalMDP{D,C,AC,P}(mode::D, params::P, actions::Vector{AC}, beta::Float64=1.0, horizon_limit::Int64=0) where {D,C,AC,P}
    modal_actions = [ModalAction(a,i) for (i,a) in enumerate(actions)]
    return ModalMDP{D,C,AC,P}(mode, modal_actions, beta, horizon_limit, Inf*ones(horizon_limit), Inf, false, params)
end

function ModalMDP{D,C,AC,P}(mode::D, params::P) where {D,C,AC,P}
    return ModalMDP{D,C,AC,P}(mode, params, Vector{ModalAction{AC}}(undef,0))
end


function set_horizon_limit!(mdp::ModalMDP, h::Int64) where {D,C,AC}
    mdp.horizon_limit = h
    mdp.min_value_per_horizon = Inf*ones(h)
end


POMDPs.actions(mdp::ModalMDP) = mdp.actions
POMDPs.n_actions(mdp::ModalMDP) = length(mdp.actions)
POMDPs.discount(mdp::ModalMDP) = 1.0 # SSP - Undiscounted
POMDPs.actionindex(mdp::ModalMDP, a::ModalAction) = a.action_idx


"""
Convert augmented state to a vector by calling the underlying convert_s function
for non-augmented state. IMPORTANT: The underlying dynamics should be for RELATIVE STATE
"""
function POMDPs.convert_s(::Type{V}, s::ModalStateAugmented, 
                          mdp::ModalMDP) where {V <: AbstractVector{Float64}}
    v = convert_s(Vector{Float64}, s.relative_state, mdp)
    push!(v, convert(Float64, s.horizon))
    return v
end

"""
Convert horizon-augmented vector to ModalStateAugmented
"""
function POMDPs.convert_s(::Type{ModalStateAugmented{C}}, v::AbstractVector{Float64},
                          mdp::ModalMDP) where {C}
    state = convert_s(C, v, mdp)
    horizon = convert(Int64,v[end])
    s = ModalStateAugmented{C}(state,horizon)
    return s
end


function POMDPs.generate_sr(mdp::ModalMDP, s::ModalStateAugmented, a::ModalAction, 
                            rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Get next state and underlying cost for dynamics
    # TODO : This should be the same regardless of relative or absolute state
    (sp,r) = generate_sr(mdp,s.relative_state,a.action,rng)
    cost = -1.0*r

    # If at end of horizon, increment with terminal cost penalty
    # if sp is not terminal
    if s.horizon == 1
        if mdp.terminal_costs_set
            if POMDPs.isterminal(mdp, sp) == false
                cost += mdp.terminal_cost_penalty
            end
        end
    end

    return ModalStateAugmented(sp, s.horizon-1), -cost
end

function POMDPs.isterminal(mdp::ModalMDP, s::ModalStateAugmented)
    return s.horizon == 0
end


"""
Compute the terminal cost penalty for the modal region, i.e. phi_CF from paper
This version uses a local function approximator.

Arguments:
    - `mdp::ModalMDP{D,C,AC}`
    - `cmssp::CMSSP{D,C,AD,AC}`
    - `mode::D`
    - `lfa::LFA` A LocalFunctionApproximator object for the value iteration
"""
function compute_terminalcost_localapprox!(mdp::ModalMDP{D,C,AC,P}, cmssp::CMSSP, mode::D,
                                          lfa::LFA) where {D,C,AD,AC,P,LFA <: LocalFunctionApproximator}
    max_contr_cost = 0.0
    max_switch_cost = 0.0

    # First iterate over continuous actions
    @info "Getting max control cost"
    for a in actions(mdp)
        for point in get_all_interpolating_points(lfa)
            state = convert_s(C, point, mdp)
            cost = -1.0*expected_reward(mdp,state,a.action)
            if cost > max_contr_cost
                max_contr_cost = cost
            end
        end
    end
    mdp.terminal_cost_penalty = max_contr_cost * mdp.horizon_limit
end

"""
Perform finite horizon value approximation on the regional MDP

Arguments:
    - `mdp::ModalMDP{D,C,AC}` 
    - `lfa::LFA`
    - `is_mdp_generative::Bool`
    - `n_generative_samples::Int64`
    - `rng::RNG`
"""
function finite_horizon_VI_localapprox!(mdp::ModalMDP, lfa::LFA,
                                       is_mdp_generative::Bool=false, n_generative_samples::Int64=0,
                                       rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG,LFA<:LocalFunctionApproximator}

    # Setup an out-horizon-approximator
    # Terminal costs are 0 in this case
    mdp.terminal_costs_set = false
    outhor_lfa = finite_horizon_extension(lfa, 0:1:mdp.horizon_limit)
    outhor_solver = LocalApproximationValueIterationSolver(outhor_lfa, max_iterations = 1,
                                                           verbose=true, rng=rng, is_mdp_generative=is_mdp_generative,
                                                           n_generative_samples=n_generative_samples)
    out_horizon_policy_augmented = solve(outhor_solver, mdp)

    # Create the true out-horizon LFA that has just one time axis
    outhor_lfa_true = finite_horizon_extension(lfa, mdp.horizon_limit+1:1:mdp.horizon_limit+1)
    all_interp_values = get_all_interpolating_values(out_horizon_policy_augmented.interp)
    all_interp_states = get_all_interpolating_points(out_horizon_policy_augmented.interp)

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

    ## Now do for in-horizon
    mdp.terminal_costs_set = true
    inhor_lfa = finite_horizon_extension(lfa, 0:1:mdp.horizon_limit)
    inhor_solver = LocalApproximationValueIterationSolver(inhor_lfa, max_iterations = 1,
                                                           verbose=true, rng=rng, is_mdp_generative=is_mdp_generative,
                                                           n_generative_samples=n_generative_samples)
    in_horizon_policy = solve(inhor_solver, mdp)

    return ModalHorizonPolicy(in_horizon_policy, out_horizon_policy)
end

"""
Compute the worst cost or minimum value per horizon and update mdp object in place.

Arguments:
    - `mdp::ModalMDP{D,C,AC}`
    - `modal_policy::ModalHorizonPolicy{P}`
"""
function compute_min_value_per_horizon_localapprox!(modal_policy::ModalHorizonPolicy)

    # Access the underlying value function approximator
    all_interp_values_inhor = get_all_interpolating_values(modal_policy.in_horizon_policy.interp)
    all_interp_states_inhor = get_all_interpolating_points(modal_policy.in_horizon_policy.interp)

    for (v,s) in zip(all_interp_values_inhor, all_interp_states_inhor)
        hor_val = convert(Int64,s[end])
        
        if hor_val == 0
            continue
        end

        # Compute lowest value
        if v < modal_policy.in_horizon_policy.mdp.min_value_per_horizon[hor_val]
            modal_policy.in_horizon_policy.mdp.min_value_per_horizon[hor_val] = v
        end
    end
end


"""
Compute the negative value of the relative state between current and target. The value is weighted by the
distribution over horizons.

Arguments:
    - `mdp::ModalMDP{D,C,AC}`
    - `modal_policy::ModalHorizonPolicy`
    - `curr_timestep::Int64` The current system time
    - `tp_dist::TPDistribution` The distribution over GLOBAL times for reaching target state
    - `curr_state::C`
    - `target_state::C`
"""
function horizon_weighted_value(modal_policy::ModalHorizonPolicy, curr_timestep::Int64,
                                tp_dist::TPDistribution, curr_state::C, target_state::C) where C

    weighted_value = 0.0
    weighted_minvalue = 0.0

    mdp = modal_policy.in_horizon_policy.mdp

    # Iterate over horizon values and weight
    # Subtract the current time-step to get the true relative time
    for (h_,p) in tp_dist
        h = h_ - curr_timestep
        if h <= 0
            continue
        end
        temp_relative_state = get_relative_state(mdp, curr_state, target_state)
        if h > mdp.horizon_limit
            temp_state = ModalStateAugmented(temp_relative_state, mdp.horizon_limit+1)
            weighted_value += p*POMDPs.value(modal_policy.out_horizon_policy,temp_state)
            
            # IMP - For any prob of out-of-horizon, never abort
            weighted_minvalue += p*(-Inf)
        else
            temp_state = ModalStateAugmented(temp_relative_state, h)
            weighted_value += p*value(modal_policy.in_horizon_policy,temp_state)
            weighted_minvalue += p*mdp.min_value_per_horizon[h]
        end
    end

    return weighted_value, weighted_minvalue
end


"""
Compute the negative value of the relative state between current and target. The value is weighted by the
distribution over horizons.

Arguments:
    - `mdp::ModalMDP{D,C,AC}`
    - `modal_policy::ModalHorizonPolicy`
    - `curr_timestep::Int64` The current system time
    - `tp_dist::TPDistribution` The distribution over GLOBAL times for reaching target state
    - `curr_state::C`
    - `target_state::C`
    - `a::ModalAction{AC}`
"""
function horizon_weighted_actionvalue(modal_policy::ModalHorizonPolicy, curr_timestep::Int64,
                                tp_dist::TPDistribution, curr_state::C, target_state::C, a::ModalAction{AC}) where {C,AC}

    total_value = 0.0

    mdp = modal_policy.in_horizon_policy.mdp

    # Iterate over horizon values and weight 
    for (h_,p) in tp_dist
        h = h_ - curr_timestep
        if h <= 0
            continue
        end
        temp_relative_state = get_relative_state(mdp, curr_state, target_state)
        if h > mdp.horizon_limit
            temp_state = ModalStateAugmented(temp_relative_state, mdp.horizon_limit+1)
            total_value += p*action_value(modal_policy.out_horizon_policy, temp_state, a)
        else
            temp_state = ModalStateAugmented(temp_relative_state, h)
            total_value += p*action_value(modal_policy.in_horizon_policy, temp_state, a)
        end
    end

    return total_value
end


"""
Returns action with lowest average cost (i.e. highest average value, since value is negative cost)

Arguments:
    - `mdp::ModalMDP{D,C,AC}`
    - `modal_policy::ModalHorizonPolicy`
    - `curr_timestep::Int64` The current system time
    - `tp_dist::TPDistribution` The distribution over GLOBAL times for reaching target state
    - `curr_state::C`
    - `target_state::C`
"""
function get_best_intramodal_action(modal_policy::ModalHorizonPolicy, curr_timestep::Int64,
                               tp_dist::TPDistribution, curr_state::C, target_state::C) where C

    mdp = modal_policy.in_horizon_policy.mdp

    # First check if abort needed
    (weighted_value, weighted_minvalue) = horizon_weighted_value(modal_policy,curr_timestep,
                                                                 tp_dist,curr_state,target_state)

    if weighted_value <= mdp.beta_threshold * weighted_minvalue
        return nothing # closed-loop interrupt
    end

    best_action = mdp.actions[1]
    best_action_val = -Inf

    for a in mdp.actions
        action_val = horizon_weighted_actionvalue(modal_policy, curr_timestep, tp_dist, curr_state, target_state, a)
        if action_val > best_action_val
            best_action_val = action_val
            best_action = a
        end
    end

    return best_action
end