"""
The overall HHPC framework solver object.

Attributes:
    - `graph_tracker::GraphTracker{D,C,AD}`
    - `modal_policies::Dict{D,ModalHorizonPolicy}` A map from the mode to the modal policies (in and out-horizon)
    - `modal_mdps::Dict{D,ModalMDP{D,C,AC}}` A map from the mode to the regional MDP
    - `replan_time_threshold::Int64` The Delta T parameter for periodic replanning (discrete time-steps)
    - `heuristic::Function` The heuristic method for the global layer
    - `max_steps::Int64` The maximum length of a problem episode (may not be applicable)
    - `start_state::CMSSPState{C,AC}`
    - `goal_modes::Vector{D}`
"""
mutable struct HHPCSolver{D,C,AD,AC,RNG <: AbstractRNG} <: Solver
    graph_tracker::GraphTracker{D,C,AD}
    modal_policies::Dict{D,ModalHorizonPolicy}
    modal_mdps::Dict{D,ModalMDP{D,C,AC}}
    replan_time_threshold::Int64
    heuristic::Function
    max_steps::Int64
    goal_modes::Vector{D}
    rng::RNG
end

function HHPCSolver{D,C,AD,AC}(N::Int64,modal_policies::Dict{D,ModalHorizonPolicy},
                               modal_mdps::Dict{D,ModalMDP{D,C,AC}}, deltaT::Int64,
                               heuristic::Function = n->0, max_steps::Int64=1000, goal_modes::Vector{D},
                               rng::RNG=Random.GLOBAL_RNG) where {D,C,AD,AC, RNG <: AbstractRNG}
    return HHPCSolver{D,C,AD,AC}(GraphTracker{D,C,AD}(N),
                                 modal_policies, modal_mdps, deltaT, heuristic, max_steps, goal_modes, rng)
end

"""
Creates the weight function for the edges, used by the global layer graph search.

Arguments:
    - `solver::HHPCSolver{D,C,AD,AC}` The HHPC solver instance
    - `cmssp::CMSSP{D,C,AD,AC}` The CMSSP instance
    - `u::OpenLoopVertex{D,C,AD}` The parent vertex
    - `v::OpenLoopVertex{D,C,AD}` The child vertex
"""
function edge_weight_fn(solver::HHPCSolver{D,C,AD,AC}, cmssp::CMSSP{D,C,AD,AC},
                        u::OpenLoopVertex{D,C,AD}, v::OpenLoopVertex{D,C,AD}) where {D,C,AD,AC}

    # u's state to v's pre-state should be in a single region
    @assert u.state.mode == v.pre_bridge_state.mode

    # We care about relative tp_dist between u and v
    # If inf, have modal policy overload it!
    mode = u.state.mode
    mdp = solver.modal_mdps[mode]
    temp_curr_time = convert(Int64,round(mean(u.tp)))
    weighted_value, weighted_minvalue = horizon_weighted_value(
                                            mdp,
                                            solver.modal_policies[mode],
                                            temp_curr_time,
                                            v.tp,
                                            u.state.continuous,
                                            v.pre_bridge_state.continuous)

    # TODO : Verify that this works
    # Filter out edges that nominally would trigger an interrupt
    if weighted_value <= mdp.beta_threshold*weighted_minvalue
        return Inf
    end

    edge_weight = -1.0*weighted_value

    # Also add the cost due to the assumed mode switch
    if v.pre_bridge_state.mode != v.state.mode
        ac_idx = mode_actionindex(v.bridging_action)
        mode_idx = mode_index(v.pre_bridge_state.mode)
        edge_weight += -1.0*cmssp.modeswitch_mdp.R[mode_idx,ac_idx]
    end

    return edge_weight
end


# TODO : Fill this in
@POMDP_require solve(solver::HHPCSolver, cmssp::CMSSP) begin

    P = typeof(cmssp)
    S = statetype(P)
    A = actiontype(P)
    D = modetype(cmssp)
    C = continuoustype(cmssp)
    AD = modeactiontype(cmssp)
    AC = controlactiontype(cmssp)

    @req isterminal(::P, ::S)
    @req generate_sr(::ModalMDP{D,C,AC} where {D,C,AC}, ::C, ::AC, ::RNG where {RNG <: AbstractRNG}) # OR transition + reward?
    @req isterminal(::ModalMDP{D,C,AC} where {D,C,AC}, ::C)

    # Global layer requirements
    @req update_vertices_with_context!(::P, ::Vector{OpenLoopVertex{D,C,AD}} where {D,C,AD}, ::Tuple{D,D} where D)
    @req generate_goal_sample_set(::P, ::C, ::Int64, ::RNG where {RNG <: AbstractRNG})
    @req generate_next_valid_modes(::P, ::D)
    @req generate_bridge_sample_set(::P, ::C, ::Tuple{D,D} where {D}, ::Int64, ::RNG where {RNG <: AbstractRNG})
    

    # Local layer requirements
    @req get_relative_state(::ModalMDP{D,C,AC}, ::C, ::C)
    @req convert_s(::Type{V} where V <: AbstractVector{Float64},::C,::ModalMDP{D,C,AC} where {D,C,AC})
    @req convert_s(::Type{C},::V where V <: AbstractVector{Float64},::ModalMDP{D,C,AC} where {D,C,AC})

    # HHPC requirements
    @req startstate_context(::P, ::RNG where {RNG <: AbstractRNG})
    @req simulate(::P, ::S, ::A, ::Int64, ::RNG where {RNG <: AbstractRNG})

end



"""
Executes the top-level behavior of HHPC. Utilizes both global and local layer logic, and interleaving.
"""
function POMDPs.solve(solver::HHPCSolver{D,C,AD,AC}, cmssp::CMSSP{D,C,AD,AC}) where {D,C,AD,AC}

    @warn_requirements solve(solver,cmssp)

    # Initialize Global flags
    t = 0
    plan = true
    last_plan_time = 0

    # Defs needed for open-loop-plan
    edge_weight_fn(u,v) = edge_weight_fn(solver, cmssp, u, v)

    # Generate initial state and context
    # This will typically be a bound method
    curr_state, curr_context = startstate_context(cmssp, solver.rng)

    # Diagnostic info
    total_cost = 0.0
    successful = false

    while t <= solver.max_steps

        if plan == true
            open_loop_plan!(cmssp, curr_state, curr_context, edge_weight_fn,
                            solver.heuristic, hhpc.goal_modes, hhpc.graph_tracker)
            last_plan_time = t
            plan = false
        end

        # now do work for macro-actions 
        next_target = plan[2] # Second vertex in full plan

        # Check if 'terminal' state as per modal MDP
        @assert curr_state.mode == next_target.pre_bridge_state.mode
        relative_state = get_relative_state(curr_state.continuous, next_target.pre_bridge_state.continuous)

        if POMDPs.isterminal(solver.modal_mdps[curr_state.mode], relative_state)
            # TODO: Guaranteed to not be truly terminal state
            # Action is just the bridging action
            curr_action = next_target.bridging_action
        else
            curr_action = get_best_intramodal_action(solver.modal_mdps[curr_state.mode], 
                                                     solver.modal_policies[curr_state.mode], 
                                                     t,
                                                     next_target.tp, 
                                                     curr_state.continuous, 
                                                     next_target.pre_bridge_state.continuous)
        end

        # If curr_action = NOTHING, means interrupt
        # So simulator should handle NOTHING
        # Will typically be bound to other environment variables
        (new_state, new_context, reward, failed_mode_switch) = simulate(cmssp, curr_state, curr_action, t)
        t = t+1
        total_cost += -1.0*reward

        # Update current state
        curr_state = new_state
        curr_context = new_context

        # Check terminal conditions
        if POMDPs.isterminal(cmssp, curr_state)
            successful = true
            break
        end

        if curr_action == nothing || failed_mode_switch || t - last_plan_time > solver.replan_time_threshold
            plan = true
        end
    end

    # TODO : Diagnostics
    return total_cost, successful
end