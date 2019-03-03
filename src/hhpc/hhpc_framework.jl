mutable struct HHPCSolver{D,C,AD,AC} <: Solver
    graph_tracker::GraphTracker{D,C,AD}
    modal_policies::Dict{D,ModalPolicy}
    modal_mdps::Dict{D,ModalMDP{C,AC}}
    replan_time_threshold::Int64
    heuristic::Function
    max_steps::Int64
end

function HHPCSolver{D,C,AD,AC}(N::Int64,modal_policies::Dict{D,ModalPolicy},
                               modal_mdps::Dict{D,ModalMDP{C,AC}}, deltaT::Int64,
                               heuristic::Function = n->0, max_steps::Int64=1000) where {D,C,AD,AC}
    return HHPCSolver{D,C,AD,AC}(GraphTracker{D,C,AD}(N),
                                 modal_policies, modal_mdps, deltaT, heuristic, max_steps)
end

function edge_weight_fn(solver::HHPCSolver{D,C,AD,AC}, cmssp::CMSSP{D,C,AD,AC},
                        u::OpenLoopVertex{D,C,AD}, v::OpenLoopVertex{D,C,AD}) where {D,C,AD,AC}

    # We care about u's state to v's pre-state
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

    # TODO : Change this?
    if weighted_value < mdp.beta_threshold*weighted_minvalue
        return Inf
    end

    edge_weight = -1.0*weighted_value

    # If change in mode for final - do that
    if v.pre_bridge_state.mode != v.state.mode
        ac_idx = mode_actionindex(v.bridging_action)
        mode_idx = mode_index(v.pre_bridge_state.mode)
        edge_weight += -1.0*cmssp.modeswitch_mdp.R[mode_idx,ac_idx]
    end

    return edge_weight
end


# TODO : Fill this in
@POMDP_require solve(solver::HHPCSolver, mdp::CMSSP) begin



function solve(solver::HHPCSolver{D,C,AD,AC}, cmssp::CMSSP{D,C,AD,AC}) where {D,C,AD,AC}

    @warn_requirements solve(solver,mdp)

    # Initialize Global flags
    t = 0
    plan = true
    last_plan_time = 0

    # Defs needed for open-loop-plan
    edge_weight_fn(u,v) = edge_weight_fn(solver, cmssp, u, v)

    # Generate initial state and context
    # This will typically be a bound method
    (start_state, start_context) = initial_state(cmssp)
    curr_state = start_state
    curr_context = start_context

    # Diagnostic info
    total_cost = 0.0
    successful = false

    while t <= solver.max_steps

        if plan == true
            open_loop_plan!(cmssp, curr_state, curr_context, edge_weight_fn,
                            solver.heuristic, hhpc.graph_tracker)
            last_plan_time = t
            plan = false
        end

        # now do work for macro-actions 
        next_target = plan[2] # Second vertex in full plan

        # Check if 'terminal' state as per modal MDP
        @assert curr_state.mode == next_target.pre_bridge_state.mode
        modal_relstate = ModalState{C}(curr_state.continuous, next_target.pre_bridge_state.continuous)

        if POMDPs.isterminal(solver.modal_mdps[curr_state.mode], modal_relstate)
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
        (new_state, new_context, reward, failed_mode_switch) = simulate(cmssp,curr_state,curr_action)
        t = t+1
        total_cost += -1.0*reward

        # Copy over things
        curr_state = new_state
        curr_context = new_context

        # Check terminal conditions
        if POMDPs.isterminal(cmssp, curr_state)
            successful = true
            break
        end

        if curr_action == Nothing || failed_mode_switch || t - last_plan_time > solver.replan_time_threshold
            plan = true
        end
    end

    # TODO : Diagnostics
    return total_cost, successful
end