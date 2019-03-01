mutable struct HHPCSolver{D,C,AD,AC} <: Solver
    graph_tracker::GraphTracker{D,C,AD}
    modal_policies::Dict{D,ModalPolicy}
    modal_mdps::Dict{D,ModalMDP{C,AC}}
    replan_time_threshold::Int64
end

function HHPCSolver{D,C,AD,AC}(N::Int64,modal_policies::Dict{D,ModalPolicy},
                               modal_mdps::Dict{D,ModalMDP{C,AC}}, deltaT::Int64) where {D,C,AD,AC}
    return HHPCSolver{D,C,AD,AC}(GraphTracker{D,C,AD}(N),
                                 modal_policies, modal_mdps, deltaT)
end

function edge_weight_fn(cmssp::CMSSP{D,C,AD,AC}, solver::HHPCSolver{D,C,AD,AC},
                        u::OpenLoopVertex{D,C,AD}, v::OpenLoopVertex{D,C,AD}) where {D,C,AD,AC}

    # We care about u's state to v's pre-state + potentially 
    @assert u.state.mode == v.pre_bridge_state.mode

    # We care about tp-dist to v
    # If inf, have modal policy overload it!
    mode = u.state.mode
    mdp = solver.modal_mdps[mode]
    weighted_value, weighted_minvalue = horizon_weighted_value(
                                            mdp,
                                            solver.modal_policies[mode]
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


function simulate(cmssp::CMSSP{D,C,AD,AC}, solver::HHPCSolver{D,C,AD,AC},
                  heuristic::Function) where {D,C,AD,AC}

    t = 0
    plan = true
    last_plan_time = 0

    # Defs needed for open-loop-plan
    edge_weight_fn(u,v) = edge_weight_fn(cmssp,solver,u,v)

    (start_state, start_context) = initial_state(cmssp)

    curr_state = start_state
    curr_context = start_context

    while True

        if plan == true
            open_loop_plan!(cmssp, curr_state, curr_context, edge_weight_fn,
                            heuristic, hhpc.graph_tracker)
            last_plan_time = t
            plan = false
        end

        # now do work for macro-actions 



    end

end