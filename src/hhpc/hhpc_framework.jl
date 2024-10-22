"""
The overall HHPC framework solver object.

Attributes:
    - `graph_tracker::GraphTracker{D,C,AD}`
    - `modal_policies::Dict{D,Policy}` A map from the mode to the modal policies (in and out-horizon)
    - `replan_time_threshold::Int64` The Delta T parameter for periodic replanning (discrete time-steps)
    - `heuristic::Function` The heuristic method for the global layer
    - `curr_state::CMSSPState{D,C}` The current state of the agent, tracked by the solver.
    - `max_steps::Int64` The maximum length of a problem episode (may not be applicable)
    - `start_state::CMSSPState{C,AC}`
    - `goal_modes::Vector{D}`
"""
mutable struct HHPCSolver{D,C,AD,M,B,RNG <: AbstractRNG} <: Solver
    graph_tracker::GraphTracker{D,C,AD,M,B,RNG}
    modal_policies::Dict
    replan_time_threshold::Int64
    goal_modes::Vector{D}
    curr_state::CMSSPState{D,C}
    rng::RNG
    heuristic::Function
    max_steps::Int64
end

function HHPCSolver{D,C,AD,M,B,RNG}(num_samples::Int64, modal_policies::Dict,
                                    deltaT::Int64, goal_modes::Vector{D},
                                    start_state::CMSSPState{D,C}, 
                                    bookkeeping::B, rng::RNG=Random.GLOBAL_RNG,
                                    heuristic::Function = n->0, max_steps::Int64=1000) where {D,C,AD,M,B,RNG <: AbstractRNG}
    return HHPCSolver(GraphTracker{D,C,AD,M,B,RNG}(num_samples, bookkeeping, rng),
                      modal_policies, deltaT, goal_modes,
                      start_state, rng, heuristic, max_steps)
end


function set_start_state!(solver::HHPCSolver, start_state::CMSSPState{D,C}) where {D,C}
    solver.curr_state = start_state
end

function set_open_loop_samples!(solver::HHPCSolver, num_samples::Int64)
    solver.graph_tracker.num_samples = num_samples
end


"""
Creates the weight function for the edges, used by the global layer graph search.

Arguments:
    - `solver::HHPCSolver{D,C,AD,AC}` The HHPC solver instance
    - `cmssp::CMSSP{D,C,AD,AC}` The CMSSP instance
    - `u::OpenLoopVertex{D,C,AD}` The parent vertex
    - `v::OpenLoopVertex{D,C,AD}` The child vertex
"""
function edge_weight_fn(solver::HHPCSolver, cmssp::CMSSP,
                        u::OpenLoopVertex, v::OpenLoopVertex)

    # u's state to v's pre-state should be in a single region
    @assert u.state.mode == v.pre_bridge_state.mode

    # We care about relative tp_dist between u and v
    # If inf, have modal policy overload it!
    mode = u.state.mode

    # Branch on finite horizon and infinite horizon
    if is_inf_hor(v.tp)

        policy = solver.modal_policies[mode].inf_hor_policy
        pol_value = inf_hor_value(policy, u.state.continuous, v.pre_bridge_state.continuous)

    else
        policy = solver.modal_policies[mode].fin_hor_policy
        mdp = get_mdp(policy)
        temp_curr_time = convert(Int64, round(mean(u.tp)))

        pol_value, weighted_minvalue = horizon_weighted_value(policy,
                                                temp_curr_time,
                                                v.tp,
                                                u.state.continuous,
                                                v.pre_bridge_state.continuous)

        # Filter out edges that nominally would trigger an interrupt
        if pol_value <= mdp.beta_threshold*weighted_minvalue
            return Inf
        end
    end

    edge_weight = -1.0*pol_value

    # Also add the cost due to the assumed mode switch
    if v.pre_bridge_state.mode != v.state.mode
        ac_idx = mode_actionindex(cmssp,v.bridging_action)
        mode_idx = mode_index(cmssp,v.pre_bridge_state.mode)
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
    CS = typeof(cmssp.curr_context_set)
    PR = typeof(cmssp.params)
    M = metadatatype(solver.graph_tracker)
    B = bookkeepingtype(solver.graph_tracker)
    MDPType = Union{ModalFinHorMDP{D,C,AC,PR},ModalInfHorMDP{D,C,AC,PR}}

    @req isterminal(::P, ::S)
    @req generate_sr(::MDPType, ::C, ::AC, ::typeof(solver.rng)) # OR transition + reward?
    @req isterminal(::MDPType, ::C)

    # Global layer requirements
    @req update_vertices_with_context!(::P, ::typeof(solver.graph_tracker), ::Int64)
    @req generate_goal_vertex_set!(::P, ::OpenLoopVertex{D, C, AD, M}, ::typeof(solver.graph_tracker), ::RNG where {RNG <: AbstractRNG})
    @req generate_next_valid_modes(::P, ::D)
    @req generate_bridge_vertex_set!(::P, ::OpenLoopVertex{D,C,AD,M}, ::Tuple{D,D}, ::typeof(solver.graph_tracker), ::AD, ::RNG where {RNG <: AbstractRNG})
    @req update_next_target!(::P, ::typeof(solver), ::Int64)
    @req zero(::Type{M})    

    # Local layer requirements
    @req get_relative_state(::MDPType, ::C, ::C)
    @req expected_reward(::MDPType, ::C, ::AC)
    @req convert_s(::Type{Vector{Float64}}, ::C, ::ModalFinHorMDP{D,C,AC,PR})
    @req convert_s(::Type{C}, ::AbstractVector{Float64}, ::ModalFinHorMDP{D,C,AC,PR})

    # HHPC requirements
    @req simulate_cmssp!(::P, ::S, ::Union{Nothing, A}, ::Int64, ::RNG where {RNG <: AbstractRNG})
    @req update_context_set!(::P, ::RNG where {RNG <: AbstractRNG}) 
    @req get_bridging_action(::OpenLoopVertex{D, C, AD, M})
    @req display_context_future(::CS, ::Int64)

end



# TODO - Update context before or after simulate step???
"""
Executes the top-level behavior of HHPC. Utilizes both global and local layer logic, and interleaving.
"""
function POMDPs.solve(solver::HHPCSolver, cmssp::CMSSP{D,C,AD,AC,P,CS}) where {D,C,AD,AC,P,CS}

    @warn_requirements solve(solver,cmssp)

    # Initialize Global flags
    t = 0
    plan = true
    last_plan_time = 0

    # Defs needed for open-loop-plan
    edge_weight(u::OpenLoopVertex, v::OpenLoopVertex) = edge_weight_fn(solver, cmssp, u, v)

    # Diagnostic info
    total_cost = 0.0
    successful = false
    mode_switches = 0

    start_metadata = zero(metadatatype(solver.graph_tracker))

    next_target = nothing

    while t <= solver.max_steps

        @debug "Curr state - ",solver.curr_state
        @debug "Curr time - ", t

        if plan == true
            open_loop_plan!(cmssp, solver.curr_state, t, edge_weight, solver.heuristic,
                                  solver.goal_modes, solver.graph_tracker, start_metadata)
            last_plan_time = t
            plan = false
            next_target = solver.graph_tracker.curr_graph.vertices[solver.graph_tracker.curr_soln_path_idxs[2]]
        else
            if is_inf_hor(next_target.tp) == false
                valid_update = update_next_target!(cmssp, solver, t)

                if valid_update == false
                    plan = false
                    continue
                end
                next_target = solver.graph_tracker.curr_graph.vertices[solver.graph_tracker.curr_soln_path_idxs[2]]
            end
        end

        # now do work for macro-actions 
         # Second vertex in full plan
        # start_metadata = next_target.metadata
        @debug "Next Target - ", next_target

        # Check if 'terminal' state as per modal MDP
        @assert solver.curr_state.mode == next_target.pre_bridge_state.mode

        relative_time = mean(next_target.tp) - t
        @debug "Time to next switch - ".relative_time

        if relative_time < 2

            curr_action = get_bridging_action(next_target)
            
        else
            if is_inf_hor(next_target.tp)
                policy = solver.modal_policies[solver.curr_state.mode].inf_hor_policy
            else
                policy = solver.modal_policies[solver.curr_state.mode].fin_hor_policy
            end
            curr_action = get_best_intramodal_action(policy, 
                                                     t,
                                                     next_target.tp, 
                                                     solver.curr_state.continuous, 
                                                     next_target.pre_bridge_state.continuous)
            if curr_action != nothing
                curr_action = curr_action.action
            end
        end

        @debug "Current Action - ", curr_action


        if curr_action != nothing
            temp_full_action = CMSSPAction{AD,AC}(curr_action, 1)
        else
            temp_full_action = nothing
        end

        # If curr_action = NOTHING, means interrupt
        # So simulator should handle NOTHING
        # Will typically be bound to other environment variables
        # update_context_set!(cmssp, solver.rng)
        (new_state, reward, failed_mode_switch, timeout) = simulate_cmssp!(cmssp, solver.curr_state, temp_full_action, t, solver.rng)
        t = t+1
        total_cost += -1.0*reward

        if curr_action == nothing || failed_mode_switch || t - last_plan_time > solver.replan_time_threshold ||
            new_state.mode != solver.curr_state.mode
            plan = true
        end

        if new_state.mode != solver.curr_state.mode
            start_metadata = next_target.metadata
            mode_switches += 1
        end

        # Update current state and context
        solver.curr_state = new_state
        update_context_set!(cmssp, solver.rng)

        @debug new_state

        # Check terminal conditions
        if POMDPs.isterminal(cmssp, solver.curr_state)
            successful = true
            break
        end

        if timeout
            successful = false
            break
        end
    end

    # TODO : Diagnostics
    return total_cost, t, successful, mode_switches
end