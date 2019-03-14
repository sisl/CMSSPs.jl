"""
Data structure used by open-loop layer to update current high-level plan.

Attributes:
    - `curr_graph::SimpleVListGraph{OpenLoopVertex{D,C,AD}}` A list of vertices generated so far by the bridge sampler
    - `mode_switch_idx_range::Dict{Tuple{D,D},MVector{2,Int64}}` The range for the subvector of vertices generated for a particular mode switch.
    Is updated in place by update_graph_tracker!
    - `curr_start_idx::Int64` The vertex index of the current start vertex in the graph, from which planning will happen. Updated each time open_loop_plan is called
    - `curr_goal_idx::Int64` The vertex index of current goal vertex. Updated each time a goal state vertex is popped.
    - `curr_soln_path_idxs::Vector{Int64}` The in-order list of indices from current start to current goal
    - `num_samples::Int64` The N parameter for the number of bridge samples to generate per mode switch
"""
mutable struct GraphTracker{D,C,AD,RNG <: AbstractRNG}
    curr_graph::SimpleVListGraph{OpenLoopVertex{D,C,AD}}
    mode_switch_idx_range::Dict{Tuple{D,D},MVector{2,Int64}}
    curr_start_idx::Int64
    curr_goal_idx::Int64
    curr_soln_path_idxs::Vector{Int}
    num_samples::Int64
    rng::RNG
end

function GraphTracker{D,C,AD}(N::Int64, rng::RNG=Random.GLOBAL_RNG) where {D,C,AD,RNG <: AbstractRNG}
    return GraphTracker{D,C}(SimpleVListGraph{OpenLoopVertex{D,C,AD}}(),
                        Dict{Tuple{D,D},MVector{2,Int64}}(),
                        0,
                        0,
                        Vector{Int64}(undef,0),
                        N,
                        rng)
end

"""
Open loop plan from current state from scratch. Return a graph tracker with solution

Arguments:
    - `cmssp::CMSSP{D,C,AD,AC}` The CMSSP instance
    - `s_t::CMSSPState{D,C}` The current state
    - `edge_weight::Function` A generic edge weight function (handed down by hhpc solver)
    - `heuristic::Function` A generic heuristic function (handed down by hhpc solver)
    - `goal_modes::Vector{D}` A set of modes that are goal conditions
    - `graph_tracker::GraphTracker{D,C}` The instance of the object that maintains the current open-loop graph

Returns:
    Updates the `graph_tracker` object in place.
"""
function open_loop_plan!(cmssp::CMSSP{D,C,AD,AC}, s_t::CMSSPState{D,C}, 
                        context_set::Vector{Any},
                        edge_weight::Function,
                        heuristic::Function,
                        goal_modes::Vector{D},
                        graph_tracker::GraphTracker{D,C}) where {D,C,AD,AC}

    # Create start vertex and insert in graph
    # Don't need explicit goal - visitor will handle
    add_vertex!(graph_tracker.curr_graph, OpenLoopVertex{D,C,AD}(s_t))
    graph_tracker.curr_start_idx = num_vertices(graph_tracker.curr_graph)
    graph_tracker.curr_goal_idx = 0

    # Clear out current solution
    empty!(graph_tracker.curr_soln_path_idxs)

    # Obtain path and current cost with A*
    astar_path_soln = astar_light_shortest_path_implicit(graph_tracker.curr_graph, edge_weight,
                                                         graph_tracker.curr_start_idx,
                                                         GoalVisitorImplicit{D,C,AD,AC}(graph_tracker, cmssp, context_set, goal_modes))

    # If unreachable, report warning
    if graph_tracker.curr_goal_idx == 0
        @warn "A goal state is currently unreachable"
        return
    end

    # Check that goal state is indeed terminal
    # And that A* has non-Inf cost path to it
    @assert astar_path_soln.dists[graph_tracker.curr_goal_idx] < Inf "Path to goal is Inf cost!"
    @assert is_terminal(cmssp, graph_tracker.vertices[graph_tracker.curr_goal_idx].state) == true "Goal state is not terminal!"

    ## Walk path back to goal
    # Insert goal in soln vector
    pushfirst!(graph_tracker.curr_soln_path_idxs, graph_tracker.curr_goal_idx)
    curr_vertex_idx = curr_goal_idx

    while curr_vertex_idx != graph_tracker.curr_start_idx
        prev_vertex_idx = astar_path_soln.parent_indices[curr_vertex_idx]
        pushfirst!(graph_tracker.curr_soln_path_idxs, prev_vertex_idx)
        curr_vertex_idx = prev_vertex_idx
    end
end

"""
Update the current open-loop graph tracker with the new context set.

Arguments:
    - `cmssp::CMSSP{D,C,AD,AC}` The CMSSP instance
    - `graph_tracker::GraphTracker{D,C}` The graph_tracker instance to update in place
    - `context_set::Vector{Any}` The current and estimated future context
"""
function update_graph_tracker!(cmssp::CMSSP{D,C,AD,AC}, graph_tracker::GraphTracker{D,C},
                               context_set::Vector{Any}) where {D,C,AD,AC}
    
    # Run through mode switch ranges and either retain or remove if context has changed too much
    keys_to_delete = Vector{Tuple{D,D}}(undef,0)

    for (switch,idx_range) in graph_tracker.mode_switch_idx_range
        # If samples in final mode, skip
        if switch[1] == switch[2]
            continue
        end

        # Copy over subvector of vertices
        range_subvector = graph_tracker.curr_graph.vertices[idx_range[1]:idx_range[2]]
        valid_update = update_vertices_with_context!(cmssp, range_subvector, context_set)

        # Delete key if not updated
        if valid_update == false
            push!(keys_to_delete,switch)
        end
    end

    # Iterate over keys_to_delete and delete those that are not updated
    for ktd in keys_to_delete
        delete(graph_tracker.mode_switch_idx_range, ktd)
    end
end


"""
The visitor object for the implicit A* search. Attributes obvious.
"""
struct GoalVisitorImplicit{D,C,AD,AC} <: AbstractDijkstraVisitor
    graph_tracker::GraphTracker
    cmssp::CMSSP{D,C,AD,AC}
    context_set::Vector{Any}
    goal_modes::Vector{D}
end


function Graphs.include_vertex!(vis::GoalVisitorImplicit{D,C,AD,AC}, 
                                u::OpenLoopVertex{D,C,AD}, v::OpenLoopVertex{D,C,AD}, d::Float64, nbrs::Vector{Int64}) where {D,C,AD,AC}

    # If popped vertex is terminal, then stop
    if is_terminal(vis.cmssp, v.state) == true
        vis.graph_tracker.curr_goal_idx = vertex_index(vis.graph_tracker.curr_graph, v)
        return false
    end

    popped_mode = v.state.mode
    popped_cont = v.state.continuous

    # If goal mode but NOT goal state, add samples from goal
    if popped_mode in vis.goal_modes
        # If leftover from previous step, just re-add those
        if haskey(vis.mode_switch_idx_range,(popped_mode,popped_mode))
            mode_switch_range = vis.mode_switch_idx_range[(popped_mode,popped_mode)]
            for nbr_idx = mode_switch_range[1] : mode_switch_range[2]
                push!(nbrs, nbr_idx)
            end
        else
            # Generate goal sample set
            goal_samples = generate_goal_sample_set(vis.cmssp, popped_cont, vis.graph_tracker.num_samples, vis.graph_tracker.rng)
            @assert length(goal_samples) > 0

            # Update mode switch map range
            range_st = num_vertices(vis.graph_tracker.curr_graph)+1
            range_end = range_st + length(goal_samples)
            vis.graph_tracker.mode_switch_idx_range[(popped_mode,popped_mode)] = MVector{2,Int64}(range_st, range_end)
            
            # Add vertices to graph and to nbrs
            # IMP - Add a default mode-switch action
            for (i,gs) in enumerate(goal_samples)
                add_vertex!(vis.graph_tracker.curr_graph, OpenLoopVertex{D,C,AD}(gs, vis.cmssp.mode_actions[1]))
                push!(nbrs,range_st+i-1)
            end
        end
    end

    # Now add for next modes
    next_valid_modes = generate_next_valid_modes(vis.cmssp, vis.context_set, popped_mode)

    for (action,nvm) in next_valid_modes
        # First check if mode switch has them, then just use those
        if haskey(vis.mode_switch_idx_range,(popped_mode,nvm))
            mode_switch_range = vis.mode_switch_idx_range[(popped_mode,nvm)]
            for nbr_idx = mode_switch_range[1] : mode_switch_range[2]
                push!(nbrs, nbr_idx)
            end
        else
            # Generate bridge samples
            bridge_samples = generate_bridge_sample_set(vis.cmssp, vis.context_set, popped_cont, 
                                                        (popped_mode, nvm), vis.graph_tracker.num_samples, vis.graph_tracker.rng)
            @assert length(bridge_samples) > 0

            # Update mode switch map range
            range_st = num_vertices(vis.graph_tracker.curr_graph)+1
            range_end = range_st + length(bridge_samples)
            vis.graph_tracker.mode_switch_idx_range[(popped_mode,nvm)] = MVector{2,Int64}(range_st, range_end)

            # Add vertices to graph and to nbrs
            for (i,bs) in enumerate(bridge_samples)
                add_vertex!(vis.graph_tracker.curr_graph, OpenLoopVertex{D,C,AD}(popped_mode,nvm,bs,action))
                push!(nbrs,range_st+i-1)
            end
        end
    end

    return true

end