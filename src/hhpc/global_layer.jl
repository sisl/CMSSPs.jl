"""
Data structure used by open-loop layer to update current high-level plan.

Attributes:
    - `curr_graph::SimpleVListGraph{OpenLoopVertex{D,C,AD}}` A list of vertices generated so far by the bridge sampler
    - `curr_start_idx::Int64` The vertex index of the current start vertex in the graph, from which planning will happen. Updated each time open_loop_plan is called
    - `curr_goal_idx::Int64` The vertex index of current goal vertex. Updated each time a goal state vertex is popped.
    - `curr_soln_path_idxs::Vector{Int64}` The in-order list of indices from current start to current goal
    - `num_samples::Int64` The N parameter for the number of bridge samples to generate per mode switch
    - `bookkeeping::B` A data structure for tracking information pertaining to the graph search
"""
mutable struct GraphTracker{D,C,AD,M,B,RNG <: AbstractRNG}
    curr_graph::SimpleVListGraph{OpenLoopVertex{D,C,AD,M}}
    curr_start_idx::Int64
    curr_goal_idx::Int64
    curr_soln_path_idxs::Vector{Int64}
    num_samples::Int64
    bookkeeping::B
    rng::RNG
end

metadatatype(::GraphTracker{D,C,AD,M,B,RNG}) where {D,C,AD,M,B,RNG <: AbstractRNG} = M
bookkeepingtype(::GraphTracker{D,C,AD,M,B,RNG}) where {D,C,AD,M,B,RNG <: AbstractRNG} = B


function GraphTracker{D,C,AD,M,B,RNG}(N::Int64, bookkeeping::B, rng::RNG=Random.GLOBAL_RNG) where {D,C,AD,M,B,RNG <: AbstractRNG}
    return GraphTracker{D,C,AD,M,B,RNG}(SimpleVListGraph{OpenLoopVertex{D,C,AD,M}}(),
                        0, 0, Vector{Int64}(undef, 0), N, bookkeeping, rng)
end

"""
Open loop plan from current state from scratch. Return a graph tracker with solution

Arguments:
    - `cmssp::CMSSP{D,C,AD,AC}` The CMSSP instance
    - `s_t::CMSSPState{D,C}` The current state
    - `curr_time::Int64` The current time step value
    - `edge_weight::Function` A generic edge weight function (handed down by hhpc solver)
    - `heuristic::Function` A generic heuristic function (handed down by hhpc solver)
    - `goal_modes::Vector{D}` A set of modes that are goal conditions
    - `graph_tracker::GraphTracker{D,C}` The instance of the object that maintains the current open-loop graph
    - `start_metadata::M` Default metadata for a new start vertex

Returns:
    Updates the `graph_tracker` object in place.
"""
function open_loop_plan!(cmssp::CMSSP, s_t::CMSSPState,
                         curr_time::Int64,
                         edge_weight::Function,
                         heuristic::Function,
                         goal_modes::Vector{D},
                         graph_tracker::GraphTracker{D,C,AD,M,B,RNG},
                         start_metadata::M) where {D,C,AD,M,B,RNG <: AbstractRNG} 

    # First update the context
    update_vertices_with_context!(cmssp, graph_tracker, curr_time)

    # Create start vertex and insert in graph
    # Don't need explicit goal - visitor will handle
    add_vertex!(graph_tracker.curr_graph, OpenLoopVertex{D,C,AD,M}(s_t, cmssp.mode_actions[1], TPDistribution([curr_time], [1.0]), start_metadata))
    graph_tracker.curr_start_idx = num_vertices(graph_tracker.curr_graph)
    graph_tracker.curr_goal_idx = 0

    # Clear out current solution
    empty!(graph_tracker.curr_soln_path_idxs)

    # Obtain path and current cost with A*
    astar_path_soln = astar_light_shortest_path_implicit(graph_tracker.curr_graph, edge_weight,
                                                         graph_tracker.curr_start_idx,
                                                         GoalVisitorImplicit(graph_tracker, cmssp, goal_modes))

    # If unreachable, report warning
    if graph_tracker.curr_goal_idx == 0
        @warn "A goal state is currently unreachable"
        return
    end

    # Check that goal state is indeed terminal
    # And that A* has non-Inf cost path to it
    @assert astar_path_soln.dists[graph_tracker.curr_goal_idx] < Inf "Path to goal is Inf cost!"

    ## Walk path back to goal
    # Insert goal in soln vector
    pushfirst!(graph_tracker.curr_soln_path_idxs, graph_tracker.curr_goal_idx)
    curr_vertex_idx = graph_tracker.curr_goal_idx

    while curr_vertex_idx != graph_tracker.curr_start_idx
        prev_vertex_idx = astar_path_soln.parent_indices[curr_vertex_idx]
        pushfirst!(graph_tracker.curr_soln_path_idxs, prev_vertex_idx)
        curr_vertex_idx = prev_vertex_idx
    end
    # @show graph_tracker.curr_soln_path_idxs
    # for idx in graph_tracker.curr_soln_path_idxs
    #     println(graph_tracker.curr_graph.vertices[idx].state," at ",graph_tracker.curr_graph.vertices[idx].tp)
    #     hor_val = graph_tracker.curr_graph.vertices[idx].tp.vals[1]
    #     if hor_val < typemax(Int64)
    #         display_context_future(context_set,hor_val)
    #     end
    # end
end



"""
The visitor object for the implicit A* search. Attributes are obvious.
"""
struct GoalVisitorImplicit <: AbstractDijkstraVisitor
    graph_tracker::GraphTracker
    cmssp::CMSSP
    goal_modes::Vector
end


function Graphs.include_vertex!(vis::GoalVisitorImplicit, 
                                u::OpenLoopVertex, v::OpenLoopVertex, d::Float64, nbrs::Vector{Int64})

    # @show v
    # If popped vertex is terminal, then stop
    if isterminal(vis.cmssp, v.state) == true
        vis.graph_tracker.curr_goal_idx = vertex_index(vis.graph_tracker.curr_graph, v)
        return false
    end

    popped_mode = v.state.mode
    popped_cont = v.state.continuous

    # If goal mode but NOT goal state, add samples from goal
    if popped_mode in vis.goal_modes
            
        # Generate goal sample set
        (vertices_to_add, nbrs_to_add) = generate_goal_vertex_set!(vis.cmssp, v, vis.graph_tracker, vis.graph_tracker.rng)
        
        for nbr_idx in nbrs_to_add
            push!(nbrs, nbr_idx)
        end

        num_new_vertices = length(vertices_to_add)

        if num_new_vertices > 0
            
            # Add vertices to graph and to nbrs
            for vtx in vertices_to_add
                add_vertex!(vis.graph_tracker.curr_graph, vtx)
                push!(nbrs, num_vertices(vis.graph_tracker.curr_graph))
            end
        end
    end

    # Now add for next modes
    next_valid_modes = generate_next_valid_modes(vis.cmssp, popped_mode)


    for (action, nvm) in next_valid_modes
            
        # Generate bridge samples
        (vertices_to_add, nbrs_to_add) = generate_bridge_vertex_set!(vis.cmssp, v, 
                                                                    (popped_mode, nvm),
                                                                    vis.graph_tracker,
                                                                    action,
                                                                    vis.graph_tracker.rng)
        for nbr_idx in nbrs_to_add
            push!(nbrs, nbr_idx)
        end

        num_new_vertices = length(vertices_to_add)

        if num_new_vertices > 0

            # Add vertices to graph and to nbrs
            for vtx in vertices_to_add
                add_vertex!(vis.graph_tracker.curr_graph, vtx)
                push!(nbrs, num_vertices(vis.graph_tracker.curr_graph))
            end
        end
    end

    # @show v
    # @show nbrs
    # readline()
    return true
end