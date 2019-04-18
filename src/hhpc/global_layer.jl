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

# TODO : Do we need the types sent as args?
# function GraphTracker(N::Int64, rng::RNG=Random.GLOBAL_RNG) where {D,C,AD,RNG <: AbstractRNG}
#     return GraphTracker(SimpleVListGraph{OpenLoopVertex{D,C,AD}}(),
#                         Dict{Tuple{D,D},MVector{2,Int64}}(),
#                         0,0,Vector{Int64}(undef,0),N,rng)
# end

function GraphTracker{D,C,AD,M,B,RNG}(N::Int64, bookkeeping::B, rng::RNG=Random.GLOBAL_RNG) where {D,C,AD,M,B,RNG <: AbstractRNG}
    return GraphTracker{D,C,AD,M,B,RNG}(SimpleVListGraph{OpenLoopVertex{D, C, AD, M}}(),
                        Dict{Tuple{D,D},MVector{2,Int64}}(),
                        0, 0, Vector{Int64}(undef, 0), N, bookkeeping, rng)
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
function open_loop_plan!(cmssp::CMSSP, s_t::CMSSPState,
                         curr_time::Int64,
                         edge_weight::Function,
                         heuristic::Function,
                         goal_modes::Vector{D},
                         graph_tracker::GraphTracker,
                         context_set::CS) where {D,CS} 

    # First update the context
    update_vertices_with_context!(cmssp, graph_tracker, context_set)

    # Create start vertex and insert in graph
    # Don't need explicit goal - visitor will handle
    add_vertex!(graph_tracker.curr_graph, OpenLoopVertex(s_t, cmssp.mode_actions[1], TPDistribution([curr_time], [1.0])))
    graph_tracker.curr_start_idx = num_vertices(graph_tracker.curr_graph)
    graph_tracker.curr_goal_idx = 0

    # Clear out current solution
    empty!(graph_tracker.curr_soln_path_idxs)

    # Obtain path and current cost with A*
    astar_path_soln = astar_light_shortest_path_implicit(graph_tracker.curr_graph, edge_weight,
                                                         graph_tracker.curr_start_idx,
                                                         GoalVisitorImplicit(graph_tracker, cmssp, goal_modes, context_set))

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
Update the current open-loop graph tracker with the new context set.

Arguments:
    - `cmssp::CMSSP{D,C,AD,AC}` The CMSSP instance
    - `graph_tracker::GraphTracker{D,C}` The graph_tracker instance to update in place
"""
# function update_graph_tracker!(cmssp::CMSSP{D, C, AD, AC}, graph_tracker::GraphTracker, context_set::CS) where {D, C, AD, AC, CS}
    
#     # Run through mode switch ranges and either retain or remove if context has changed too much
#     keys_to_delete = Vector{Tuple{D,D}}(undef, 0)

#     for (switch, idx_range) in graph_tracker.mode_switch_idx_range
#         # If samples in final mode, skip
#         if switch[1] == switch[2]
#             continue
#         end

#         # Copy over subvector of vertices
#         # range_subvector = graph_tracker.curr_graph.vertices[idx_range[1]:idx_range[2]]
#         (valid_update, new_idx_range) = update_vertices_with_context!(cmssp, graph_tracker, switch, context_set)

#         # Delete key if not updated
#         if valid_update == false
#             push!(keys_to_delete,switch)
#         else
#             graph_tracker.mode_switch_idx_range = new_idx_range
#         end
#     end

#     # Iterate over keys_to_delete and delete those that are not updated
#     for ktd in keys_to_delete
#         delete!(graph_tracker.mode_switch_idx_range, ktd)
#     end
# end


"""
The visitor object for the implicit A* search. Attributes obvious.
"""
struct GoalVisitorImplicit{CS} <: AbstractDijkstraVisitor
    graph_tracker::GraphTracker
    cmssp::CMSSP
    goal_modes::Vector
    context_set::CS
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
        (samples_to_add, nbrs_to_add) = generate_goal_sample_set!(vis.cmssp, v, vis.context_set, vis.graph_tracker, vis.graph_tracker.rng)
        
        for nbr_idx in nbrs_to_add
            push!(nbrs, nbr_idx)
        end

        num_new_samples = length(samples_to_add)

        if num_new_samples > 0
            
            # Add vertices to graph and to nbrs
            for gs in samples_to_add
                add_vertex!(vis.graph_tracker.curr_graph, OpenLoopVertex(gs, vis.cmssp.mode_actions[1]))
                push!(nbrs, num_vertices(vis.graph_tracker.curr_graph))
            end
        end
    end

    # Now add for next modes
    next_valid_modes = generate_next_valid_modes(vis.cmssp, popped_mode, vis.context_set)

    # @show next_valid_modes
    # @show vis.graph_tracker.mode_switch_idx_range

    for (action, nvm) in next_valid_modes
            
        # Generate bridge samples
        (samples_to_add, nbrs_to_add) = generate_bridge_sample_set!(vis.cmssp, v, 
                                                                    (popped_mode, action, nvm),
                                                                    vis.context_set, vis.graph_tracker,
                                                                    vis.graph_tracker.rng)
        for nbr_idx in nbrs_to_add
            push!(nbrs, nbr_idx)
        end

        num_new_samples = length(samples_to_add)

        if num_new_samples > 0

            # Add vertices to graph and to nbrs
            for bs in samples_to_add
                add_vertex!(vis.graph_tracker.curr_graph, OpenLoopVertex(popped_mode, nvm, bs, action))
                push!(nbrs, num_vertices(vis.graph_tracker.curr_graph))
            end
        end
    end

    #@show nbrs
    # readline()
    return true
end