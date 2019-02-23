function open_loop_plan!(cmssp::CMSSP{D,C,AD,AC}, s_t::CMSSPState{D,C}, 
                        context_set::Vector{Any}, N::Int64,
                        edge_weight::Function, is_valid_edge::Function) where {D,C,AD,AC}

# Creates graph inside and just returns what is needed



end

"""
Open loop plan from current state after updating an existing graph. Return a Vector{V} (user doesn't care about indices)
"""
function open_loop_replan!(cmssp::CMSSP{D,C,AD,AC}, s_t::CMSSPState{D,C}, 
                           context_set::Vector{Any}, N::Int64,
                           edge_weight::Function, is_valid_edge::Function,
                           graph_tracker::GraphTracker{D,C}) where {D,C,AD,AC}

# Updates graph tracker and returns what is needed

end


mutable struct GraphTracker{D,C}
    curr_graph::SimpleVListGraph{OpenLoopVertex{D,C}}
    mode_switch_map::Dict{Tuple{D,D},Set{Int64}}
    curr_goal_idx::Int64
    next_start_idx::Int64
    has_next_start::Bool
    curr_soln_path_idxs::Vector{Int64}
end

function GraphTracker{D,C}()
    return GraphTracker(SimpleVListGraph{OpenLoopVertex{D,C}}(),
                        Dict{Tuple{D,D},Set{Int64}}(),
                        0,
                        0,
                        false,
                        Vector{Int64}(undef,0))
end



struct GoalVisitorImplicit <: AbstractDijkstraVisitor
    graph_tracker::GraphTracker
    next_valid_modes::Function
    bridge_sample::Function
end


function Graphs.include_vertex!(vis::GoalVisitorImplicit, u::V, v::V, d::Float64, nbrs::Vector{Int64}) where {V}

    # Implement expand

end