function open_loop_plan(cmssp::CMSSP, s_t::CMSSPState, 
                        context_set::Vector{Any}, N::Int64,
                        edge_weight::Function)



end

struct GoalVisitorImplicit <: AbstractDijkstraVisitor
    cmssp::CMSSP
    next_valid_modes::Function
    bridge_sample::Function
end

function Graphs.include_vertex!(vis::GoalVisitorImplicit, u::V, v::V, d::Float64, nbrs::Vector{Int64}) where {V}

    # Implement expand

end