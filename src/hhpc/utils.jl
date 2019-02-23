const TPDistribution = SparseCat{Vector{Int64},Vector{Float64}}

const VKey = MVector{2,Float64}

"""
Vertex type for open-loop layer
"""
mutable struct OpenLoopVertex{D,C}
    state::CMSSPState{D,C}
    key::VKey
    tp::TPDistribution
end