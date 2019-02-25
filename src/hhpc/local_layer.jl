# This implements the functionality required by the local layer
# For now, we're doing LocalApproximationVI, so the user has to 
# provide the interpolation object and define the necessary convert functions
# struct ModalPolicy
#     in_horizon_policy::LocalApproximationValueIterationPolicy
#     out_horizon_policy::LocalApproximationValueIterationPolicy
# end

# Dict{D,ModalPolicy}

# function get_modal_policy{D,C}(cmssp, mode, interp,...)
# function compute_penalty
# function convert_s (use convert_s for underlying state)
# struct CMSSPAugmented (for horizon)
# function incorporate_closedloop_interrupt

"""
CMSSP State augmented with horizon value. Needed for LocalApproximationVI
"""
struct CMSSPStateAugmented{D,C}
    state::CMSSPState{D,C}
    horizon::Int64
end

function CMSSPStateAugmented{D,C}(state::CMSSPState{D,C}) where {D,C}
    return CMSSPStateAugmented{D,C}(state, 0)
end


"""
Convert augmented state to a vector by calling the underlying convert_s function
for non-augmented state
"""
function POMDPs.convert_s(::Type{V},s::CMSSPStateAugmented{D,C}, 
                          cmssp::CMSSP{D,C,AD,AC}) where {D,C,AD,AC,V <: AbstractVector{Float64}}
    v = convert_s(Vector{Float64}, s.state, cmssp)
    push!(v, convert(Float64, s.horizon))
    return v
end

"""
Convert horizon-augmented vector to CMSSPStateAugmented
"""
function POMDPs.convert_s(::Type{CMSSPStateAugmented}, v::AbstractVector{Float64},
                          cmssp::CMSSP{D,C,AD,AC}) where {D,C,AD,AC}
    state = convert_s(CMSSPState, v, cmssp)
    horizon = convert(Int64,v[end])
    s = CMSSPStateAugmented{D,C}(state,horizon)
    return s
end

