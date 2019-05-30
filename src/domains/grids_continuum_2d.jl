struct GridsContinuum2DParams
    epsilon::Float64
    num_generative_samples::Int64
    num_bridge_samples::Int64
    horizon_limit::Int64
    vals_along_axis::Int64 # Number of points along axis for interpolation
end



const GridsContinuum2DStateType = CMSSPState{Int64, Vec2}
const GridsContinuum2DActionType = CMSSPAction{Int64, Vec2}
const GridsContinuum2DContextType = Dict{Tuple{Int64, Int64}, Vec2}


mutable struct GridsContinuum2DContextSet
    curr_timestep::Int64
    curr_context::GridsContinuum2DContextType
    future_context::Vector{GridsContinuum2DContextType}
end



const GridsContinuumCMSSPType = CMSSP{Int64, Vec2, Int64, Vec2, GridsContinuum2DParams}