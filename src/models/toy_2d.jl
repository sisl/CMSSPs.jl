struct Toy2DContState
    x::Float64
    y::Float64
end
norm(s::Toy2DContState) = norm(SVector{2,Float64}(s.x,s.y))

struct Toy2DContAction
    xdot::Float64
    ydot::Float64
end

struct Toy2DParameters
    speed_limit::Float64
    speed_resolution::Float64
    epsilon::Float64
end

# Const types
const Toy2DStateType = CMSSPState{Int64,Toy2DContState}
const Toy2DActionType = CMSSPAction{Int64,Toy2DContAction}
const Toy2DCMSSPType = CMSSP{Int64,Toy2DStateType,Int64,Toy2DContAction}
const Toy2DModalMDPType = ModalMDP{Int64,Toy2DContState,Toy2DContAction}
const ContextSetType = Vector{Toy2DContState}

# Const values
const TOY2D_MODES = [1,2,3,4,5]
const GOAL_CENTRE = Toy2DContState(0.5,0.5)
const GOAL_MODE = 5

# Hardcoded - only 1 modeswitch action
# Get control actions based on parameters
function get_toy2d_actions(params::Toy2DParameters)

    actions = Vector{Toy2DActionType}(undef,0)

    idx = 1
    push!(actions, Toy2DActionType(1,idx))
    idx = 2
    push!(actions, Toy2DActionType(2,idx))


    # Now enter control actions based on parameters
    vel_vals = range(-params.speed_limit, stop=params.speed_limit, step=params.speed_resolution)
    for xdot in vel_vals
        for ydot in vel_vals
            idx += 1
            push!(actions, Toy2DActionType(Toy2DContAction(xdot,ydot), idx))
        end
    end

    return actions
end


# Hardcoded tabular MDP
# 1 -- 2 -- 5
# |         |
# 3 -- 4 -- |
function get_tabular_mdp()

    # Initialize T as 5 x 1 x 5
    T = zeros(5, 2, 5)
    T[1,1,2] = 1.0
    T[1,2,3] = 1.0
    T[2,1,5] = 1.0
    T[3,1,4] = 1.0
    T[3,2,5] = 1.0
    T[4,1,5] = 1.0

    R = zeros(5,2)
    R[1,1] = -1.0 # For T[1,1,2]
    R[1,3] = -2.0 # For T[1,2,3]
    R[2,5] = -8.0
    R[3,4] = -2.0
    R[3,5] = -2.0
    R[4,5] = -3.0

    # Undiscounted
    return TabularMDP(T, R, 1.0)
end



# Needs to be bound to parameters in runtime script
function isterminal(mdp::Toy2DModalMDPType, relative_state::Toy2DContState, params::Toy2DParameters)
    return norm(relative_state) < params.epsilon
end

function get_relative_state(source::Toy2DContState, target::Toy2DContState)
    return Toy2DContState(target.x - source.x, target.y - source.y)
end

function get_relative_state(mdp::Toy2DModalMDPType, source::Toy2DContState, target::Toy2DContState)
    return get_relative_state(source, target) 
end

function isterminal(cmssp::Toy2DCMSSPType, state::Toy2DStateType, params::Toy2DParameters)
    return state.mode == GOAL_MODE && norm(get_relative_state(state,GOAL_CENTRE)) < params.epsilon
end

function sample_toy2d(rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    d = Distributions.Uniform(0.0,1.0)
    return Toy2DContState(rand(d,rng), rand(d,rng))
end

function startstate_context(cmssp::Toy2DCMSSPType, start_context::ContextSetType, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    start_state = Toy2DStateType(1, sample_toy2d())
    return start_state, start_context
end

# TODO
# generate_sr for modes
# global layer reqs