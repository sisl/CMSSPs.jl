struct Toy2DContState
    x::Float64
    y::Float64
end
norm(s::Toy2DContState) = norm(SVector{2,Float64}(s.x,s.y))

struct Toy2DContAction
    xdot::Float64
    ydot::Float64
end
norm(a::Toy2DContAction) = norm(SVector{2,Float64}(a.x,a.y))

# Note - Hardcoding 1 x 1 grid because it can be scale-invariant
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
const Toy2DOpenLoopVertex = OpenLoopVertex{Int64,Toy2DStateType,Int64}
# Context is current collection of all valid transition points at each timestep
const Toy2DModePair = Tuple{Int64}
const Toy2DContextType = Vector{Tuple{Toy2DModePair,Toy2DContState}}

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

# Needs to be bound to parameters in runtime script
function isterminal(cmssp::Toy2DCMSSPType, state::Toy2DStateType, params::Toy2DParameters)
    return state.mode == GOAL_MODE && norm(get_relative_state(state,GOAL_CENTRE)) < params.epsilon
end

function sample_toy2d(rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    d = Uniform(0.0,1.0)
    return Toy2DContState(rand(d,rng), rand(d,rng))
end

function startstate_context(cmssp::Toy2DCMSSPType, start_context::ContextSetType, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    start_state = Toy2DStateType(1, sample_toy2d(rng))
    return start_state, start_context
end


# TODO
# generate_sr for modes - done
# global layer reqs

# Just used by generate_sr - euclidean distance + squared velocity cost
function POMDPs.reward(mdp::Toy2DModalMDPType, relative_state::Toy2DContState,
                       cont_action::Toy2DContAction, relative_statep::Toy2DContState)
    delta_relstate = Toy2DContState(relative_statep.x - relative_state.x, relative_statep.y - relative_state.y)
    reward = -1.0*(norm(delta_relstate) + 0.5*norm(cont_action)^2)    
end


# In this case, independent of actual mode
function POMDPs.generate_sr(mdp::Toy2DModalMDPType, relative_state::Toy2DContState, 
                            cont_action::Toy2DContAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # NOTE that this is the relative state
    noise_dist = MvNormal([0.01 + cont_action.xdot/5.0, 0.01 + cont_action.ydot/5.0])
    xy_noise = rand(noise_dist, rng)
    new_rel_x = clamp(relative_state.x + cont_action.xdot + xy_noise[1], -1.0, 1.0)
    new_rel_y = clamp(relative_state.y + cont_action.ydot + xy_noise[2], -1.0, 1.0)

    new_rel_state = Toy2DContState(new_rel_x, new_rel_y)
    reward_val = reward(mdp, relative_state, cont_action, new_rel_state)

    return new_rel_state, reward_val
end

# Needs to be bound to context set in runtime script
# We want that the state of the vertex be within the bounds
# of the transition point. If it is, set that to be the horizon
# Otherwise, set vertex to t = 0 with probability 1.0
function update_vertices_with_context!(cmssp::Toy2DCMSSPType, vertices::Vector{Toy2DOpenLoopVertex},
                                       context_set::Vector{Toy2DContextType})

    # Note that this is for a specific mode, so only consider those with the same mode switch
    # Iterate over context set
    # If the mode switch tuple is the same as in the open-loop vertex
    # Then check that the pre_bridge_state is in the range of the first
    # The post bridge state is just the same as from where the mode switch is attempted