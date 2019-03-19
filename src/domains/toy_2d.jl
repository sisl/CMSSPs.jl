struct Toy2DContState
    x::Float64
    y::Float64
end
norm(s::Toy2DContState) = LinearAlgebra.norm(SVector{2,Float64}(s.x,s.y))

struct Toy2DContAction
    xdot::Float64
    ydot::Float64
end
norm(a::Toy2DContAction) = LinearAlgebra.norm(SVector{2,Float64}(a.xdot,a.ydot))

# Note - Hardcoding 1 x 1 grid because it can be scale-invariant
struct Toy2DParameters
    speed_limit::Float64
    speed_vals::Int64
    epsilon::Float64
    num_generative_samples::Int64
    num_bridge_samples::Int64
end

# Const types
const Toy2DStateType = CMSSPState{Int64,Toy2DContState}
const Toy2DActionType = CMSSPAction{Int64,Toy2DContAction}
const Toy2DCMSSPType = CMSSP{Int64,Toy2DContState,Int64,Toy2DContAction}
const Toy2DModalMDPType = ModalMDP{Int64,Toy2DContState,Toy2DContAction}
const Toy2DOpenLoopVertex = OpenLoopVertex{Int64,Toy2DContState,Int64}
# Context is current collection of all valid transition points at each timestep
const Toy2DModePair = Tuple{Int64, Int64}
const Toy2DContextType = Vector{Tuple{Toy2DModePair,Toy2DContState}}

# Const values
const TOY2D_MODES = [1,2,3,4,5]
const TOY2D_GOAL_CENTRE = Toy2DContState(0.5,0.5)
const TOY2D_GOAL_MODE = 5

# Hardcoded - only 1 modeswitch action
# Get control actions based on parameters
function get_toy2d_actions(params::Toy2DParameters)

    actions = Vector{Toy2DActionType}(undef,0)

    idx = 1
    push!(actions, Toy2DActionType(1,idx))
    idx = 2
    push!(actions, Toy2DActionType(2,idx))

    vel_vals = polyspace_symmetric(params.speed_limit, params.speed_vals)

    # Now enter control actions based on parameters
    #vel_vals = range(-params.speed_limit, stop=params.speed_limit, step=params.speed_resolution)
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
function get_toy2d_switch_mdp()

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
    R[1,2] = -2.0 # For T[1,2,3]
    R[2,1] = -8.0
    R[3,1] = -2.0
    R[3,2] = -2.0
    R[4,1] = -3.0

    # Undiscounted
    return TabularMDP(T, R, 1.0)
end


# Create Toy2DCMSSP based on params
function create_toy2d_cmssp(params::Toy2DParameters)

    actions = get_toy2d_actions(params)
    modes = TOY2D_MODES
    switch_mdp = get_toy2d_switch_mdp()

    return Toy2DCMSSPType(actions, modes, switch_mdp)
end


# Needs to be bound to parameters in runtime script
function isterminal(mdp::Toy2DModalMDPType, relative_state::Toy2DContState, params::Toy2DParameters)
    return norm(relative_state) < params.epsilon
end

function get_relative_state(source::Toy2DContState, target::Toy2DContState)
    return Toy2DContState(source.x - target.x, source.y - target.y)
end

function HHPC.get_relative_state(mdp::Toy2DModalMDPType, source::Toy2DContState, target::Toy2DContState)
    return get_relative_state(source, target) 
end


function POMDPs.convert_s(t::Type{V}, c::Toy2DContState, mdp::Toy2DModalMDPType) where V <: AbstractVector{Float64}
    v_ = SVector{2,Float64}(c.x, c.y)
    v = convert(t,v_)
    return v
end

function POMDPs.convert_s(::Type{Toy2DContState}, v::AbstractVector{Float64}, mdp::Toy2DModalMDPType)
    c = Toy2DContState(v[1], v[2])
    return c
end




# Needs to be bound to parameters in runtime script
function isterminal(cmssp::Toy2DCMSSPType, state::Toy2DStateType, params::Toy2DParameters)
    return state.mode == TOY2D_GOAL_MODE && norm(get_relative_state(state,TOY2D_GOAL_CENTRE)) < params.epsilon
end

function sample_toy2d(rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    d = Uniform(0.0,1.0)
    return Toy2DContState(rand(rng,d), rand(rng,d))
end

function startstate_context(cmssp::Toy2DCMSSPType, rng::RNG, start_context::Toy2DContextType) where {RNG <: AbstractRNG}
    start_state = Toy2DStateType(1, sample_toy2d(rng))
    return (start_state, start_context)
end


# TODO
# generate_sr for modes - done
# global layer reqs

# Just used by generate_sr - euclidean distance + squared velocity cost
function POMDPs.reward(mdp::Toy2DModalMDPType, state::Toy2DContState,
                       cont_action::Toy2DContAction, statep::Toy2DContState)
    delta_state = Toy2DContState(statep.x - state.x, statep.y - state.y)
    reward = -1.0*(norm(delta_state) + 0.0*norm(cont_action)^2)
    return reward  
end


function expected_reward(mdp::Toy2DModalMDPType, state::Toy2DContState,
                         cont_action::Toy2DContAction, rng::RNG, params::Toy2DParameters) where {RNG <: AbstractRNG}
    avg_rew = 0.0
    for i = 1:params.num_generative_samples
        (_, r) = next_state_reward(mdp,state, cont_action, rng)
        avg_rew += r
    end
    avg_rew /= params.num_generative_samples
    return avg_rew
end 
    


function next_state_reward(mdp::Toy2DModalMDPType, state::Toy2DContState, 
                           cont_action::Toy2DContAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    noise_dist = MvNormal([0.0001 + cont_action.xdot/100.0, 0.0001 + cont_action.ydot/100.0])
    xy_noise = rand(rng,noise_dist)
    new_x = clamp(state.x + cont_action.xdot + xy_noise[1], -1.0, 1.0)
    new_y = clamp(state.y + cont_action.ydot + xy_noise[2], -1.0, 1.0)

    new_state = Toy2DContState(new_x, new_y)
    reward_val = reward(mdp, state, cont_action, new_state)

    return (new_state, reward_val)
end

    

# In this case, independent of actual mode
function POMDPs.generate_sr(mdp::Toy2DModalMDPType, state::Toy2DContState, 
                            cont_action::Toy2DContAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    # can be used with relative state too
    return next_state_reward(mdp, state, cont_action, rng)
end

# Needs to be bound to context set in runtime script
# We want that the state of the vertex be within the bounds
# of the transition point. If it is, set that to be the horizon
# Otherwise, set vertex to t = 0 with probability 1.0
function update_vertices_with_context!(cmssp::Toy2DCMSSPType, vertices::Vector{Toy2DOpenLoopVertex},
                                       switch::Toy2DModePair, context_set::Vector{Toy2DContextType}, params::Toy2DParameters)

    # Note that this is for a specific mode, so only consider those with the same mode switch
    # Iterate over context set
    # If the mode switch tuple is the same as in the open-loop vertex
    # Then check that the pre_bridge_state is in the range of the first
    # The post bridge state is just the same as from where the mode switch is attempted
    @assert (vertices[1].pre_bridge_state.mode, vertices[1].state.mode) == switch "Switch argument does not match OpenLoop Vertices"

    # IMP - Only look at future context, so start with index 2
    context_hor = length(context_set)

    # TODO : Search more efficiently by starting from current v.tp???
    for v in vertices

        tp_time = 0

        for (hor,context) in enumerate(context_set[2:end])

            for switch_points in context
                if switch_points[1] != switch
                    continue
                end
                pt_dist = norm(get_relative_state(v.pre_bridge_state, switch_points[2]))
                if pt_dist < params.epsilon
                    tp_time = hor
                    break
                end
            end

            if tp_time > 0
                break
            end
        end

        # Will either be [hor,1.0] or [0,1.0]
        v.tp = TPDistribution([tp_time],[1.0])
    end
end


# Just generate points around centre within +- box
function generate_goal_sample_set(cmssp::Toy2DCMSSPType, popped_cont::Toy2DContState,
                                  num_samples::Int64, rng::RNG, params::Toy2DParameters) where {RNG <: AbstractRNG}

    goal_samples = Vector{Toy2DStateType}(undef,0)

    dx = Uniform(TOY2D_GOAL_CENTRE.x - params.epsilon/2.0, TOY2D_GOAL_CENTRE.x + params.epsilon/2.0)
    dy = Uniform(TOY2D_GOAL_CENTRE.y - params.epsilon/2.0, TOY2D_GOAL_CENTRE.y + params.epsilon/2.0)

    for i = 1:num_samples
        push!(goal_samples, Toy2DStateType(TOY2D_GOAL_MODE, Toy2DContState(rand(rng,dx), rand(rng,dy))))
    end

    return goal_samples
end


function generate_next_valid_modes(cmssp::Toy2DCMSSPType, mode::Int64, context_set::Vector{Toy2DContextType})

    action_nextmodes = Vector{Tuple{Int64,Int64}}(undef,0)
    next_modes = Set{Int64}()

    for context in context_set[2:end]

        for switch_points in context
            if switch_points[1][1] == mode
                next_mode = switch_points[1][2]

                if next_mode in next_modes == false
                    mode_idx = mode_index(cmssp, mode)
                    next_mode_idx = mode_index(cmssp, next_mode)
                    
                    # Need to find action to take to next mode
                    for (ac_idx,action) in cmssp.mode_actions
                        if cmssp.modeswitch_mdp.T[mode_idx,ac_idx,next_mode_idx] > 0.0
                            push!(action_nextmodes,(ac_idx,next_mode_idx))
                            push!(next_modes, next_mode)
                            break
                        end
                    end
                end
            end
        end
    end
    return action_nextmodes
end

function generate_bridge_sample_set(cmssp::Toy2DCMSSPType, cont_state::Toy2DContState, 
                                    mode_pair::Toy2DModePair, num_samples::Int64,
                                    rng::RNG, context_set::Vector{Toy2DContextType}, params::Toy2DParameters) where {RNG <: AbstractRNG}

    bridge_samples = Vector{BridgeSample{Toy2DContState}}(undef,0)

    avg_samples_per_context = convert(Int64,round(num_samples/length(context_set)))

    for (hor,context) in enumerate(context_set[2:end])

        for switch_points in context

            if switch_points[1] == mode_pair
                # Generate bridge points around switch
                dx = Uniform(switch_points[2].x - params.epsilon/2.0, switch_points[2].x + params.epsilon/2.0)
                dy = Uniform(switch_points[2].y - params.epsilon/2.0, switch_points[2].y + params.epsilon/2.0)

                for _ = 1:avg_samples_per_context
                    bpx = rand(rng,dx)
                    bpy = rand(rng,dy)
                    state = Toy2DContState(bpx,bpy)
                    push!(bridge_samples, BridgeSample{Toy2DContState}(state, state, TPDistribution([hor],[1.0])))
                end
            end
        end
    end
    return bridge_samples
end



function simulate(cmssp::Toy2DCMSSPType, state::Toy2DStateType, a::Toy2DActionType, t::Int64, rng::RNG,
                  context_dict::Dict{Int64, Vector{Toy2DContextType}}, params::Toy2DParameters) where {RNG <: AbstractRNG}

    curr_contextset = context_dict[t]
    curr_context = curr_contextset[1]
    new_context = context_dict[t+1]

    curr_mode = state.mode
    curr_cont = state.continuous

    if typeof(a.action) <: Int64
        # action is mode switch - attempt
        mode_idx = mode_index(cmssp, curr_mode)
        mode_ac_idx = mode_actionindex(cmssp, a.action)
        next_mode_idx = findfirst(x -> x > 0.0, cmssp.modeswitch_mdp[mode_idx, mode_ac_idx,:])
        next_mode = cmssp.modes[next_mode_idx]

        # Action cost incurred no matter what
        reward = cmssp.R[mode_idx, mode_ac_idx]

        # Check if it can be done based on context
        for switch_points in curr_context
            if switch_points[1] == (curr_mode,next_mode)
                if norm(get_relative_state(curr_cont,switch_points[2])) < params.epsilon
                    @info "Mode switch possible!"
                    next_state = Toy2DStateType(next_mode, curr_cont)
                    return (next_state, new_context, reward, false)
                else
                    @info "Mode switch failed!"
                    next_state = curr_state
                    return (next_state, new_context, reward, true)
                end
            end
        end

        @info "Attempted mode switch unavailable from context!"
        next_state = curr_state
        return (new_state, new_context, reward, true)
    else
        # Control action within mode
        new_cont_state, reward = next_state_reward(Toy2DModalMDPType(curr_mode), curr_cont, a.action, rng)
        next_state = Toy2DStateType(curr_mode, new_cont_state)
        return (next_state, new_context, reward, false)
    end
end