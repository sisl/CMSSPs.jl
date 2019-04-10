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
    horizon_limit::Int64
    axis_vals::Int64
    valid_vertices_threshold::Float64
    time_coefficient::Float64
end

# Const types
const Toy2DStateType = CMSSPState{Int64,Toy2DContState}
const Toy2DActionType = CMSSPAction{Int64,Toy2DContAction}
const Toy2DCMSSPType = CMSSP{Int64,Toy2DContState,Int64,Toy2DContAction,Toy2DParameters}
const Toy2DModalMDPType = ModalMDP{Int64,Toy2DContState,Toy2DContAction,Toy2DParameters}
const Toy2DOpenLoopVertex = OpenLoopVertex{Int64,Toy2DContState,Int64}
# Context is current collection of all valid transition points at each timestep
const Toy2DModePair = Tuple{Int64, Int64}
const Toy2DContextType = Vector{Tuple{Toy2DModePair,Toy2DContState}}

# Const values
const TOY2D_MODES = [1,2,3,4,5]
const TOY2D_GOAL_MODE = 5
const TOY2D_QUADRANTS = [(0.0,pi/2), (pi/2,pi), (pi,3*pi/2), (3*pi/2,2*pi)]

mutable struct Toy2DContextSet
    curr_time::Int64
    switch_bias_dict::Dict{Toy2DModePair,Int64}
    curr_context_set::Vector{Toy2DContextType}
end

function Toy2DContextSet()
    return Toy2DContextSet(0,Dict{Toy2DModePair,Int64}(),Vector{Toy2DContextType}(undef,0))
end

const Toy2DSolverType = HHPCSolver{Int64,Toy2DContState,Int64,Toy2DContAction,Toy2DContextSet,Toy2DParameters}

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
    R[2,1] = -10.0
    R[3,1] = -2.0
    R[3,2] = -8.0
    R[4,1] = -1.0

    # Undiscounted
    return TabularMDP(T, R, 1.0)
end


# Create Toy2DCMSSP based on params
function create_toy2d_cmssp(params::Toy2DParameters)

    actions = get_toy2d_actions(params)
    modes = TOY2D_MODES
    switch_mdp = get_toy2d_switch_mdp()

    return Toy2DCMSSPType(actions, modes, switch_mdp, params)
end


function POMDPs.isterminal(mdp::Toy2DModalMDPType, relative_state::Toy2DContState)
    return norm(relative_state) < mdp.params.epsilon
end

function POMDPs.isterminal(cmssp::Toy2DCMSSPType, state::Toy2DStateType)
    return state.mode == TOY2D_GOAL_MODE
end

function rel_state(source::Toy2DContState, target::Toy2DContState)
    return Toy2DContState(source.x - target.x, source.y - target.y)
end

function HHPC.get_relative_state(mdp::Toy2DModalMDPType, source::Toy2DContState, target::Toy2DContState)
    return rel_state(source, target) 
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


function sample_toy2d(rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    d = Uniform(0.0,1.0)
    return Toy2DContState(rand(rng,d), rand(rng,d))
end

function generate_start_state(cmssp::Toy2DCMSSPType, rng::RNG) where {RNG <: AbstractRNG}
    return Toy2DStateType(1, sample_toy2d(rng))
end


# Just used by generate_sr - euclidean distance + squared velocity cost
function POMDPs.reward(mdp::Toy2DModalMDPType, state::Toy2DContState,
                       cont_action::Toy2DContAction, statep::Toy2DContState)
    delta_state = Toy2DContState(statep.x - state.x, statep.y - state.y)
    reward = -1.0*(norm(delta_state)) - mdp.params.time_coefficient
    return reward  
end


function HHPC.expected_reward(mdp::Toy2DModalMDPType, state::Toy2DContState,
                         cont_action::Toy2DContAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    params = mdp.params
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


function next_state_reward_sim(mdp::Toy2DModalMDPType, state::Toy2DContState, 
                           cont_action::Toy2DContAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    noise_dist = MvNormal([0.0001 + cont_action.xdot/100.0, 0.0001 + cont_action.ydot/100.0])
    xy_noise = rand(rng,noise_dist)
    new_x = clamp(state.x + cont_action.xdot + xy_noise[1], 0.0, 1.0)
    new_y = clamp(state.y + cont_action.ydot + xy_noise[2], 0.0, 1.0)

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
function HHPC.update_vertices_with_context!(cmssp::Toy2DCMSSPType, vertices::Vector{Toy2DOpenLoopVertex},
                                       switch::Toy2DModePair, toy2d_context_set::Toy2DContextSet)

    # Note that this is for a specific mode, so only consider those with the same mode switch
    # Iterate over context set
    # If the mode switch tuple is the same as in the open-loop vertex
    # Then check that the pre_bridge_state is in the range of the first
    # The post bridge state is just the same as from where the mode switch is attempted
    # @assert (vertices[1].pre_bridge_state.mode, vertices[1].state.mode) == switch "Switch argument does not match OpenLoop Vertices"

    # # IMP - Only look at future context, so start with index 2
    # context_hor = length(toy2d_context_set)

    # num_verts = length(vertices)
    # num_valid_updates = 0

    # # TODO : Search more efficiently by starting from current v.tp???
    # for v in vertices

    #     tp_time = Vector{Int64}(undef,0)
    #     tp_prob = Vector{Float64}(undef,0)

    #     for (hor,context) in enumerate(toy2d_context_set[2:end])

    #         for switch_points in context
    #             if switch_points[1] != switch
    #                 continue
    #             end
    #             pt_dist = norm(rel_state(v.pre_bridge_state, switch_points[2]))
    #             if pt_dist < params.epsilon
    #                 push!(tp_time,hor)
    #                 push!(tp_prob,1.0/pt_dist)
    #             end
    #         end
    #     end

    #     # Will either be [0,1.0] or a weighted probability dist of inverse distance
    #     if isempty(tp_time)
    #         v.tp = TPDistribution([0],[1.0])
    #     else
    #         tp_prob = tp_prob/sum(tp_prob)
    #         v.tp = TPDistribution(tp_time, tp_prob)
    #         num_valid_updates += 1
    #     end
    # end

    # # If too many vertices invalidated, then return false and resample
    # return (num_valid_updates > params.valid_vertices_threshold*num_verts)

    # OR : JUST RETURN FALSE EACH TIME AND RE-SAMPLE
    return false
end


# Just generate points around centre within +- box
function HHPC.generate_goal_sample_set(cmssp::Toy2DCMSSPType, popped_cont::Toy2DContState,
                                  num_samples::Int64, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # params = cmssp.params
    # goal_samples = Vector{Toy2DStateType}(undef,0)

    # dx = Uniform(TOY2D_GOAL_CENTRE.x - params.epsilon/2.0, TOY2D_GOAL_CENTRE.x + params.epsilon/2.0)
    # dy = Uniform(TOY2D_GOAL_CENTRE.y - params.epsilon/2.0, TOY2D_GOAL_CENTRE.y + params.epsilon/2.0)

    # for i = 1:num_samples
    #     push!(goal_samples, Toy2DStateType(TOY2D_GOAL_MODE, Toy2DContState(rand(rng,dx), rand(rng,dy))))
    # end
    goal_samples = [Toy2DStateType(TOY2D_GOAL_MODE, popped_cont)]

    return goal_samples
end


function HHPC.generate_next_valid_modes(cmssp::Toy2DCMSSPType, mode::Int64, toy2d_context_set::Toy2DContextSet)

    action_nextmodes = Vector{Tuple{Int64,Int64}}(undef,0)
    next_modes = Set{Int64}()
    curr_context_set = toy2d_context_set.curr_context_set

    for context in curr_context_set[2:end]

        for switch_points in context
            if switch_points[1][1] == mode
                next_mode = switch_points[1][2]

                if next_mode in next_modes
                    continue
                end

                mode_idx = mode_index(cmssp, mode)
                next_mode_idx = mode_index(cmssp, next_mode)

                # Need to find action to take to next mode
                for (ac_idx,action) in enumerate(cmssp.mode_actions)
                    if cmssp.modeswitch_mdp.T[mode_idx,ac_idx,next_mode_idx] > 0.0
                        push!(action_nextmodes,(ac_idx,next_mode_idx))
                        push!(next_modes, next_mode)
                        break
                    end
                end
            end
        end
    end
    return action_nextmodes
end

function HHPC.generate_bridge_sample_set(cmssp::Toy2DCMSSPType, cont_state::Toy2DContState, 
                                    mode_pair::Toy2DModePair, num_samples::Int64,
                                    toy2d_context_set::Toy2DContextSet, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    params = cmssp.params
    bridge_samples = Vector{BridgeSample{Toy2DContState}}(undef,0)
    curr_context_set = toy2d_context_set.curr_context_set
    curr_time = toy2d_context_set.curr_time

    avg_samples_per_context = convert(Int64,ceil(num_samples/length(curr_context_set)))

    for (hor,context) in enumerate(curr_context_set[2:end])

        for switch_points in context

            if switch_points[1] == mode_pair
                # Generate bridge points around switch
                dx = Uniform(switch_points[2].x - params.epsilon/2.0, switch_points[2].x + params.epsilon/2.0)
                dy = Uniform(switch_points[2].y - params.epsilon/2.0, switch_points[2].y + params.epsilon/2.0)

                for _ = 1:avg_samples_per_context
                    bpx = clamp(rand(rng,dx),0.0,1.0)
                    bpy = clamp(rand(rng,dy),0.0,1.0)
                    state = Toy2DContState(bpx,bpy)
                    push!(bridge_samples, BridgeSample(state, state, TPDistribution([curr_time+hor],[1.0])))
                end
            end
        end
    end
    return bridge_samples
end



function HHPC.simulate_cmssp(cmssp::Toy2DCMSSPType, state::Toy2DStateType, a::Toy2DActionType, t::Int64,
                        toy2d_context_set::Toy2DContextSet, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    curr_context_set = toy2d_context_set.curr_context_set
    curr_context = curr_context_set[1]

    curr_mode = state.mode
    curr_cont_state = state.continuous
    params = cmssp.params

    if typeof(a.action) <: Int64
        # action is mode switch - attempt
        mode_idx = mode_index(cmssp, curr_mode)
        mode_ac_idx = mode_actionindex(cmssp, a.action)
        next_mode_idx = findfirst(x -> x > 0.0, cmssp.modeswitch_mdp.T[mode_idx, mode_ac_idx,:])
        next_mode = cmssp.modes[next_mode_idx]

        # Action cost incurred no matter what
        reward = cmssp.modeswitch_mdp.R[mode_idx, mode_ac_idx]

        # Check if it can be done based on context
        for switch_points in curr_context
            if switch_points[1] == (curr_mode,next_mode)
                if norm(rel_state(curr_cont_state,switch_points[2])) < 2.5*params.epsilon
                    @info "Successful mode switch to mode ",next_mode
                    next_state = Toy2DStateType(next_mode, curr_cont_state)
                    return (next_state, reward, false)
                else
                    @show switch_points[2]
                    @info "Mode switch failed!"
                    next_state = curr_state
                    return (next_state,  reward, true)
                end
            end
        end

        @info "Attempted mode switch unavailable from context!"
        next_state = curr_state
        return (new_state, reward, true)
    else
        # Control action within mode
        new_cont_state, reward = next_state_reward_sim(Toy2DModalMDPType(curr_mode, cmssp.params), curr_cont_state, a.action, rng)
        next_state = Toy2DStateType(curr_mode, new_cont_state)
        return (next_state, reward, false)
    end
end

function toy2d_parse_params(filename::AbstractString)

    params_key = TOML.parsefile(filename)

    return Toy2DParameters(params_key["SPEED_LIMIT"],
                           params_key["SPEED_VALS"],
                           params_key["EPSILON"],
                           params_key["NUM_GEN_SAMPLES"],
                           params_key["NUM_BRIDGE_SAMPLES"],
                           params_key["HORIZON_LIMIT"],
                           params_key["AXIS_VALS"],
                           params_key["VALID_VERTICES_THRESHOLD"],
                           params_key["TIME_COEFFICIENT"])
end


# Context functions
function generate_start_context_set(cmssp::Toy2DCMSSPType, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    params = cmssp.params
    toy2d_context_set = Toy2DContextSet()
    init_context = Toy2DContextType(undef,0)

    # Maintains the context bias for each mode switch
    switch_bias_dict = toy2d_context_set.switch_bias_dict

    for m1 in cmssp.modes
        for m2 in cmssp.modes
            for a in cmssp.mode_actions
                if cmssp.modeswitch_mdp.T[m1,a,m2] > 0.0
                    switch_point = sample_toy2d(rng)
                    push!(init_context,(Toy2DModePair((m1,m2)), switch_point))
                    switch_bias_dict[Toy2DModePair((m1,m2))] = rand(rng, 1:4)
                end
            end
        end
    end

    push!(toy2d_context_set.curr_context_set, init_context)

    r = params.epsilon/2.0

    # Now have init_context, need to propagate forward and keep pushing
    for idx = 1:params.horizon_limit
        temp_context = toy2d_context_set.curr_context_set[idx]

        next_context = Toy2DContextType(undef,0)

        # Iterate over each and propagate with epsilon/2.0 in some radial direction
        for (switch,point) in temp_context
            theta_dist = Uniform(TOY2D_QUADRANTS[switch_bias_dict[switch]][1], TOY2D_QUADRANTS[switch_bias_dict[switch]][2])
            theta = rand(rng,theta_dist)
            new_x = clamp(point.x + r*cos(theta),0.0,1.0)
            new_y = clamp(point.y + r*sin(theta),0.0,1.0)
            new_point = Toy2DContState(new_x,new_y)

            push!(next_context,(switch,new_point))

            # If either outside range, switch to opposite TOY2D_QUADRANTS
            if new_y*new_x == 0.0 || new_x == 1.0 || new_y == 1.0
                switch_bias_dict[switch] = (switch_bias_dict[switch] + 2) % 4 + 1
            end
        end

        # Now append to context
        push!(toy2d_context_set.curr_context_set, next_context)
    end

    @assert length(toy2d_context_set.curr_context_set) == params.horizon_limit+1
    return toy2d_context_set
end

function HHPC.update_context_set!(cmssp::Toy2DCMSSPType, toy2d_context_set::Toy2DContextSet,
                            rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    switch_bias_dict = toy2d_context_set.switch_bias_dict
    curr_context_set = toy2d_context_set.curr_context_set
    params = cmssp.params

    # For each switch, with 50% probability, have it either stay in place
    # Or go to the next point. In the first case, just maintain the same vector
    # Otherwise, perturb the last point and add, and left-shift
    next_context = Toy2DContextType(undef,0)
    last_context = Toy2DContextType(undef,0)

    curr_context = curr_context_set[1]
    curr_next_context = curr_context_set[2]
    curr_last_context = curr_context_set[end]

    theta_dist = Uniform(0,2*pi)
    r = params.epsilon/2.0

    for (i,switch_point) in enumerate(curr_context)
        toss = rand(rng,Uniform(0.0,1.0))
        switch = switch_point[1]
        point = switch_point[2]

        if toss < 0.35
            # Stay and most_future is the same
            push!(next_context,(switch,point))
            push!(last_context,(switch,curr_last_context[i][2]))
        else
            # Go to next one
            push!(next_context,(switch,curr_next_context[i][2]))
            
            # Generate new point for last context
            curr_last_context_pt = curr_last_context[i][2]
            theta_dist = Uniform(TOY2D_QUADRANTS[switch_bias_dict[switch]][1], TOY2D_QUADRANTS[switch_bias_dict[switch]][2])
            theta = rand(rng,theta_dist)
            new_x = clamp(curr_last_context_pt.x + r*cos(theta),0.0,1.0)
            new_y = clamp(curr_last_context_pt.y + r*sin(theta),0.0,1.0)
            new_point = Toy2DContState(new_x,new_y)

            push!(last_context,(switch,new_point))

            # If either outside range, switch to opposite TOY2D_QUADRANTS
            if new_y*new_x == 0.0 || new_x == 1.0 || new_y == 1.0
                switch_bias_dict[switch] = (switch_bias_dict[switch] + 2) % 4 + 1
            end
        end
    end

    # Now re-assign
    toy2d_context_set.curr_context_set[1] = next_context
    toy2d_context_set.curr_context_set[end] = last_context

    # Finally, update time
    toy2d_context_set.curr_time = toy2d_context_set.curr_time + 1
end

function HHPC.display_context_future(toy2d_context_set::Toy2DContextSet,future_time::Int64)

    # IMP - Because if 0, then first
    hor = future_time - toy2d_context_set.curr_time + 1
    println(toy2d_context_set.curr_context_set[hor])
end

## TODO : NEED TO IMPLEMENT generate_sr for MCTS