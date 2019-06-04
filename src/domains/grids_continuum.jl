struct GridsContinuumParams
    epsilon::Float64
    num_generative_samples::Int64
    num_bridge_samples::Int64
    horizon_limit::Int64
    vals_along_axis::Int64 # Number of points along axis for interpolation
    cworlds::Dict{Int64,CWorld} # The exemplar continuum worlds
end


const GridsContinuumStateType = CMSSPState{Int64, Vec2}
const GridsContinuumActionType = CMSSPAction{Int64, Vec2}
const GridsContinuumContextType = Dict{Tuple{Int64, Int64}, Vec2}


mutable struct GridsContinuumContextSet
    curr_timestep::Int64
    curr_context::GridsContinuumContextType
    future_context::Vector{GridsContinuumContextType}
end

Base.zero(Nothing) = nothing


const GridsContinuumCMSSPType = CMSSP{Int64, Vec2, Int64, Vec2, GridsContinuumParams}
const GridsContinuumMDPType = ModalFinHorMDP{Int64, Vec2, Vec2, GridsContinuumParams}
const GridsContinuumBridgeSample = BridgeSample{Vec2, Nothing}
const GridsContinuumOLV = OpenLoopVertex{Int64, Vec2, Int64, Nothing}
const GridsContinuumGraphTracker = GraphTracker{Int64, Vec2, Int64, Nothing, Nothing, RNG} where {RNG <: AbstractRNG}
const GridsContinuumSolverType = HHPCSolver{Int64, Vec2, Int64, Nothing, Nothing, RNG} where {RNG <: AbstractRNG}

# Const mode values
const CONTINUUM_MODES = [1, 2, 3, 4]
const CONTINUUM_GOAL_MODE = 4



# Discrete MDP - 4 states, 2 actions
# 1 -- 2 -- 4
# |         |
# 3 --------|
function get_grids_continuum_mdp()

    T = zeros(4, 2, 4)
    T[1, 1, 2] = 1.0
    T[1, 2, 3] = 1.0
    T[2, 1, 4] = 1.0
    T[3, 1, 4] = 1.0

    R = zeros(4, 2)
    R[1, 1] = 15
    R[1, 2] = 5
    R[2, 1] = 5
    R[3, 1] = 5

    return TabularMDP(T, R, 1.0)
end


function get_grids_continuum_actions(params::GridsContinuumParams)

    actions = Vector{GridsContinuumActionType}(undef, 0)

    # First enter the mode actions
    push!(actions, GridsContinuumActionType(1, 1))
    push!(actions, GridsContinuumActionType(2, 2))

    # Then enter the control actions
    # Use actions of CWorld 1
    cworld_actions = params.cworlds[1].actions

    for (i, cwa) in enumerate(cworld_actions)
        idx = i+2
        push!(actions, GridsContinuumActionType(cwa, idx))
    end

    return actions
end


function create_continuum_cmssp(params::GridsContinuumParams)

    actions = get_grids_continuum_actions(params)
    switch_mdp = get_grids_continuum_mdp()

    return GridsContinuumCMSSPType(actions, CONTINUUM_MODES, switch_mdp, params)

end

function POMDPs.isterminal(mdp::GridsContinuumMDPType, relative_state::Vec2)
    return norm(relative_state) < mdp.params.epsilon
end

function POMDPs.isterminal(cmssp::GridsContinuumCMSSPType, state::Vec2)
    return state.mode = CONTINUUM_GOAL_MODE
end

function HHPC.get_relative_state(mdp::GridsContinuumMDPType, source::GridsContinuumStateType, target::GridsContinuumStateType)
    return (source - target)
end

function sample_continuum_state(rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    d = Uniform(0.0, 1.0)
    return Vec2(rand(rng, d), rand(rng, d))
end

function generate_start_state(cmssp::GridsContinuumCMSSPType, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    return GridsContinuumStateType(1, sample_continuum_state(rng))
end

function POMDPs.reward(mdp::GridsContinuumMDPType, state::Vec2, action::Vec2, statep::Vec2)

    # Copy over reward of underlying Continuum World
    cworld = mdp.params.cworlds[mdp.mode]
    return POMDPs.reward(cworld, state, action, statep)

end

function HHPC.expected_reward(mdp::GridsContinuumMDPType, state::Vec2,
                              action::Vec2, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    params = mdp.params
    cworld = params.cworlds[mdp.mode]
    avg_reward = 0.0

    for i = 1:params.num_generative_samples
        statep = generate_s(cworld, state, action, rng)
        avg_reward += POMDPs.reward(cworld, state, action, statep)
    end

    avg_reward /= params.num_generative_samples
    return avg_reward
end


# Will never actually be called because any vertices in grid 4 satisfy the goal condition
# Just return the same vertex and the current goal idx
function HHPC.generate_goal_vertex_set!(cmssp::GridsContinuumCMSSPType, vertex::GridsContinuumOLV,
                                        graph_tracker::GridsContinuumGraphTracker, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    return [], [graph_tracker.curr_goal_idx]
end


# Just generate a new set of neighbours at each timestep
# So vertices_to_add has vertices and nbrs_to_add is empty
function HHPC.generate_bridge_vertex_set!(cmssp::GridsContinuumCMSSPType, vertex::GridsContinuumOLV,
                                          mode_pair::Tuple{Int64, Int64}, graph_tracker::GridsContinuumGraphTracker,
                                          action::Int64, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    context_set = cmssp.curr_context_set
    future_context = context_set.future_context # Vector of contexts
    curr_timestep = context_set.curr_timestep
    params = cmssp.params

    vertices_to_add = Vector{GridsContinuumOLV}(undef, 0)

    @assert !isempty(future_context)
    avg_samples = convert(Int64, ceil(cmssp.params.num_bridge_samples/length(future_context)))

    for (hor, context) in enumerate(future_context)

        # Iterate over (M1,M2) -> point
        for (mpair, switch_point) in context

            # Generate bridge point
            if mpair == mode_pair

                d1 = Uniform(switch_point[1] - params.epsilon/2.0, switch_point[1] + params.epsilon/2.0)
                d2 = Uniform(switch_point[2] - params.epsilon/2.0, switch_point[2] + params.epsilon/2.0)

                for i = 1:avg_samples
                    
                    bp1 = clamp(rand(rng, d1), 0.0, 1.0)
                    bp2 = clamp(rand(rng, d2), 0.0, 1.0)
                    cont_state = Vec2(bp1, bp2)

                    bridge_sample = GridsContinuumBridgeSample(cont_state, cont_state,
                                        TPDistribution([curr_timestep + hor], [1.0]), nothing)
                    new_bridge_vtx = GridsContinuumOLV(mode_pair[1], mode_pair[2], bridge_sample, action)

                    push!(vertices_to_add, new_bridge_vtx)
                end
            end
        end
    end

    # Return all new vertices and no existing neighbours
    return vertices_to_add, []
end


function HHPC.generate_next_valid_modes(cmssp::GridsContinuumCMSSPType, mode::Int64)

    context_set = cmssp.curr_context_set
    future_context = context_set.future_context

    next_actions_modes = Vector{Tuple{Int64, Int64}}(undef, 0)

    for (hor, context) in enumerate(future_context) # Vector of dictionaries

        for (mpair, switch_point) in context # Iterate over current dictionary

            if mpair[1] == mode # Switch from current mode

                next_mode = mpair[2]

                for action = 1:2
                    if cmssp.modeswitch_mdp.T[mode, action, next_mode] > 0.0
                        push!(next_modes_actions, (action, next_mode))
                    end
                end
            end
        end
    end

    return next_modes_actions
end

# Dummy function as vertices will always be regenerated
function HHPC.update_vertices_with_context!(cmssp::GridsContinuumCMSSPType, graph_tracker::GridsContinuumGraphTracker,
                                            curr_timestep::Int64)
    return
end


function HHPC.update_next_target!(cmssp::GridsContinuumCMSSPType, solver::GridsContinuumSolverType, curr_timestep::Int64)
    return
end


function HHPC.simulate_cmssp!(cmssp::GridsContinuumCMSSPType, state::GridsContinuumStateType, a::Union{Nothing, GridsContinuumActionType},
                              curr_timestep::Int64, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Default - will delete when timeout removed
    timeout = false

    curr_context_set = cmssp.curr_context_set
    curr_context = curr_context_set.curr_context

    curr_mode = state.mode
    curr_cont_state = state.continuous
    params = cmssp.params

    if typeof(a.action) <: Int64

        # Action is mode switch
        next_mode = findfirst(x -> x > 0.0, cmssp.modeswitch_mdp.T[curr_mode, a.action, :])
        reward = cmssp.modeswitch_mdp.R[curr_mode, a.action]

        # Check if it can be done based on current context
        for (mpair, switch_point) in curr_context

            if mpair == (curr_mode, next_mode)

                if norm(switch_point - curr_cont_state) < 2.0*params.epsilon
                    @info "Successful mode switch to grid ",next_mode
                    next_state = GridsContinuumStateType(next_mode, switch_point)
                    return (next_state, reward, false, timeout)
                else
                    @info "Mode switch to grid ",next_mode, "failed "
                    next_state = state
                    return (next_state, reward, true, timeout)
                end
            end
        end

        @warn "Attempted mode switch to ", next_mode," unavailable from context"
        next_state = curr_state
        return (next_state, reward, true, timeout)
    else
        # control action within mode
        cworld = params.cworlds[curr_mode]
        new_cont_state = generate_s(cworld, state, a.action, rng)
        reward = POMDPs.reward(cworld, state, a.action, new_cont_state)
        next_state = GridsContinuumStateType(curr_mode, new_cont_state)
        return (next_state, reward, false, timeout)
    end
end

# Still need to generate initial context set and update context set.