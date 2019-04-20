abstract type UAVDynamicsModel end
abstract type UAVState end
abstract type UAVAction end

"""
Represents state of simple 2D multirotor, with position and velocity in each direction.
"""
struct MultiRotorUAVState <: UAVState
    x::Float64
    y::Float64
    xdot::Float64
    ydot::Float64
end

function get_position(state::MultiRotorUAVState)
    return Point(state.x, state.y)
end

function get_speed(state::MultiRotorUAVState)
    return sqrt(state.xdot^2 + state.ydot^2)
end 

function rel_state(source::MultiRotorUAVState, target::MultiRotorUAVState)
    return MultiRotorUAVState(source.x - target.x, source.y - target.y, source.xdot, source.ydot)
end
"""
Represents control action for simple 2D multirotor, with acceleration in each direction.
"""
struct MultiRotorUAVAction <: UAVAction
    xddot::Float64
    yddot::Float64
end

"""
Simulates physical dynamics of UAV, mapping state and control to next state
"""
struct MultiRotorUAVDynamicsModel <: UAVDynamicsModel
    timestep::Float64
    noise::Distributions.ZeroMeanDiagNormal
    params::Parameters
end

"""
    MultiRotorUAVDynamicsModel(t::Float64, params::Parameters, sig::Float64)

Define a dynamics model where t is the timestep duration, params is the collection of system parameters,
and sig is the standard deviation along each axis (sig_xy) is vector of standard deviations for each corresponding axis
"""
function MultiRotorUAVDynamicsModel(params::Parameters)
    # Generate diagonal covariance matrix
    noise = Distributions.MvNormal([params.scale_params.ACC_NOISE_STD, params.scale_params.ACC_NOISE_STD])
    return MultiRotorUAVDynamicsModel(params.time_params.MDP_TIMESTEP, noise, params)
end

function MultiRotorUAVDynamicsModel(sig_xy::StaticVector{2,Float64}, params::Parameters)
    # Generate diagonal covariance matrix
    noise = Distributions.MvNormal([sig_xy[1], sig_xy[2]])
    return MultiRotorUAVDynamicsModel(t, noise, params)
end

"""
    get_uav_dynamics_actions(model::MultiRotorUAVDynamicsModel)

Compiles a vector of all multirotor acceleration actions within limits, based on the resolution parameters
"""
function get_uav_dynamics_actions(::Type{MultiRotorUAVAction}, params::Parameters)

    multirotor_actions = Vector{MultiRotorUAVAction}(undef, 0)
    acc_vals = range(-params.scale_params.ACCELERATION_LIM, stop=params.scale_params.ACCELERATION_LIM, length=params.scale_params.ACCELERATION_NUMVALS)

    for xddot in acc_vals
        for yddot in acc_vals
            push!(multirotor_actions, MultiRotorUAVAction(xddot, yddot))
        end
    end

    return multirotor_actions
end

"""
    generate_start_state(model::MultiRotorUAVDynamicsModel, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

Generate a MultiRotorUAVState with a random location inside the grid and at rest
"""
function generate_start_state(model::MultiRotorUAVDynamicsModel, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    x = rand(rng, Uniform(-model.params.scale_params.XY_LIM, model.params.scale_params.XY_LIM))
    y = rand(rng, Uniform(-model.params.scale_params.XY_LIM, model.params.scale_params.XY_LIM))
    
    xdot = 0.
    ydot = 0.

    return MultiRotorUAVState(x, y, xdot, ydot)
end


function generate_start_state(::Type{MultiRotorUAVState}, params::Parameters, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    x = rand(rng, Uniform(-params.scale_params.XY_LIM, params.scale_params.XY_LIM))
    y = rand(rng, Uniform(-params.scale_params.XY_LIM, params.scale_params.XY_LIM))
    
    xdot = 0.
    ydot = 0.

    return MultiRotorUAVState(x, y, xdot, ydot)
end

"""
    get_state_at_rest(model::MultiRotorUAVDynamicsModel, p::Point)

Given a Point instance, generate the corresponding MultiRotorUAVState at rest at that position
"""
function get_state_at_rest(::Type{MultiRotorUAVState}, p::Point)
    return MultiRotorUAVState(p.x, p.y, 0.0, 0.0)
end

"""
get_relative_state_to_goal(model::MultiRotorUAVDynamicsModel, goal_pos::Point, state::MultiRotorUAVState)

Given some goal position, get the MultiRotorUAVState with relative position and own velocity
"""
function get_relative_state_to_goal(::Type{MultiRotorUAVState}, goal_pos::Point, state::MultiRotorUAVState)
    return MultiRotorUAVState(state.x - goal_pos.x, state.y - goal_pos.y, state.xdot, state.ydot)
end

"""
    apply_controls(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

Propagate the MultiRotorUAVState through the dynamics model (without noise)
"""
function apply_controls(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

    # Update position and velocity exactly
    xdot = clamp(state.xdot + xddot*model.timestep, -model.params.scale_params.XYDOT_LIM, model.params.scale_params.XYDOT_LIM)
    ydot = clamp(state.ydot + yddot*model.timestep, -model.params.scale_params.XYDOT_LIM, model.params.scale_params.XYDOT_LIM)

    # Get true effective accelerations
    true_xddot = (xdot - state.xdot)/model.timestep
    true_yddot = (ydot - state.ydot)/model.timestep

    x = state.x + state.xdot*model.timestep + 0.5*true_xddot*(model.timestep^2)
    x = clamp(x, -model.params.scale_params.XY_LIM, model.params.scale_params.XY_LIM)

    y = state.y + state.ydot*model.timestep + 0.5*true_yddot*(model.timestep^2)
    y = clamp(y, -model.params.scale_params.XY_LIM, model.params.scale_params.XY_LIM)

    return MultiRotorUAVState(x, y, xdot, ydot)
end

function apply_controls(params::Parameters, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

    timestep = params.time_params.MDP_TIMESTEP

    # Update position and velocity exactly
    xdot = clamp(state.xdot + xddot*timestep, -params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_LIM)
    ydot = clamp(state.ydot + yddot*timestep, -params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_LIM)

    # Get true effective accelerations
    true_xddot = (xdot - state.xdot)/timestep
    true_yddot = (ydot - state.ydot)/timestep

    x = state.x + state.xdot*timestep + 0.5*true_xddot*(timestep^2)
    x = clamp(x, -params.scale_params.XY_LIM, params.scale_params.XY_LIM)

    y = state.y + state.ydot*timestep + 0.5*true_yddot*(timestep^2)
    y = clamp(y, -params.scale_params.XY_LIM, params.scale_params.XY_LIM)

    return MultiRotorUAVState(x, y, xdot, ydot)
end


"""
    next_state(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

Generate the next state by propagating the action noisily with the dynamics
"""
function next_state(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Sample acceleration noise vector    
    noiseval = rand(rng,model.noise)

    # Augment actual acceleration with noise
    xddot = action.xddot + noiseval[1]
    yddot = action.yddot + noiseval[2]

    return apply_controls(model, state, xddot, yddot)
end

function next_state(params::Parameters, state::MultiRotorUAVState, action::MultiRotorUAVAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Sample acceleration noise vector
    noise = Distributions.MvNormal([params.scale_params.ACC_NOISE_STD, params.scale_params.ACC_NOISE_STD])   
    noiseval = rand(rng, noise)

    # Augment actual acceleration with noise
    xddot = action.xddot + noiseval[1]
    yddot = action.yddot + noiseval[2]

    return apply_controls(params, state, xddot, yddot)
end


"""
    function dynamics_cost(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, next_state::MultiRotorUAVState)

Compute the energy/dynamics cost of going from one state to another, based on the cost parameters in the model.
N.B - In the most general case, cost should depend on the action too.
"""
function dynamics_cost(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, next_state::MultiRotorUAVState)
    old_point = Point(state.x, state.y)
    new_point = Point(next_state.x, next_state.y)
    dyn_dist = point_dist(old_point, new_point)

    cost = 0.0

    # Compute cost due to flying or due to hovering
    if dyn_dist < model.timestep*model.params.scale_params.EPSILON &&
        sqrt(next_state.xdot^2 + next_state.ydot^2) < model.timestep*model.params.scale_params.EPSILON
        cost += model.params.cost_params.HOVER_COEFFICIENT*model.timestep
    else
        cost += model.params.cost_params.FLIGHT_COEFFICIENT*dyn_dist
    end

    return cost
end

function dynamics_cost(params::Parameters, state::MultiRotorUAVState, next_state::MultiRotorUAVState)
    old_point = Point(state.x, state.y)
    new_point = Point(next_state.x, next_state.y)
    dyn_dist = point_dist(old_point, new_point)

    cost = 0.0

    # Compute cost due to flying or due to hovering
    if dyn_dist < params.time_params.MDP_TIMESTEP*params.scale_params.EPSILON &&
        sqrt(next_state.xdot^2 + next_state.ydot^2) < params.time_params.MDP_TIMESTEP*params.scale_params.EPSILON
        cost += params.cost_params.HOVER_COEFFICIENT*params.time_params.MDP_TIMESTEP
    else
        cost += params.cost_params.FLIGHT_COEFFICIENT*dyn_dist
    end

    return cost
end