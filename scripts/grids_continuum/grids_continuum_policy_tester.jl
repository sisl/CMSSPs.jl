using GridInterpolations
using LocalFunctionApproximation
using StaticArrays
using POMDPs
using JLD2, FileIO
using Random
using ContinuumWorld
using CMSSPs

trials = 5
rng = MersenneTwister(2973)

params_fn = "grids-continuum-params1.toml"
params = continuum_parse_params(params_fn)

inhor_fn = "grids-continuum-params1-inhor.jld2"
outhor_fn = "grids-continuum-params1-outhor.jld2"
modal_horizon_policy = load_modal_horizon_policy_localapprox(inhor_fn, outhor_fn)

# Create CWorld actions
grid_cworld_actions = [Vec2(params.move_amt, 0.0), Vec2(-params.move_amt, 0.0),
                       Vec2(0.0, params.move_amt), Vec2(0.0, -params.move_amt), Vec2(0.0, 0.0)]

# Use one cworld for all of them
cworld = CWorld(xlim = (0.0, 1.0), ylim = (0.0, 1.0),
                reward_regions = [], rewards = [], terminal = [],
                stdev = params.move_std, actions = grid_cworld_actions, discount = 1.0)
cworlds = Dict(1=>cworld, 2=>cworld, 3=>cworld, 4=>cworld)
params.cworlds = cworlds

# Only need to find policy for one mode, since shared
modal_mdp = GridsContinuumMDPType(1, params, grid_cworld_actions, params.beta, params.horizon_limit)

for i = 1:trials

    start_horizon = convert(Int64, rand(rng, params.horizon_limit/2:params.horizon_limit))
    target_state = sample_continuum_state(rng)
    curr_state = Vec2(0.0, 0.0)
    tp_dist = TPDistribution([start_horizon],[1.0])

    @show target_state, start_horizon

    for t = 0:1:start_horizon-1
        a = get_best_intramodal_action(modal_horizon_policy, t, tp_dist, curr_state, target_state)
        if a == nothing
            @warn "ABORT!"
            break
        end

        (new_state, reward) = generate_sr(modal_mdp, curr_state, a.action, rng)
        curr_state = new_state
    end

    if isterminal(modal_mdp, curr_state - target_state)
        @info "Success!"
    else
        @info "Failure!"
    end
end