using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPModelTools
using StaticArrays
using JLD2, FileIO
using Random
using Logging
using Distributions
using ContinuumWorld
using CMSSPs

# Read parameters from file
# params_fn = ARGS[1]
# inhor_fn = ARGS[2]
# outhor_fn = ARGS[3]

params_fn = "grids-continuum-params1.toml"
inhor_fn = "grids-continuum-params1-inhor.jld2"
outhor_fn = "grids-continuum-params1-outhor.jld2"

rng = MersenneTwister(10)

# Create CWorld actions
grid_cworld_actions = [Vec2(0.1, 0.0), Vec2(-0.1, 0.0), Vec2(0.0, 0.1), Vec2(0.0, -0.1), Vec2(0.0, 0.0)]

# Use one cworld for all of them
cworld = CWorld(xlim = (0.0, 1.0), ylim = (0.0, 1.0),
                reward_regions = [], rewards = [], terminal = [],
                stdev = 0.05, actions = grid_cworld_actions, discount = 1.0)
cworlds = Dict(1=>cworld, 2=>cworld, 3=>cworld, 4=>cworld)
params = continuum_parse_params(params_fn, cworlds)

# Only need to find policy for one mode, since shared
modal_mdp = GridsContinuumMDPType(1, params, grid_cworld_actions, 0.75, params.horizon_limit)

# Create grid interpolator
xy_spacing = polyspace_symmetric(1.0, params.vals_along_axis, 3)
xy_grid = RectangleGrid(xy_spacing, xy_spacing)
lfa = LocalGIFunctionApproximator(xy_grid)

# Compute and update the terminal cost
compute_terminalcost_localapprox!(modal_mdp, lfa)

# Compute policy with localApproxVI
modal_policy = finite_horizon_VI_localapprox!(modal_mdp, lfa, true,
                                              params.num_generative_samples, rng)

# Compute min_cost_per_horizon
compute_min_value_per_horizon_localapprox!(modal_policy)

# Save horizon policy to file
save_modal_horizon_policy_localapprox(modal_policy, inhor_fn,outhor_fn, modal_mdp)