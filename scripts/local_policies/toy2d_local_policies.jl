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
using CMSSPs

# Read parameters from file
params_fn = ARGS[1]
inhor_fn = ARGS[2]
outhor_fn = ARGS[3]

rng = MersenneTwister(10)

# Create parameters and then CMSSP
params = toy2d_parse_params(params_fn)
cmssp = create_toy2d_cmssp(params)

# Set up the modal mdp
# Need to do it for just one mode
modal_mdp = Toy2DModalMDPType(1, params, cmssp.control_actions, 0.8, params.horizon_limit)

# Create grid interpolator
xy_spacing = polyspace_symmetric(1.0, params.axis_vals, 3)
xy_grid = RectangleGrid(xy_spacing, xy_spacing)
lfa = LocalGIFunctionApproximator(xy_grid)


# Compute and update the terminal cost
compute_terminalcost_localapprox!(modal_mdp, cmssp, 1, lfa)

# Compute policy with localApproxVI
modal_policy = finite_horizon_VI_localapprox!(modal_mdp, lfa, true,
                                              params.num_generative_samples, rng)

# Compute min_cost_per_horizon
compute_min_value_per_horizon_localapprox!(modal_policy)

# Save horizon policy to file
save_modal_horizon_policy_localapprox(modal_policy, inhor_fn,outhor_fn, modal_mdp)