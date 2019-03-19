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

# Constants for parameters - read from file later??
const SPEED_LIMIT = 0.075
const SPEED_RESOLUTION = 0.015
const EPSILON = 0.01
const N_GEN_SAMPLES = 20
const NUM_BRIDGE_SAMPLES = 20

const HORIZON_LIMIT = 20
const AXIS_VALS = 41

rng = MersenneTwister(10)

# Create parameters and then CMSSP
params = Toy2DParameters(SPEED_LIMIT, SPEED_RESOLUTION, EPSILON, N_GEN_SAMPLES, NUM_BRIDGE_SAMPLES)
cmssp = create_toy2d_cmssp(params)

# Set up the modal mdp
# Need to do it for just one mode
modal_mdp = Toy2DModalMDPType(1, cmssp.control_actions, 0.8, HORIZON_LIMIT)

# Create grid interpolator
xy_spacing = polyspace_symmetric(1.0, AXIS_VALS, 3)
xy_grid = RectangleGrid(xy_spacing, xy_spacing)
lfa = LocalGIFunctionApproximator(xy_grid)

# Bind expected reward to parameters
CMSSPs.HHPC.expected_reward(mdp,state,action) = CMSSPs.CMSSPDomains.expected_reward(mdp,state,action,rng,params)
POMDPs.isterminal(mdp::Toy2DModalMDPType,state::Toy2DContState) = CMSSPs.CMSSPDomains.isterminal(mdp,state,params)

# Compute and update the terminal cost
compute_terminalcost_localapprox!(modal_mdp, cmssp, 1, lfa)

# Compute policy with localApproxVI
modal_policy = finite_horizon_VI_localapprox!(modal_mdp, lfa, true,
                                              params.num_generative_samples, rng)

# Compute min_cost_per_horizon
compute_min_value_per_horizon_localapprox!(modal_policy)

# Save horizon policy to file
save_modal_horizon_policy_localapprox(modal_policy, "toy2d-grid1-inhor.jld2", 
                                      "toy2d-grid1-outhor.jld2", modal_mdp)