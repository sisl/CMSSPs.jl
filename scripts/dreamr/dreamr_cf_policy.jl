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
scale_file = ARGS[1]
simtime_file = ARGS[2]
cost_file = ARGS[3]
policy_name = ARGS[4]
poly_or_exp = ARGS[5]
beta = parse(Float64, ARGS[6])

# scale_file = "./paramsets/scale-1.toml"
# simtime_file = "./paramsets/simtime-1.toml"
# cost_file = "./paramsets/cost-1.toml"
# policy_name = "dreamr-cf-params1-betapt75"
# poly_or_exp = "poly"

rng = MersenneTwister(10)

params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

cf_mdp = get_cf_mdp(MultiRotorUAVState, MultiRotorUAVAction, params, beta)

if poly_or_exp == "poly"

    xy_spacing = polyspace_symmetric(params.scale_params.XY_LIM, params.scale_params.XY_AXISVALS)
    xydot_spacing = polyspace_symmetric(params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_AXISVALS)

elseif poly_or_exp == "exp"
    
    xy_spacing = log2space_symmetric(params.scale_params.XY_LIM, params.scale_params.XY_AXISVALS)
    xydot_spacing = log2space_symmetric(params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_AXISVALS)

end

state_grid = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing)
@show length(vertices(state_grid))
lfa = LocalGIFunctionApproximator(state_grid)

compute_terminalcost_localapprox!(cf_mdp, lfa)

cf_policy = finite_horizon_VI_localapprox!(cf_mdp, lfa, true, params.scale_params.MC_GENERATIVE_NUMSAMPLES, rng)

compute_min_value_per_horizon_localapprox!(cf_policy)

inhor_fn = string(policy_name, "-inhor.jld2")
outhor_fn = string(policy_name, "-outhor.jld2")
save_modal_horizon_policy_localapprox(cf_policy, inhor_fn,outhor_fn, cf_mdp)