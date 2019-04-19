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


# scale_file = ARGS[1]
# simtime_file = ARGS[2]
# cost_file = ARGS[3]
# policy_name = ARGS[4]
# poly_or_exp = ARGS[5]

scale_file = "../../paramsets/dreamr/scale-1.toml"
simtime_file = "../../paramsets/dreamr/simtime-1.toml"
cost_file = "../../paramsets/dreamr/cost-1.toml"
policy_name = "dreamr-uf-smalltest"
poly_or_exp = "poly"


rng = MersenneTwister(10)

params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

uf_mdp = get_uf_mdp(MultiRotorUAVState, MultiRotorUAVAction, params)

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

uf_policy = infinite_horizon_VI_localapprox(uf_mdp, lfa, 20, true, params.scale_params.MC_GENERATIVE_NUMSAMPLES, rng)

policy_fn = string(policy_name,".jld2")
save_localapproxvi_policy_to_jld2(policy_fn, uf_policy, uf_mdp)