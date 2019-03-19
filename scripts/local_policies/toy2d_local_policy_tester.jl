using POMDPs
using POMDPModels
using POMDPModelTools
using StaticArrays
using JLD2, FileIO
using Random
using Logging
using Distributions
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using CMSSPs

const SPEED_LIMIT = 0.075
const SPEED_RESOLUTION = 0.015
const EPSILON = 0.01
const N_GEN_SAMPLES = 10
const NUM_BRIDGE_SAMPLES = 20
const HORIZON_LIMIT = 20


inhor_file = ARGS[1]
outhor_file = ARGS[2]
trials = 5

rng = MersenneTwister(12)

modal_horizon_policy = load_modal_horizon_policy_localapprox(inhor_file, outhor_file)

# Create parameters and then CMSSP
params = Toy2DParameters(SPEED_LIMIT, SPEED_RESOLUTION, EPSILON, N_GEN_SAMPLES, NUM_BRIDGE_SAMPLES)
cmssp = create_toy2d_cmssp(params)
mdp = Toy2DModalMDPType(1, cmssp.control_actions, 1.0, HORIZON_LIMIT)

POMDPs.isterminal(mdp,state) = CMSSPs.CMSSPDomains.isterminal(mdp,state,params)

for i = 1:trials

    tot_cost = 0.0

    start_horizon = convert(Int64,rand(HORIZON_LIMIT/2:HORIZON_LIMIT))
    target_state = sample_toy2d(rng)
    curr_state = Toy2DContState(0.0,0.0)
    tp_dist = TPDistribution([start_horizon],[1.0])

    @show target_state, start_horizon
    @show horizon_weighted_value(modal_horizon_policy, 0, tp_dist, curr_state, target_state)
    readline()

    for t = 0:1:start_horizon-1
        a = get_best_intramodal_action(modal_horizon_policy, t, tp_dist, curr_state, target_state)
        if a == nothing
            println("ABORT!")
            break
        end
        (new_state, reward) = generate_sr(mdp, curr_state, a.action, rng)
        @show t,a,new_state
        readline()
        curr_state = new_state
        tot_cost += -1.0*reward
    end

    if POMDPs.isterminal(mdp, CMSSPs.CMSSPDomains.get_relative_state(curr_state,target_state))
        @show "Success!"
    else
        @show "Failure!"
    end
end