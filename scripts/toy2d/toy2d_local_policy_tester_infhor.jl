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

inhor_file = ARGS[1]
outhor_file = ARGS[2]
params_fn = ARGS[3]
trials = 5

rng = MersenneTwister(5542)

modal_horizon_policy = load_modal_horizon_policy_localapprox(inhor_file, outhor_file)

# Create parameters and then CMSSP
params = toy2d_parse_params(params_fn)
cmssp = create_toy2d_cmssp(params)

# Set up the modal mdp
# Need to do it for just one mode
modal_mdp = Toy2DModalMDPType(1, params, cmssp.control_actions, 0.8, params.horizon_limit)

for i = 1:trials

    tot_cost = 0.0

    target_state = sample_toy2d(rng)
    curr_state = Toy2DContState(0.0,0.0)

    @show target_state
    inf_hor_rch = 0

    while true
        (a,inf_hor_rch) = get_best_intramodal_action_infhor(modal_horizon_policy, inf_hor_rch, curr_state, target_state)
        (new_state, reward) = generate_sr(modal_mdp, curr_state, a.action, rng)
        curr_state = new_state
        tot_cost += -1.0*reward
        if POMDPs.isterminal(modal_mdp,get_relative_state(modal_mdp, curr_state,target_state))
            println("Success!")
            break
        end
    end
end