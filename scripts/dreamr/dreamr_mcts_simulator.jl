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
using JSON
using CMSSPs
using MCTS
using Logging

global_logger(SimpleLogger(stderr, Logging.Debug))

rng = MersenneTwister(3456)

param_files = ["./paramsets/scale-1.toml","./paramsets/simtime-1.toml","./paramsets/cost-1.toml"]

scale_file = param_files[1]
simtime_file = param_files[2]
cost_file = param_files[3]

episode_args = ["./trial-data/mcts-test-ep", 1]

ep_file_prefix = episode_args[1]
num_eps = episode_args[2]

params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

const DEPTH = 100

for iter=1:num_eps

    episode_dict = Dict

    ep_file = string(ep_file_prefix,"-",iter,".json")

    @show ep_file

    open(ep_file,"r") do f
        episode_dict = JSON.parse(f)
    end

    context_set = get_dreamr_episode_context(episode_dict["epochs"])

    start_pos = Point(episode_dict["start_pos"][1], episode_dict["start_pos"][2])
    goal_pos = Point(episode_dict["goal_pos"][1], episode_dict["goal_pos"][2])

    start_state = DREAMRStateType{MultiRotorUAVState}(FLIGHT, get_state_at_rest(MultiRotorUAVState, start_pos))

    @show start_state
    @show goal_pos

    # Create CMSSP
    cmssp = create_dreamr_cmssp(MultiRotorUAVState, MultiRotorUAVAction, context_set, goal_pos, params)

    dmcts = DREAMRMCTSType(cmssp)

    # Create solver and planner
    solver = DPWSolver(depth=DEPTH, rng=rng, estimate_value=estimate_value_dreamr)
    planner = solve(solver, dmcts)

    curr_dmcts_state = DREAMRMCTSState(start_state)
    t = 0
    tot_cost = 0

    while true
        
        @show curr_dmcts_state
        a = action(planner, curr_dmcts_state)
        @show a
        # readline()
        (next_state, reward, _) = simulate_cmssp!(dmcts.cmssp, curr_dmcts_state.cmssp_state, a, t, rng)

        tot_cost += -1.0*reward

        curr_dmcts_state = DREAMRMCTSState(next_state, cmssp.curr_context_set.curr_car_id, 0)

        t += 1
        @show t

        if isterminal(dmcts, curr_dmcts_state)
            break
        end
    end

    @show tot_cost
end