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
using TOML


struct MCTSParams
    depth::Int64
    exploration::Float64
    init_N::Int64
    n_iters::Int64
end

function parse_mcts_params(filename::String)
    params_key = TOML.parsefile(filename)
    return MCTSParams(params_key["DEPTH"], params_key["C"], params_key["INITN"], params_key["NITERS"])
end

global_logger(SimpleLogger(stderr, Logging.Warn))

rng = MersenneTwister(3456)

param_files = ["./paramsets/scale-1.toml","./paramsets/simtime-1.toml","./paramsets/cost-1.toml"]

scale_file = param_files[1]
simtime_file = param_files[2]
cost_file = param_files[3]

# Load flight policy and create an alias for estimate value
infhor_fn = "./dreamr-uf-params111.jld2"
flight_policy = load_localapproxvi_policy_from_jld2(infhor_fn)
estimate_value(dmcts::DREAMRMCTSType, state::DREAMRMCTSState, depth::Int64) = estimate_value_dreamr(flight_policy, dmcts, state, depth)

init_Q(dmcts::DREAMRMCTSType, state::DREAMRMCTSState, a::DREAMRActionType) = init_q_dreamr(flight_policy, dmcts, state, a)


ep_file_prefix = "/scratch/shushman/HitchhikingDrones/set-2-hard/set-2-100-to-1000"
num_eps = parse(Int64, ARGS[3])
mcts_param_file = ARGS[1]
outfn = ARGS[2]


# ep_file_prefix = ARGS[1]
# num_eps = parse(Int64, ARGS[2])
# mcts_param_file = ARGS[3]
# outfn = ARGS[4]

params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)
mcts_params = parse_mcts_params(mcts_param_file)


costs = Vector{Float64}(undef, 0)
steps = Vector{Int64}(undef, 0)
mode_switches = Vector{Int64}(undef, 0)
unsuccessful = Vector{Int64}(undef, 0)


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

    # @show start_state
    # @show goal_pos
    # readline()

    # Create CMSSP
    cmssp = create_dreamr_cmssp(MultiRotorUAVState, MultiRotorUAVAction, context_set, goal_pos, params)

    dmcts = DREAMRMCTSType(cmssp)

    # Create solver and planner
    solver = DPWSolver(depth=mcts_params.depth,
                       exploration_constant = mcts_params.exploration,
                       init_N = mcts_params.init_N,
                       n_iterations = mcts_params.n_iters,
                       estimate_value=estimate_value,
                       # init_Q = init_Q,
                       rng=rng)
    planner = solve(solver, dmcts)

    curr_dmcts_state = DREAMRMCTSState(start_state)
    
    timesteps = 0
    cost = 0
    num_mode_switches = 0
    successful = false

    while true
        
        # @debug curr_dmcts_state
        a = action(planner, curr_dmcts_state)
        # @debug a

        # readline()

        (next_state, reward, _, timeout) = simulate_cmssp!(dmcts.cmssp, curr_dmcts_state.cmssp_state, a, timesteps, rng)

        if next_state.mode != curr_dmcts_state.cmssp_state.mode
            num_mode_switches += 1
        end

        cost += -1.0*reward

        curr_dmcts_state = DREAMRMCTSState(next_state, cmssp.curr_context_set.car_id, 0)

        timesteps += 1
        # @debug timesteps

        if isterminal(dmcts, curr_dmcts_state)
            successful = true
            break
        end

        if timeout
            @debug "Timeout!"
            break
        end
    end

    if successful
        energy_cost = cost - params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP*timesteps
        @show energy_cost
        push!(costs, energy_cost)
        push!(steps, timesteps)
        push!(mode_switches, num_mode_switches)
    else
        @info "unsuccessful"
        push!(unsuccessful, iter)
    end
end

results_dict = Dict("costs"=>costs, "steps"=>steps, "mode_switches"=>mode_switches, "failed_iters"=>unsuccessful)


open(outfn, "w") do f
    JSON.print(f, results_dict, 2)
end