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

global_logger(SimpleLogger(stderr, Logging.Error))

rng = MersenneTwister(3456)

param_files = ["./paramsets/scale-1.toml","./paramsets/simtime-1.toml","./paramsets/cost-1.toml"]

scale_file = param_files[1]
simtime_file = param_files[2]
cost_file = param_files[3]


inhor_fn = ARGS[1]
outhor_fn = ARGS[2]
infhor_fn = "./policies/dreamr-uf-params111.jld2"

ep_file_prefix = ARGS[3]
num_eps = parse(Int64, ARGS[4])
outfn = ARGS[5]

# Parse parameters
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

@info "Loading policies"
flight_policy = load_localapproxvi_policy_from_jld2(infhor_fn)
hopon_policy = load_modal_horizon_policy_localapprox(inhor_fn, outhor_fn)
ride_policy = DREAMRDeterministicPolicy(params)

# Create modal policies
modal_policies = Dict(FLIGHT => ModalFinInfHorPolicy(hopon_policy, flight_policy),
                      RIDE => ModalFinInfHorPolicy(ride_policy,flight_policy))

deltaT = convert(Int64, (params.time_params.MAX_REPLAN_TIMESTEP/params.time_params.MDP_TIMESTEP))

costs = Vector{Float64}(undef, 0)
steps = Vector{Int64}(undef, 0)
mode_switches = Vector{Int64}(undef, 0)
unsuccessful = Vector{Int64}(undef, 0)


for iter = 1:num_eps

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


    # Create CMSSP
    cmssp = create_dreamr_cmssp(MultiRotorUAVState, MultiRotorUAVAction, context_set, goal_pos, params)

    # Create solver
    dreamr_solver = DREAMRSolverType{MultiRotorUAVState,typeof(rng)}(0, modal_policies,
                                    deltaT, [FLIGHT], start_state, DREAMRBookkeeping(), rng)

    (cost, timesteps, successful, num_mode_switches) = solve(dreamr_solver, cmssp)
    
    if successful
        energy_cost = cost - params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP*timesteps
        push!(costs, energy_cost)
        push!(steps, timesteps)
        push!(mode_switches, num_mode_switches)
    else
        push!(unsuccessful, iter)
    end
end

results_dict = Dict("costs"=>costs, "steps"=>steps, "mode_switches"=>mode_switches, "failed_iters"=>unsuccessful)


open(outfn, "w") do f
    JSON.print(f, results_dict, 2)
end