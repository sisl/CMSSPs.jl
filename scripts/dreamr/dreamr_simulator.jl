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

rng = MersenneTwister(3456)

inhor_fn = "./dreamr-cf-params111-betapt75-inhor.jld2"
outhor_fn = "./dreamr-cf-params111-betapt75-outhor.jld2"
infhor_fn = "./dreamr-uf-params111.jld2"

param_files = ["./paramsets/scale-1.toml","./paramsets/simtime-1.toml","./paramsets/cost-1.toml"]

scale_file = param_files[1]
simtime_file = param_files[2]
cost_file = param_files[3]

episode_args = ["/scratch/shushman/HitchhikingDrones/set-1-easy/set-1-100-to-1000", 2]

ep_file_prefix = episode_args[1]
num_eps = episode_args[2]

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

    @show start_pos, goal_pos

    start_state = DREAMRStateType{MultiRotorUAVState}(FLIGHT, get_state_at_rest(MultiRotorUAVState, start_pos))

    @show start_state

    # Create CMSSP
    cmssp = create_dreamr_cmssp(MultiRotorUAVState, MultiRotorUAVAction, Point(0.0, 0.0), params)

    set_dreamr_goal!(cmssp, goal_pos)

    # Create solver
    dreamr_solver = DREAMRSolverType{MultiRotorUAVState,MultiRotorUAVAction,typeof(rng)}(0, modal_policies,
                                    deltaT, [FLIGHT], start_state, DREAMRBookkeeping(), rng)

    (cost, timesteps, successful) = solve(dreamr_solver, cmssp)
    if successful
        @info "Success!"
    else
        @info "Failure!"
    end

    @show timesteps
    @show cost
end