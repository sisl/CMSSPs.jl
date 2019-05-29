using POMDPs
using POMDPModels
using POMDPModelTools
using StaticArrays
using JLD2, FileIO
using Random
using Logging
using Distributions
using Statistics
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using CMSSPs

rng = MersenneTwister(5678)

inhor_fn = ARGS[1]
outhor_fn = ARGS[2]
beta = parse(Float64, ARGS[3])

arguments = ["./paramsets/scale-1.toml","./paramsets/simtime-1.toml",
             "./paramsets/cost-1.toml", "30"]

scale_file = arguments[1]
simtime_file = arguments[2]
cost_file = arguments[3]
num_trials = parse(Int64, arguments[4])

# First, parse parameter files to get filenames and construct params object
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

# Get MDP and policy - DIFFERENT SEEDS
cf_mdp = get_cf_mdp(MultiRotorUAVState, MultiRotorUAVAction, params, beta)
hopon_policy = load_modal_horizon_policy_localapprox(inhor_fn, outhor_fn)


costs = Vector{Float64}(undef, 0)
successes = Vector{Int64}(undef, 0)

sim = HopOnOffSingleCarSimulator(params, rng)

for i = 1:num_trials

    tot_cost = 0.0

    # Go to 0.0, 0.0
    curr_state = generate_start_state(MultiRotorUAVState, params, rng)
    goal_state = get_state_at_rest(MultiRotorUAVState, Point(0.0, 0.0))

    reset_sim(sim)

    @show curr_state, mean(sim.time_to_finish)

    while true

        # @show curr_state, mean(sim.time_to_finish)

        tp_dist = get_arrival_time_distribution(0, mean(sim.time_to_finish), params, rng)

        a = get_best_intramodal_action(hopon_policy, 0, tp_dist, curr_state, goal_state)
        # @show a
        # readline()

        if a == nothing
            println("ABORT!")
            break
        end

        (new_state, reward) = generate_sr(cf_mdp, curr_state, a.action, rng)
        curr_state = new_state
        tot_cost += -1.0*reward

        is_done = step_sim(sim)
    
        if is_done
            if isterminal(cf_mdp, get_relative_state(cf_mdp, curr_state, goal_state))
                @info "Success!"
                push!(successes, 1)
            else
                @info "Failure!"
                push!(successes, 0)
            end
            break
        end
    end

    push!(costs, tot_cost)
end

@show costs
@show successes
