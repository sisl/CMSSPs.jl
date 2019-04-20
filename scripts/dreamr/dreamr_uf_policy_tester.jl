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

rng = MersenneTwister(1234)

flight_policy_name = "dreamr-uf-params111.jld2"

arguments = ["./paramsets/scale-1.toml","./paramsets/simtime-1.toml",
        "./paramsets/cost-1.toml", "5"]

scale_file = arguments[1]
simtime_file = arguments[2]
cost_file = arguments[3]
num_trials = parse(Int64, arguments[4])

# First, parse parameter files to get filenames and construct params object
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

uf_mdp = get_uf_mdp(MultiRotorUAVState, MultiRotorUAVAction, params)

flight_policy = load_localapproxvi_policy_from_jld2(flight_policy_name)

costs = Vector{Float64}(undef, 0)

for i = 1 : num_trials

    curr_state = generate_start_state(MultiRotorUAVState, params, rng)
    goal_state = get_state_at_rest(MultiRotorUAVState, Point(0.0, 0.0))

    tot_cost = 0.0

    @show value(flight_policy, get_relative_state(uf_mdp, curr_state, goal_state))

    while true

        # @show curr_state
        inf_tp_dist = TPDistribution([typemax(Int64)], [1.0])

        a = get_best_intramodal_action(flight_policy, 0, inf_tp_dist, curr_state, goal_state)
        # @show a
        # readline()
        (next_state, reward) = generate_sr(uf_mdp, curr_state, a.action, rng)

        curr_state = next_state

        tot_cost += -1.0*reward

        if isterminal(uf_mdp, get_relative_state(uf_mdp, curr_state, goal_state))
            @info "Reached!"
            break
        end
    end

    push!(costs, tot_cost)
end

@show costs