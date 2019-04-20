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

inhor_file = "../local_policies/toy2d-inhor-1.jld2"
outhor_file = "../local_policies/toy2d-outhor-1.jld2"
params_fn = "../paramsets/toy2d-params1.toml"

const TRIALS = 5

rng = MersenneTwister(1234)

@info "Loading policies"
modal_horizon_policy = load_modal_horizon_policy_localapprox(inhor_file, outhor_file)

# Create parameters and then CMSSP
params = toy2d_parse_params(params_fn)
cmssp = create_toy2d_cmssp(params)


# Create reqments for solver by loading modal policies and modal MDPs etc
modal_policies = Dict{Int64,ModalHorizonPolicy}()
modal_mdps = Dict{Int64,Toy2DModalMDPType}()
for m in TOY2D_MODES
    global modal_policies
    global modal_mdps
    modal_policies[m] = modal_horizon_policy
    modal_mdps[m] = modal_horizon_policy.in_horizon_policy.mdp
end

# Other params for solver
deltaT = 5
goal_modes = [TOY2D_GOAL_MODE]

for trial = 1:TRIALS
    @info "Generating start state and context"
    # Get start state and context
    start_state = generate_start_state(cmssp, rng)
    start_context = generate_start_context_set(cmssp, rng)

    

    @info "Creating solver"
    toy2d_solver = HHPCSolver(Int64, params.num_bridge_samples, modal_policies,
                                   modal_mdps, deltaT, goal_modes,
                                   start_state, start_context, rng)
    solve(toy2d_solver, cmssp)
end