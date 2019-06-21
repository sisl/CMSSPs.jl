using GridInterpolations
using LocalFunctionApproximation
using StaticArrays
using JLD2, FileIO
using Random
using ContinuumWorld
using CMSSPs

# global_logger(SimpleLogger(stderr, Logging.Debug))

rng = MersenneTwister(1234)
const TRIALS = 10

@info "Loading policies"
inhor_fn = "grids-continuum-params1-inhor.jld2"
outhor_fn = "grids-continuum-params1-outhor.jld2"
modal_horizon_policy = load_modal_horizon_policy_localapprox(inhor_fn, outhor_fn)

# Create reqments for solver by loading modal policies and modal MDPs etc
modal_policies = Dict{Int64, ModalFinInfHorPolicy}()
modal_mdps = Dict{Int64, GridsContinuumMDPType}()
for m in CONTINUUM_MODES
    global modal_policies
    global modal_mdps
    modal_policies[m] = ModalFinInfHorPolicy(modal_horizon_policy, nothing)
    modal_mdps[m] = modal_horizon_policy.in_horizon_policy.mdp
end

# Define the params struct and the dmssp problem objectß
params = modal_horizon_policy.in_horizon_policy.mdp.params
cmssp = create_continuum_cmssp(params, rng)

# Other parameters required by the solver - the replanning period (in terms of timesteps) and the goal modes
deltaT = 5
goal_modes = [CONTINUUM_GOAL_MODE]

for trial = 1:TRIALS

    @info "Generating start state and context"
    start_state = generate_start_state(cmssp, rng)
    set_start_context_set!(cmssp, rng)

    @debug start_state

    @info "Creating solver"
    continuum_solver = GridsContinuumSolverType{typeof(rng)}(params.num_bridge_samples,
                                                modal_policies,
                                                deltaT,
                                                goal_modes,
                                                start_state,
                                                zero(ContinuumDummyBookkeeping),
                                                rng)

    solve(continuum_solver, cmssp)
end    