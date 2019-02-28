"""
    save_localapproxvi_policy_to_jld2(policy_fn::String, policy::LocalApproximationValueIterationPolicy, 
                                           mdp::Union{MDP,POMDP}, mersenne_seed::Int)

Custom saving method for storing a LocalApproximationValueIterationPolicy as a JLD2 object. It saves the individual
components based on the fields of the object. NOTE - Can only work with load_localapproxvi_policy_from_jld2
"""
function save_localapproxvi_policy_to_jld2(policy_fn::String, policy::LocalApproximationValueIterationPolicy, 
                                           mdp::MDP)

    save(policy_fn, "interp", policy.interp,
                    "mdp", mdp,
                    "is_mdp_generative",policy.is_mdp_generative,
                    "n_generative_samples",policy.n_generative_samples,
                    "seed",policy.rng.seed)
end


"""
    load_localapproxvi_policy_from_jld2(policy_fn::String)

Custom loading method for retrieving a LocalApproximationValueIterationPolicy from a JLD2 file. It loads the individual
fields for the object. NOTE - Can only work with save_localapproxvi_policy_to_jld2
"""
function load_localapproxvi_policy_from_jld2(policy_fn::String, rng_type::Type{RNG} = MersenneTwister) where {RNG <: AbstractRNG}

    policy_interp = load(policy_fn, "interp")
    policy_mdp = load(policy_fn, "mdp")
    policy_isgen = load(policy_fn,"is_mdp_generative")
    policy_n_gen_samples = load(policy_fn,"n_generative_samples")
    policy_seed = load(policy_fn, "seed")

    return LocalApproximationValueIterationPolicy(policy_interp, ordered_actions(policy_mdp), policy_mdp, policy_isgen,
                                                  policy_n_gen_samples, rng_type(policy_seed))
end

function load_modal_policy_localapprox(inhor_policy_fn::String, outhor_policy_fn::String,
                                       rng_type::Type{RNG} = MersenneTwister) where {RNG <: AbstractRNG}

    inhor_policy = load_localapproxvi_policy_from_jld2(inhor_policy_fn, rng_type)
    outhor_policy = load_localapproxvi_policy_from_jld2(outhor_policy_fn, rng_type)

    @assert inhor_policy.action_map == outhor_policy.action_map "Action Maps for in and out horizon must match!"

    return ModalPolicy(inhor_policy, outhor_policy)
end

function save_modal_policy_localapprox(modal_policy::ModalPolicy, inhor_policy_fn::String, outhor_policy_fn::String,
                                       mdp::MDP)
    save_localapproxvi_policy_to_jld2(inhor_policy_fn, modal_policy.in_horizon_policy, mdp)
    save_localapproxvi_policy_to_jld2(outhor_policy_fn, modal_policy.out_horizon_policy, mdp)
end