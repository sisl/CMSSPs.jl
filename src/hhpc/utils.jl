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

function load_modal_horizon_policy_localapprox(inhor_policy_fn::String, outhor_policy_fn::String,
                                       rng_type::Type{RNG} = MersenneTwister) where {RNG <: AbstractRNG}

    inhor_policy = load_localapproxvi_policy_from_jld2(inhor_policy_fn, rng_type)
    outhor_policy = load_localapproxvi_policy_from_jld2(outhor_policy_fn, rng_type)

    @assert inhor_policy.action_map == outhor_policy.action_map "Action Maps for in and out horizon must match!"

    return ModalHorizonPolicy(inhor_policy, outhor_policy)
end

function save_modal_horizon_policy_localapprox(modal_policy::ModalHorizonPolicy, inhor_policy_fn::String, outhor_policy_fn::String,
                                       mdp::MDP)
    save_localapproxvi_policy_to_jld2(inhor_policy_fn, modal_policy.in_horizon_policy, mdp)
    save_localapproxvi_policy_to_jld2(outhor_policy_fn, modal_policy.out_horizon_policy, mdp)
end

"""
    log2space_symmetric(symm_val::Float64, n::Int64, base_val::Int64=2)

Returns a vector of logarithmically spaced numbers from -x to +x, symmetrically
spaced around 0. The number of values must be odd to reflect 0 and a symmetric
number of arguments around it.
"""
function log2space_symmetric(symm_val::Float64, n::Int64, base_val::Int64=2)

    # Ensure that the number of terms is odd (so around 0.0)
    @assert n%2==1

    # Ensure that the value itself is positive
    @assert symm_val > 0

    vals = Vector{Float64}(undef,0)
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    for i=1:midpt
        push!(vals, -(symm_val/idx))
        idx = idx*base_val
    end

    # Add 0.0 to axis
    symm_vect = reverse(-vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end


"""
    polyspace_symmetric(symm_val::Float64, n::Int64, exp_val::Int64=3)

Returns a vector of polynomially spaced numbers from -x to +x, symmetrically
spaced around 0. The number of values must be odd to reflect 0 and a symmetric
number of arguments around it.
"""
function polyspace_symmetric(symm_val::Float64, n::Int64, exp_val::Int64=3)

    # Ensure that the number of terms is odd (so around 0.0)
    @assert n%2==1
    # Ensure that the value itself is positive
    @assert symm_val > 0

    vals = Vector{Float64}(undef,0)
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    x = (symm_val/(midpt^exp_val))^(1/exp_val)

    for i=midpt:-1:1
        val = -1*(i*x)^exp_val
        push!(vals,val)
    end

    symm_vect = reverse(-vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end