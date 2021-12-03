using Parameters
using Distributions
using DataStructures
using Random
using Plots

@with_kw struct SimulationParameters
    N::Int       = 200
    T_max::Int   = 10000
    seed::Int    = 1234
end

@with_kw struct Pars
    λ₀::Float64             = 0.1; @assert λ₀ > 0
    α_fraction::Float64     = 0.4; @assert α_fraction > 0 && α_fraction < 1
    δ::Float64              = 0.5; @assert δ > 0
    η                       = undef
    α                       = undef
end

pars = OrderedDict(
    :λ₀         => 0.1,
    :α_fraction => 0.4,
    :δ          => 0.5,
    :η          => 0.5)


pars = Pars(η = 0.5, α_fraction = 0.565685424949238)

# specify intervals only for those variables which should be transformed
intervals = Dict(:λ₀ => [0, Inf], :α_fraction => [0, 1],
                 :δ => [0,2], :η => [0, Inf])

"""


Transform parameters so that they lie in pre-specified intervals.

# Inputs
    - pars          parameters dictionary with parameter name and value pairs
    - intervals     array of intervals
    - back          boolean if parameters should be transformed back, default = false

# Output
    - pars          array of transformed parameters
"""
function transpars(pars, intervals; back = false)
    
    
    OrderedDict(type2dict(pars))

    # do not use parametrs struct for these model aprameters, use ordered dict. initialize them via key value pairs?

    par_dict = type2dict(pars)
    par_tr = copy(par_dict)

    # change those parameters with provided intervals
    for (par,value) in par_dict
        if haskey(intervals, par)
    
            # get associated bounds
            left_bound = intervals[par][1]
            right_bound = intervals[par][2]

            # transform variables
            if ~back
                if left_bound == -Inf
                    new_value = log(-(value - right_bound))
                elseif right_bound == Inf
                    new_value = log(value - left_bound)
                else
                    aux = (value - left_bound) / (right_bound - left_bound)
                    new_value = log(aux / (1-aux))
                end

            # transform variables back
            else
                if left_bound == -Inf
                    new_value = -exp(value) + right_bound
                elseif right_bound == Inf
                    new_value = exp(value) + left_bound
                else
                    aux = exp(value) / (1 + exp(value))
                    new_value = (right_bound - left_bound) * aux + left_bound
                end
            end

            # update parameter value
            par_tr[par] = new_value
        end
    end

    # pack parameters into new Parameters struct
    # extract defined value in order top to bottom
    return [(key, val) for (key, val) in par_tr if val != undef]
end


Dict([(key, val) for (key, val) in par_tr if val != undef])




"""
    ϕ_exp(α::Float64, δ::Float64, τ::Float64)

Exponential kernel for Hawkes process.

# Inputs
    - α ≥ 0 
    - δ > 0
    - α < δ
    - τ   inter-arrival time
"""
function ϕ_exp(α::Float64, δ::Float64, τ::Float64)
    α * exp(-δ * τ)
end


"""
    ϕ_power(α::Float64, δ::Float64, η::Float64, τ::Float64)

Power-law kernel for Hawkes process.

# Inputs
    - α ≥ 0 
    - δ,η > 0
    - α < η δ^η
    - τ   inter-arrival time
"""
function ϕ_power(α::Float64, δ::Float64, η::Float64, τ::Float64)
    α / (τ + δ)^(η + 1)
end


"""
    λ(λ₀::Float64, t::Float64, vT; right = false, α::Float64, δ::Float64[, η::Float64])

Compute the intensity of a Hawkes process with the exponential or power-law kernel.

If η is specified, use power-law kernel.

# Inputs
        λ₀          baseline intensity
        t           current time in process
        vT          vector of previous event times
        right       true if computing limit from right side
        α, δ, η        kernel parameters

# Output
        λ(t)        the intensity of the Hawkes process at time t
"""
function λ(λ₀::Float64, t::Float64, vT; right = false, α::Float64, δ::Float64)

    ∑ϕ = 0.0

    # provide option for computing right limit
    if right
        vt = vcat(vT, [t])
    else
        vt = vT
    end

    for T in vt
        ∑ϕ += ϕ_exp(α, δ, t-T)
    end

    return λ₀ + ∑ϕ
end

function λ(λ₀::Float64, t::Float64, vT; right = false, α::Float64, δ::Float64, η::Float64)

    ∑ϕ = 0.0

    # provide option for computing right limit
    if right
        vt = vcat(vT, [t])
    else
        vt = vT
    end

    for T in vt
        ∑ϕ += ϕ_power(α, δ, η, t-T)
    end

    return λ₀ + ∑ϕ
end


# test_results = []
# vt = [0.5, 1.0, 2.0, 4.0, 4.5]
# for t in vt
#     prev_t = [T for T in vt if T < t]
#     push!(test_results, log(λ(0.1, t, prev_t, right = true, α = 0.2, δ = 0.5, η = 0.5)))
# end # gives same results for right = true and false



"""
    sim_hawkes(N::Int, T_max::Int, pars)

Simulate a Hawkes process, i.e. event times.

# Inputs
    N       number of event times to simulate
    T_max   upper bound on simulated time horizon
    seed    seed for replicability

# Output
    vT      array of event times

"""
function sim_hawkes(simpars, pars)

    # convert α_fraction to alpha
    if isnothing(pars.η)
        θ = Pars(α = pars.α_fraction * pars.δ)
    else
        θ = Pars(α = pars.α_fraction * pars.η * pars.δ ^ pars.η,
            	η = pars.η)
    end

    # initialization
    Random.seed!(simpars.seed)
    vT = []
    n_events = 0
    T = 0.0
    exp_dist = Exponential(1/pars.λ₀)

    # draw event times
    while (n_events < simpars.N) && (T< simpars.T_max)
        # set upper bound
        if isnothing(pars.η)
            λᵘ = λ(θ.λ₀, T, vT, right = true, α = θ.α, δ = θ.δ)
        else
            λᵘ = λ(θ.λ₀, T, vT, right = true, α = θ.α, δ = θ.δ, η = θ.η)
        end
        
        # draw interarrival time
        τ = rand(exp_dist)

        # update current time
        T += τ

        # draw random uniform s for reject sampling
        s = rand()

        # compute new λ(T)
        if isnothing(pars.η)
            λₜ = λ(θ.λ₀, T, vT, right = false, α = θ.α, δ = θ.δ)
        else
            λₜ = λ(θ.λ₀, T, vT, right = false, α = θ.α, δ = θ.δ, η = θ.η)
        end

        # check if accept sample
        if s ≤ (λₜ / λᵘ)
            push!(vT, T)
            n_events += 1
        end
    end
    return vT
end



function plot_hawkes(vT, pars)

    # convert α_fraction to alpha
    if isnothing(pars.η)
        θ = Pars(α = pars.α_fraction * pars.δ)
    else
        θ = Pars(α = pars.α_fraction * pars.η * pars.δ ^ pars.η,
            	η = pars.η)
    end

    # generate grid
    maxT = ceil(maximum(vT))
    t_grid = collect(LinRange(0.0, maxT, 10))

    # concatenate with vT
    t_grid = vcat(t_grid, vT)
    sort!(t_grid)

    # get intensities for events
    λ_events = []

    for (i,T) in enumerate(vT)
        prev_times = vT[begin:i-1]

        if isnothing(pars.η)
            λₜ = λ(θ.λ₀, T+0.1, prev_times, right = true, α = θ.α, δ = θ.δ)
        else
            λₜ = λ(θ.λ₀, T+0.1, prev_times, right = true, α = θ.α, δ = θ.δ, η = θ.η)
        end
        push!(λ_events, λₜ)
    end

    # get intensities for grid points
    λ_grid = []

    for (i,T) in enumerate(t_grid)
        prev_times = [time for time in vT if time < T]

        if isnothing(pars.η)
            λₜ = λ(θ.λ₀, T, prev_times, right = true, α = θ.α, δ = θ.δ)
        else
            λₜ = λ(θ.λ₀, T, prev_times, right = true, α = θ.α, δ = θ.δ, η = θ.η)
        end
        push!(λ_grid, λₜ)
    end

    # construct plot
    gr()
    plot(t_grid, λ_grid)
    plot!(vT, λ_events, seriestype = :scatter, leg = false)
    xlabel!("time t")
    ylabel!("λ(t)")
    annotate!(140,1.4,text("λ₀ = $(round(θ.λ₀, digits = 3)), α = $(round(θ.α, digits = 3)), δ = $(round(θ.δ, digits = 3)), η = $(round(θ.η, digits = 3))", 10))

end

for (i,T) in enumerate(t_grid)
    if T in vT
        index = findfirst(T .== vT)
        l1 = λ_events[index]

        index = findfirst(T .== t_grid)
        l2 = λ_grid[index]

        println("T = $(T), λ_event = $l1, λ_grid = $l2")
    end
end



# define parameters to simulate Hawkes process
simpars = SimulationParameters(N = 2)
pars = Pars(λ₀ = 2.4, η = 0.5)

# simulate data
vT = sim_hawkes(simpars, pars);

# plot
plot_hawkes(vT, pars)

θ.α / (0.1 + θ.δ)^(θ.η + 1)
# at max when tau = 0, so t = T





