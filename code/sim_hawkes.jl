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


"""
    transpars(pars, intervals; back = false)

Transform parameters so that they lie in pre-specified intervals.

# Inputs
    - pars          array of parameters
    - intervals     array of intervals (in same order as pars)
    - back          boolean if parameters should be transformed back, default = false

# Output
    - pars          array of transformed parameters
"""
function transpars(pars, intervals; back = false)
    
    # init list of transformed parameters
    par_tr = []

    # change those parameters with provided intervals
    for (i,par) in enumerate(pars)

        # get associated bounds
        lower_bound = intervals[i][1]
        upper_bound = intervals[i][2]

        # do not transform if both upper and lower bound are Inf
        if lower_bound == -Inf && upper_bound == Inf
            push!(par_tr, par)
        end

        # transform variables
        if ~back
            if lower_bound == -Inf
                new_value = log(-(par - upper_bound))
            elseif upper_bound == Inf
                new_value = log(par - lower_bound)
            else
                aux = (par - lower_bound) / (upper_bound - lower_bound)
                new_value = log(aux / (1-aux))
            end

        # transform variables back
        else
            if lower_bound == -Inf
                new_value = -exp(par) + upper_bound
            elseif upper_bound == Inf
                new_value = exp(par) + lower_bound
            else
                aux = exp(par) / (1 + exp(par))
                new_value = (upper_bound - lower_bound) * aux + lower_bound
            end
        end

        # update parameter value
        push!(par_tr, new_value)

    end

    return par_tr
end


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

    if right # complete limit from the right
        for T in vT
            if t-T ≥ 0
                ∑ϕ += ϕ_exp(α, δ, t-T)
            else
                ∑ϕ += 0
            end
        end
    else
        for T in vT
            if t-T > 0 
                ∑ϕ += ϕ_exp(α, δ, t-T)
            else
                ∑ϕ += 0
            end
        end 
    end

    return λ₀ + ∑ϕ
end

function λ(λ₀::Float64, t::Float64, vT; right = false, α::Float64, δ::Float64, η::Float64)

    ∑ϕ = 0.0

    if right # complete limit from the right
        for T in vT
            if t-T ≥ 0
                ∑ϕ += ϕ_power(α, δ, η, t-T)
            else
                ∑ϕ += 0
            end
        end
    else
        for T in vT
            if t-T > 0 
                ∑ϕ += ϕ_power(α, δ, η, t-T)
            else
                ∑ϕ += 0
            end 
        end
    end

    return λ₀ + ∑ϕ
end


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
    if length(pars) == 3
        λ₀, α_fraction, δ = pars
        α = α_fraction * δ
    else
        λ₀, α_fraction, δ, η = pars
        α = α_fraction * η * δ^η
    end

    # initialization
    Random.seed!(simpars.seed)
    vT = []
    n_events = 0
    T = 0.0
    exp_dist = Exponential(1/λ₀)

    # draw event times
    while (n_events < simpars.N) && (T < simpars.T_max)
        # set upper bound
        if length(pars) == 3
            λᵘ = λ(λ₀, T, vT, right = true, α = α, δ = δ)
        else
            λᵘ = λ(λ₀, T, vT, right = true, α = α, δ = δ, η = η)
        end
        
        # draw interarrival time
        τ = rand(exp_dist)

        # update current time
        T += τ

        # draw random uniform s for reject sampling
        s = rand()

        # compute new λ(T)
        if length(pars) == 3
            λₜ = λ(λ₀, T, vT, right = false, α = α, δ = δ)
        else
            λₜ = λ(λ₀, T, vT, right = false, α = α, δ = δ, η = η)
        end

        # check if accept sample
        if s ≤ (λₜ / λᵘ)
            push!(vT, T)
            n_events += 1
        end
    end
    return vT
end

vT = [0.5, 1.0, 2.5, 3.0]

function plot_hawkes(vT, pars)

    # convert α_fraction to alpha
    if length(pars) == 3
        λ₀, α_fraction, δ = pars
        α = α_fraction * δ
    else
        λ₀, α_fraction, δ, η = pars
        α = α_fraction * η * δ^η
    end

    # generate grid
    maxT = ceil(maximum(vT))
    t_grid = collect(LinRange(0.0, maxT, 1000))

    # concatenate with vT
    t_grid = vcat(t_grid, vT)
    sort!(t_grid)
    unique!(t_grid)

    # get intensities for events
    λ_events = []

    for T in vT
        prev_times = [time for time in vT if time ≤ T]

        if length(pars) == 3
            λₜ = λ(λ₀, T, prev_times, right = true, α = α, δ = δ)
        else
            λₜ = λ(λ₀, T, prev_times, right = true, α = α, δ = δ, η = η)
        end
        push!(λ_events, λₜ)
    end

    # get intensities for grid points
    λ_grid = []

    for T in t_grid
        prev_times = [time for time in vT if time ≤ T]

        if length(pars) == 3
            λₜ = λ(λ₀, T, prev_times, right = true, α = α, δ = δ)
        else
            λₜ = λ(λ₀, T, prev_times, right = true, α = α, δ = δ, η = η)
        end
        push!(λ_grid, λₜ)
    end

    # construct plot
    plot(t_grid, λ_grid)
    plot!(vT, λ_events, seriestype = :scatter, leg = false)
    xlabel!("time t")
    ylabel!("λ(t)")
    if length(pars) == 3
        annotate!(140,1.4,text("λ₀ = $(round(λ₀, digits = 3)), α = $(round(α, digits = 3)), δ = $(round(δ, digits = 3))", 10))
    else
        annotate!(140,1.4,text("λ₀ = $(round(λ₀, digits = 3)), α = $(round(α, digits = 3)), δ = $(round(δ, digits = 3)), η = $(round(η, digits = 3))", 10))
    end

end


function nll(pars, vT)

    # convert α_fraction to alpha
    if length(pars) == 3
        λ₀, α_fraction, δ = pars
        α = α_fraction * δ
    else
        λ₀, α_fraction, δ, η = pars
        α = α_fraction * η * δ^η
    end

    # compute sum log lambda
    ∑logλ = 0.0

    for T in vT
        prev_T = [time for time in vT if time < T]
        ∑logλ += log(λ(λ₀, T, prev_T, α = α, δ = δ, η = η))
    end

    # compute integral (compensator function)
    last_T = vT[end] # can also be set to end of observation period
    integral_part = 0.0

    for T in vT
        integral_part += (1.0 / (δ^η) - 1.0/((last_T - T + δ)^η)) * α/η
    end

    return -(∑logλ - λ₀*last_T - integral_part)
end


# define parameters to simulate Hawkes process
simpars = SimulationParameters(N = 2000);

# specify array of parameters
# λ₀, α_fraction, δ, η

λ₀ = 0.5
α = 0.4
δ = 0.4
η = 0.2
pars = [λ₀, α/(η * δ^η), δ, η]

# specify intervals only for those variables, [-Inf, Inf] if no transform needed
intervals = [[0,Inf], [0,3], [0,2], [0,Inf]]



# simulate data
vT = sim_hawkes(simpars, pars);

# plot
plot_hawkes(vT, pars)

# estimate parameters
using Optim
using NLSolversBase
using ForwardDiff

f(x) = nll(transpars(x, intervals, back = true), vT)/length(vT)

par0 = zeros(Float64, length(pars))


par0 = transpars(pars, intervals, back = false)

# f = TwiceDifferentiable(par0 -> nll(par0, vT)/length(vT))

func= TwiceDifferentiable(pars -> f(pars), par0)

func= TwiceDifferentiable(pars -> f(pars), par0; autodiff = :forward)

opt = optimize(func, par0, LBFGS(), Optim.Options(show_trace = true,
                show_every = 10, iterations = 200))

transpars(Optim.minimizer(opt), intervals, back = true)

# problem: cannot estimate the parameters!

# what else as extension? e.g. let lambda increase incrementally over time

# prep data

# setup ocnditional estimation









