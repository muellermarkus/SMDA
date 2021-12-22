using Parameters
using Distributions
using DataStructures
using Random
using Plots
using Optim
using NLSolversBase
using ForwardDiff
using DataFrames
using Parquet
using LinearAlgebra
using LineSearches

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
    par_tr = Vector{eltype(pars)}()

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
function ϕ_exp(α, δ, τ)
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
function ϕ_power(α, δ, η, τ)
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
function λ_exp(λ₀, t, vT; right = false, α, δ)

    ∑ϕ = 0.0

    if right # compute limit from the right
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



function λ_pwr(λ₀, t, vT; right = false, α, δ, η)

    ∑ϕ = 0.0

    if right # compute limit from the right
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
    
    # draw event times
    while (n_events < simpars.N) && (T < simpars.T_max)
        # set upper bound
        if length(pars) == 3
            λᵘ = λ_exp(λ₀, T, vT, right = true, α = α, δ = δ)
        else
            λᵘ = λ_pwr(λ₀, T, vT, right = true, α = α, δ = δ, η = η)
        end

        # draw interarrival time
        exp_dist = Exponential(1/λᵘ)
        τ = rand(exp_dist)

        # update current time
        T += τ

        # draw random uniform s for reject sampling
        s = rand()

        # compute new λ(T)
        if length(pars) == 3
            λₜ = λ_exp(λ₀, T, vT, right = false, α = α, δ = δ)
        else
            λₜ = λ_pwr(λ₀, T, vT, right = false, α = α, δ = δ, η = η)
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
            λₜ = λ_exp(λ₀, T, prev_times, right = true, α = α, δ = δ)
        else
            λₜ = λ_pwr(λ₀, T, prev_times, right = true, α = α, δ = δ, η = η)
        end
        push!(λ_events, λₜ)
    end

    # get intensities for grid points
    λ_grid = []

    for T in t_grid
        prev_times = [time for time in vT if time ≤ T]
        if length(pars) == 3
            λₜ = λ_exp(λ₀, T, prev_times, right = true, α = α, δ = δ)
        else
            λₜ = λ_pwr(λ₀, T, prev_times, right = true, α = α, δ = δ, η = η)
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


function nll_exp(pars, vT)
    last_T = vT[end]
    ∑logλ = 0.0
    integral_part = 0.0

    # construct alpha parameter
    λ₀, α_fraction, δ = pars
    α = α_fraction * δ

    # compute sum log lambda
    for T in vT
        prev_T = [time for time in vT if time < T]
        ∑logλ += log(λ_exp(λ₀, T, prev_T, α = α, δ = δ))
    end

    # compute integral (compensator function)
    for T in vT
        integral_part += α/δ * (1-exp(-δ*(last_T - T)))
    end

    return -(∑logλ - λ₀*last_T - integral_part)
end

function nll_pwr(pars, vT)
    last_T = vT[end]
    ∑logλ = 0.0
    integral_part = 0.0

    # construct alpha parameter
    λ₀, α_fraction, δ, η = pars
    α = α_fraction * η * δ^η

    # compute sum log lambda
    for T in vT
        prev_T = [time for time in vT if time < T]
        ∑logλ += log(λ_pwr(λ₀, T, prev_T, α = α, δ = δ, η = η))
    end

    # compute integral (compensator function)
    for T in vT
        integral_part += (1.0 / (δ^η) - 1.0/((last_T - T + δ)^η)) * α/η
    end

    return -(∑logλ - λ₀*last_T - integral_part)
end

function get_results(opt, func, intervals, vT, varnames; original_pars = nothing)
    estθ = transpars(Optim.minimizer(opt), intervals, back = true)
    # use delta method to get standard errors
    # as I used transformed parameters
    num_hess = -length(vT) .* hessian!(func, Optim.minimizer(opt))
    inv_num_hess = -inv(num_hess)
    num_jacob = ForwardDiff.jacobian(x -> transpars(x, intervals, back = true), Optim.minimizer(opt))
    var_cov = num_jacob * inv_num_hess * transpose(num_jacob)
    se = sqrt.(diag(var_cov))
    ttest = estθ ./ se
    normal_dist = Normal()
    pval = 2*pdf.(normal_dist, -abs.(ttest))
    if isnothing(original_pars)
        results = DataFrame(name = varnames, estθ = estθ, se = se, ttest = ttest, pval = pval)
    else
        results = DataFrame(name = varnames, θ = original_pars, estθ = estθ, se = se, ttest = ttest, pval = pval)
    end
    return results
end

function run_optimization(par0, intervals, vT, f)
    # define start parameters and differentiation
    par0_tr = transpars(par0, intervals)
    func = TwiceDifferentiable(f, par0_tr; autodiff = :forward);

    # run optimizer
    options = Optim.Options(show_trace = true, show_every = 10, iterations = 300, g_tol = 1e-5, f_tol = 2.2e-9)
    opt = optimize(func, par0_tr, LBFGS(; linesearch = LineSearches.HagerZhang(linesearchmax = 30)), options)
    print(opt)

    return opt, func
end

# define parameters to simulate Hawkes process
simpars = SimulationParameters(N = 6000);

# specify array of parameters
λ₀ = 0.1
α_fraction = 0.1
δ = 0.8

# specify array of parameters and parameter names
varnames = ["λ₀", "α_fraction", "δ"]
pars = [λ₀, α_fraction, δ]

# specify intervals for the parameter transformations
intervals = [[0,Inf], [0,1], [0,Inf]];

# simulate data
vT = sim_hawkes(simpars, pars);

# plot first 20 data points
plot_hawkes(vT[begin:20], pars)

# define related nll function
if length(pars) == 3
    f(x) = nll_exp(transpars(x, intervals, back = true), vT)/length(vT);
else
    f(x) = nll_pwr(transpars(x, intervals, back = true), vT)/length(vT);
end

# init parameters for estimation
par0 = copy(pars);

opt, func = run_optimization(par0, intervals, vT, f);

results = get_results(opt, func, intervals, vT, varnames, original_pars = pars)

##########
# ESTIMATE PARAMETERS ON CLIMATE DISASTER NEWS DATA

# load disaster data
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "disasters.gzip");
df = DataFrame(read_parquet(path));
vT = df.date;

# define starting values
par0 = [0.1, 0.01, 0.5];

# define related nll function
if length(pars) == 3
    f(x) = nll_exp(transpars(x, intervals, back = true), vT)/length(vT);
else
    f(x) = nll_pwr(transpars(x, intervals, back = true), vT)/length(vT);
end

# one run
opt, func = run_optimization(par0, intervals, vT, f)
results = get_results(opt, func, intervals, vT, varnames)

# output to latex
using Latexify
latexify(results, env=:table, fmt="%.3f")

# plot results
plot_hawkes(vT[begin:100], transpars(Optim.minimizer(opt), intervals, back = true))




##########
# ROBUSTNESS CHECK (vary initial guess)

optlist = []
parlist = []
funclist = []

# create grid of value to optimize over
pargrid = Iterators.product(range(0.1, 2.0, 10), range(0.1, 0.9, 10), range(0.1, 5.0, 10))

for par0 in pargrid
    println("optimize using $par0")
    try
        opt, func = run_optimization(par0, intervals, vT, f)
    catch
        opt, func = ["error", "error"]
    end

    push!(optlist, opt)
    push!(parlist, par0)
    push!(funclist, func)
end



get_results(optlist[1], funclist[1], intervals, vT, varnames)

