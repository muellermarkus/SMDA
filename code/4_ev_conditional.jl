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
using DelimitedFiles

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
            continue
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



function plot_hawkes(vT, pars, climate_intensities)


    λ₀, α_fraction, δ, β = pars
    α = α_fraction * δ

    # generate grid
    maxT = ceil(maximum(vT))
    t_grid = collect(LinRange(0.0, maxT, 1000))

    # concatenate with vT
    t_grid = vcat(t_grid, vT)
    sort!(t_grid)
    unique!(t_grid)

    # get intensities for events
    λ_events = []

    for (i,T) in enumerate(vT)
        prev_times = [time for time in vT if time ≤ T]
        λₜ = λ_exp(λ₀, T, prev_times, right = true, α = α, δ = δ) + β * climate_intensities[i]

        push!(λ_events, λₜ)
    end

    # get intensities for grid points
    λ_grid = []

    for (i,T) in enumerate(t_grid)
        prev_times = [time for time in vT if time ≤ T]
        λₜ = λ_exp(λ₀, T, prev_times, right = true, α = α, δ = δ) + β * climate_intensities[i]

        push!(λ_grid, λₜ)
    end

    # construct plot
    plot(t_grid, λ_grid)
    plot!(vT, λ_events, seriestype = :scatter, leg = false)
    xlabel!("time t (15 min intervals)")
    ylabel!("λ(t)")
end


function nll_exp(pars, vT, climate_integral, climate_intensities)
    last_T = vT[end]
    ∑logλ = 0.0
    integral_ev_part = 0.0

    # construct alpha parameter
    λ₀, α_fraction, δ, β = pars
    α = α_fraction * δ

    # compute sum log lambda
    for (i,T) in enumerate(vT)
        prev_T = [time for time in vT if time < T]

        ∑logλ += log(λ_exp(λ₀, T, prev_T, α = α, δ = δ) + β * climate_intensities[i])
    end

    # compute integral (compensator function)
    for T in vT
        integral_ev_part += α/δ * (1-exp(-δ*(last_T - T)))
    end

    return -(∑logλ - λ₀*last_T - integral_ev_part - β * climate_integral)
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
    options = Optim.Options(show_trace = true, show_every = 1, iterations = 300, g_tol = 1e-5, f_tol = 2.2e-9)
    opt = optimize(func, par0_tr, LBFGS(; linesearch = LineSearches.HagerZhang(linesearchmax = 30)), options)
    print(opt)

    return opt, func
end


##########
# ESTIMATE PARAMETERS ON EV DATA CONDITION ON CLIMATE PROCESS

# load EV data
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "electricvehicles.gzip");
df = DataFrame(read_parquet(path));
vT = df.date;

# load climate precomputed bits
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "climate_intensities.txt");
climate_intensities = vec(readdlm(path));

path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "climate_integral.txt");
climate_integral = readdlm(path)[1];

# define starting values
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "ev_start_values.txt");
ev_start = vec(readdlm(path));

par0 = vcat(ev_start, 0.5)

# par0 = [0.1, 0.5, 1.5, 0.1];

varnames = ["λ₀", "α_fraction", "δ", "β"];
intervals = [[0,Inf], [0,1], [0,Inf], [0, Inf]];

# define related nll function
f(x) = nll_exp(transpars(x, intervals, back = true), vT, climate_integral, climate_intensities)/length(vT);

# one run
opt, func = run_optimization(par0, intervals, vT, f);
results = get_results(opt, func, intervals, vT, varnames)

# output to latex
using Latexify
latexify(results, env=:table, fmt="%.3f")

# plot results
plot_hawkes(vT[begin:100], transpars(Optim.minimizer(opt), intervals, back = true), climate_intensities)

# save figure
path = joinpath(dirname(dirname(@__FILE__)), "output", "conditional_ev_plot.png")
savefig(path)