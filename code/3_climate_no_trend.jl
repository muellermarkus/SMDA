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
    ??_exp(??::Float64, ??::Float64, ??::Float64)

Exponential kernel for Hawkes process.

# Inputs
    - ?? ??? 0 
    - ?? > 0
    - ?? < ??
    - ??   inter-arrival time
"""
function ??_exp(??, ??, ??)
    ?? * exp(-?? * ??)
end


"""
    ??_power(??::Float64, ??::Float64, ??::Float64, ??::Float64)

Power-law kernel for Hawkes process.

# Inputs
    - ?? ??? 0 
    - ??,?? > 0
    - ?? < ?? ??^??
    - ??   inter-arrival time
"""
function ??_power(??, ??, ??, ??)
    ?? / (?? + ??)^(?? + 1)
end


"""
    ??(?????::Float64, t::Float64, vT; right = false, ??::Float64, ??::Float64[, ??::Float64])

Compute the intensity of a Hawkes process with the exponential or power-law kernel.

If ?? is specified, use power-law kernel.

# Inputs
        ?????          baseline intensity
        t           current time in process
        vT          vector of previous event times
        right       true if computing limit from right side
        ??, ??, ??        kernel parameters

# Output
        ??(t)        the intensity of the Hawkes process at time t
"""
function ??_exp(?????, t, vT; right = false, ??, ??)

    ????? = 0.0

    if right # compute limit from the right
        for T in vT
            if t-T ??? 0
                ????? += ??_exp(??, ??, t-T)
            else
                ????? += 0
            end
        end
    else
        for T in vT
            if t-T > 0 
                ????? += ??_exp(??, ??, t-T)
            else
                ????? += 0
            end 
        end
    end

    return ????? + ?????
end



function ??_pwr(?????, t, vT; right = false, ??, ??, ??)

    ????? = 0.0

    if right # compute limit from the right
        for T in vT
            if t-T ??? 0
                ????? += ??_power(??, ??, ??, t-T)
            else
                ????? += 0
            end
        end
    else
        for T in vT
            if t-T > 0 
                ????? += ??_power(??, ??, ??, t-T)
            else
                ????? += 0
            end 
        end
    end

    return ????? + ?????
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

    # convert ??_fraction to alpha
    if length(pars) == 3
        ?????, ??_fraction, ?? = pars
        ?? = ??_fraction * ??
    else
        ?????, ??_fraction, ??, ?? = pars
        ?? = ??_fraction * ?? * ??^??
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
            ????? = ??_exp(?????, T, vT, right = true, ?? = ??, ?? = ??)
        else
            ????? = ??_pwr(?????, T, vT, right = true, ?? = ??, ?? = ??, ?? = ??)
        end

        # draw interarrival time
        exp_dist = Exponential(1/?????)
        ?? = rand(exp_dist)

        # update current time
        T += ??

        # draw random uniform s for reject sampling
        s = rand()

        # compute new ??(T)
        if length(pars) == 3
            ????? = ??_exp(?????, T, vT, right = false, ?? = ??, ?? = ??)
        else
            ????? = ??_pwr(?????, T, vT, right = false, ?? = ??, ?? = ??, ?? = ??)
        end

        # check if accept sample
        if s ??? (????? / ?????)
            push!(vT, T)
            n_events += 1
        end
    end
    return vT
end

function plot_hawkes(vT, pars)

    # convert ??_fraction to alpha
    if length(pars) == 3
        ?????, ??_fraction, ?? = pars
        ?? = ??_fraction * ??
    else
        ?????, ??_fraction, ??, ?? = pars
        ?? = ??_fraction * ?? * ??^??
    end

    # generate grid
    maxT = ceil(maximum(vT))
    t_grid = collect(LinRange(0.0, maxT, 1000))

    # concatenate with vT
    t_grid = vcat(t_grid, vT)
    sort!(t_grid)
    unique!(t_grid)

    # get intensities for events
    ??_events = []

    for T in vT
        prev_times = [time for time in vT if time ??? T]
        if length(pars) == 3
            ????? = ??_exp(?????, T, prev_times, right = true, ?? = ??, ?? = ??)
        else
            ????? = ??_pwr(?????, T, prev_times, right = true, ?? = ??, ?? = ??, ?? = ??)
        end
        push!(??_events, ?????)
    end

    # get intensities for grid points
    ??_grid = []

    for T in t_grid
        prev_times = [time for time in vT if time ??? T]
        if length(pars) == 3
            ????? = ??_exp(?????, T, prev_times, right = true, ?? = ??, ?? = ??)
        else
            ????? = ??_pwr(?????, T, prev_times, right = true, ?? = ??, ?? = ??, ?? = ??)
        end
        push!(??_grid, ?????)
    end

    # construct plot
    plot(t_grid, ??_grid)
    plot!(vT, ??_events, seriestype = :scatter, leg = false)
    xlabel!("time t (15 min intervals)")
    ylabel!("??(t)")
end


function nll_exp(pars, vT)
    last_T = vT[end]
    ???log?? = 0.0
    integral_part = 0.0

    # construct alpha parameter
    ?????, ??_fraction, ?? = pars
    ?? = ??_fraction * ??

    # compute sum log lambda
    for T in vT
        prev_T = [time for time in vT if time < T]
        ???log?? += log(??_exp(?????, T, prev_T, ?? = ??, ?? = ??))
    end

    # compute integral (compensator function)
    for T in vT
        integral_part += ??/?? * (1-exp(-??*(last_T - T)))
    end

    return -(???log?? - ?????*last_T - integral_part)
end

function nll_pwr(pars, vT)
    last_T = vT[end]
    ???log?? = 0.0
    integral_part = 0.0

    # construct alpha parameter
    ?????, ??_fraction, ??, ?? = pars
    ?? = ??_fraction * ?? * ??^??

    # compute sum log lambda
    for T in vT
        prev_T = [time for time in vT if time < T]
        ???log?? += log(??_pwr(?????, T, prev_T, ?? = ??, ?? = ??, ?? = ??))
    end

    # compute integral (compensator function)
    for T in vT
        integral_part += (1.0 / (??^??) - 1.0/((last_T - T + ??)^??)) * ??/??
    end

    return -(???log?? - ?????*last_T - integral_part)
end

function get_results(opt, func, intervals, vT, varnames; original_pars = nothing)
    est?? = transpars(Optim.minimizer(opt), intervals, back = true)
    # use delta method to get standard errors
    # as I used transformed parameters
    num_hess = -length(vT) .* hessian!(func, Optim.minimizer(opt))
    inv_num_hess = -inv(num_hess)
    num_jacob = ForwardDiff.jacobian(x -> transpars(x, intervals, back = true), Optim.minimizer(opt))
    var_cov = num_jacob * inv_num_hess * transpose(num_jacob)
    se = sqrt.(diag(var_cov))
    ttest = est?? ./ se
    normal_dist = Normal()
    pval = 2*pdf.(normal_dist, -abs.(ttest))
    if isnothing(original_pars)
        results = DataFrame(name = varnames, est?? = est??, se = se, ttest = ttest, pval = pval)
    else
        results = DataFrame(name = varnames, ?? = original_pars, est?? = est??, se = se, ttest = ttest, pval = pval)
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
????? = 0.1
??_fraction = 0.1
?? = 0.8

# specify array of parameters and parameter names
varnames = ["?????", "??_fraction", "??"]
pars = [?????, ??_fraction, ??]

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
par0 = [0.1, 0.5, 0.5];

# define related nll function
if length(par0) == 3
    f(x) = nll_exp(transpars(x, intervals, back = true), vT)/length(vT);
else
    f(x) = nll_pwr(transpars(x, intervals, back = true), vT)/length(vT);
end

# one run
opt, func = run_optimization(par0, intervals, vT, f)
results = get_results(opt, func, intervals, vT, varnames)

# save parameters
climate_pars = transpars(Optim.minimizer(opt), intervals, back = true)

# output to latex
using Latexify
latexify(results, env=:table, fmt="%.3f")

# plot results
plot_hawkes(vT[begin:100], transpars(Optim.minimizer(opt), intervals, back = true))

# save figure
path = joinpath(dirname(dirname(@__FILE__)), "output", "climate_plot.png")
savefig(path)

##########
# ROBUSTNESS CHECK (vary initial guess)

# optlist = []
# parlist = []
# funclist = []

# # create grid of value to optimize over
# pargrid = Iterators.product(range(0.1, 1.0, 5), range(0.1, 0.9, 5), range(0.1, 2.0, 5))

# for par0 in pargrid
#     println("optimize using $par0")
#     try
#         opt, func = run_optimization(par0, intervals, vT, f)
#     catch
#         opt, func = ["error", "error"]
#     end

#     push!(optlist, opt)
#     push!(parlist, par0)
#     push!(funclist, func)
# end

# for (i, opt) in enumerate(optlist)
#     try
#         res = get_results(opt, funclist[i], intervals, vT, varnames)
#         print(res)
#     catch
#         println("estimation not successful for parameters $(parlist[i])")
#     end
# end    

##########
# PREP DATA FOR EV HAWKES PROCESS

# load electricvehicle data
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "electricvehicles.gzip");
df = DataFrame(read_parquet(path));
vT_ev = df.date;

# compute climate process intensity at ev time points
?????, ??_fraction, ?? = climate_pars
?? = ??_fraction * ??

climate_intensities = []
for T in vT_ev
    prev_T = [time for time in vT if time < T]
    push!(climate_intensities, ??_exp(?????, T, prev_T, ?? = ??, ?? = ??))
end

# compute integral of climate process at ev time points
last_T = vT_ev[end]
climate_integral = 0.0
for T in vT_ev
    climate_integral +=  ??/?? * (1-exp(-??*(last_T - T)))
end
climate_integral += ????? * last_T

# save to txt files
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "climate_intensities.txt");
writedlm(path, climate_intensities)

path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "climate_integral.txt");
writedlm(path, climate_integral)

##########
# PREP DATA FOR CAR HAWKES PROCESS

# load electricvehicle data
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "carcomments.gzip");
df = DataFrame(read_parquet(path));
vT_ev = df.date;

# compute climate process intensity at ev time points
?????, ??_fraction, ?? = climate_pars
?? = ??_fraction * ??

climate_intensities = []
for T in vT_ev
    prev_T = [time for time in vT if time < T]
    push!(climate_intensities, ??_exp(?????, T, prev_T, ?? = ??, ?? = ??))
end

# compute integral of climate process at ev time points
last_T = vT_ev[end]
climate_integral = 0.0
for T in vT_ev
    climate_integral +=  ??/?? * (1-exp(-??*(last_T - T)))
end
climate_integral += ????? * last_T

# save to txt files
path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "climate_intensities_car.txt");
writedlm(path, climate_intensities)

path = joinpath(dirname(dirname(@__FILE__)), "data", "processed", "climate_integral_car.txt");
writedlm(path, climate_integral)