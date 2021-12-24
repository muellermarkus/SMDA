import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt

def exp_decay(x, alpha, delta):
    '''
    Exponential function which is used as a monotonically decreasing kernel in Hawkes process
    
    alpha >= 0
    delta > 0
    alpha < delta
    '''
    return alpha * np.exp(-delta * x)

def power_law_kernel(x, alpha = 0.2, delta = 0.5, eta = 0.5):
    '''
    Power-law kernel
    
    alpha >= 0
    delta > 0
    eta > 0
    alpha < eta * delta ^ eta
    '''
    
    assert alpha < eta * delta ** eta

    return alpha / (x + delta) ** (eta + 1)

def homogenous_hawkes_process(lambda0, lambda1, kernel, t, vT, right = False, **kargs):
    """
    Compute the intensity of a hawkes process given time t and previous event times vT
    
    lambda0: baseline intensity (here homogenous, constant over time)
    kernel: kernel function, power law or exp decay
    t: current time
    vT: times of previous events
    right: set True if you want get limit from the right side and account for t-T = 0
    
    """
    kernel_sum = 0.0
    
    for T in vT:
        if right:
            if t-T >= 0:
                kernel_sum += kernel(t-T, **kargs)
            else:
                kernel_sum += 0
        else:
            if t-T > 0:
                kernel_sum += kernel(t-T, **kargs)
            else:
                kernel_sum += 0

    return lambda0 + lambda1*t + kernel_sum


def simulate_hawkes(theta, N, T_max):
    '''
    simulate event times of hawkes process

    theta: vector of parameters,
    N: maximum number of events to draw
    T_max: maximum total time to simulate

    Output:
        vT: vector of event times
    '''
    
    # prep parameters
    pars = theta.copy()
    if len(pars) == 5:
        pars[2] = pars[2] * pars[4] * pars[3] ** pars[4]
    else:
        pars[2] = pars[2] * pars[3]
        
    # scale lambda1 as fraction of lambda0 that trend increases over a year
    pars[1] = pars[0] * pars[1]  / (365 * 24 * 4)
    
    # init data structures
    vT = []
    n_events = 0
    T = 0 

    while (n_events < N) and (T < T_max):
        # set upper bound
        if len(pars) == 5:
            Lambda_upper_bound = homogenous_hawkes_process(pars[0], pars[1], power_law_kernel, T, vT, right = True, alpha = pars[2], delta = pars[3], eta = pars[4])
        else:
            Lambda_upper_bound = homogenous_hawkes_process(pars[0], pars[1], exp_decay, T, vT, right = True, alpha = pars[2], delta = pars[3])

        # sample interarrival time
        tau = st.expon.rvs(scale= 1/Lambda_upper_bound)
        
        # update current time
        T += tau
        
        # draw s to see if sample is accepted
        s = np.random.uniform()
    
        # compute new lambda(T)
        if len(pars) == 5:
            LambdaT = homogenous_hawkes_process(pars[0], pars[1], power_law_kernel, T, vT, alpha = pars[2], delta = pars[3], eta = pars[4])
        else:
            LambdaT = homogenous_hawkes_process(pars[0], pars[1], exp_decay, T, vT, alpha = pars[2], delta = pars[3])
        
        if s <= LambdaT / Lambda_upper_bound:
            # accept sample
            vT.append(T)
            n_events += 1
    
    return vT


def plot_hawkes(theta, vT):
    
    # prep alpha parameter
    pars = theta.copy()
    if len(pars) == 5:
        pars[2] = pars[2] * pars[4] * pars[3] ** pars[4]
    else:
        pars[2] = pars[2] * pars[3]
        
    # scale lambda1 as fraction of lambda0 that trend increases over a year
    pars[1] = pars[0] * pars[1]  / (365 * 24 * 4)
    
    # generate grid of data points
    iN= len(vT)
    iT= np.ceil(max(vT))
    vt= np.arange(0, iT, iT/1000)

    # concatenate times
    vt = np.unique(np.concatenate((vt, vT)))
    vt = np.sort(vt)

    event_intensities = []

    for T in vT:
        prev_times = [time for time in vT if time <= T]
        if len(pars) == 5:  
            l = homogenous_hawkes_process(pars[0], pars[1], power_law_kernel, T, prev_times, right = True, alpha = pars[2], delta = pars[3], eta = pars[4])
        else:
            l = homogenous_hawkes_process(pars[0], pars[1], exp_decay, T, prev_times, right = True, alpha = pars[2], delta = pars[3])
        event_intensities.append(l)

    # get intensity function values in between events
    other_intensities = []

    for T in vt:
        prev_times = [time for time in vT if time <= T]
        if len(pars) == 5:
            l = homogenous_hawkes_process(pars[0], pars[1], power_law_kernel, T, prev_times, right = True, alpha = pars[2], delta = pars[3], eta = pars[4])
        else:
            l = homogenous_hawkes_process(pars[0], pars[1], exp_decay, T, prev_times, right = True, alpha = pars[2], delta = pars[3])
        other_intensities.append(l)
        
    # create plot
    plt.figure()
    plt.plot(vt, other_intensities)
    plt.plot(vT, event_intensities, 'o')

def nll(theta, vT):
    '''
    Negative log-likelihood
    
    Baseline intensity
    theta[0] = lambda0
    
    Kernel parameters
    theta[1] = alpha-fraction
    theta[2] = delta
    theta[3] = eta
    
    vT: vector of event times
    
    '''
    
    # prep alpha parameter
    alpha = theta[2] * theta[4] * theta[3] ** theta[4]
    
    # scale lambda1 as fraction of lambda0 that trend increases over a year
    theta[1] = theta[0] * theta[1]  / (365 * 24 * 4)
    
    # compute sum log lambda(T_i)
    sum_list = []
    
    for i,T in enumerate(vT):
        
        prev_T = [time for time in vT if time < T]
        sum_list.append(np.log(homogenous_hawkes_process(theta[0], theta[1], power_law_kernel, T, prev_T, right = False, alpha = alpha, delta = theta[3], eta = theta[4])))
        
        # this part gives the exact same results as Charles, holds to right = true and right = false
        
    sum_log_lambda = np.sum(sum_list)

    # compute integral lambda(t)dt (compensator function)
    last_T = vT[-1]
     
    sum_list = []
    
    for T in vT:
        sum_part = 1.0/(theta[3] ** theta[4]) - 1.0/((last_T - T + theta[3]) ** theta[4])
        sum_list.append(alpha/theta[4] * sum_part) 
    
    integral_part = theta[0]*last_T + theta[1]*(last_T**2)/2 + np.sum(sum_list)
    
    nll = -(sum_log_lambda - integral_part)
    
    return nll


def nll_exp(theta, vT):
    '''
    Negative log-likelihood
    
    Baseline intensity
    theta[0] = lambda0
    
    Kernel parameters
    theta[1] = alpha-fraction
    theta[2] = delta
    
    vT: vector of event times
    
    '''
    
    # prep alpha parameter
    alpha = theta[2] * theta[3]
    
    # scale lambda1 as fraction of lambda0 that trend increases over a year
    theta[1] = theta[0] * theta[1]  / (365 * 24 * 4)
    
    # compute sum log lambda(T_i)
    sum_list = []
    
    for i,T in enumerate(vT):
        
        prev_T = [time for time in vT if time < T]
        sum_list.append(np.log(homogenous_hawkes_process(theta[0], theta[1], exp_decay, T, prev_T, right = False, alpha = alpha, delta = theta[3])))
        
    sum_log_lambda = np.sum(sum_list)

    # compute integral lambda(t)dt (compensator function)
    last_T = vT[-1] 
    
    sum_list = []
    
    for T in vT:
        sum_part = (alpha / theta[3]) * (1 - np.exp(-theta[3] * (last_T - T)))
        sum_list.append(sum_part) 
    
    integral_part = np.sum(sum_list)
    
    nll = -(sum_log_lambda - theta[0]*last_T - theta[1]*(last_T**2)/2 - integral_part)
    
    return nll

##############################################################################################

# run simulation
np.random.seed(1234)
# theta = [1.2, 0.5, 4.0, 1.2]
theta = [1.2, 0.01, 0.5, 1.0]
N = 4000
T = 100000

vT = simulate_hawkes(theta, N, T)

plot_hawkes(theta, vT[0:21])

# standard errors

# try exponential kernel

# grid of deltas, 5 - 6 different values, 3-4 values for alphas
# collect results, compare if they converge to same points
# collect LLns, and select best if no consensus
# [1.0, 0.5, delta, 1.0]


# save to file
# data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
# df.to_csv(data_path + '/simulations/hawkes.csv', index = False)


def transform(theta):
    # transform parameters for optimization
    l0 = theta[0]
    l1 = theta[1]
    alphaf = theta[2]
    delta = theta[3]
    
    # restrict lambda to interval (0,inf)
    # right bound, < inf
    l0_tr = np.log(l0 - 0)
    
    # restrict l1 fraction to be between 0 and 1
    aux = (l1 - 0)/(1- 0)
    l1_tr = np.log(aux / (1-aux))

    # restrict alpha fraction to be between 0 and 1
    aux = (alphaf - 0)/(1- 0)
    alphaf_tr = np.log(aux / (1-aux))

    # restrict delta to be between 0 and 5
    # aux = (delta- 0)/(5 - 0)
    # delta_tr = np.log(aux / (1-aux))
    delta_tr = np.log(delta - 0) # (0,Inf)
    
    # restrict eta to be between 0 and inf
    if len(theta) == 5:
        eta = theta[4]
        eta_tr = np.log(eta - 0)
        return [l0_tr, l1_tr, alphaf_tr, delta_tr, eta_tr]

    return [l0_tr, l1_tr, alphaf_tr, delta_tr]

# ensure parameters are transformed back when passed to lik
def TransformBack(P_tr):
    l0 = np.exp(P_tr[0]) + 0 
    aux= np.exp(P_tr[1]) / (1+np.exp(P_tr[1]))
    l1 = (1 - 0)*aux + 0
    aux= np.exp(P_tr[2]) / (1+np.exp(P_tr[2]))
    alphaf = (1 - 0)*aux + 0
    # aux= np.exp(P_tr[2]) / (1+np.exp(P_tr[2]))
    # delta = (5 - 0)*aux + 0
    delta = np.exp(P_tr[3]) + 0
    
    if len(P_tr) == 5:
        eta = np.exp(P_tr[4]) + 0
        return [l0, l1, alphaf, delta, eta]
    return [l0, l1, alphaf, delta]

# run optimizer
# lik_func = lambda P_tr: nll(TransformBack(P_tr), vT)/len(vT)
lik_func = lambda P_tr: nll_exp(TransformBack(P_tr), vT)/len(vT)

pars0 = transform(theta)
theta = TransformBack(pars0)
lik_model = opt.minimize(lik_func, pars0,
                     method='L-BFGS-B', options={'disp': True})

lik_func(pars0)
TransformBack(pars0)

# compare to julia
# theta = [1.2, 0.5, 1.0, 0.5]
# theta = [1.2, 0.5, 1.0]
# pars0 = transform(theta)
# pars = TransformBack(pars0)
# vT = [1.5, 20.0, 24.2, 30.0]
# if len(theta) == 4:
#     lik_func = lambda P_tr: nll(TransformBack(P_tr), vT)/len(vT)
# else:
#     lik_func = lambda P_tr: nll_exp(TransformBack(P_tr), vT)/len(vT)
# lik_func(pars0)

# get parameter estimates
TransformBack(opt.OptimizeResult(lik_model)["x"])
# [1.2, 0.5, 1.0] exp
# [1.1964091730218844, 0.48811245338029835, 1.0646455899418472]


# theta = [1.2, 0.5, 4.0, 1.2] pwr
# estimate = [1.2491359667938302,0.4653976151197698,3.6660888661965614,1.4757812642108619]

# [1.2, 0.01, 0.5, 1.0] 6k observations
# [1.1134183704232128,0.010528186094057423,0.48425925588717755,1.0306036547177806]



# optimize over grid
results = []

for delta in [0.2, 0.6, 1.0, 1.4, 1.8]:
    pars0 = transform([1.0, 0.4, delta, 1.0])
    lik_model = opt.minimize(lik_func, pars0,
                     method='L-BFGS-B', options={'disp': True})
    res = TransformBack(opt.OptimizeResult(lik_model)["x"])
    results.append(res)
    



[1.0, 0.5, delta, 1.0]






# EXTENSIONS:

# let lambda for both processes increase with T, which functional form? can let it vary by number of posts in certain window around T
# see https://arxiv.org/pdf/1603.09449.pdf 

# model log lambda0 as a linear regression model with basis functions
# https://arxiv.org/pdf/1702.04443.pdf 

# for lambda0 for disasters just use a rough cubic estimate from the literature?

# let lambda of second process depend on lambda of first (exp(lambda0 + beta lambda_disaster))

# have to increase bounds dramatically, tutorial paper has delta = 250!

# compute lmabda_climate before, include in linear model for lambda_car (alpha + beta * lambda_climate), include CO2 level in atmosphere.

# whether having an event in whether increases intensity of
# does beta0 have an effect?
# run linear trend on CO2 levels, or linear time trend!

# report: show data, where it comes from, why climate change data, intensity plots (huge events, leading toe xtra communication, could be causedby increase co2 or because we aremore ware of this, our thinking  about this does it impact the electricvehilces awareness,does one affect the other?)

# limit memory of past processes to last 3 days, etc.? can speed up code immensely, e.g.  50 days, get approximate likelihood

# compute beforehand integrated lambda climate




##############################################################################################
# ESTIMATE PROCESS FOR DISASTERS

# load data for estimation
main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
disasters = pd.read_parquet(main_path + '/data/processed/disasters.gzip')

# create vT list
vT_dis = disasters['date'].tolist()

# model disasters as a simple hawkes process, no influence of upvotes, etc.
lik_func = lambda P_tr: nll_exp(TransformBack(P_tr), vT_dis)/len(vT_dis)
par0 = transform([0.5,0.1,0.5,0.5])
disaster_model = opt.minimize(lik_func, par0,
                     method='L-BFGS-B', options={'disp': True})

dis_par = TransformBack(opt.OptimizeResult(disaster_model)["x"])

# save parameters
with open(main_path + '/output/results/disaster_parameters.txt', 'w') as f:
    for parameter in dis_par:
        f.write(str(parameter) + '\n')
        
# load parameters
with open(main_path + '/output/results/disaster_parameters.txt', 'r') as f:
    test = f.read().splitlines()

# plot first 20 observations
first20 = vT_dis[0:21]

plot_hawkes(dis_par, first20)
plt.savefig(main_path + 'output/disaster_first20.png', bbox_inches='tight', dpi=300)



##############################################################################################
# ESTIMATE PROCESS FOR 

