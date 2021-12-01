import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as st

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

def homogenous_hawkes_process(lambda0, kernel, t, vT, right = False, **kargs):
    """
    Compute the intensity of a hawkes process given time t and previous event times vT
    
    lambda0: baseline intensity (here homogenous, constant over time)
    kernel: kernel function, power law or exp decay
    t: current time
    vT: times of previous events
    right: set True if you want get limit from the right side and account for t-T = 0
    
    """
    kernel_sum = 0
    
    if right:
        time_list = np.concatenate((vT, [t]))
    else:
        time_list = vT.copy()
        
    for T in time_list:
        kernel_sum += kernel(t-T, **kargs)
        
    return lambda0 + kernel_sum


# Q: how to make baseline intensity change over time? take increasing function like log?


    
# simulate from hawkes process

# init current time to 0


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
    # theta[1] = theta[1] * theta[3] * theta[2] ** theta[3]
    
    theta[1] = theta[1] * theta[2]
    
    # init counters and output list
    # vT = [0.25, 0.33, 0.44, 1.0, 2.0, 2.5, 2.55]
    # n_events = 1
    # T = 10.0
    # dt = T
    # vt = dt

    # IntensityHawkes(dt, vP, vT) # gives exact same results
    
    vT = []
    n_events = 0
    T = 0

    while (n_events < N) and (T < T_max):
        # set upper bound
        Lambda_upper_bound = homogenous_hawkes_process(theta[0], power_law_kernel, T, vT, right = True, alpha = theta[1], delta = theta[2], eta = theta[3])
        
        # homogenous_hawkes_process(theta[0], power_law_kernel, T, vT, right = False, alpha = theta[1], delta = theta[2], eta = theta[3]) # NO DIFFERENCE

        # sample interarrival time
        # tau = -np.log(np.random.uniform())/Lambda_upper_bound
        tau = st.expon.rvs(scale= 1/Lambda_upper_bound)
        
        # update current time
        T += tau
        
        # draw s to see if sample is accepted
        s = np.random.uniform()
    
        # compute new lambda(T)
        LambdaT = homogenous_hawkes_process(theta[0], power_law_kernel, T, vT, alpha = theta[1], delta = theta[2], eta = theta[3])
        
        if s <= LambdaT / Lambda_upper_bound:
            # accept sample
            vT.append(T)
            n_events += 1
    
    return vT


# run simulation
np.random.seed(1234)

theta = [.1, .4, .5, .5]     # Parameters lambda, alpha-frac, delta, eta, where alpha= alpha-frac*eta*delta^eta
N = 200
T = 10000

vT_markus = simulate_hawkes(theta, N, T)

# my numbers always start at >8, yours at >2
# your numbers tend to have greater differences in between


# plot 
iN= len(vT_markus)
iT= np.ceil(max(vT_markus))
vt= np.arange(0, iT, iT/1000)

# concatenate times
vt = np.unique(np.concatenate((vt, vT_markus)))
vt = np.sort(vt)

# adjust T so that intensities for events are plotting as right limit
# df.loc[:, "T"] = df["T"] + 1e-6 # NOT WORKING?!

event_intensities = []

for i,T in enumerate(vT_markus):
    prev_times = vT_markus[0:i]
    l = homogenous_hawkes_process(theta[0], power_law_kernel, T, prev_times, right = True, alpha = theta[1], delta = theta[2], eta = theta[3])
    event_intensities.append(l)

# get intensity function values in between events
other_intensities = []

for T in vt:
    prev_times = [time for time in vT_markus if time < T]
    l = homogenous_hawkes_process(theta[0], power_law_kernel, T, prev_times, right = True, alpha = theta[1], delta = theta[2], eta = theta[3])
    other_intensities.append(l)
    
plt.figure()
plt.plot(vt, other_intensities)
plt.plot(vT_markus, event_intensities, 'o')


# save to file
# data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
# df.to_csv(data_path + '/simulations/hawkes.csv', index = False)
    
# HOW TO SAMPLE INTERDEPENDENT PROCESSES? ONE DEPENDENT ON THE OTHER SHOULD BE POSSIBLE -> first sample climate change process, then define linear model and from this sample electric cars process


# max lik estimation
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
    
    alpha = theta[1] * theta[3] * theta[2] ** theta[3]
    
    # compute sum log lambda(T_i)
    sum_list = []
    
    for i,T in enumerate(vT):
        
        prev_T = [time for time in vT if time < T]
        sum_list.append(np.log(homogenous_hawkes_process(theta[0], power_law_kernel, T, prev_T, alpha = alpha, delta = theta[2], eta = theta[3])))
        
        # how to ensure assertion holds when optimizing? alpha < eta * delta ** eta
        
    sum_log_lambda = np.sum(sum_list)
    
    # compute integral lambda(t)dt (compensator function)
    last_T = vT[-1]
     
    sum_list = []
    
    for T in vT:
        if T == last_T:
            sum_list.append(0.0)
        sum_part = ( 1/ ((-theta[2]) ** theta[3]) - 1/((last_T - T - theta[2]) ** theta[3])) / theta[3]
        sum_list.append(theta[0] * last_T + alpha * sum_part)
        
        # problem now: last_T - Ti is zero for last observation, so taking negative number to negative exponent --> nan
        # is T = last T in paper? or when observation period ended? e.g. could set at last_T + 1
        
    integral_part = np.sum(sum_list)
    
    nll = -(sum_log_lambda - integral_part)
    
    return nll
    
    
    
from scipy.optimize import minimize, Bounds




# adtPar= [{'name': 'l0', 'p': 1, 'trans': [0, np.inf]},
#              {'name': 'alphaf', 'p': .2, 'trans': [0, 1]},
#              {'name': 'delta', 'p': .5, 'trans': [0, 2]},
#              {'name': 'eta', 'p': .5, 'trans': [0, np.inf]}]

# tranform initial parameters similar to dict above


l0 = theta[0]
alphaf = theta[1]
delta = theta[2]
eta = theta[3]


# restrict lambda to interval (0,inf)
# right bound, < inf
l0_tr = np.log(l0 - 0)

# restrict alpha fraction to be between 0 and 1
aux = (alphaf - 0)/(1- 0)
alphaf_tr = np.log(aux / (1-aux))

# restrict delta to be between 0 and 2
aux = (delta- 0)/(2- 0)
delta_tr = np.log(aux / (1-aux))

# restrict eta to be between 0 and inf
eta_tr = np.log(eta - 0)

# construct parameter vector
theta0 = [l0_tr, alphaf_tr, delta_tr, eta_tr]



# ensure parameters are transformed back when passed to lik
def TranformBack(P_tr):
    l0 = np.exp(P_tr[0]) + 0 
    aux= np.exp(theta[1]) / (1+np.exp(theta[1]))
    alphaf = (1 - 0)*aux + 0
    aux= np.exp(theta[2]) / (1+np.exp(theta[2]))
    delta = (2 - 0)*aux + 0
    eta = np.exp(P_tr[3]) + 0
    
    return [l0, alphaf, delta, eta]

TranformBack(theta0)


lik_func = lambda P_tr: nll(TranformBack(P_tr), vT_markus)



lik_model = minimize(lik_func, theta0,
                     method='L-BFGS-B', options={'disp': True})

# assertion error!!!



