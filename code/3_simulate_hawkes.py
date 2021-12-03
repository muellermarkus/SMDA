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

def homogenous_hawkes_process(lambda0, kernel, t, vT, right = False, **kargs):
    """
    Compute the intensity of a hawkes process given time t and previous event times vT
    
    lambda0: baseline intensity (here homogenous, constant over time)
    kernel: kernel function, power law or exp decay
    t: current time
    vT: times of previous events
    right: set True if you want get limit from the right side and account for t-T = 0
    
    """
    kernel_sum = 0.0
    
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
    theta[1] = theta[1] * theta[3] * theta[2] ** theta[3]
    
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

theta = [.1, .56568, .5, .5]     # Parameters lambda, alpha-frac, delta, eta, where alpha= alpha-frac*eta*delta^eta
vP = theta



T =2.0
alpha = theta[1] * theta[3] * theta[2] ** theta[3]
vT = [0.0,1.0] # for my method have to remove current t from vT!

homogenous_hawkes_process(theta[0], power_law_kernel, T, vT, right = True, alpha = alpha, delta = theta[2], eta = theta[3])

vT = [0.0,1.0,2.0] # for charles method have to include current time in vT!
IntensityHawkes(np.array([T]), vP, vT, right= True)






N = 2009
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
        sum_list.append(np.log(homogenous_hawkes_process(theta[0], power_law_kernel, T, prev_T, right = False, alpha = alpha, delta = theta[2], eta = theta[3])))
        
        # this part gives the exact same results as Charles, holds to right = true and right = false
        
    sum_log_lambda = np.sum(sum_list)

    # compute integral lambda(t)dt (compensator function)
    last_T = vT[-1] # can also set last_T to end of observation horizon
     
    sum_list = []
    
    for T in vT:
        sum_part = 1.0/(theta[2] ** theta[3]) - 1.0/((last_T - T + theta[2]) ** theta[3])
        sum_list.append(alpha/theta[3] * sum_part) 
    
    integral_part = theta[0]*last_T + np.sum(sum_list)
    
    nll = -(sum_log_lambda - integral_part)
    
    return nll
    

theta = [.1,  0.565685424949238, .5, .5]

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
def TransformBack(P_tr):
    l0 = np.exp(P_tr[0]) + 0 
    aux= np.exp(P_tr[1]) / (1+np.exp(P_tr[1]))
    alphaf = (1 - 0)*aux + 0
    aux= np.exp(P_tr[2]) / (1+np.exp(P_tr[2]))
    delta = (2 - 0)*aux + 0
    eta = np.exp(P_tr[3]) + 0
    
    return [l0, alphaf, delta, eta]

test = TransformBack(theta0)

alphatest = test[1] * test[3] * test[2] ** test[3]
assert alphatest < test[3] * test[2] ** test[3]

lik_func = lambda P_tr: nll(TransformBack(P_tr), vT)/len(vT)

lik_model = opt.minimize(lik_func, theta0,
                     method='L-BFGS-B', options={'disp': True})







TransformBack(opt.OptimizeResult(lik_model)["x"])
TransformBack(theta0)





