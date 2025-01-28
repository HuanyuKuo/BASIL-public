# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:58:27 2021

@author: huanyu

Test liklihood function

0209.2021:
    NUM_SAMPLE = 10000 -> NUM_SAMPLE = 5000
    sol = minimizeCompass(func= neg_likelihood_function_global_variable_multiprocessing, bounds=bounds, x0 =  x0, args=(glob, lins), feps=1e-3, deltatol=0.1, paired=False)
    => sol = minimizeCompass(func= neg_likelihood_function_global_variable_multiprocessing, bounds=bounds, x0 =  x0, args=(glob, lins), feps=1e-2, deltatol=0.1, paired=False)
    errorcontrol =True -> errorcontrol = False

"""

from multiprocessing import Process, Queue
import numpy as np
import scipy
import scipy.stats
#from scipy import optimize
from noisyopt import minimizeCompass

#from matplotlib import pyplot as plt
import myConstant as mc

LINEAGE_TAG = mc.LINEAGE_TAG #{'UNK': 'Unknown', 'NEU': 'Neutral', 'ADP': 'Adaptive'}
NUMBER_OF_PROCESSES = mc.NUMBER_OF_PROCESSES
NUM_SAMPLE = 5000
ERRORCONTROL = False
#=====================================
#
#Two dimensional likelihood function
#
#====================================
'''
def maximum_likelihood_function_global_variable2d(glob, lins, const, t):
    
    r0 = 0
    r1 = 0
    for lin in lins:
        if lin.TYPETAG == LINEAGE_ TAG['ADP']:
            r0 += lin.r0 * np.exp(lin.sm.post_parm_NormS_mean*glob.C)
            r1 += lin.r1
        else:
            r0 += lin.r0
            r1 += lin.r1
            
    f1 = r1/const.Rt[t]
    f0 = r0/const.Rt[t-1]
    meanfitness_ini =   -(np.log(f1) - np.log(f0))/glob.C
    log10eps_min = min(-1 , -1.*np.log10(glob.R) + 1)
    log10eps_max = 5
    log10eps_arr = np.arange(log10eps_min,log10eps_max,(log10eps_max-log10eps_min)/20)
    
    #eps_max = max(np.log10(glob.R)-1, 0.1)
    #eps_arr = [10**(-i*eps_max/20) for i in range(20)]
    llk_arr = []
    
    for log10eps in log10eps_arr:
        nllk = neg_likelihood_function_global_variable_multiprocessing2d([meanfitness_ini, log10eps], glob, lins)
        llk_arr.append(-1.*nllk)
    
    #print(eps_arr, llk_arr)
    log10eps_ini = log10eps_arr[llk_arr.index(max(llk_arr))]
    
    sol = []
    x0 = [meanfitness_ini , log10eps_ini]
    bounds = [[-1*const.Dt[t], const.Dt[t]], [log10eps_min, log10eps_max]]
    sol = minimizeCompass(func= neg_likelihood_function_global_variable_multiprocessing2d, 
                          bounds=bounds, x0 =  x0, args=(glob, lins), 
                          feps=0.001, deltatol=0.01, errorcontrol =ERRORCONTROL, funcNinit=30, paired=False)
    print(sol)
    
    return sol
'''

def maximum_likelihood_function_global_variable2d(glob, lins, const, t):
    r0 = 0
    r1 = 0
    alpha = 3
    for lin in lins:
        s_mean = lin.sm.post_parm_NormS_mean
        s_std  = np.sqrt(lin.sm.post_parm_NormS_var)

        if abs(s_mean)-alpha*s_std<0:
            lin.TYPETAG = LINEAGE_TAG['NEU']
            k_NModel = lin.sm.post_parm_Gamma_k/(1+lin.sm.post_parm_Gamma_k*lin.sm.post_parm_Gamma_b**2)
            theta_NModel = np.exp(np.log(lin.sm.post_parm_Gamma_a)+ lin.sm.post_parm_Gamma_b**2/2 -np.log(k_NModel))
            lin.nm.UPDATE_POST_PARM(k=k_NModel, theta=theta_NModel, log_norm=lin.sm.log_noramlization_const, log_prob_survive=lin.sm.log_noramlization_const)
            r0 += lin.r0
            r1 += lin.r1
        else:
            r0 += lin.r0 * np.exp(lin.sm.post_parm_NormS_mean * glob.C)
            r1 += lin.r1

    f1 = r1 / const.Rt[t]
    f0 = r0 / const.Rt[t - 1]
    meanfitness_ini = -(np.log(f1) - np.log(f0)) / glob.C
    log10eps_min = min(-1, -1. * np.log10(glob.R) + 1)
    log10eps_max = 5
    log10eps_arr = np.arange(log10eps_min, log10eps_max, (log10eps_max - log10eps_min) / 20)

    # eps_max = max(np.log10(glob.R)-1, 0.1)
    # eps_arr = [10**(-i*eps_max/20) for i in range(20)]
    llk_arr = []

    for log10eps in log10eps_arr:
        nllk = neg_likelihood_function_global_variable_multiprocessing2d([meanfitness_ini, log10eps], glob, lins)
        llk_arr.append(-1. * nllk)

    # print(eps_arr, llk_arr)
    log10eps_ini = log10eps_arr[llk_arr.index(max(llk_arr))]

    sol = []
    x0 = [meanfitness_ini, log10eps_ini]
    bounds = [[-1 * const.Dt[t], const.Dt[t]], [log10eps_min, log10eps_max]]
    sol = minimizeCompass(func=neg_likelihood_function_global_variable_multiprocessing2d,
                          bounds=bounds, x0=x0, args=(glob, lins),
                          feps=0.001, deltatol=0.01, errorcontrol=ERRORCONTROL, funcNinit=30, paired=False)
    print(sol)

    return sol

def neg_likelihood_function_global_variable_multiprocessing2d(x0, glob, lins):
    meanfitness = x0[0]
    epsilon = 10**(x0[1])

    llk = likelihood_function(meanfitness, epsilon, lins, glob)
    print(meanfitness, epsilon, llk)
    return -1.*llk

#=====================================
#
#One dimensional likelihood function
#
#====================================


def maximum_likelihood_function_epsilon(glob, lins, const, t, meanfitness):
    #
    # get initial guess of meanfitness and epsilon
    #
    log10eps_min = min(-1 , -1.*np.log10(glob.R) + 1)
    sol = []
    x0 = [-1.]
    #bounds = ((0, 10),)
    bounds = ((log10eps_min, 0.),)
    sol = minimizeCompass(func= neg_likelihood_function_epsilon_multiprocessing, 
                          bounds=bounds, x0 = x0, args=(glob, lins, meanfitness), 
                          feps=0.0001, deltatol=0.001, errorcontrol =ERRORCONTROL, funcNinit=30, paired=False)
    print(sol)
    
    return sol

def neg_likelihood_function_epsilon_multiprocessing(x0, glob, lins, meanfitness):
    meanfitness = meanfitness
    epsilon = 10**(x0[0])
    
    llk = likelihood_function(meanfitness, epsilon, lins, glob)
    print(meanfitness, epsilon, llk)
    
    return -1.*llk

def maximum_likelihood_function_meanfitness(glob, lins, const, t, eps):
    #
    # get initial guess of meanfitness and epsilon
    #
    r0 = 0
    r1 = 0
    for lin in lins:
        if lin.TYPETAG == LINEAGE_TAG['ADP']:
            r0 += lin.r0 * np.exp(lin.sm.post_parm_NormS_mean*glob.C)
            r1 += lin.r1
        else:
            r0 += lin.r0
            r1 += lin.r1
            
    f1 = r1/const.Rt[t]
    f0 = r0/const.Rt[t-1]
    meanfitness_ini =   -(np.log(f1) - np.log(f0))/glob.C
    
    sol = []
    x0 = [meanfitness_ini]
    bounds = ((-1, np.inf),)
    sol = minimizeCompass(func= neg_likelihood_function_meanfitness_multiprocessing, 
                          bounds=bounds, x0 = x0, args=(glob, lins, eps), 
                          feps=0.0001, deltatol=0.001, errorcontrol =ERRORCONTROL, funcNinit=30, paired=False)
    print(sol)
    
    return sol

def neg_likelihood_function_meanfitness_multiprocessing(x0, glob, lins, eps):
    meanfitness = x0[0]
    epsilon = eps#x0[1]
    
    llk = likelihood_function(meanfitness, epsilon, lins, glob)
    print(meanfitness, epsilon, llk)
    
    return -1.*llk

#========================================================
#
# Compute likelihood function value for all lineages including NEU & ADP
#
#===========================================================
def likelihood_function(meanfitness, epsilon, lins, glob):
    llk = 0
    lins_NEU = []
    lins_ADP = []
    
    for lin in lins:
        if lin.TYPETAG == LINEAGE_TAG['ADP']:
            lins_ADP.append(lin)
        else:
            lins_NEU.append(lin)
            
    if len(lins_NEU)>0:
        llk += compute_likelihood_multiprocessing(LINEAGE_TAG['NEU'], lins_NEU, glob, meanfitness, epsilon)
    if len(lins_ADP) > 0:
        llk += compute_likelihood_multiprocessing(LINEAGE_TAG['ADP'], lins_ADP, glob, meanfitness, epsilon)
        
    llk /= len(lins)
    
    return llk
    


#===========================================================
# Function apply multiprocessing to run function,
#        in parallel for all lineages in the 'lins' 
#        the posterior information is saved in Queue
#===========================================================        
def compute_likelihood_multiprocessing(lineage_type, lins, glob, meanfitness, epsilon):
    
    #model_name = run_dict['model_name']
    
    Input_data_array = get_Input_data_array(lineage_type, lins, glob, meanfitness, epsilon)
    
    # TASK is a list of task for lineages with function-to-work and input data
    
    if lineage_type == LINEAGE_TAG['NEU']:
        TASKS = [(posterior_constant_neutral_multiporcessing, Input_data_array[i])  for i in range(len(lins))]    
    elif  lineage_type == LINEAGE_TAG['ADP']:
        TASKS = [(posterior_constant_selective_multiporcessing, Input_data_array[i])  for i in range(len(lins))]
    
    # Create queues
    task_queue = Queue()
    done_queue = Queue()
    
    # Submit tasks (put into queue)
    for task in TASKS:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        p=Process(target=worker, args=(task_queue, done_queue))
        p.start()
    
    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')
    
    llk = 0
    # queue to likelihood
    for i in range(len(lins)):
        result = done_queue.get()
        llk += np.log( float(result['p']))
    
    # return  likelihood
    out = llk#/len(lins)
    return out

def get_Input_data_array(lineage_type, lins, glob, meanfitness, epsilon):
    RD = glob.R
    N = glob.N
    dilution_ratio = glob.D
    cycle = glob.C
    
    Input_data_array=[]
    
    if lineage_type == LINEAGE_TAG['ADP']:
        Input_data_array = [{ 'meanfitness': meanfitness, 'epsilon': epsilon, 'k': lin.sm.post_parm_Gamma_k, 'a': lin.sm.post_parm_Gamma_a,
                             'b': lin.sm.post_parm_Gamma_b, 'mean_s': lin.sm.post_parm_NormS_mean, 'var_s': lin.sm.post_parm_NormS_var,
                             'dilution_ratio': dilution_ratio,  'cycle': cycle,'read_depth': RD, 'population_size': N, 
                             'read': lin.r1} for lin in lins]
    else:
        Input_data_array = [{ 'meanfitness': meanfitness, 'epsilon': epsilon, 'k': lin.nm.post_parm_Gamma_k, 'theta': lin.nm.post_parm_Gamma_theta, 
                             'dilution_ratio': dilution_ratio,  'cycle': cycle,'read_depth': RD, 'population_size': N, 
                             'read': lin.r1} for lin in lins]
    
    return Input_data_array

#
# Function run by worker processes
#   
def worker(input_q, output_q):

    for func, args in iter(input_q.get, 'STOP'):
        # here 'func' = MCMC_sampling, 'args' = item of Input_Data_Array
        result = func(args)
        # put returning result into queue
        output_q.put(result) 
        
#=============================#
#
# Main Functions be computed in multiprocessing
#
#=============================#
def posterior_constant_neutral_multiporcessing(input_dict):
    meanfitness = input_dict['meanfitness']
    epsilon = input_dict['epsilon']
    k = input_dict['k']
    theta = input_dict['theta']
    read = input_dict['read']
    D = input_dict['dilution_ratio']
    C = input_dict['cycle']
    R = input_dict['read_depth']
    N = input_dict['population_size']
    
    # if cell number = 0, sample zero reads
    prob_zero_cell, k1, theta1 = extinct_prob_cycle(k, theta, D, C, meanfitness, 0)
    cellnum_arr = np.random.gamma(shape=k1, scale= theta1, size=NUM_SAMPLE)
    
    #n_arr_parameter_observation_NB = cellnum_arr * np.exp( np.log(R)- np.log(N) -np.log(epsilon))
    #p_parameter_observation_NB = 1/(1+epsilon)
    #prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_arr_parameter_observation_NB,p=p_parameter_observation_NB)
    
    n_parameter_observation_NB = 1/epsilon
    ratio = np.exp( np.log(R) - np.log(N) + np.log(epsilon) )
    p_arr_parameter_observation_NB = np.exp(-np.log(1+ cellnum_arr*ratio  ))
    
    prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_parameter_observation_NB,p=p_arr_parameter_observation_NB)
    
    expected_prob_observation = (read==0)*prob_zero_cell + (1-prob_zero_cell)*np.mean(prob_observation_NB)
    p = expected_prob_observation

    return {'p':p}


def posterior_constant_selective_multiporcessing(input_dict):
    meanfitness = input_dict['meanfitness']
    epsilon = input_dict['epsilon']
    read = input_dict['read']
    k = input_dict['k']
    a = input_dict['a']
    b = input_dict['b']
    mean_s  = input_dict['mean_s']
    std_s = np.sqrt(input_dict['var_s'])
    D = input_dict['dilution_ratio']
    C = input_dict['cycle']
    R = input_dict['read_depth']
    N = input_dict['population_size']
    
    #p_parameter_observation_NB = 1/(1+epsilon)
    #ratio = np.exp( np.log(R)- np.log(N) -np.log(epsilon))
    
    s_arr = np.random.normal(loc=mean_s, scale=std_s, size=NUM_SAMPLE)
    theta_arr = np.exp(b*(s_arr-mean_s)/std_s + np.log(a) - np.log(k))
    prob_zero_cell_arr, k1_arr, theta1_arr = extinct_prob_cycle(k, theta_arr, D, C, meanfitness, s_arr)
    cellnum_arr = np.random.gamma(shape=k1_arr, scale= theta1_arr)
    
    
    n_parameter_observation_NB = 1/epsilon
    ratio = np.exp( np.log(R) - np.log(N) + np.log(epsilon) )
    p_arr_parameter_observation_NB = np.exp(-np.log(1+ cellnum_arr*ratio  ))
    
    #n_arr_parameter_observation_NB = cellnum_arr * ratio
    
    prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_parameter_observation_NB,p=p_arr_parameter_observation_NB)
    
    prob_zero_cell = np.mean(prob_zero_cell_arr)
    expected_prob_observation = (read==0)*prob_zero_cell + (1-prob_zero_cell)*np.mean(prob_observation_NB)
    p = expected_prob_observation
        
    return {'p':p}

#=============================#
#
# Daughter Functions
#
#=============================#

def survival_prob_one_dilution(k, theta, D):
    p_extinction = np.exp(-k*np.log(1+theta/D))
    return 1-p_extinction

def extinct_prob_cycle(k, theta, D, C, meanfitness, s):
    
    growth_factor = np.exp(s-meanfitness)
    
    p_survive = 1.
    
    _k = k
    _theta = theta
    
    for i in range(int(C)):
        
        q = survival_prob_one_dilution(_k, _theta, D)
        
        variance_by_conditional_on_survive = np.exp(np.log(_k)+np.log(_theta))*(1.-1/q)
        
        variance_before_growth = D + _theta + variance_by_conditional_on_survive
        
        p_survive *= q
        _k *= _theta/variance_before_growth/q
        _theta = growth_factor * variance_before_growth
        
        '''
        p_survive *= survival_prob_one_dilution(_k, _theta, D)
        _k *= _theta/ (D + _theta)
        _theta = (D + _theta)* np.exp(s-meanfitness)
        '''
    
    p = 1.-p_survive 
    return p, _k, _theta   



# 
# Old Functions
#

'''
def maximum_likelihood_function_global_variable(glob, lins):
    bounds = optimize.Bounds([0., 0], [np.inf, np.inf])
    method = 'nelder-mead'
    sol = optimize.minimize(fun= likelihood_function_global_variable, x0 = np.array([0., 1.]), args=(glob, lins), method=method,bounds=bounds)
    print(sol)
    return sol


def likelihood_function_global_variable(x0, glob, lins):
    meanfitness = x0[0]
    epsilon = x0[1]
    lins_ADP = []
    lins_NEU = []
    lins_UNK = []
    p_arr=[]
    llk = 0
    count = 0
    D = glob.D
    C = glob.C
    transition = np.exp(-1.*meanfitness * C+ np.log(D) + np.log(glob.R) - np.log(glob.N) )
    
    for i in range(len(lins)):
        if lins[i].TYPETAG == LINEAGE_TAG['ADP']:
            lins_ADP.append(lins[i])
        elif lins[i].TYPETAG == LINEAGE_TAG['NEU']:
            lins_NEU.append(lins[i])
        else:
            lins_UNK.append(lins[i])
    #print(len(lins_ADP))
    for lin in lins_ADP:
        k = lin.sm.post_parm_Gamma_k
        a = lin.sm.post_parm_Gamma_a
        b = lin.sm.post_parm_Gamma_b
        mean_s = lin.sm.post_parm_NormS_mean
        var_s = lin.sm.post_parm_NormS_var
        read = lin.r1
        #p = posterior_constant_selective(transition, epsilon, k, a, b, mean_s, var_s, read, D, C)
        #llk += p
    #print(len(lins_NEU))
    for lin in lins_UNK+lins_NEU:
        
        k = lin.nm.post_parm_Gamma_k
        theta = lin.nm.post_parm_Gamma_theta
        read = lin.r1
        #p=posterior_constant_neutral(transition, epsilon, k, theta, read, D)
        p=posterior_constant_neutral(meanfitness, epsilon, k, theta, read, glob)
        p_arr.append(p)
        llk += np.log(p)
        count += 1
    llk /= count
    #llk = np.log(llk) - np.log(len(lins))
    
    #print(x0, llk)
    return  -1.*llk
    #return llk#, p_arr
    
def posterior_constant_neutral(meanfitness, epsilon, k, theta, read, glob):
    #
    # R = total read, D=dilution ratio, N=population size, 
    # bar_s = mean fitness per cycle, C = cycle legth 
    # 
    #NUM_SAMPLE = 10000
    #
    # if cell number = 0, sample zero reads
    #
    #prob_zero_cell = scipy.stats.nbinom.pmf(k=0,n=k,p=1/(1+theta/D))
    prob_zero_cell, k1, theta1 = extinct_prob_cycle(k, theta, glob.D, glob.C, meanfitness, 0)
    cellnum_arr = np.random.gamma(shape=k1, scale= theta1, size=NUM_SAMPLE)
    n_arr_parameter_observation_NB = cellnum_arr * np.exp( np.log(glob.R)- np.log(glob.N) -np.log(epsilon))
    
    #alpha_arr = np.random.negative_binomial(n=k, p=1/(1+theta/glob.D), size=NUM_SAMPLE)
    #alpha_arr = alpha_arr[alpha_arr>0]
    #n_arr_parameter_observation_NB = alpha_arr * transition / epsilon
    #print(n_arr_parameter_observation_NB)
    p_parameter_observation_NB = 1/(1+epsilon)
    prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_arr_parameter_observation_NB,p=p_parameter_observation_NB)
    
    #prob_observation_NB = np.ma.masked_invalid(prob_observation_NB)
    expected_prob_observation = (read==0)*prob_zero_cell + (1-prob_zero_cell)*np.mean(prob_observation_NB)
    p = expected_prob_observation
    
    # condition on non-zero read
    #prob_observe_zeroread_nonzerocell = scipy.stats.nbinom.pmf(k=0,n=n_arr_parameter_observation_NB,p=p_parameter_observation_NB)
    #prob_observe_zeroread_nonzerocell = np.exp(n_arr_parameter_observation_NB * np.log(prob_observation_NB))
    #p = np.mean(prob_observation_NB)/(1-np.mean(prob_observe_zeroread_nonzerocell))
    
    #return expected_prob_observation
    return p

def posterior_constant_selective(transition, epsilon, k, a, b, mean_s, var_s, read, D, C):
    #
    # transition = RD/N * exp(-bar_s * C),
    # where R = total read, D=dilution ratio, N=population size, bar_s = mean fitness per cycle, C = cycle legth 
    #
    #NUM_SAMPLE = 100000
    #
    # x = ( s - mean_s )/sigma_s
    # sigma_s = standard deviation of selection s for P(s) = prior dist
    # 
    x_arr = np.random.normal(size=NUM_SAMPLE)
    theta_arr = a/k * np.exp(b*x_arr)
    #print(theta_arr)
    p_arr_parameter_analytical_Prior = 1/(1+theta_arr/D)
    alpha_arr = np.random.negative_binomial(n=k, p=p_arr_parameter_analytical_Prior)
    #print(alpha_arr)
    
    s_arr = x_arr*(var_s**.5) + mean_s
    transition_arr = transition * np.exp(s_arr*C)
    
    n_arr_parameter_observation_NB = alpha_arr * transition_arr / epsilon
    p_parameter_observation_NB = 1/(1+epsilon)
    #print('n',n_arr_parameter_observation_NB)
    #print('p',p_parameter_observation_NB)
    #print('expect',n_arr_parameter_observation_NB*(1-p_parameter_observation_NB)/p_parameter_observation_NB)
    prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_arr_parameter_observation_NB,p=p_parameter_observation_NB)
    #print(prob_observation_NB)
    prob_observation_NB = np.ma.masked_invalid(prob_observation_NB)
    prob_observation_NB = prob_observation_NB.filled(0)

    #prob_selection = scipy.stats.norm.pdf(x_arr)
    
    #prob_2d = prob_selection* prob_observation_NB
    
    expected_prob_observation = np.mean(prob_observation_NB)
    
    return expected_prob_observation

def maximum_likelihood_function_global_variable(glob, lins, const, t):
    #bounds = optimize.Bounds([0., 0], [np.inf, np.inf])
    #method = 'L-BFGS-B'
    #bounds = ((0,0), (np.inf, np.inf))
    #method = 'powell'
    #method = 'TNC'
    #sol = optimize.minimize(fun= likelihood_function_global_variable_multiprocessing, x0 = np.array([0., 5.]), args=(glob, lins), 
    #                        method=method, bounds=bounds, options={'disp': True, 'eps': np.array([0.1, 1.])})
    
    #
    # get initial guess of meanfitness and epsilon
    #
    r0 = 0
    r1 = 0
    for lin in lins:
        if (lin.TYPETAG == LINEAGE_TAG['NEU']) or (lin.TYPETAG == LINEAGE_TAG['UNK']):
            r0 += lin.r0
            r1 += lin.r1
        elif lin.TYPETAG == LINEAGE_TAG['ADP']:
            r0 += lin.r0 * np.exp(lin.sm.post_parm_NormS_mean*glob.C)
            r1 += lin.r1
    f1 = r1/const.Rt[t]
    f0 = r0/const.Rt[t-1]
    meanfitness_ini =   (np.log(f1) - np.log(f0))/glob.C
    #print(f0, f1, meanfitness_ini)
    eps_arr = np.arange(0.001,20,1)
    llk_arr = []
    
    for eps in eps_arr:
        nllk = neg_likelihood_function_global_variable_multiprocessing([meanfitness_ini, eps], glob, lins)
        llk_arr.append(-1.*nllk)
    
    #print(eps_arr, llk_arr)
    eps_ini = eps_arr[llk_arr.index(max(llk_arr))]
    
    plt.figure()
    plt.plot(eps_arr, llk_arr, 'ko-', label='arg max {:.0f}'.format(eps_ini))
    plt.title('Likelihood function (epsilon, meanfitness={:.5f})'.format(meanfitness_ini))
    plt.xlabel('epsilon')
    plt.legend()
    plt.savefig('./initial_guess.png',dpi=150)
    
    
    sol = []
    x0 = [meanfitness_ini , eps_ini]
    bounds = [[-1.0, np.inf], [0.0, 20.0]]
    sol = minimizeCompass(func= neg_likelihood_function_global_variable_multiprocessing, 
                          bounds=bounds, x0 =  x0, args=(glob, lins), 
                          feps=0.001, deltatol=0.01, errorcontrol =False, funcNinit=30, paired=False)
    print(sol)
    
    return sol

'''
    
