# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:07:41 2020

@author: huanyu
"""
import numpy as np
import scipy
import scipy.stats
from scipy import optimize
'''
def sampling_alpha(D, k, theta, num):
    #num = 5000 # num of sampling per lineage
    nb_p = 1/(1+D/theta) 
    nb_r = k
    # the definition of negative_binomial in Numpy is different to the Wikipedia, so I use 1-p in the second term
    alpha = np.random.negative_binomial(n=nb_r, p=1-nb_p, size=num)
    #mu_r = alpha * np.exp(transition)
    return alpha#mu_r
'''
def new_MLE(glob, lins, optimize_method):
    NL = len(lins)
    alphas=[]
    reads = []
    for i in range(NL):
        #alpha_arr.append(sampling_alpha(glob.D, lins[i].nm.post_parm_Gamma_k, lins[i].nm.post_parm_Gamma_theta,num))
        alphas.append( np.exp(np.log(lins[i].nm.post_parm_Gamma_k)+ np.log(lins[i].nm.post_parm_Gamma_theta) -np.log(glob.D)))
        reads.append(lins[i].r1)
    method = optimize_method
    transition = np.exp(- glob.meanfitness * glob.C ) # initial value
    bounds = optimize.Bounds([0], [1.])
    sol = optimize.minimize(fun= new_llk_trans, x0 = np.array([transition]), args=(alphas, reads, glob), method=method,bounds=bounds)
    if sol.success == True:
        transition = sol.x[0]
    else:
        print("\n Warning: MLE fails to find optimal value for meanfitness")

    expected_reads = np.asarray(alphas)* transition* np.exp(np.log(glob.D) + np.log(glob.R) - np.log(glob.N))
    
    epsilon = 10. # initial value
    bounds = optimize.Bounds([0.1], [np.inf])
    sol2 = optimize.minimize(fun= new_llk_epsilon, x0 = np.array([epsilon]), args=(reads, expected_reads), method=method,bounds=bounds)
    if sol2.success == True:
        epsilon = sol2.x[0]
    else:
        print("\n Warning: MLE fails to find optimal value for epsilon")
    
    meanfitness_est = max(-1*np.log(transition)/glob.C, 0)
    epsilon_est = epsilon
    return meanfitness_est, epsilon_est, sol, sol2

def new_llk_trans(transition, alphas, reads, glob):
    result = 0
    for i in range(len(reads)):
        result = result + (reads[i] - alphas[i] *transition* np.exp(np.log(glob.D) + np.log(glob.R) - np.log(glob.N)) )**2
    return result

def new_llk_epsilon(epsilon, reads, expected_reads):
    result = 0
    p = epsilon/(1+epsilon)
    n_arr = expected_reads/ epsilon
    for i in range(len(reads)):
        result = result + scipy.stats.nbinom.logpmf(k=reads[i], n=n_arr[i], p =p)
    return -1.*result
'''
def mle(glob, lins):
    s_ini = 0.1#glob.meanfitness
    s_bar = -1
    method = 'Powell'#'nelder-mead'#'TNC'#'L-BFGS-B'#'TNC' #'L-BFGS-B'
    sol = optimize.minimize(fun= giant_neutral_likelihood, x0 = np.array([s_ini]), args=(glob, lins), method=method,  options={'xatol': 1e-2, 'disp': True})
    if sol.success == True:
        s_bar = sol.x[0]
    else:
        print("\n Warning: MLE fails to find optimal value for meanfitness")
    return s_bar, sol

def giant_neutral_likelihood(s_bar, glob, lins):
    NL = len(lins)
    transition = np.log(glob.D) - ((s_bar-0) * glob.C) 
    read_total = 0
    expected_reads =0
    sampling_per_lin = 1000
    for i in range(NL):
        read_total = read_total + lins[i].r1
        sampling_cellnum = np.mean(sampling_next_prior(glob.D, lins[i].nm.post_parm_Gamma_k, lins[i].nm.post_parm_Gamma_theta, transition, sampling_per_lin))
        expected_reads = expected_reads + sampling_cellnum * np.exp(np.log(glob.R) - np.log(glob.N))
    return abs(expected_reads-read_total)

def nb_lik_arr(epsilon, mu, r1):#testing
    logp_cont = r1 * np.log(epsilon/(epsilon+1))  - scipy.special.gammaln(1+r1)
    logp_arr = [-(u/epsilon)*np.log(epsilon+1) + scipy.special.gammaln(r1+u/epsilon) - scipy.special.gammaln(u/epsilon)  for u in mu]
    return np.exp(logp_cont+logp_arr)

def llk(s_bar, epsilon, glob, Lin): #testing
    NL = len(Lin)
    likk = 0
    j=0
    transition = np.log(glob.D) - ((s_bar-0) * glob.C)
    sampling_per_lin = 1000
    for i in range(NL):
        #r1 = Lin[i].r1
        mu_arr = sampling_next_prior(glob.D, Lin[i].nm.post_parm_Gamma_k, Lin[i].nm.post_parm_Gamma_theta, transition, sampling_per_lin)
        mu_arr = mu_arr * np.exp(np.log(glob.R) - np.log(glob.N) )
        pob = np.mean(nb_lik_arr(epsilon, mu_arr, r1 = Lin[i].r1))
        if pob >0:
            j=j+1
            likk = likk + np.log(pob)
    #p = p_arr
    return -likk/j#-np.log(p)

def sampling_next_prior(D, k, theta, transition, num):
    #num = 5000 # num of sampling per lineage
    nb_p = 1/(1+D/theta) 
    nb_r = k
    # the definition of negative_binomial in Numpy is different to the Wikipedia, so I use 1-p in the second term
    alpha = np.random.negative_binomial(n=nb_r, p=1-nb_p, size=num)
    mu_r = alpha * np.exp(transition)
    return mu_r
'''
def analytical_Posterior_cellnum_SModel(x_arr, k, a, b, data_s):
    mean_s = np.mean(data_s)
    std_s = np.std(data_s)
    ds_arr = b*(data_s-mean_s)/std_s
    p_gamma = []
    for x in x_arr:
        x2theta_arr =[ x/a*k / np.exp(ds) for ds in ds_arr]
        ave = np.mean([(x2th**k) * np.exp(-x2th) for x2th in x2theta_arr ])
        p_gamma.append( ave/x/scipy.special.gamma(k))
    return np.asarray(p_gamma)

def analytical_Posterior_log10cellnum_SModel(x_arr, k, a, b, data_s):
    mean_s = np.mean(data_s)
    std_s = np.std(data_s)
    ds_arr = b*(data_s-mean_s)/std_s
    p_gamma = []
    x_arr = np.exp(x_arr*np.log(10))
    for x in x_arr:
        x2theta_arr =[ x/a*k / np.exp(ds) for ds in ds_arr]
        ave = np.mean([(x2th**k) * np.exp(-x2th) for x2th in x2theta_arr ])
        p_gamma.append( ave/scipy.special.gamma(k)*np.log(10) )
    return np.asarray(p_gamma)

class S_model_posterior():
    def __init__(self, data_cell_num, data_selection_coefficient, log_joint_prob):
        # data = MC sampling of cellnumber, selection coefficient
        data_cellnum = np.asarray(data_cell_num)
        data_s = np.asarray(data_selection_coefficient)
        self.data_logcellnum = np.log(data_cellnum)
        
        self.var_s = np.var(data_s)
        self.mean_s = np.mean(data_s)
        self.delta_s = (data_s- self.mean_s)/ np.std(data_s)
        self.mean_data_logcellnum = np.mean(self.data_logcellnum)
        
        self.b0 = np.cov(self.data_logcellnum, data_s)[1,0]/np.std(data_s)
        self.a0 = np.mean(data_cellnum) * np.exp(-1.*self.b0 *self.b0 /2)
        sol = optimize.root_scalar(f = self._find_k0, bracket=[0, 1000000], method='brentq')
        self.k0 = max(sol.root, 1)
        
        self.k = self.k0
        self.a = self.a0
        self.b = self.b0
        
        logp_post = self._log_posterior(data_cell_num, data_selection_coefficient)
        self.log_normalization_const = self._get_log_normalization_const(logp_post, log_joint_prob)
        
    def _find_k0(self, x):
        return scipy.special.polygamma(1,x) - self.var_s
    
    def _fn_llk_neg(self, x):
        k = x[0]
        a = x[1]
        b = x[2]
        arr = self.data_logcellnum - b * self.delta_s
        y0 = k * self.mean_data_logcellnum
        y1 = -1.*k*np.mean(np.exp(arr))/a
        y2 = k*(np.log(k)-np.log(a)) - scipy.special.gammaln(k)
        y = y0+y1+y2
        
        return -1.*y
    
    def maximum_llk_S_Model_GammaDist_Parameters(self): 
        # Maximum Likelihood Estimates MLE: estimates of parameters (k, a, b) of gamm distribuiton
        
        # bounds of parmater: 0 < k < inf, 0 < a < inf, 0 < b < inf
        bounds = optimize.Bounds([1.0, 1.0, 0.0], [np.inf, np.inf, np.inf])
        method = 'L-BFGS-B'#'TNC' #'L-BFGS-B'
        sol = optimize.minimize(fun= self._fn_llk_neg, x0 = np.array([self.k0, self.a0, self.b0]), method=method, bounds=bounds)
        if sol.success == True:
            self.k = sol.x[0]
            self.a = sol.x[1]
            self.b = sol.x[2]
        else:
            print("\n Warning: MLE fails to find optimal value for parameters (k, theta) of gamma distribuiton.")

        return sol
    
    def _get_log_normalization_const(self, log_posterior, log_joint_prob):
        arr = log_joint_prob-log_posterior
        return np.median(arr)
    
    def _log_posterior(self, data_cellnum, data_s):
        # return log of posterior P(s, n) = log P(s) + log P(n|s)
        std_s = np.std(data_s)
        n2th  = [np.exp( np.log(self.k) + np.log(n) - np.log(self.a) - self.b*(s-self.mean_s)/std_s) for n,s in zip(data_cellnum, data_s)] 
        logps = [-1.*(s-self.mean_s)*(s-self.mean_s)/(2*self.var_s)  for s in data_s] 
        logps = logps + -1.*np.log(2*np.pi*std_s*std_s)/2.
        logpn = [self.k*np.log(x) - x - np.log(n) for n,x in zip(data_cellnum, n2th)] 
        logpn = logpn - scipy.special.gammaln(self.k)
        logp  = logps + logpn
        return logp
    
    
class N_model_posterior():
    def __init__(self, data, log_joint_prob):
        # data = MC sampling of cellnumber
        # log_joint_prob = log of Joint probability (Prior X Likelihood)
        
        self._mean_cellnumber = np.mean(data)
        self._var_cellnumber = np.var(data)
        self._mean_logcellnumber = np.mean(np.log(data))
        
        self.theta0 = self._var_cellnumber/self._mean_cellnumber
        self.k0 = max(self._mean_cellnumber/self.theta0, 1)
        
        self.k = self.k0
        self.theta = self.theta0
        
        self.sol = self._maximum_llk_N_Model_GammaDist_Parameters()
        
        logp_post = self._log_posterior_cellnumber(data)
        self.log_normalization_const = self._get_log_normalization_const(logp_post, log_joint_prob)
    
    def _log_gamma_dist(self, x, k, theta):
        # return log pdf of gamma distribution of x with parameters shape = k, scale = theta
        logp = (k-1.)*np.log(x) - np.exp(np.log(x)-np.log(theta)) - scipy.special.gammaln(k) - k*np.log(theta)
        return logp

    def _fn_llk_neg(self, x):
        # negative likelihood function of fitting parameter (k, theta) of gamma distribution 
        k = x[0]
        theta = x[1]
        y =  (k-1.)*self._mean_logcellnumber - np.exp(np.log(self._mean_cellnumber)-np.log(theta)) - scipy.special.gammaln(k) - k*np.log(theta)
        return -1.*y

    def _maximum_llk_N_Model_GammaDist_Parameters(self):
        # Maximum Likelihood Estimates MLE: estimates of parameters (k, theta) of gamm distribuiton

        # bounds of parmater: 1 < k < inf, and 0 < theta < inf
        bounds = optimize.Bounds([1.0, 0.00001], [np.inf, np.inf])
        method = 'L-BFGS-B'#'TNC' #'L-BFGS-B'
        sol = optimize.minimize(fun= self._fn_llk_neg, x0 = np.array([ self.k0, self.theta0]), method=method, bounds=bounds)
        if sol.success == True:
            self.k = sol.x[0]
            self.theta = sol.x[1]
        else:
            print("\n Warning: MLE fails to find optimal value for parameters (k, theta) of gamma distribuiton.")
        return sol
            
    def _log_posterior_cellnumber(self, y):
        # return log of P(y) is gamma distribution
        k = self.k
        theta = self.theta 
        #y = np.exp(x)
        logpy = self._log_gamma_dist(y, k, theta)
        return logpy
    
    def _get_log_normalization_const(self, log_posterior, log_joint_prob):
        arr = log_joint_prob-log_posterior
        return np.median(arr) 