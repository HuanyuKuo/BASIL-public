# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:29:59 2020

@author: huanyu

Build Neutral Model (NModel) using Pystan
Three parts of code
A. Stan code model, defines data, parameters, and stochastic process.
B. Compile the c++ code model in part A, and save the compiled code.
C. Run the compiled code & test.
"""

# for compiling stan code
import pystan
import pickle

# for testing the compiled code
import numpy as np
from matplotlib import pyplot as plt
#from MEE_Variables import Lineage
import MEE_Functions as mf
#import seaborn as sns
#from scipy.special import gammaln
import scipy
import time

import os
import sys
import threading
import logging
# silence logger, there are better ways to do this
# see PyStan docs
logging.getLogger("pystan").propagate=False



MODEL_NAME = {'I': 'InitModel', 'N': 'NModel', 'SN': 'SModel_N', 'SS': 'SModel_S'}

# Part A
# Pystan code (c++) for model building
#   Initialization model : creates model for posterior probability of cell_number given barcode-count at t=0
#

pystan_modelcode_N_model = """
// ================================================
// function: custom function
// ================================================

functions {

        // Prior Distribution of cellnumber, log of probability density function (lpdf)
        real myDist_NM_transition_prior_lpdf(real cell_num, vector transition_parameters)
        {
            real q = transition_parameters[1]; // survival probabiity
            real k_transition =  transition_parameters[2];
            real theta_transition = transition_parameters[3];
            
            real logp_cellnum = gamma_lpdf(cell_num | k_transition , 1/theta_transition);
            
            return logp_cellnum;
        }
        real survival_prob_one_dilution(real k, real theta, real dilution_ratio)
        {
            real out = 1-exp(-k*log(1+theta/dilution_ratio));
            return out;
        }
        vector get_transition_parameters(real k_, real theta_, real cycle_, real dilution_ratio, real growth_factor)
        {
            vector[3] out;    
            real p_survive = 1.;
            real k = k_;
            real theta = theta_;
            
            int c = 0;
            while(c < cycle_){
                    c += 1;
            }
            //int c=2;
            for (n in 1:c)
            {
                real ps = survival_prob_one_dilution(k, theta, dilution_ratio);
                real variance_by_conditional_on_survive = exp(log(k) + log(theta))* (1.- 1/ps);
                real variance_before_growth = dilution_ratio + theta + variance_by_conditional_on_survive;
                
                p_survive *= ps;
                k *=  theta/variance_before_growth/ps;
                theta = growth_factor * variance_before_growth;
            }
            
            out[1] = p_survive;
            out[2] = k;
            out[3] = theta;
            
            return out;
        }
        
}
// ================================================
// data: input information including constant and observation value
// ================================================
data {
      // barcode count is the observation in the stochastic model
      //    note taht barcode_count is a free random variable in the stochastic model, 
      //    but for the posterior probability its value is given as observation 
      
      int<lower=0> barcode_count;
      
      // parameter for Analytical Prior distribution
      
      real<lower=0> k_;                            // shape parameter of Gamma distribution
      real<lower=0> theta_;                        // scale parameter of Gamma distribution
      
      // the following are global variables = constant for stochastic model, 
      //    but set by input becasue value could be different at different time point
      
      real<lower=0> cycle;                        // # of transition cycle between two measurement time points
      real          meanfitness;                  // mean fitness (per cycle)
      real<lower=1> dilution_ratio;               // dilution ratio, dilution_ratio = 1000 if dilute 1000:1
      real<lower=0> read_depth;                   // total barcode counts for all barcodes
      real<lower=0> population_size;              // total cell number in the flask before measurement
      //real<lower=0> total_distinct_barcodes;      // total number of distinct barcodes (Number of lineages)
      //
      // for a measure of barcode read counts: variance = mean + epsilon * mean ^2 (define epsilon 0-1)
      // if epsilon = 0, then measure process becomes Poisson (VMR=1)
      // if epsilon = 1  then variance ~ mean ^2 for all results except of zero read count (VMR ~2)
      //
      real<lower=0> epsilon;                      // measurement dispersion

}

// ================================================
// parameters: the free random variable in posterior distribution, 
//              usually the variable of interest
// ================================================
parameters {

            real<lower=0> cell_num; 
}

// ================================================
// transformed parameters: the variable with value determined by random variable or input global variables
// ================================================
transformed parameters {

        //real            transition_ratio    = -log(dilution_ratio) + (meanfitness - 0.) * cycle;
        //real<lower=0> ratio_PS2TDB        = exp(log(population_size) - log(total_distinct_barcodes));  // average of cell_number per barcode
        //real<lower=0> ratio_RD2PS         = exp(log(read_depth)-log(population_size));                  // average of barcode-count per cell
        //real            gamma_Ci           = - lgamma(k_) - k_*log(1+ theta_/dilution_ratio);    
        
        real<lower=0> growth_factor         = exp(0.-meanfitness);
        vector[3] transition_parameters     = get_transition_parameters(k_, theta_, cycle, dilution_ratio, growth_factor);
        real            log_mu_r           = log(cell_num+0.001) + log(read_depth) - log(population_size);                                             // the expected value of measured barcode-count based on random variable cell-number
        real<lower=0>   phi                = 1/epsilon;                                                 // phi is the dispersion parameter of Negative Binomial distribtion in Stan
}

// ================================================
// model: describe the stochastic process (probability distribtion) 
//        that link random varibes, including prior and likelihood
// ================================================
model {

        // prior: cell_number is given by custom Prior distribution
        
        //cell_num ~ myDist_NM_analytical_prior(transition_ratio, k_, theta_, dilution_ratio, gamma_Ci);
        
        cell_num ~ myDist_NM_transition_prior(transition_parameters);
        
        // measurement process: defined as Negative Binomial Distribution. See function distribution here: https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
        
        barcode_count ~ neg_binomial_2_log(log_mu_r, phi);
}

// ================================================
// generated quantities: quantitiy not necessary in stochastic model 
// ================================================
generated quantities {
        
        // log of joint probabity of parameters = log of Prior + log of Likelihood 
        real log_joint_prob = myDist_NM_transition_prior_lpdf(cell_num | transition_parameters)
                            + neg_binomial_2_log_lpmf(barcode_count | log_mu_r, phi);
        
        //real log_joint_prob = myDist_NM_analytical_prior_lpdf(cell_num | transition_ratio, k_, theta_, dilution_ratio, gamma_Ci) 
        //                    + neg_binomial_2_log_lpmf(barcode_count | log_mu_r, phi);
        //real log_joint_prob = normal_lpdf(cell_num | k_*theta_, k_*theta_*theta_) 
        //                    + neg_binomial_2_lpmf(barcode_count | mu_r, phi);
        
        real prob_survive = transition_parameters[1];
}

"""
def drain_pipe(captured_stdout, stdout_pipe):
    while True:
        data = os.read(stdout_pipe[0], 1024)
        if not data:
            break
        captured_stdout += data
def capture_output(function, *args, **kwargs):
    """
    https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
    """
    stdout_fileno = sys.stdout.fileno()
    stdout_save = os.dup(stdout_fileno)
    stdout_pipe = os.pipe()
    os.dup2(stdout_pipe[1], stdout_fileno)
    os.close(stdout_pipe[1])

    captured_stdout = b''

    t = threading.Thread(target=lambda:drain_pipe(captured_stdout, stdout_pipe))
    t.start()
    # run user function
    result = function(*args, **kwargs)
    os.close(stdout_fileno)
    t.join()
    os.close(stdout_pipe[0])
    os.dup2(stdout_save, stdout_fileno)
    os.close(stdout_save)
    return result, captured_stdout.decode("utf-8")

if __name__ == '__main__':

    # Part B
    # Compile the stan code
    
    #
    # Choose model code as Initialization Model
    #
    #model_name = 'Init_Model' # File name
    model_name = MODEL_NAME['N']
    
    model_code = pystan_modelcode_N_model
    
    #
    # Compile stan model code and Save compiled model by pickle
    # 
    
    # StanModel: Model described in Stan’s modeling language compiled from C++ code.
    print('Compile stan code model')
    #model = pystan.StanModel(model_code=model_code, model_name=model_name) # compile
    model, _ = capture_output(pystan.StanModel, model_code=model_code, model_name=model_name) # compile
    
    # save the compiled code
    #print('Save compiled model')
    with open(model_name+'.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    # Part C
    # Run and test pystan model
    # 
    #print('Open compiled model')
    with open(model_name+'.pkl', 'rb') as pickle_file:
        model_load = pickle.load(pickle_file)
    #print('Set up input data')
    # Define input data 
    RD = 10**6
    N = float(10**7)
    TDB = 10**5
    bc_count = 1000
    epsilon = 0.1
    meanfitness = 0.
    cycle = 2.
    k = 100
    theta = bc_count/RD*N/k
    init_value = bc_count/RD*N
    dilution_ratio = float(10000)
    
    
    '''
    input_data = { 'barcode_count': bc_count, 'read_depth': RD, 'population_size': N, 
                 'epsilon': epsilon, 'k_': k, 'theta_': theta, 'meanfitness': meanfitness,
                 'cycle': cycle, 'dilution_ratio': dilution_ratio}
    '''
    
    input_data = { 'barcode_count': bc_count, 'read_depth': RD, 'population_size': N, 
                 'epsilon': epsilon, 'k_': k, 'theta_': theta, 'meanfitness': 0.25,
                 'cycle': 2.0, 'dilution_ratio': 100.0}
    '''
    input_data = {'barcode_count': 98, 'read_depth': 1008692.0, 'population_size': 20149527.0, 
                  'epsilon': 17, 'k_': 5.160454010938178, 'theta_': 100.38396105755787, 
                  'meanfitness': 0.25580110678309087, 'cycle': 2.0, 'dilution_ratio': 100.0}
    '''
    
    #print('Setting for sampling')
    # setting of MCMC sampling
    now = time.time()
    chain_num = 2
    iter_steps = 3500
    burns = 500
    pars = ['cell_num', 'log_joint_prob', 'prob_survive'] # only output those parameters
    algorithm = 'NUTS'
    n_jobs = 1
    # MCMC sampling
    #print('Start sampling')
    #fit = model_load.sampling(pars=pars ,data=input_data, warmup=burns, iter=iter_steps, 
    #                          chains=chain_num, n_jobs=n_jobs, algorithm=algorithm)
    #
    fit, _ = capture_output( model_load.sampling, pars=pars ,data=input_data, warmup=burns, iter=iter_steps, 
                              chains=chain_num, n_jobs=n_jobs, algorithm=algorithm, refresh = 0)
    
    # Print out results
    #print(fit)
    #print(f'Time (s) on sampling: {time.time()-now}')
    result = fit.extract(permuted=True)
    cell_num = result['cell_num']
    log_joint_prob = result['log_joint_prob']
    p_survive = result['prob_survive'][0]
    log_cellnumber = np.log(cell_num)
    mean = np.mean(log_cellnumber)
    #cell_num2=cell_num[cell_num>0.5]
    #log_joint_prob = log_joint_prob[cell_num>0.5]
    posterior = mf.N_model_posterior(data = cell_num, log_joint_prob=log_joint_prob)
    #
    # Plot posterior
    #
    xmin = np.min(cell_num)
    xmax = np.max(cell_num)
    x_grid = np.linspace(xmin,xmax) 
    dx = x_grid[1]-x_grid[0]
    px_gamma = scipy.stats.gamma.pdf(x=x_grid, a=posterior.k, scale=posterior.theta)#(1-result['transition_parameters'][0][0])
    #expected = np.exp(np.log(N)-np.log(RD) + np.log(bc_count))
    
    expected = np.exp(np.log(N)-np.log(RD) + np.log(bc_count+0.1))
    #expected=0
    plt.figure()
    plt.hist(cell_num, bins=100, density=True, alpha=0.3, color='grey',label='expected mean %.2E'%(expected))
    plt.plot(x_grid, px_gamma, color='g', alpha=0.5 ,label='Gamma distribution\nk (shape) %.2f    \ntheta (scale) %.1E'%(posterior.k, posterior.theta), lw=5);
    plt.xlabel('cell number')
    #plt.ylim(0,0.0001)
    plt.legend()
    plt.title('Posterior fit by Gamma-Distribution', fontsize=18)
    plot_file_name = 'posterior_'+model_name+".png"
    plt.savefig(plot_file_name)
    plt.close('all')
    
    #print("The End of Code.")
    
    
    
    


