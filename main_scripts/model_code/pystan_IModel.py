# -*- coding: utf-8 -*-
"""
edited on Aug 1, 2020:
    change prior of cell_num as uninformative
Created on Mon Jul 20 18:29:59 2020

@author: huanyu

Build Initialization Model using Pystan
Three parts of code
A. Stan code model, defines data, parameters, and stochastic process.
B. Compile the c++ code model in part A, and save the compiled code.
C. Run the compiled code as test.
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

pystan_modelcode_Initialization_model = """
// ================================================
// data: input information including constant and observation value
// ================================================
data {
      // barcode count is the observation in the stochastic model
      //    note taht barcode_count is a free random variable in the stochastic model, 
      //    but for the posterior probability its value is given as observation 
      
      int<lower=0> barcode_count;
     
      // the following are global variables = constant for stochastic model, 
      //    but set by input becasue value could be different at different time point
      
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

        //real<lower=0> ratio_PS2TDB = exp(log(population_size) - log(total_distinct_barcodes));  // average of cell_number per barcode
        //real<lower=0> ratio_RD2PS = exp(log(read_depth)-log(population_size));                  // average of barcode-count per cell
        real            log_mu_r    = log(cell_num) + log(read_depth)-log(population_size);       // the expected value of measured barcode-count based on random variable cell-number
        //real<lower=0>   mu_r        = exp(log_mu_r);
        real<lower=0>   phi         = 1/epsilon;                                                  // phi is the dispersion parameter of Negative Binomial distribtion in Stan
        }

// ================================================
// model: describe the stochastic process (probability distribtion) 
//        that link random varibes, including prior and likelihood
// ================================================
model {

        // prior: cell_number is given by uniform distribution == Prior distribution at t=0.
        
        cell_num ~ uniform(1, population_size/10);
        
        // measurement process: defined as Negative Binomial Distribution. 
        // See function distribution here: https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
        
        barcode_count ~ neg_binomial_2_log(log_mu_r, phi);
        //barcode_count ~ neg_binomial_2(mu_r, phi);
}

// ================================================
// generated quantities: quantitiy not necessary in stochastic model 
// ================================================
generated quantities {
        
        // log of joint probabity of parameters = log of Prior + log of Likelihood 
        real log_joint_prob = uniform_lpdf(cell_num |1,population_size) + neg_binomial_2_log_lpmf(barcode_count | log_mu_r, phi);
        //real log_joint_prob = uniform_lpdf(cell_num |1,population_size) + neg_binomial_2_lpmf(barcode_count | mu_r, phi);
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
    model_name = MODEL_NAME['I'] # File name
    
    
    model_code = pystan_modelcode_Initialization_model
    
    #
    # Compile stan model code and Save compiled model by pickle
    # 
    
    # StanModel: Model described in Stan’s modeling language compiled from C++ code.
    #model = pystan.StanModel(model_code=model_code, model_name=model_name) # compile
    
    model, _ = capture_output(pystan.StanModel, model_code=model_code, model_name=model_name) # compile
    # save the compiled code
    with open(model_name+'.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Part C
    # Run and test pystan model
    # 
    
    with open(model_name+'.pkl', 'rb') as pickle_file:
        model_load = pickle.load(pickle_file)
    
    # Define input data
    RD = 10**6
    N = 2*10**7
    #TDB = 10**5
    bc_count = 570
    epsilon = 0.01
    init_value = bc_count/RD*N
    input_data = { 'barcode_count': bc_count, 'read_depth': RD, 'population_size': N, 
                 'epsilon': epsilon }
    
    # setting of MCMC sampling
    chain_num = 2
    iter_steps = 3500
    burns = 1000
    pars = ['cell_num', 'log_joint_prob'] # only output those parameters
    algorithm = 'NUTS'
    n_jobs = 1
    # MCMC sampling
    #fit = model_load.sampling(pars=pars ,data=input_data, warmup=burns, iter=iter_steps, 
    #                          chains=chain_num, n_jobs=n_jobs, algorithm=algorithm)
    fit, _  = capture_output(model_load.sampling, pars=pars ,data=input_data, warmup=burns, iter=iter_steps, 
                             chains=chain_num, n_jobs=n_jobs, algorithm=algorithm,  refresh = -1)
    # Print out results
    #print(fit)
    print(fit)
    result = fit.extract(permuted=True)
    print(result)
    cell_num = result['cell_num']
    log_joint_prob = result['log_joint_prob']
    
    log_cellnumber = np.log(cell_num)
    mean = np.mean(log_cellnumber)
    posterior = mf.N_model_posterior(data = cell_num, log_joint_prob=log_joint_prob)
    #
    # Plot posterior
    #
    xmin = np.min(cell_num)
    xmax = np.max(cell_num)
    x_grid = np.linspace(xmin,xmax) 
    px_gamma = scipy.stats.gamma.pdf(x=x_grid, a=posterior.k, scale=posterior.theta)
    expected = np.exp(np.log(N)-np.log(RD) + np.log(bc_count+0.1))
    plt.figure()
    plt.hist(cell_num, bins=100, density=True, alpha=0.3, color='grey',label='expected mean %.2E'%(expected))
    plt.plot(x_grid, px_gamma, color='g', alpha=0.5 ,label='Gamma distribution\nk (shape) %.2f    \ntheta (scale) %.1E'%(posterior.k, posterior.theta), lw=5);
            
    plt.xlabel('cell number')
    plt.legend()
    plt.title('Posterior fit by Gamma-Distribution', fontsize=18)
    plot_file_name = 'posterior_'+model_name+".png"
    plt.savefig(plot_file_name)
    plt.close('all')
    
    '''
    expected = np.log(N)-np.log(RD) + np.log(bc_count)
    
    plt.figure()
    plt.title(f"Posterior Dist of log_cellnumber (expected={expected})")
    plt.hist(log_cellnumber, bins = 50, label=f"Mu={mean}")
    plt.xlabel('log cellnumber')
    plt.legend()
    plot_file_name = model_name+"_Sampling_log_cellnumber.png"
    plt.savefig(plot_file_name)
    
    
    # Optional Plot
    
    # derive the parameter k, theta of gamma distirbuiton to fit posterior
    posterior = mf.N_model_posterior(data = cell_num, log_joint_prob=log_joint_prob)
    k = posterior.k
    theta = posterior.theta

    xmin = np.min(cell_num)
    xmax = np.max(cell_num)
    x_grid = np.linspace(xmin,xmax) 
    px_gamma = scipy.stats.gamma.pdf(x=x_grid, a=k, scale=theta)
    plt.figure()
    plt.hist(cell_num, bins=100, density=True, alpha=0.3)
    plt.plot(x_grid, px_gamma, label='Gamma-dist k %.2f, th %.2f'%(k, theta), lw=5);
    plot_file_name = model_name+f"_posterior_Fitting.png"
    plt.xlabel('cell number')
    plt.legend()
    plt.title('Sampling of Posterior Probability fit by Gamma-Distribution')
    plt.savefig('Plot_posterior_fitting.png')
    '''
    print("The End of Code.")
    
    
    


