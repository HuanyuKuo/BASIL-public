# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:29:59 2020

@author: huanyu

Build Selective Model (SModel) for Neutral lineages using Pystan
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
#   S Model for Neutral lineage: creates model for posterior probability of (cell_number, selection coefficient) given barcode-count
#   the prior of selection coefficient is non-informative

pystan_modelcode_S_model_N = """
// ================================================
// function: custom function
// ================================================
functions {

        // Prior Distribution of cellnumber condition on selection coefficient, log of probability density function (lpdf)
        real myDist_SM_transition_prior_lpdf(real cell_num, vector transition_parameters)
        {
            real q = transition_parameters[1]; // survival probabiity
            real k_transition =  transition_parameters[2];
            real theta_transition = transition_parameters[3];
            
            real logp_cellnum = gamma_lpdf(cell_num | k_transition , 1/theta_transition);
            
            return logp_cellnum;
        }
        real survival_prob_one_dilution(real k, real theta, real dilution_ratio)
        {
            return 1-exp(-k*log(1+theta/dilution_ratio));
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
      
      // parameters of Analytical Prior distribution
      
      real<lower=0>    k_;               // shape parameter of Gamma distribution
      real<lower=0>    theta_;           // scale parameter of Gamma distribution
      
      // the following are global variables = constant for stochastic model, 
      //    but set by input becasue value could be different at different time point
      
      real             s_min;            // maximum value of unimform distriubiton of selection coefficient
      real             s_max;            // minimum value of unimform distriubiton of selection coefficient
      real<lower=0> cycle;                        // # of transition cycle between two measurement time points
      real          meanfitness;                  // mean fitness (per cycle)
      real<lower=1> dilution_ratio;               // dilution ratio, dilution_ratio = 1000 if dilute 1000:1
      real<lower=0> read_depth;                   // total barcode counts for all barcodes
      real<lower=0> population_size;              // total cell number in the flask before measurement
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

            real<lower=0>   cell_num; 
            real            selection_coefficient;
}

// ================================================
// transformed parameters: the variable with value determined by random variable or input global variables
// ================================================
transformed parameters {

        //real            transition_ratio    = -log(dilution_ratio) + (meanfitness - selection_coefficient) * cycle;
        //real<lower=0> ratio_RD2PS         = exp(log(read_depth)-log(population_size));                                // average of barcode-count per cell
        //real            gamma_Ci            = - lgamma(k_) - k_*log(1+ theta_/dilution_ratio);                          // phi is the dispersion parameter of Negative Binomial distribtion in Stan
        
        real<lower=0> growth_factor         = exp(selection_coefficient-meanfitness);
        vector[3] transition_parameters     = get_transition_parameters(k_, theta_, cycle, dilution_ratio, growth_factor);
        real            log_mu_r            = log(cell_num+0.001) +  log(read_depth) - log(population_size);                                                    // the expected value of measured barcode-count based on random variable cell-number
        real<lower=0>   phi                 = 1/epsilon;             
        
}

// ================================================
// model: describe the stochastic process (probability distribtion) 
//        that link random varibes, including prior and likelihood
// ================================================
model {

        // prior: 
            
        //  selectio coefficient is samplied from Uniform distribution
        
        selection_coefficient ~ uniform(s_min, s_max);
        
        // cell_number is given from custom Prior distribution conditional on selection coefficient
        
        //cell_num ~ myDist_SM_analytical_prior(transition_ratio, k_, theta_, dilution_ratio, gamma_Ci);
        cell_num ~ myDist_SM_transition_prior(transition_parameters);
        
        // measurement process: defined as Negative Binomial Distribution. See function distribution here: https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
        
        barcode_count ~ neg_binomial_2_log(log_mu_r, phi);
}

// ================================================
// generated quantities: quantitiy not necessary in stochastic model 
// ================================================
generated quantities {
        
        // log of joint probabity of parameters = log of Prior + log of Likelihood 
        
        real log_joint_prob = uniform_lpdf(selection_coefficient | s_min, s_max) 
        + myDist_SM_transition_prior_lpdf(cell_num | transition_parameters)
        //+ myDist_SM_analytical_prior_lpdf(cell_num | transition_ratio, k_, theta_, dilution_ratio, gamma_Ci) 
        + neg_binomial_2_log_lpmf(barcode_count | log_mu_r, phi);
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
    model_name = MODEL_NAME['SN']
    
    model_code = pystan_modelcode_S_model_N
    
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
    N = float(10**11)
    TDB = 10**5
    bc_count = 100
    epsilon = 0.1
    meanfitness = 0.
    cycle = 1.
    k = 10.
    theta = 100#(bc_count/10)/RD*N/k
    mean_s = 0.5
    var_s = 0.2
    b_ = cycle * np.sqrt(var_s)
    a_ = np.exp(13.)
    init_value = bc_count/RD*N
    dilution_ratio = float(10000)
    input_data = { 'barcode_count': bc_count, 'read_depth': RD, 'population_size': N, 
                 'epsilon': epsilon, 'k_': k, 'theta_':theta,  'meanfitness': meanfitness,
                 'cycle': cycle, 'dilution_ratio': dilution_ratio, 
                 's_max': 10, 's_min': -10
        }
    #print('Setting for sampling')
    # setting of MCMC sampling
    now = time.time()
    chain_num = 4
    iter_steps = 3500
    burns = 500
    pars = ['cell_num', 'selection_coefficient' , 'log_joint_prob', 'prob_survive'] # only output those parameters
    
    algorithm = 'NUTS'
    n_jobs = 1
    # MCMC sampling
    #print('Start sampling')
    fit, _ = capture_output( model_load.sampling, pars=pars ,data=input_data, warmup=burns, iter=iter_steps, 
                              chains=chain_num, n_jobs=n_jobs, algorithm=algorithm, refresh = 0)
    #fit = model_load.sampling(pars=pars ,data=input_data, warmup=burns, iter=iter_steps, 
    #                          chains=chain_num, n_jobs=n_jobs, algorithm=algorithm)
    
    #expected_logn = np.log(N)-np.log(RD) + np.log(bc_count)
    #expected_s = (expected_logn - np.log(k) - np.log(theta))/cycle - meanfitness
    # Print out results
    #print(fit)
    #print(f'Time (s) on sampling: {time.time()-now}')
    
    result = fit.extract(permuted=True)
    cell_num = result['cell_num']
    selection_coefficient = result['selection_coefficient']
    log_joint_prob = result['log_joint_prob']
    p_survive = result['prob_survive'][0]
    posterior = mf.S_model_posterior(cell_num, selection_coefficient, log_joint_prob)
    posterior.maximum_llk_S_Model_GammaDist_Parameters()
    
    log10_cellnum = np.log10(cell_num)
            
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    c_plot = fig.add_subplot(grid[-1,0], xticklabels=[], yticklabels=[])
            
    s_mean = np.mean(selection_coefficient)
    s_var = np.var(selection_coefficient)
            
    # Text box at left bottom
    c_plot.axis('off')
    c_plot.text(-0.2,0.1,'parameters:\n s mean %.3f\n s var %.3f \n k %.3f\n a %.0E\n b %.3f'%(s_mean,s_var,posterior.k, posterior.a, posterior.b))
            
            
    # scatter points on the main axes
    main_ax.plot(selection_coefficient, log10_cellnum, 'ko', alpha=0.05, ms=5)
                 #,label="expect cellnum\n %.1E"%(np.exp(expected_logn)))
    main_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    main_ax.set_title('Posterior S Model', y=1.02, fontsize=18)
    #main_ax.legend(loc='best')       
    # histogram on the attached axes
    #
    # x subplot
    xmin = np.min(selection_coefficient)
    xmax = np.max(selection_coefficient)
    x_grid = np.linspace(xmin,xmax) 
    px_normal = [np.exp(-1.*((s-s_mean)**2)/s_var/2)/np.sqrt(2*np.pi*s_var) for s in list(x_grid)]
            
    x_hist.hist(selection_coefficient, bins=100, histtype='stepfilled',
                        orientation='vertical', density=True, alpha=0.3, color='gray')
    x_hist.set_title('selection coefficient (1/cycle)', y= -0.5)
    x_hist.plot(x_grid, px_normal, color='g', alpha=0.5, lw=3)
    x_hist.invert_yaxis()
            
    #
    # y subplot
    xmin = np.min(log10_cellnum)
    xmax = np.max(log10_cellnum)
    x_grid = np.linspace(xmin,xmax) 
    px_gamma = mf.analytical_Posterior_log10cellnum_SModel(x_grid, posterior.k0, posterior.a0, posterior.b0, selection_coefficient)
    px_gamma2 = mf.analytical_Posterior_log10cellnum_SModel(x_grid, posterior.k, posterior.a, posterior.b, selection_coefficient)
            
    y_hist.plot(px_gamma, x_grid, label='moment-method', lw=3, color='k', alpha=0.5)
    y_hist.plot(px_gamma2, x_grid, label='max-likelihood', lw=3, color='b', alpha=0.5)
    y_hist.hist(log10_cellnum, bins=100, density=True, alpha=0.3, color='gray', histtype='stepfilled',
                        orientation='horizontal')
    y_hist.set_title('Log10 Cell Number', y=0.25, x=-0.5, rotation = 90)
    y_hist.legend(loc=(0, 1.02),prop={'size': 6})
    y_hist.invert_xaxis()
    plot_file_name = 'posterior_'+model_name+".png"
            
    fig.savefig(plot_file_name, dpi=800)        
    
   
    #print("The End of Code.")
    
    
    
    


