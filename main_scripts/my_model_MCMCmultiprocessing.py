# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:50:50 2021

@author: huanyu
"""

import myFunctions as mf
import myConstant as mc
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import numpy as np
import os.path
import signal
import logging
import sys
import threading
from matplotlib import pyplot as plt


# ---- global variable ----
NUMBER_OF_PROCESSES = mc.NUMBER_OF_PROCESSES
MODEL_NAME = mc.MODEL_NAME
LINEAGE_TAG = mc.LINEAGE_TAG
OutputFileDir = mc.OutputFileDir
FLAG_PLOT_POSTERIOR_N = False
FLAG_PLOT_POSTERIOR_SN = False
FLAG_PLOT_POSTERIOR_SS = False
DEBUG_STAN = False
MAX_TASK_PER_CHILD = 10
_STAN_MODEL = None



# =====================================================================================
#
# Part A: compute posterior via MCMC with multiprocessing
#
# the body of multiprocessing: run_model_MCMCmultiprocessing
# the body of MCMC: MCMC_sampling
# 
# fucntions: run_model_MCMCmultiprocessing, worker, MCMC_sampling
#
# =====================================================================================

#
# Function apply multiprocessing to run MCMC sampling of specified model,
#        in parallel for all lineages in the 'lins' 
#        the posterior information is saved in Queue then output to file
#

def run_model_MCMCmultiprocessing(run_dict, lins, glob):
    """Run MCMC for millions of lineages without using queues or loading all results in memory"""
    model_name = run_dict['model_name']
    lineage_name = run_dict['lineage_name']

    # output file
    outfilename = f"{OutputFileDir}posterior_{lineage_name}_{model_name}_T{glob.current_timepoint}.txt"
    print("Writing results to", outfilename)

    # open file first
    with open(outfilename, 'w') as f:
        # write header
        if model_name == mc.MODEL_NAME['N']:
            f.write("listID\tBCID\tTAG\tk\ttheta\tlog_normalization\tlog_prob_survive_cummulated\n")
        else:
            f.write("listID\tBCID\tTAG\tk\ta\tb\ts_mean\ts_var\tlog_normalization\tlog_prob_survive_cummulated\n")

        n_ok = 0
        n_fail = 0
        FLUSH_EVERY = 100
        count = 0
        # create Pool
        with Pool(NUMBER_OF_PROCESSES, initializer=init_worker, initargs=(model_name,),maxtasksperchild=MAX_TASK_PER_CHILD,) as pool:
            # iterate generator of inputs
            for result in tqdm(pool.imap_unordered(MCMC_sampling, get_Input_data_generator(model_name, lins, glob)),
                               total=len(lins) if hasattr(lins, "__len__") else None, desc="MCMC Sampling"):

                # write successful results
                if result.get("status") == "ok":
                    n_ok += 1
                    if model_name == mc.MODEL_NAME['N']:
                        f.write(f"{result['listID']}\t{result['BCID']}\t{result['TAG']}\t"
                                f"{result['k']}\t{result['theta']}\t{result['log_norm']}\t"
                                f"{result['log_prob_survive_cummulated']}\n")
                        count += 1
                        if count % FLUSH_EVERY == 0:
                            f.flush()

                    else:
                        f.write(f"{result['listID']}\t{result['BCID']}\t{result['TAG']}\t"
                                f"{result['k']}\t{result['a']}\t{result['b']}\t"
                                f"{result['mean_s']}\t{result['var_s']}\t"
                                f"{result['log_norm']}\t{result['log_prob_survive_cummulated']}\n")
                        count += 1
                        if count % FLUSH_EVERY == 0:
                            f.flush()

                else:
                    # log failures
                    n_fail += 1
                    print(f"Failed lineage {result['listID']} (BCID {result.get('BCID')}): {result.get('error')}")

        total = n_ok + n_fail
        if total > 0:
            print(
                f"Finished MCMC: "
                f"{n_ok} succeeded, {n_fail} failed "
                f"({100 * n_fail / total:.2f}% failure rate)"
            )
        else:
            print("Finished MCMC: no lineages processed")
#
# Function run by worker processes
#
# ---- global variable ----
_STAN_MODEL = None

def init_worker(model_name):
    global _STAN_MODEL
    # silence_stdout()
    with open(f'./model_code/{model_name}.pkl', 'rb') as f:
        _STAN_MODEL = pickle.load(f)

# def worker(input_q, output_q):
#     for func, args in iter(input_q.get, 'STOP'):
#         # here 'func' = MCMC_sampling, 'args' = item of Input_Data_Array
#         result = func(args)
#         # put returning result into queue
#         output_q.put(result)
#     #


# Function derives posterior sampling via MCMC, return fitting parameters
#
# def MCMC_sampling(inputdict):
#
#     global _STAN_MODEL
#     model_load = _STAN_MODEL
#     model_name = inputdict['model_name']
#
#     try:
#         #
#         # Sampling MCMC
#         #
#         META_KEYS = {'t', 'BCID', 'model_name', 'listID', 'tag', 'log_prob_survive'}
#         input_data = {k: v for k, v in inputdict.items() if k not in META_KEYS}
#
#         chain_num = 2
#         n_jobs = 1
#         burns = 1000
#         sampling_step = 3500
#         algorithm = 'NUTS'
#
#         pars_name = get_pars_interest(model_name=model_name)
#         '''
#         initial_value = get_pars_initialvalue(model_name=model_name,
#                                               input_data = input_data,
#                                               chain_num = chain_num)
#         '''
#         initial_value = 0
#         # print(input_data, pars_name)
#
#         fit, _ = capture_output(model_load.sampling,data= input_data, init= initial_value, pars= pars_name,
#                                 warmup= burns, iter= sampling_step, chains= chain_num,
#                                 n_jobs= n_jobs, algorithm = algorithm, control={'adapt_delta':0.8},
#                                 refresh = 0)
#
#         # fit = model_load.sampling(data=input_data, init=initial_value, pars=pars_name,
#         #                           warmup=burns, iter=sampling_step, chains=chain_num,
#         #                           n_jobs=n_jobs, algorithm=algorithm, control={'adapt_delta': 0.8}, refresh=-1)
#
#         # parametric posterior
#         #
#         outputdict  = get_posterior_info(inputdict, fit)
#         outputdict["status"] = "ok"
#         return outputdict
#
#     except Exception as e:
#         return {
#             "listID": inputdict["listID"],
#             "BCID": inputdict["BCID"],
#             "TAG": inputdict["tag"],
#             "status": "failed",
#             "error": str(e),
#         }

def MCMC_sampling(inputdict):
    global _STAN_MODEL
    if _STAN_MODEL is None:
        raise RuntimeError("Stan model not initialized in worker")

    model_name = inputdict['model_name']

    META_KEYS = {'t', 'BCID', 'model_name', 'listID', 'tag', 'log_prob_survive'}
    input_data = {k: v for k, v in inputdict.items() if k not in META_KEYS}

    try:
        # NOTE: SIGALRM works only on Unix/Linux (not Windows)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(600)  # 10 minutes max per lineage

        sampling_kwargs = dict(
            data=input_data,
            init=0,
            pars=get_pars_interest(model_name),
            warmup=1000,
            iter=3500,
            chains=1,
            seed =  10000 + inputdict['listID'],
            n_jobs=1,
            algorithm='NUTS',
            control={'adapt_delta': 0.8},
            refresh=0,
        )

        if DEBUG_STAN:
            # show Stan output (useful for debugging hangs / divergences)
            fit = _STAN_MODEL.sampling(**sampling_kwargs)
        else:
            # suppress Stan stdout
            fit, _ = capture_output(_STAN_MODEL.sampling, **sampling_kwargs)

        outputdict = get_posterior_info(inputdict, fit)
        outputdict["status"] = "ok"
        signal.alarm(0)
        return outputdict


    except TimeoutError as e:

        return _fail(inputdict, "TimeoutError", e)

    except RuntimeError as e:

        return _fail(inputdict, "StanRuntimeError", e)

    except ValueError as e:

        return _fail(inputdict, "ValueError", e)

    except Exception as e:

        return _fail(inputdict, "UnknownError", e)

def _fail(inputdict, etype, e):
    return {
        "listID": inputdict["listID"],
        "BCID": inputdict["BCID"],
        "TAG": inputdict["tag"],
        "status": "failed",
        "error": f"{etype}: {e}",
    }
# =====================================================================================
# Part B: functions to gather informations for MCMC in part A
#
# functions for inputs:
#   get_pars_interest, get_Input_data_array,
#
# function for outputs:
#   get_posterior_info, put_posterior_to_file
# =====================================================================================

#
# Function returns the a list of parameter, which is the output parameter of specific model
#
def get_pars_interest(model_name):
    if model_name == MODEL_NAME['N']:
        return ['cell_num', 'log_joint_prob', 'prob_survive']
    elif model_name in (MODEL_NAME['SN'], MODEL_NAME['SS']):
        return ['cell_num', 'selection_coefficient', 'log_joint_prob', 'prob_survive']
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

#
# Function returns input data for feeding specified model
#
# def get_Input_data_array(model_name, lins, glob):
#     RD = glob.R
#     N = glob.N
#     dilution_ratio = glob.D
#     cycle = glob.C
#     meanfitness = glob.meanfitness
#     epsilon = glob.epsilon
#     t = glob.current_timepoint
#     Input_data_array = []
#
#     if model_name not in MODEL_NAME.values():
#         print(
#             f"ERROR! function \"get_Input_data_array\" could not identify the input model_name.model_name should be one of {MODEL_NAME}. ")
#
#     elif model_name == MODEL_NAME['N']:
#
#         # do Not change hte order of Input_data_array!
#         Input_data_array = [
#             {'t': t, 'BCID': lins[i].BCID, 'model_name': model_name, 'listID': i, 'tag': lins[i].TYPETAG,
#              'log_prob_survive': lins[i].nm.log_prob_survive,
#              'barcode_count': np.int64(lins[i].r1), 'read_depth': RD, 'population_size': N, 'epsilon': epsilon,
#              'k_': lins[i].nm.post_parm_Gamma_k, 'theta_': lins[i].nm.post_parm_Gamma_theta,
#              'meanfitness': meanfitness, 'cycle': cycle, 'dilution_ratio': dilution_ratio
#              } for i in range(len(lins))]
#
#     elif model_name == MODEL_NAME['SS']:
#         # do Not change hte order of Input_data_array!
#         Input_data_array = [
#             {'t': t, 'BCID': lins[i].BCID, 'model_name': model_name, 'listID': i, 'tag': lins[i].TYPETAG,
#              'log_prob_survive': lins[i].sm.log_prob_survive,
#              'barcode_count': np.int64(lins[i].r1), 'read_depth': RD, 'population_size': N, 'epsilon': epsilon,
#              'k_': lins[i].sm.post_parm_Gamma_k, 'a_': lins[i].sm.post_parm_Gamma_a, 'b_': lins[i].sm.post_parm_Gamma_b,
#              'meanfitness': meanfitness, 'cycle': cycle, 'dilution_ratio': dilution_ratio,
#              'mean_s': lins[i].sm.post_parm_NormS_mean, 'var_s': lins[i].sm.post_parm_NormS_var
#              } for i in range(len(lins))]
#
#     return Input_data_array
# def get_Input_data_generator(model_name, lins, glob):
#     RD = glob.R
#     N = glob.N
#     D = glob.D
#     C = glob.C
#     meanfitness = glob.meanfitness
#     epsilon = glob.epsilon
#     t = glob.current_timepoint
#
#     for i, lin in enumerate(lins):
#         if model_name == mc.MODEL_NAME['N']:
#             yield {'t': t, 'BCID': lin.BCID, 'model_name': model_name, 'listID': i, 'tag': lin.TYPETAG,
#                    'log_prob_survive': lin.nm.log_prob_survive,
#                    'barcode_count': np.int64(lin.r1), 'read_depth': RD, 'population_size': N, 'epsilon': epsilon,
#                    'k_': lin.nm.post_parm_Gamma_k, 'theta_': lin.nm.post_parm_Gamma_theta,
#                    'meanfitness': meanfitness, 'cycle': C, 'dilution_ratio': D}
#         else:
#             yield {'t': t, 'BCID': lin.BCID, 'model_name': model_name, 'listID': i, 'tag': lin.TYPETAG,
#                    'log_prob_survive': lin.sm.log_prob_survive,
#                    'barcode_count': np.int64(lin.r1), 'read_depth': RD, 'population_size': N, 'epsilon': epsilon,
#                    'k_': lin.sm.post_parm_Gamma_k, 'a_': lin.sm.post_parm_Gamma_a, 'b_': lin.sm.post_parm_Gamma_b,
#                    'meanfitness': meanfitness, 'cycle': C, 'dilution_ratio': D,
#                    'mean_s': lin.sm.post_parm_NormS_mean, 'var_s': lin.sm.post_parm_NormS_var}
def get_Input_data_generator(model_name, lins, glob):
    RD = glob.R
    N = glob.N
    D = glob.D
    C = glob.C
    meanfitness = glob.meanfitness
    epsilon = glob.epsilon
    t = glob.current_timepoint

    if model_name not in MODEL_NAME.values():
        raise ValueError(f"Unknown model_name: {model_name}")

    for i, lin in enumerate(lins):
        base = {
            't': t,
            'BCID': lin.BCID,
            'model_name': model_name,
            'listID': i,
            'tag': lin.TYPETAG,
            'barcode_count': np.int64(lin.r1),
            'read_depth': RD,
            'population_size': N,
            'epsilon': epsilon,
            'meanfitness': meanfitness,
            'cycle': C,
            'dilution_ratio': D,
        }

        if model_name == MODEL_NAME['N']:
            yield {
                **base,
                'log_prob_survive': lin.nm.log_prob_survive,
                'k_': lin.nm.post_parm_Gamma_k,
                'theta_': lin.nm.post_parm_Gamma_theta,
            }
        else:  # SS / SN
            yield {
                **base,
                'log_prob_survive': lin.sm.log_prob_survive,
                'k_': lin.sm.post_parm_Gamma_k,
                'a_': lin.sm.post_parm_Gamma_a,
                'b_': lin.sm.post_parm_Gamma_b,
                'mean_s': lin.sm.post_parm_NormS_mean,
                'var_s': lin.sm.post_parm_NormS_var,
            }
#
# Function returns a dictionary of posterior information.
#   (1) fitting parametric function of posterior onto sampling data.
#   (2) optional: Plots of posterior distribution.
#
def get_posterior_info(inputdict, fit):
    model_name = inputdict['model_name']
    BCID = inputdict['BCID']
    # TAG = inputdict['tag']
    current_time = inputdict['t']

    outputdict = {'listID': inputdict['listID'], 'BCID': BCID, 'TAG': inputdict['tag']}

    if (model_name == MODEL_NAME['N']):
        #
        # Extract results
        #
        results = fit.extract(permuted=True)
        del fit
        cell_num = results['cell_num']
        log_joint_prob = results['log_joint_prob']
        # Rhat_cellnum = fit.summary("cell_num")["summary"][0][9]
        #
        # Fit the posterior probability, then output fitting parameters
        #
        posterior = mf.N_model_posterior(data=cell_num, log_joint_prob=log_joint_prob)
        # outputdict.update({ 'k': posterior.k, 'theta': posterior.theta,
        #                   'log_norm': posterior.log_normalization_const, 'Rhat_cellnum': Rhat_cellnum})
        outputdict.update({'k': posterior.k, 'theta': posterior.theta, 'log_norm': posterior.log_normalization_const})

        log_prob_survive_cummulated = inputdict['log_prob_survive'] + np.log(results['prob_survive'][0])
        outputdict.update({'log_prob_survive_cummulated': log_prob_survive_cummulated})

        '''        
        if FLAG_PLOT_POSTERIOR_N:
            #
            # Plot posterior
            #
            xmin = np.min(cell_num)
            xmax = np.max(cell_num)
            x_grid = np.linspace(xmin,xmax) 
            px_gamma = scipy.stats.gamma.pdf(x=x_grid, a=posterior.k, scale=posterior.theta)
            N = inputdict['population_size']
            RD = inputdict['read_depth']
            bc_count = inputdict['barcode_count']
            expected = np.exp(np.log(N)-np.log(RD) + np.log(bc_count))
            plt.figure()
            plt.hist(cell_num, bins=100, density=True, alpha=0.3, color='grey',label='expected mean %.2E'%(expected))
            plt.plot(x_grid, px_gamma, color='g', alpha=0.5 ,label='Gamma distribution\nk (shape) %.2f    \ntheta (scale) %.1E'%(posterior.k, posterior.theta), lw=5);

            plt.xlabel('cell number')
            plt.legend()
            titletext = f'BCID {BCID}    T {current_time}'
            plt.title(titletext, fontsize=18)
            plot_file_name = 'posterior_'+model_name+f"_T{current_time}_BC_{BCID}.png"
            plt.savefig(OutputFileDir+plot_file_name)
            plt.close('all')
        '''


    elif (model_name == MODEL_NAME['SS']):  # or (model_name == MODEL_NAME['SN']) :
        #
        # Extract results
        #
        results = fit.extract(permuted=True)
        del fit
        cell_num = results['cell_num']
        selection_coefficient = results['selection_coefficient']
        log_joint_prob = results['log_joint_prob']
        # Rhat_cellnum = fit.summary("cell_num")["summary"][0][9]
        # Rhat_selection = fit.summary("selection_coefficient")["summary"][0][9]
        #
        # Fit the posterior probability, then output fitting parameters
        #

        posterior = mf.S_model_posterior(cell_num, selection_coefficient, log_joint_prob)

        posterior.maximum_llk_S_Model_GammaDist_Parameters()
        # outputdict.update ({ 'k': posterior.k, 'a': posterior.a, 'b': posterior.b,
        #                    'mean_s': posterior.mean_s, 'var_s': posterior.var_s,
        #              'log_norm': posterior.log_normalization_const,
        #              'Rhat_cellnum': Rhat_cellnum, 'Rhat_selection': Rhat_selection})

        outputdict.update(
            {'k': posterior.k, 'a': posterior.a, 'b': max(10 ** (-30), posterior.b), 'mean_s': posterior.mean_s,
             'var_s': posterior.var_s, 'log_norm': posterior.log_normalization_const})
        log_prob_survive_cummulated = inputdict['log_prob_survive'] + np.log(results['prob_survive'][0])
        outputdict.update({'log_prob_survive_cummulated': log_prob_survive_cummulated})

        '''
        if ((FLAG_PLOT_POSTERIOR_SN) and (model_name == MODEL_NAME['SN'])) or ((FLAG_PLOT_POSTERIOR_SS) and (model_name == MODEL_NAME['SS'])):

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
            main_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            main_ax.set_title(f'BCID {BCID}    T {current_time}', y=1.02, fontsize=18)

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
            plot_file_name = 'posterior_'+model_name+f"_T{current_time}_BC_{BCID}.png"

            fig.savefig(OutputFileDir+plot_file_name, dpi=400)
        '''

    return outputdict #, posterior


#
# Function output posterior information to the file
#
def put_posterior_to_file(run_dict, done_queue, TASK_size, glob):
    model_name = run_dict['model_name']
    lineage_name = run_dict['lineage_name']

    # output lineage posterior to file
    #
    outfilename = 'posterior_' + lineage_name + '_' + model_name + f"_T{glob.current_timepoint}.txt"
    print(outfilename)

    if (model_name == MODEL_NAME['N']):
        file = open(OutputFileDir + outfilename, 'w')
        file.write("listID\t BCID\t TAG\t k\t theta\t log_normalization\t log_prob_survive_cummulated\n")

        for i in range(TASK_size):
            result = done_queue.get()
            file.write(
                f"{result['listID']}\t{result['BCID']}\t{result['TAG']}\t{result['k']}\t{result['theta']}\t{result['log_norm']}\t{result['log_prob_survive_cummulated']}\n")
        file.close()
    elif (model_name == MODEL_NAME['SN']) or (model_name == MODEL_NAME['SS']):
        file = open(OutputFileDir + outfilename, 'w')
        file.write(
            "listID\t  BCID\tTAG\t k\t a\t b\t s_mean\t s_var\t log_normalization \t log_prob_survive_cummulated\n")

        for i in range(TASK_size):
            result = done_queue.get()
            # print(result['BCID'])  # 20251201
            file.write(
                f"{result['listID']}\t {result['BCID']}\t{result['TAG']}\t{result['k']}\t {result['a']}\t {result['b']}\t {result['mean_s']}\t {result['var_s']}\t {result['log_norm']}\t {result['log_prob_survive_cummulated']}\n")
        file.close()
    else:
        print(
            f"ERROR! function \"put_posterior_to_file\" could not identify the input model_name. model_name should be one of {MODEL_NAME}. ")

def write_results(results, run_dict, glob):
    model_name = run_dict['model_name']
    lineage_name = run_dict['lineage_name']

    outfilename = (
        f"posterior_{lineage_name}_{model_name}_T{glob.current_timepoint}.txt"
    )
    print("Outputing results to", outfilename)

    # optional: preserve deterministic order
    results = sorted(results, key=lambda r: r['listID'])
    results_ok = [r for r in results if r.get("status") == "ok"]
    results_fail = [r for r in results if r.get("status") != "ok"]

    with open(OutputFileDir + outfilename, 'w') as f:

        if model_name == MODEL_NAME['N']:
            f.write(
                "listID\tBCID\tTAG\tk\ttheta\t"
                "log_normalization\tlog_prob_survive_cummulated\n"
            )
            for r in results_ok:
                f.write(
                    f"{r['listID']}\t{r['BCID']}\t{r['TAG']}\t"
                    f"{r['k']}\t{r['theta']}\t{r['log_norm']}\t"
                    f"{r['log_prob_survive_cummulated']}\n"
                )

        elif model_name in (MODEL_NAME['SN'], MODEL_NAME['SS']):
            f.write(
                "listID\tBCID\tTAG\tk\ta\tb\t"
                "s_mean\ts_var\tlog_normalization\t"
                "log_prob_survive_cummulated\n"
            )
            for r in results_ok:
                f.write(
                    f"{r['listID']}\t{r['BCID']}\t{r['TAG']}\t"
                    f"{r['k']}\t{r['a']}\t{r['b']}\t"
                    f"{r['mean_s']}\t{r['var_s']}\t{r['log_norm']}\t"
                    f"{r['log_prob_survive_cummulated']}\n"
                )

        else:
            raise ValueError(f"Unknown model: {model_name}")
# =====================================================================================
# Part C: Work on lineage
#
# function to create lineages: readfile2lineage
#
# functions:
#   readfile2lineage
#
# =====================================================================================

#
# Function creates the tag list of lineage from the past time point
#
def create_lineage_list_by_pastTag(lins, current_time, lineage_info, const):
    last_time = current_time - 1

    # Update the reads value to current time
    for lin in lins:
        lin.set_reads(last_time=last_time)

    #
    # Initialization
    #
    if last_time == 0:  # # initial time point
        # Initilization of lineage
        for lin in lins:
            mu_r = float((0.001 + lin.r0))
            k = mu_r / (1 + mu_r * const.eps)
            theta = (1 + mu_r * const.eps) / const.Rt[0] * const.Nt[0]
            # lin.nm.UPDATE_POST_PARM(k=lin.r0+0.001, theta=float(const.Nt[0]/const.Rt[0]),  log_norm= 0., log_prob_survive=0.)
            lin.nm.UPDATE_POST_PARM(k=k, theta=theta, log_norm=0., log_prob_survive=0.)
            lin.sm.UPDATE_POST_PARM(k=k, a=theta * k, b=0, mean_s=0.00 * np.log2(mc.D),
                                    var_s=(0.1 * np.log2(mc.D)) ** 2, log_norm=0, log_prob_survive=0)
            lin._init_TAG()
    #
    # Read The PAST information from file and read PastTAG of lineage
    #
    elif last_time > 0:
        lins_survive = []
        for lin in lins:
            if lin.T_END > current_time:
                lins_survive.append(lin)

        lins = lins_survive
        lins = readfile2lineage(lins, lineage_info['lineage_name'], last_time=last_time)

    return lins


def readfile2lineage(lins, lineage_name, last_time):
    t = last_time

    if len(lins) > 0:
        #
        # Create a dictionary to store the bcid to lin-index:
        bcid_dict = {}
        for i in range(len(lins)):
            bcid_dict.update({lins[i].BCID: i})

        #
        # Read SModel file 2 lineages
        read_model_name = MODEL_NAME['SS']

        readfilename = 'posterior_' + lineage_name + '_' + read_model_name + f"_T{t}.txt"
        # print(readfilename)

        if os.path.exists(OutputFileDir + readfilename):

            f = open(OutputFileDir + readfilename, 'r')
            list_of_lines = f.readlines()
            f.close()

            list_of_lines[0] = list_of_lines[0].split('\n')[0] + '\t log10 Bayes Factor\n'

            for j in range(1, len(list_of_lines)):
                read_line_0 = list_of_lines[j].split('\n')[0]
                read_line = read_line_0.split('\t')
                BCID = int(read_line[1])
                if BCID in bcid_dict.keys():
                    i = bcid_dict[BCID]
                    # print(read_line)
                    lins[i].sm.UPDATE_POST_PARM(k=float(read_line[3]), a=float(read_line[4]), b=float(read_line[5]),
                                                mean_s=float(read_line[6]), var_s=float(read_line[7]),
                                                log_norm=float(read_line[8]), log_prob_survive=float(read_line[9]))

                    log10_BF = lins[i].log10_BayesFactor()
                    list_of_lines[j] = read_line_0 + '\t' + str(log10_BF) + '\n'

            #
            # Add log10 Bayes factor to SModel file
            f = open(OutputFileDir + readfilename, 'w')
            f.writelines(list_of_lines)
            f.close()

        #
        # Classify Putative lineage class at t+1 based on the (past) Bayes factor
        for i in range(len(lins)):
            lins[i].reTAG(last_time=t)  # Get current putative lineage class

    return lins

#
# silence_pystan_output credited by https://gist.github.com/ahartikainen/06192df9719031cf1a22b887b7b5d67b
#
# silence logger, there are better ways to do this
# see PyStan docs
logging.getLogger("pystan").propagate=False
logger = logging.getLogger("httpstan")
logger.setLevel(logging.ERROR)
def silence_stdout():
    sys.stdout.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
# def drain_pipe(captured_stdout, stdout_pipe):
def drain_pipe(buffer, stdout_pipe):
    while True:
        data = os.read(stdout_pipe[0], 1024)
        if not data:
            break
        # captured_stdout += data
        buffer.append(data)

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
    buffer = []
    t = threading.Thread(target=drain_pipe, args=(buffer, stdout_pipe))
    # t = threading.Thread(target=lambda:drain_pipe(captured_stdout, stdout_pipe))
    t.start()
    # run user function
    result = function(*args, **kwargs)
    os.close(stdout_fileno)
    t.join()
    os.close(stdout_pipe[0])
    os.dup2(stdout_save, stdout_fileno)
    os.close(stdout_save)
    # return result, captured_stdout.decode("utf-8")
    return result, b"".join(buffer).decode("utf-8")

#
# Add wall-clock timeout
#
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Stan sampling timeout")