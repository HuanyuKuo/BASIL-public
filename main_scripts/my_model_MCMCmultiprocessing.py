# -*- coding: utf-8 -*-
"""
Refactored MCMC multiprocessing for BASIL
@author: huanyu
"""

import os
import sys
import pickle
import threading
import logging
from collections import namedtuple
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import myFunctions as mf
import myConstant as mc

# ----------------- global variables -----------------
NUMBER_OF_PROCESSES = mc.NUMBER_OF_PROCESSES
MODEL_NAME = mc.MODEL_NAME
OutputFileDir = mc.OutputFileDir
WRITE_CHUNK = 1000
_STAN_MODEL = None

# ----------------- Stan output logging -----------------
logging.getLogger("pystan").propagate = False
logger = logging.getLogger("httpstan")
logger.setLevel(logging.ERROR)

# ----------------- MCMC worker and multiprocessing -----------------
def init_worker(model_name):
    """Initialize worker with pre-loaded Stan model"""
    global _STAN_MODEL
    with open(f'./model_code/{model_name}.pkl', 'rb') as f:
        _STAN_MODEL = pickle.load(f)

def MCMC_sampling_worker(lineage_input):
    """Worker function to run MCMC for a single lineage"""
    global _STAN_MODEL
    try:
        if _STAN_MODEL is None:
            raise RuntimeError("Stan model not initialized in worker")

        META_KEYS = {'t', 'BCID', 'model_name', 'listID', 'tag', 'log_prob_survive'}
        input_data = {k: getattr(lineage_input, k) for k in lineage_input._fields if k not in META_KEYS}

        fit, _ = capture_output(
            _STAN_MODEL.sampling,
            data=input_data,
            init=0,
            pars=get_pars_interest(lineage_input.model_name),
            warmup=1000,
            iter=3500,
            chains=1,
            seed=10000 + lineage_input.listID,
            n_jobs=1,
            algorithm='NUTS',
            control={'adapt_delta': 0.8},
            refresh=0,
        )

        outputdict = get_posterior_info(lineage_input._asdict(), fit)
        outputdict["status"] = "ok"
        return outputdict

    except Exception as e:
        return {
            "listID": lineage_input.listID,
            "BCID": lineage_input.BCID,
            "TAG": lineage_input.tag,
            "status": "failed",
            "error": str(e)
        }

def run_model_MCMCmultiprocessing(run_dict, lins, glob):
    """Run MCMC on all lineages with multiprocessing and streaming output"""
    model_name = run_dict['model_name']
    lineage_name = run_dict['lineage_name']

    outfilename = os.path.join(OutputFileDir,
                               f"posterior_{lineage_name}_{model_name}_T{glob.current_timepoint}.txt")

    inputs = get_Input_data_generator(model_name, lins, glob)

    # write header
    with open(outfilename, 'w') as f:
        if model_name == MODEL_NAME['N']:
            f.write("listID\tBCID\tTAG\tk\ttheta\tlog_normalization\tlog_prob_survive_cummulated\n")
        else:
            f.write("listID\tBCID\tTAG\tk\ta\tb\ts_mean\ts_var\tlog_normalization\tlog_prob_survive_cummulated\n")

    buffer = []

    with Pool(processes=NUMBER_OF_PROCESSES, initializer=init_worker, initargs=(model_name,), maxtasksperchild=10) as pool:
        total = len(lins) if hasattr(lins, "__len__") else None
        for res in tqdm(pool.imap_unordered(MCMC_sampling_worker, inputs, chunksize=5), total=total):
            buffer.append(res)
            if len(buffer) >= WRITE_CHUNK:
                _write_results_chunk(buffer, outfilename, model_name)
                buffer.clear()
        if buffer:
            _write_results_chunk(buffer, outfilename, model_name)

    print(f"Finished writing: {outfilename}")

def _write_results_chunk(results, filename, model_name):
    """Write results to file with flush for streaming"""
    with open(filename, "a", buffering=1) as f:
        for r in results:
            if r.get("status") != "ok":
                continue
            assert isinstance(r['listID'], int), r
            assert isinstance(r['BCID'], int), r

            if model_name == MODEL_NAME['N']:
                f.write(f"{r['listID']}\t{r['BCID']}\t{r['TAG']}\t"
                        f"{r['k']}\t{r['theta']}\t{r['log_norm']}\t{r['log_prob_survive_cummulated']}\n")
            else:
                f.write(f"{r['listID']}\t{r['BCID']}\t{r['TAG']}\t"
                        f"{r['k']}\t{r['a']}\t{r['b']}\t{r['mean_s']}\t{r['var_s']}\t"
                        f"{r['log_norm']}\t{r['log_prob_survive_cummulated']}\n")
        f.flush()
        os.fsync(f.fileno())

# ----------------- Input and output helpers -----------------
LineageInput = namedtuple(
    "LineageInput",
    [
        "listID", "BCID", "tag", "model_name",
        "barcode_count", "read_depth", "population_size", "epsilon",
        "meanfitness", "cycle", "dilution_ratio",
        "log_prob_survive", "k_", "theta_", "a_", "b_", "mean_s", "var_s", "t"
    ]
)

def get_Input_data_generator(model_name, lins, glob):
    """Generate input data for MCMC sampling"""
    for i, lin in enumerate(lins):
        RD, N, D, C, meanfitness, epsilon, t = glob.R, glob.N, glob.D, glob.C, glob.meanfitness, glob.epsilon, glob.current_timepoint
        if model_name == MODEL_NAME['N']:
            yield LineageInput(
                listID=i,
                BCID=lin.BCID,
                tag=lin.TYPETAG,
                model_name=model_name,
                barcode_count=np.int64(lin.r1),
                read_depth=RD,
                population_size=N,
                epsilon=epsilon,
                meanfitness=meanfitness,
                cycle=C,
                dilution_ratio=D,
                log_prob_survive=lin.nm.log_prob_survive,
                k_=lin.nm.post_parm_Gamma_k,
                theta_=lin.nm.post_parm_Gamma_theta,
                a_=0, b_=0, mean_s=0, var_s=0,
                t=t
            )
        else:  # SS / SN
            yield LineageInput(
                listID=i,
                BCID=lin.BCID,
                tag=lin.TYPETAG,
                model_name=model_name,
                barcode_count=np.int64(lin.r1),
                read_depth=RD,
                population_size=N,
                epsilon=epsilon,
                meanfitness=meanfitness,
                cycle=C,
                dilution_ratio=D,
                log_prob_survive=lin.sm.log_prob_survive,
                k_=lin.sm.post_parm_Gamma_k,
                theta_=0,
                a_=lin.sm.post_parm_Gamma_a,
                b_=lin.sm.post_parm_Gamma_b,
                mean_s=lin.sm.post_parm_NormS_mean,
                var_s=lin.sm.post_parm_NormS_var,
                t=t
            )

def get_pars_interest(model_name):
    """Return parameters of interest for each model"""
    if model_name == MODEL_NAME['N']:
        return ['cell_num', 'log_joint_prob', 'prob_survive']
    elif model_name in (MODEL_NAME['SN'], MODEL_NAME['SS']):
        return ['cell_num', 'selection_coefficient', 'log_joint_prob', 'prob_survive']
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def get_posterior_info(inputdict, fit):
    """Extract posterior info from Stan fit"""
    model_name = inputdict['model_name']
    results = fit.extract(permuted=True)
    del fit
    outputdict = {'listID': inputdict['listID'], 'BCID': inputdict['BCID'], 'TAG': inputdict['tag']}

    if model_name == MODEL_NAME['N']:
        posterior = mf.N_model_posterior(data=results['cell_num'], log_joint_prob=results['log_joint_prob'])
        outputdict.update({
            'k': posterior.k,
            'theta': posterior.theta,
            'log_norm': posterior.log_normalization_const,
            'log_prob_survive_cummulated': inputdict['log_prob_survive'] + np.log(results['prob_survive'][0])
        })
    else:  # SS / SN
        posterior = mf.S_model_posterior(results['cell_num'], results['selection_coefficient'], results['log_joint_prob'])
        posterior.maximum_llk_S_Model_GammaDist_Parameters()
        outputdict.update({
            'k': posterior.k,
            'a': posterior.a,
            'b': max(1e-30, posterior.b),
            'mean_s': posterior.mean_s,
            'var_s': posterior.var_s,
            'log_norm': posterior.log_normalization_const,
            'log_prob_survive_cummulated': inputdict['log_prob_survive'] + np.log(results['prob_survive'][0])
        })

    return outputdict

# ----------------- Helper to capture stdout from Stan -----------------
def drain_pipe(buffer, stdout_pipe):
    while True:
        data = os.read(stdout_pipe[0], 1024)
        if not data:
            break
        buffer.append(data)

def capture_output(function, *args, **kwargs):
    stdout_fileno = sys.stdout.fileno()
    stdout_save = os.dup(stdout_fileno)
    stdout_pipe = os.pipe()
    os.dup2(stdout_pipe[1], stdout_fileno)
    os.close(stdout_pipe[1])

    buffer = []
    t = threading.Thread(target=drain_pipe, args=(buffer, stdout_pipe))
    t.start()
    result = function(*args, **kwargs)
    os.close(stdout_fileno)
    t.join()
    os.close(stdout_pipe[0])
    os.dup2(stdout_save, stdout_fileno)
    os.close(stdout_save)
    return result, b"".join(buffer).decode("utf-8")

# ----------------- Lineage helpers -----------------
# def create_lineage_list_by_pastTag(lins, current_time, lineage_info, const):
#     """Create lineage list using past timepoint information"""
#     last_time = current_time - 1
#
#     for lin in lins:
#         lin.set_reads(last_time=last_time)
#
#     if last_time == 0:
#         for lin in lins:
#             mu_r = float(0.001 + lin.r0)
#             k = mu_r / (1 + mu_r * const.eps)
#             theta = (1 + mu_r * const.eps) / const.Rt[0] * const.Nt[0]
#             lin.nm.UPDATE_POST_PARM(k=k, theta=theta, log_norm=0., log_prob_survive=0.)
#             lin.sm.UPDATE_POST_PARM(k=k, a=theta*k, b=0, mean_s=0.0*np.log2(mc.D),
#                                     var_s=(0.1*np.log2(mc.D))**2, log_norm=0., log_prob_survive=0)
#             lin._init_TAG()
#     elif last_time > 0:
#         lins = [lin for lin in lins if lin.T_END > current_time]
#         lins = readfile2lineage(lins, lineage_info['lineage_name'], last_time=last_time)
#
#     return lins

def create_lineage_list_by_pastTag(lins, current_step, lineage_info, const ):


    # Update the reads value to current time
    for lin in lins:
        lin.set_reads(last_time=current_step -1)
    #
    # Initialization
    #
    if (current_step==1) and (lineage_info['initializing_lineage_filename'] is None):
        print('Initializing lineage list')
        #
        # Initilization of lineage by default
        #
        for lin in lins:
            mu_r = float((0.001+lin.r0))
            k = mu_r/(1+mu_r*const.eps)
            theta = (1+mu_r*const.eps)/const.Rt[0]*const.Nt[0]
            #lin.nm.UPDATE_POST_PARM(k=lin.r0+0.001, theta=float(const.Nt[0]/const.Rt[0]),  log_norm= 0., log_prob_survive=0.)
            lin.nm.UPDATE_POST_PARM(k=k, theta=theta, log_norm= 0., log_prob_survive=0.)
            lin.sm.UPDATE_POST_PARM(k=k, a=theta*k, b=0, mean_s=0.00*np.log2(mc.D), var_s=(0.1*np.log2(mc.D))**2, log_norm=0, log_prob_survive=0)
            lin._init_TAG()
    else:
        #
        # Read lineage information from file
        #
        if current_step == 1:
            readfilename = lineage_info['initializing_lineage_filename']
            print('Initializing lineage list from file', readfilename)
        else:
            last_step = current_step - 1
            T_file_to_read = lineage_info['file_start_time'] - 1 + last_step
            readfilename = 'posterior_' + lineage_info['lineage_name'] + '_' + MODEL_NAME['SS'] + f"_T{T_file_to_read}.txt"

        lins_survive = []
        for lin in lins:
            if lin.T_END > current_step:
                lins_survive.append(lin)
        lins = lins_survive
        lins = readfile2lineage(lins, readfilename=readfilename, last_step=current_step-1)

    return lins

def readfile2lineage(lins, readfilename, last_step):
    """Update lineage objects from previous posterior file"""
    t = last_step
    if len(lins) == 0:
        return lins

    bcid_dict = {lin.BCID: i for i, lin in enumerate(lins)}
    read_model_name = MODEL_NAME['SS']
    # readfilename = os.path.join(OutputFileDir, f'posterior_{lineage_name}_{read_model_name}_T{t}.txt')

    if os.path.exists(readfilename):
        with open(readfilename, 'r') as f:
            list_of_lines = f.readlines()

        list_of_lines[0] = list_of_lines[0].strip() + '\t log10 Bayes Factor\n'

        for j in range(1, len(list_of_lines)):
            read_line = list_of_lines[j].strip().split('\t')
            BCID = int(read_line[1])
            if BCID in bcid_dict:
                i = bcid_dict[BCID]
                lins[i].sm.UPDATE_POST_PARM(
                    k=float(read_line[3]),
                    a=float(read_line[4]),
                    b=float(read_line[5]),
                    mean_s=float(read_line[6]),
                    var_s=float(read_line[7]),
                    log_norm=float(read_line[8]),
                    log_prob_survive=float(read_line[9])
                )
                log10_BF = lins[i].log10_BayesFactor()
                list_of_lines[j] = list_of_lines[j].strip() + f'\t{log10_BF}\n'

        with open(readfilename, 'w') as f:
            f.writelines(list_of_lines)

        for lin in lins:
            lin.reTAG(last_time=t)

    return lins
