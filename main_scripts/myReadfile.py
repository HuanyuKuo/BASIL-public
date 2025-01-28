# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:15:08 2021

@author: huanyu
"""

import myConstant as mc
from myVariables import (Constant, Global, Lineage)
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os.path
#
# read input Barcode count data
def my_readfile(filename):
    
    filedirname = mc.InputFileDir + filename 
    Lins = []
    
    f = open(filedirname)
    line = f.readline()
    t_arr = line.split('\n')[0].split('\t')[1::]
    t_cycle = [ float(t.split('cycle')[0].split('=')[1]) for t in t_arr]
    #cycles = [t_cycle[i+1]-t_cycle[i] for i in range(len(t_cycle)-1)]
    totalread = np.zeros(len(t_cycle))
    line = f.readline()
    while line:
        reads = line.split('\n')[0].split('\t')
        BCID = int(reads[0])
        reads = [ int(r) for r in reads[1::]]
        totalread += np.asarray(reads)
        Lins.append(Lineage(reads=reads, BCID=BCID))
        line = f.readline()
    f.close()
    #return Lins, totalread, cycles
    return Lins, totalread, t_cycle

#
# output global parameters
def output_global_parameters_BFM(lineage_info, const):
    eps_Bayes = []
    meanfitness_Bayes = []
    fdir =  mc.OutputFileDir  
    for t in range(1, const.T):
        fname = fdir + 'glob_'+ lineage_info['lineage_name'] + f'_T{t}.txt'
        f =open(fname)
        f.readline()
        f.readline()
        r = f.readline()
        eps = float( r.split('\n')[0].split('\t')[1])
        eps_Bayes.append(eps)
        r = f.readline()
        sbar = float( r.split('\n')[0].split('\t')[1])
        meanfitness_Bayes.append(sbar)
        f.close()
        
    #_t_Bayes = [sum(const.Ct[1:i]) for i in range(1,len(const.Ct)+1)]
    #t_Bayes = [(_t_Bayes[i]+_t_Bayes[i+1])/2 for i in range(len(_t_Bayes)-1)]
    t_Bayes = [(const.T_cycle[i]+const.T_cycle[i+1])/2 for i in range(len(const.T_cycle)-1)]
    # Output Mean-fitness file
    f = open(mc.OutputFileDir + 'Bayesian_global_parameters_'+lineage_info['lineage_name']+'.txt','w')
    f.write('Time (cycle)\tMean-fitness(1/cycle)\tEpsilon\n')
    for i in range(len(t_Bayes)):
        f.write(str(t_Bayes[i])+'\t'+str(meanfitness_Bayes[i])+'\t'+str(eps_Bayes[i])+'\n')
    f.close()
    #
    # Make plots
    plt.figure()
    plt.plot(t_Bayes, meanfitness_Bayes, 'bo-', label= lineage_info['lineage_name'])
    plt.legend()
    plt.xlabel('time (cycle)')
    plt.title('Meanfitness(1/cycle)')
    plt.xlim(0, max(t_Bayes)+1)
    plt.savefig(mc.OutputFileDir+'meanfitness_trajectory_Bayes_'+lineage_info['lineage_name']+'.png',dpi=200)
    
    plt.figure()
    plt.plot(t_Bayes, eps_Bayes, 'bo-',label= lineage_info['lineage_name'])
    plt.legend()
    plt.xlabel('time (cycle)')
    plt.xlim(0, max(t_Bayes)+1)
    plt.title('Systematic Error Epsilon')
    plt.savefig(mc.OutputFileDir +'Epsilon_trajectory_Bayes_'+lineage_info['lineage_name']+'.png',dpi=200)
    
def read_global_parameters_BFM(lineage_info):
    t_arr_cycle = []
    meanfitness_Bayes_cycle = []
    epsilon_Bayes = []
    f = open(mc.OutputFileDir + 'Bayesian_global_parameters_'+lineage_info['lineage_name']+'.txt','r')
    f.readline()
    line = f.readline()
    while(line):
        line = line.split('\n')[0].split('\t')
        t_arr_cycle.append(float(line[0]))
        meanfitness_Bayes_cycle.append(float(line[1]))
        epsilon_Bayes.append(float(line[2]))
        line = f.readline()
    f.close()
    return meanfitness_Bayes_cycle, epsilon_Bayes, t_arr_cycle

def output_Posterior_parameters_Bayes_v5(lineage_info, datafilename): # 2025

    # output the parameters (k,a,b,s_mean, s_var) of parametric posterior for all lineages for all timepoints

    BFM_result, t_arr = read_Posterior_parameters_Bayes_v5(lineage_info, datafilename)

    print(len(BFM_result))

    # Prepare for output
    case_name = lineage_info['lineage_name']
    f = open(mc.OutputFileDir + 'BASIL_All_posterior_parameters_' + case_name + '.txt', 'w')

    headline_list = ['k_T{:d}'.format(t)+ '\t' + 'a_T{:d}'.format(t)+ '\t'+ 'b_T{:d}'.format(t)+ '\t'
                + 's_mean_T{:d}'.format(t) + '\t' + 's_var_T{:d}'.format(t) +'\t'+ 'logZ_T{:d}'.format(t) for t in t_arr]
    headline_str ='BCID\t' +  '\t'.join(headline_list) +'\n'
    f.write(headline_str)

    for j in range(len(BFM_result)):
        out = BFM_result[j]
        f.write(str(out['BCID'])+'\t')
        for t in t_arr:
            idx = np.where(np.asarray(out['t_arr']) == t)[0]
            if len(idx)>0:
                idx = idx[0]
                f.write(str(out['k_arr'][idx])+'\t')
                f.write(str(out['a_arr'][idx]) + '\t')
                f.write(str(out['b_arr'][idx]) + '\t')
                f.write(str(out['s_mean_arr'][idx]) + '\t')
                f.write(str(out['s_var_arr'][idx]) + '\t')
                f.write(str(out['log_norm_S_arr'][idx]) + '\t')
            else:
                f.write('\t\t\t\t\t\t')


        f.write('\n')

    f.close()
    return BFM_result

def read_Posterior_parameters_Bayes_v5(lineage_info, datafilename): # 2025

    lins, totalread, cycles = my_readfile(datafilename)
    BCIDs = [lins[i].BCID for i in range(len(lins))]
    dict_BCID_2_idx = {}
    for i in range(len(BCIDs)):
        dict_BCID_2_idx.update({BCIDs[i]: i})
    BFM_result = [{'BCID': lins[i].BCID, 'T_END': lins[i].T_END, 't_arr':[],
                   's_mean_arr':[], 's_var_arr':[], 'k_arr': [],  'a_arr':[], 'b_arr':[],
                   'log_norm_N_arr':[], 'log_norm_S_arr':[],
                   'n_mean_S_arr':[], 'n_std_S_arr':[], 'n_mean_N_arr':[], 'n_std_N_arr':[]}  for i in range(len(lins))]

    total_timepoint = len(totalread)
    tend = total_timepoint - 1

    #t_arr = [tend - i for i in range(0, tend )]
    t_arr = [i for i in range(1, total_timepoint)]

    for t in t_arr:
        # print(t)
        tmp_bcids_SS, tmp_s_mean_SS, tmp_s_var_SS, tmp_log_norm_SS, tmp_expected_n_mean_SS, tmp_expected_n_std_SS, tmp_param_k, tmp_param_a, tmp_param_b = get_Posterior_parameters_from_BayesFiles_v5(lineage_info, t)
        #print(len(tmp_bcids_SS), len(tmp_bcids_N))
        for i in range(len(tmp_bcids_SS)):

            bcid =tmp_bcids_SS[i]
            j  =dict_BCID_2_idx[bcid]

            BFM_result[j]['t_arr'].append(t)
            BFM_result[j]['s_mean_arr'].append(tmp_s_mean_SS[i])
            BFM_result[j]['s_var_arr'].append(tmp_s_var_SS[i])
            BFM_result[j]['k_arr'].append(tmp_param_k[i])
            BFM_result[j]['a_arr'].append(tmp_param_a[i])
            BFM_result[j]['b_arr'].append(tmp_param_b[i])
            BFM_result[j]['log_norm_S_arr'].append(tmp_log_norm_SS[i])
            BFM_result[j]['n_mean_S_arr'].append(tmp_expected_n_mean_SS[i])
            BFM_result[j]['n_std_S_arr'].append(tmp_expected_n_std_SS[i])

    # reduce size
    BFM_result = [BFM_result[j] for j in range(len(BFM_result)) if len(BFM_result[j]['t_arr'])>1]

    return BFM_result, t_arr

def get_Posterior_parameters_from_BayesFiles_v5(lineage_info, t):
    MODEL_NAME = mc.MODEL_NAME
    OutputFileDir = mc.OutputFileDir
    lineage_name = lineage_info['lineage_name']

    tmp_bcids_SS, tmp_s_mean_SS, tmp_s_var_SS, tmp_log_norm_SS, tmp_expected_n_mean_SS, tmp_expected_n_std_SS = [], [], [], [], [],[]
    tmp_param_k, tmp_param_a, tmp_param_b = [], [], []

    readfilename = 'posterior_' + lineage_name + '_' + MODEL_NAME['SS'] + f"_T{t}.txt"
    # print(readfilename)
    if os.path.exists(OutputFileDir + readfilename) is False:
        print('No file ' + OutputFileDir + readfilename + '\n')
    else:
        f = open(OutputFileDir + readfilename, 'r')
        a = f.readlines()
        f.close()
        for i in range(1, len(a)):
            b = a[i].split('\n')[0].split('\t')
            if len(b) > 8:
                tmp_bcids_SS.append(int(b[1]))
                tmp_s_mean_SS.append(float(b[6]))
                tmp_s_var_SS.append(float(b[7]))
                tmp_log_norm_SS.append(float(b[8]))
                param_k = float(b[3])
                tmp_param_k.append(param_k)
                param_a = float(b[4])
                tmp_param_a.append(param_a)
                param_b = float(b[5])
                tmp_param_b.append(param_b)
                tmp_expected_n_mean_SS.append(param_a*np.exp(param_b**2/2))
                tmp_expected_n_std_SS.append(param_a*np.exp(param_b**2/2)*param_b)


    return tmp_bcids_SS, tmp_s_mean_SS, tmp_s_var_SS, tmp_log_norm_SS, tmp_expected_n_mean_SS, tmp_expected_n_std_SS, tmp_param_k, tmp_param_a, tmp_param_b

def output_Selection_Coefficient_Bayes_v5(lineage_info, datafilename, BFM_result=None, beta=mc.beta):

    if BFM_result == None:
        BFM_result, _ = read_Posterior_parameters_Bayes_v5(lineage_info, datafilename)

    # Prepare for output
    case_name = lineage_info['lineage_name']
    f = open(mc.OutputFileDir + 'BASIL_Selection_Coefficient_' + case_name + '_ConfidenceFactorBeta={:.2f}'.format(beta)+'.txt', 'w')
    f.write('ADP_BCID_Bayes\tADP_s_mean_Bayes\tADP_s_std_Bayes\tADP_s_time\n')

    bcid_all_result = {}
    bcid_adp_fit = []

    for j in range(len(BFM_result)):
        out = BFM_result[j]
        out.update({'TYPE': 'NEU'})
        bcid_all_result.update({out['BCID']: j})

        s_var_arr = out['s_var_arr']
        idx = np.where(np.asarray(s_var_arr)==min(s_var_arr))[0][0]
        s_mean = out['s_mean_arr'][idx]
        s_std = np.sqrt(out['s_var_arr'][idx])

        out['s_mean'] = s_mean
        out['s_std'] = s_std

        if (s_mean- beta*s_std)>0:

            f.write(str(out['BCID']) + '\t')                        # bcid
            f.write(str(s_mean) + '\t')                             # s mean
            f.write(str(s_std) + '\t')                              # s std
            f.write(str(out['t_arr'][idx]) + '\n')                  # s time
            BFM_result[j]['TYPE']='ADP'
            bcid_adp_fit.append(out['BCID'])
    f.close()

    s_mean_ADP = [BFM_result[bcid_all_result[bcid]]['s_mean'] for bcid in bcid_adp_fit]
    s_std_ADP = [BFM_result[bcid_all_result[bcid]]['s_std'] for bcid in bcid_adp_fit]
    bcid_neutral_fit = list(set(list(bcid_all_result.keys())) - set(bcid_adp_fit) )
    s_mean_NEU = [BFM_result[bcid_all_result[bcid]]['s_mean'] for bcid in bcid_neutral_fit]
    s_std_NEU = [BFM_result[bcid_all_result[bcid]]['s_std'] for bcid in bcid_neutral_fit]

    # Plot the mean and standard deviation of selection coefficients
    plt.figure()
    plt.plot(s_mean_NEU / np.log2(mc.D)*100, s_std_NEU / np.log2(mc.D)*100, 'k.', ms=0.5, alpha=0.02,
             label=f'Neutral (n={len(s_mean_NEU)})')
    plt.plot(s_mean_ADP / np.log2(mc.D)*100, s_std_ADP / np.log2(mc.D)*100, 'r.', ms=1, alpha=0.3,
             label=f'Adaptive (n={len(s_mean_ADP)})')
    xmin = -25 # min(min(s_mean_NEU),min(s_mean_ADP))
    xmax = 25 # max(max(s_mean_NEU),max(s_mean_ADP))
    ymin = 0
    ymax = 10 # max(max(s_std_NEU),max(s_std_ADP))

    plt.plot([0, xmax], [0, xmax/beta], ':', color='b',label='conficence factor (beta) = {:.1f}'.format(beta))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('s mean (%)')
    plt.ylabel('s std (%)')
    plt.legend()
    plt.savefig(mc.OutputFileDir + 'BASIL_Selection_Coefficient_' + case_name + '_ConfidenceFactorBeta={:.2f}'.format(beta) + '.pdf')

    #
    # Plot the lineages trajectories, colored by inferred s
    #
    lins, totalread, t_cycles = my_readfile(datafilename)

    log10_total_reads = np.log10(totalread)

    BCIDs = [lins[i].BCID for i in range(len(lins))]
    dict_BCID_2_idx = {}
    for i in range(len(BCIDs)):
        dict_BCID_2_idx.update({BCIDs[i]: i})

    idx_adaptive, idx_neutral = [],[]
    s_sort_idx = {}
    for idx in range(0, len(lins)):
        lin = lins[idx]
        if lin.BCID in bcid_adp_fit:
            idx_adaptive.append(idx)
        else:
            idx_neutral.append(idx)


    fig, ax = plt.subplots(1,1,figsize=(8, 5.5))
    # define color map
    cmap = matplotlib.cm.get_cmap("jet")
    # need to normalize because color maps are defined in [0, 1]
    s_max = int(max(s_mean_ADP / np.log2(mc.D)*100))
    s_min = int(0)
    norm = matplotlib.colors.Normalize(s_min, s_max)


    # Plot neutral lineages (do every 20th elements)
    for bcid in bcid_neutral_fit[0::20]:
        idx = dict_BCID_2_idx[bcid]
        lin = lins[idx]
        bc_count = lin.rt
        bc_count = bc_count + (bc_count == 0) * (0.1)  # avoid inf in log10
        log10_bc_count = np.log10(bc_count)
        log10freq = log10_bc_count - log10_total_reads
        ax.plot(t_cycles, log10freq, color='grey', linewidth=0.3)

    # Plot adaptive lineages (colored by s)

    for bcid in bcid_adp_fit:
        idx = dict_BCID_2_idx[bcid]
        lin = lins[idx]
        bc_count = lin.rt
        bc_count = bc_count + (bc_count == 0) * (0.1)  # avoid inf in log10
        log10_bc_count = np.log10(bc_count)
        log10freq = log10_bc_count - log10_total_reads

        s = BFM_result[bcid_all_result[bcid]]['s_mean'] / np.log2(mc.D) * 100
        col_idx = int(max(s_min, min(s_max, (s))))
        ax.plot(t_cycles, log10freq, color=cmap(norm(col_idx)), linewidth=0.3)

    ax.set_xlabel('time (cycle)')
    ax.set_ylabel('Log10 Barcode Frequency')

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))
    cbar.ax.set_title('s, %', )


    plt.savefig(mc.OutputFileDir+ 'BASIL_Barcode_Trajectory_' + case_name + '_ConfidenceFactorBeta={:.2f}'.format(beta) + '.pdf')

    return BFM_result

if __name__ == '__main__':
    
    datafilename =  'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    
    lins, totalread, cycles = my_readfile(datafilename)
    _const = Constant(totalread, cycles)
    for t in range(1, _const.T):
        _const.Ct[t] = cycles[t-1]
    
    
    
