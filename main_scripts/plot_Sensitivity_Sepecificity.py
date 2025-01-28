# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 00:45:55 2022

@author: huanyu
"""

from myVariables import (Constant, Global, Lineage)
import myConstant as mc
import myReadfile as mr

from matplotlib import pyplot as plt
import numpy as np

def read_meanfitness_Simulation(filedirname):
    f = open(filedirname,  'r')
    line = f.readline()
    meanfitness_Simulation = []
    while(line):
        meanfitness_Simulation.append(float(line.split('\n')[0].split('\t')[1]))
        line = f.readline()
    f.close()
    return meanfitness_Simulation

def read_selection_Simulation(filedirname):
    s_arr_Simulation = []
    f = open(filedirname,'r')
    line = f.readline()
    line = f.readline()
    while(line):
        # file in the order of Barcode index 0, 1, 2, .., so we store value without sorting by index
        s_arr_Simulation.append(float(line.split('\n')[0].split('\t')[1]))
        line = f.readline()
    f.close()
    return s_arr_Simulation

def true_positive(true_BCID, classified_BCID):
    return len(set(classified_BCID).intersection(set(true_BCID)))

def sensitivity(simuADP_BCID, classifiedADP_BCID):
    # positive = adpative
    # sensitivity =  true positive rate
    true_positive = set(classifiedADP_BCID).intersection(set(simuADP_BCID))
    _sensitivity = len(true_positive)/len(simuADP_BCID)
    return _sensitivity

def specificity(simuADP_BCID, classified_BCID, total_BCID_list):
    # negative = neutral
    # specificity =  true negative rate
    simuNEU_BCID = set(total_BCID_list) - set(simuADP_BCID)
    classifiedNEU_BCID = set(total_BCID_list) - set(classified_BCID)
    true_negative = classifiedNEU_BCID.intersection(simuNEU_BCID)
    _specificity = len(true_negative)/len(simuNEU_BCID)
    return _specificity
def precision(simuADP_BCID, classifiedADP_BCID):
    # positive = adpative
    # precision =  true positive / predict postive
    true_positive = set(classifiedADP_BCID).intersection(set(simuADP_BCID))
    _precision = len(true_positive)/len(classifiedADP_BCID)
    return _precision
if __name__ == '__main__':
    
    datafilename = 'Data_BarcodeCount_simuMEE_20220226' + '.txt'  
    #datafilename = '41586_2015_BFnature14279_MOESM90_ESM_cycle_downsample'+ '.txt'  
    lins, totalread, cycles = mr.my_readfile(datafilename)
    const = Constant(totalread, cycles)
    
    #case_name = 'Simulation_20220226_NL=10e5' #'Simulation_20222013_Population1' 
    #case_name = 'nature2015_Rep1'#'Simulation_20222013_Population1' 
    
    case_name = 'Simulation_20220226_NL=10e5__testRandLintoFitGlob' 
    lineage_info =  {'lineage_name': case_name}
    
    total_BCID_list = [lin.BCID for lin in lins]
    # ##################################################
    #
    # Collect output from Bayesian Method and Save results
    #
    # ###################################################
    #eps_Bayes, meanfitness_Bayes = read_meanfitness_BTM(lineage_info, const)
    #ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time = read_selection_Bayes(lins, const, lineage_info)
    
    '''
    simu_name = 'simuMEE_20220226'
    meanfitness_Simulation = read_meanfitness_Simulation('../simulation_MEE/'+'simulation_meanfitness_'+simu_name+ '.txt')
    t_simulation = np.arange(0,len(meanfitness_Simulation),1)
    
    s_Simulation = read_selection_Simulation('../simulation_MEE/' + 'simulation_selection_coefficient_'+simu_name+'.txt')
    s_Simulation = np.asarray(s_Simulation)
    ADP_BCID_Simulation = np.where(s_Simulation>0)[0]
    
    ADP_BCID_Simulation_survive = []
    for bcid in ADP_BCID_Simulation:
        if lins[bcid].T_END > 2:
            ADP_BCID_Simulation_survive.append(bcid)
    ADP_BCID_Simulation = ADP_BCID_Simulation_survive
    s_Simulation_adpative = s_Simulation[ADP_BCID_Simulation]
    
    arr_critical_log10_BF = np.arange(-0.20,2.0,0.02)
    arr_sensitivity = []
    arr_specificity = []
    arr_precision = []
    arr_correlation = []
    arr_falsepositiverate = []
    for critical_log10_BF in arr_critical_log10_BF:
        #critical_log10_BF = 1
        critical_counts = 1
        
        ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time = mr.read_selection_Bayes(lineage_info, datafilename, critical_log10_BF, critical_counts)
        
        sen = sensitivity(ADP_BCID_Simulation, ADP_BCID_Bayes)
        spe = specificity(ADP_BCID_Simulation, ADP_BCID_Bayes, total_BCID_list)
        pre = precision(ADP_BCID_Simulation, ADP_BCID_Bayes)
        arr_sensitivity.append(sen)
        arr_specificity.append(spe)
        arr_precision.append(pre)
        arr_falsepositiverate.append(1-spe)
        idx_i = []
        s2s_Simu = []
        for i in range(len(ADP_BCID_Bayes)):
            bcid = ADP_BCID_Bayes[i]
            if bcid in ADP_BCID_Simulation:
                s2s_Simu.append(s_Simulation[bcid])
                idx_i.append(i)
        s2s_Bayes = [ADP_s_mean_Bayes[i] for i in idx_i]
        s2s_Bayes_std = [ADP_s_std_Bayes[i] for i in idx_i]
        r = np.corrcoef(s2s_Simu, s2s_Bayes)
        arr_correlation.append(r[0][1])
        print(critical_log10_BF, sen, spe, pre, r[0][1])
    
    filename = f'Precision_c={critical_counts}_'+case_name
    '''
    '''
    plt.figure()
    plt.plot(arr_critical_log10_BF, arr_precision, 'ko-', label='Precision'); 
    plt.plot(arr_critical_log10_BF, arr_sensitivity, 'ro-', label='True positive rate'); 
    plt.plot(arr_critical_log10_BF, arr_specificity, 'bo-', label='True negative rate');  
    plt.plot(arr_critical_log10_BF, arr_correlation, 'mo-', label='correlation coeff of s')
    plt.legend(); 
    plt.ylim(0,1.1);
    plt.xlabel('log10 critical Bayes Factor')
    plt.title(f'Positive if condition is satisfied >= {critical_counts} time point(s)')
    
    filename = f'Precision_c={critical_counts}_'+case_name
    plt.savefig(mc.OutputFileDir+filename+'.png',dpi=200)
    '''
    '''
    f=open(mc.OutputFileDir + filename + '.txt','w')
    f.write('log10_critical_BayesFactor\tPrecision\tTrue positive rate\tTrue negative rate\tCorrelation coeff\n')
    for i in range(len(arr_critical_log10_BF)):
        f.write(str(arr_critical_log10_BF[i])+'\t'+str(arr_precision[i])+'\t'+str(arr_sensitivity[i])+'\t'+str(arr_specificity[i])+'\t'+str(arr_correlation[i])+'\n')
    f.close()
    '''
    '''
    critical_counts = 2
    critical_log10_BF = 0.2
    ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time = mr.read_selection_Bayes(lineage_info, datafilename, critical_log10_BF, critical_counts)
    true_positive = list(set(ADP_BCID_Bayes).intersection(set(ADP_BCID_Simulation)))
    
    idx_i = []
    s2s_Simu = []
    for i in range(len(ADP_BCID_Bayes)):
        bcid = ADP_BCID_Bayes[i]
        if bcid in ADP_BCID_Simulation:
            s2s_Simu.append(s_Simulation[bcid])
            idx_i.append(i)
    s2s_Bayes = [ADP_s_mean_Bayes[i] for i in idx_i]
    s2s_Bayes_std = [ADP_s_std_Bayes[i] for i in idx_i]
    r = np.corrcoef(s2s_Simu, s2s_Bayes)
    '''
    '''
    plt.figure()
    plt.plot(arr_falsepositiverate, arr_sensitivity, 'ko-')
    #plt.plot([0,1],[0,1],'k--');
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.title(f'Positive if condition is satisfied >= {critical_counts} time point(s)')
    filename = f'ROC_c={critical_counts}_'+case_name;
    plt.savefig(mc.OutputFileDir+filename+'.png',dpi=200)
    '''
    
    critical_counts = 1
    filename = f'Precision_c={critical_counts}_'+case_name
    f=open(mc.OutputFileDir + filename + '.txt','r')
    print(f.readline())
    line = f.readline()
    
    log10BF_threshold = []
    arr_sensitivity = []
    arr_falsepositiverate = []
    while(line):
        line = line.split('\n')[0].split('\t')
        log10BF_threshold.append(float(line[0]))
        arr_sensitivity.append(float(line[2]))
        arr_falsepositiverate.append(1-float(line[3]))
        line = f.readline()
    f.close()
    
    log10BF_threshold_1 = log10BF_threshold
    arr_sensitivity_1=arr_sensitivity
    arr_falsepositiverate_1 = arr_falsepositiverate
    
    critical_counts = 2
    filename = f'Precision_c={critical_counts}_'+case_name
    f=open(mc.OutputFileDir + filename + '.txt','r')
    print(f.readline())
    line = f.readline()
    
    log10BF_threshold = []
    arr_sensitivity = []
    arr_falsepositiverate = []
    while(line):
        line = line.split('\n')[0].split('\t')
        log10BF_threshold.append(float(line[0]))
        arr_sensitivity.append(float(line[2]))
        arr_falsepositiverate.append(1-float(line[3]))
        line = f.readline()
    f.close()
    
    log10BF_threshold_2 = log10BF_threshold
    arr_sensitivity_2=arr_sensitivity
    arr_falsepositiverate_2 = arr_falsepositiverate
    
    plt.figure()
    plt.plot(arr_falsepositiverate_1, arr_sensitivity_1, 'b--', lw=3,label='condition is satisfied >= 1 ');
    plt.plot(arr_falsepositiverate_2, arr_sensitivity_2, 'k--', lw=3,  label='condition is satisfied >= 2');
    idx1=5
    plt.plot(arr_falsepositiverate_2[idx1], arr_sensitivity_2[idx1], 'ks', markerfacecolor = 'none', ms=12, label='threshold =  {:.2f}'.format(10**log10BF_threshold_2[idx1]));
    idx2=10
    plt.plot(arr_falsepositiverate_2[idx2], arr_sensitivity_2[idx2], 'k^', markerfacecolor = 'none', ms=12, label='threshold = {:.2f}'.format(10**log10BF_threshold_2[idx2]));
    idx3=25
    plt.plot(arr_falsepositiverate_2[idx3], arr_sensitivity_2[idx3], 'k<', markerfacecolor = 'none', ms=12, label='threshold =  {:.2f}'.format(10**log10BF_threshold_2[idx3]));
    
    idx3=60
    plt.plot(arr_falsepositiverate_1[idx3], arr_sensitivity_1[idx3], 'bo', markerfacecolor = 'none', ms=12, label='threshold =  {:.2f}'.format(10**log10BF_threshold_2[idx3]));
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate'); 
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc=1, ncol=2); 
    #plt.plot([0,1],[1,0],'k--')
    plt.xlim(-0.002,0.022); 
    plt.ylim(0.6,1.); 
    plt.yscale('log'); 
    plt.savefig(mc.OutputFileDir+case_name+'_ROC_2curves.png',dpi=200)
    
    d1 = [np.sqrt((1-a)*(1-a)+b*b) for a, b in zip(arr_sensitivity_1,arr_falsepositiverate_1)]
    
    d2 = [np.sqrt((1-a)*(1-a)+b*b) for a, b in zip(arr_sensitivity_2,arr_falsepositiverate_2)]
    
    plt.figure()
    plt.plot(log10BF_threshold_1, d1,  'b--', lw=3,label='condition is satisfied >= 1 ');
    plt.plot(log10BF_threshold_2, d2,  'k--', lw=3,label='condition is satisfied >= 2 ');
    plt.ylabel('Distance to optimal on ROC')
    plt.xlabel('Log10 threshold')
    