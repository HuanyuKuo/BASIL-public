# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:50:31 2020

@author: huanyu
"""

import numpy as np

import myConstant as mc
import myReadfile as mr
from myVariables import (Constant, Global, Lineage)
from my_model_MCMCmultiprocessing import run_model_MCMCmultiprocessing, create_lineage_list_by_pastTag

MODEL_NAME = mc.MODEL_NAME
LINEAGE_TAG = mc.LINEAGE_TAG
OutputFileDir = mc.OutputFileDir
NUMBER_RAND_NEUTRAL = mc.NUMBER_LINEAGE_MLE

#
# Function randomly chooses lineages from lins
#   
def select_random_lineages(lins):
    
    lins_choice =[]
    
    for lin in lins:
        if lin.r0 > 0:
            lins_choice.append(lin)
    
    length = min(NUMBER_RAND_NEUTRAL,len(lins_choice))
    rand_index = np.random.choice(a=len(lins_choice), size=length, replace=False)
    lins_ = [ lins_choice[i] for i in list(rand_index)]
    
    return lins_


def run_lineages(lins, start_time, end_time, const, lineage_info):
    #s_bar = []
    if (end_time <= const.T) and (start_time >=0 ):    
        glob = Global(const)    
        
        for current_time in range(start_time, end_time):
            
            if current_time >0 :
                
                # READ LINEAGE FROM THE PAST FILES
                lins = create_lineage_list_by_pastTag(lins, current_time, lineage_info, const)
                
                # UPDATE GLOBAL VARIABLE
                # step1: Choose random lineage for likelihood function
                lins_RAND = select_random_lineages(lins)

                # step2: Maximum likelihood estimate
                glob.UPDATE_GLOBAL(current_time, const, lineage_info, lins_RAND, '2d')

                # run SModel_S for all lineages
                run_dict = {'model_name': MODEL_NAME['SS'], 'lineage_name': lineage_info['lineage_name']}
                run_model_MCMCmultiprocessing(run_dict, lins, glob)
                
    else:
        print(f"the input start_time ={start_time} must >=0 & the end_time ={end_time} must <= total time point {const.T}")
    #print(s_bar)
    #return lins_RAND, glob, s_bar#lins

if __name__ == '__main__':
    
    # ##################################
    # Set your filename and case_name
    # ################################## 
    
    #
    # 1. FileName of Barcode Count data
    # 
    datafilename = 'Data_BarcodeCount_simuMEE_20220213' + '.txt'
    # read file
    lins, totalread, t_cycles = mr.my_readfile(datafilename)
    const = Constant(totalread, t_cycles)

    #
    # 2. Name of This Run Case
    #
    case_name = 'Simulation_20220213'


    # ##################################
    # Run & output results
    # ##################################

    #
    # 3. Run program
    #
    start_time = 1
    end_time = const.T
    lineage_info =  {'lineage_name': case_name +'_v6'}
    #run_lineages(lins, start_time, end_time, const, lineage_info)

    #
    # 4. Output mean fitness,  Output selection coefficient
    #

    # output mean fitness
    #mr.output_global_parameters_BFM(lineage_info,const)
    #meanfitness_Bayes_cycle, epsilon_Bayes, t_arr_cycle = mr.read_global_parameters_BFM(lineage_info)

    # output parameters of Bayesian estimate for all lineages
    BFM_result = mr.output_Posterior_parameters_Bayes_v5(lineage_info, datafilename)

    # output the beneficial lineages with classifying threshold beta. You can play with different threshold values.
    beta = mc.beta # mc.beta = 3.3 by default
    mr.output_Selection_Coefficient_Bayes_v5(lineage_info, datafilename, BFM_result=BFM_result, beta=beta)
    beta = 4
    mr.output_Selection_Coefficient_Bayes_v5(lineage_info, datafilename, BFM_result=BFM_result, beta=beta)
    beta = 5
    mr.output_Selection_Coefficient_Bayes_v5(lineage_info, datafilename, BFM_result=BFM_result, beta=beta)


