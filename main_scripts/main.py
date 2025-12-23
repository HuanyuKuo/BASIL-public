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
    valid = [lin for lin in lins if lin.r0 > 0]

    if not valid:
        return []

    n = min(NUMBER_RAND_NEUTRAL, len(valid))
    idx = np.random.choice(len(valid), size=n, replace=False)
    return [valid[i] for i in idx]


#def run_lineages(lins, start_time, end_time, const, lineage_info):
def run_lineages(lins, const, lineage_info):

    glob = Global(const)

    for current_step in range(1, const.T):

        # READ LINEAGE FROM THE PAST FILES
        lins = create_lineage_list_by_pastTag(lins, current_step, lineage_info, const)

        # UPDATE GLOBAL VARIABLE
        # step1: Choose random lineage for likelihood function
        lins_RAND = select_random_lineages(lins)

        # step2: Maximum likelihood estimate
        glob.UPDATE_GLOBAL(current_step, const, lineage_info, lins_RAND, '2d')

        # run SModel_S for all lineages
        run_dict = {'model_name': MODEL_NAME['SS'], 'lineage_name': lineage_info['lineage_name']}
        run_model_MCMCmultiprocessing(run_dict, lins, glob)

    # output result
    mr.output_global_parameters_BFM(lineage_info, const)
    mr.output_Selection_Coefficient_Bayes_v5(lineage_info, datafilename, beta=[mc.beta])

if __name__ == '__main__':
    
    # ##################################
    # Set your filename and case_name
    # ################################## 
    
    #
    # 1. Input Files
    # 
    datafilename = 'Data_BarcodeCount_simuMEE_20220213' + '.txt'  # FileName of Barcode Count data
    initializing_lineage_filename =  'posterior_Simulation_20220213_v7_SModel_S_T1.txt'
    file_start_time = 100 # default = 1
    #
    # 2. Name of This Run Case
    #
    case_name = 'Simulation_test_initializing_feature'

    lineage_info = {'lineage_name': case_name + '_v7'}
    lineage_info.update({'initializing_lineage_filename': mc.initializing_lineage_filename})
    lineage_info.update({'file_start_time': mc.FILE_START_TIME})

    # ##################################
    # Run & output results
    # ##################################

    #
    # 3. Run program
    #
    lins, totalread, t_cycles = mr.my_readfile(datafilename)
    const = Constant(totalread, t_cycles)
    run_lineages(lins, const, lineage_info)

    #
    # 4. (Optional) Play with different beta-threshold for lineage calling (default beta = 3.3)
    #
    mr.output_Selection_Coefficient_Bayes_v5(lineage_info, datafilename, beta=[3, 4, 5, 6])




