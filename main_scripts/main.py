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


def run_BASIL(start_step=1):

    # -------Read barcode read count-----------#
    lins, totalread, t_cycles = mr.my_readfile(mc.data)

    # -------Set up Constant and Global variable ---#
    const = Constant(totalread, t_cycles)
    glob = Global(const)

    # -------Start BASIL analysis for each time point ----#
    for current_step in range(start_step, const.T):
        print('Time step ' + str(current_step) )

        # READ LINEAGE FROM THE PAST FILES
        lins = create_lineage_list_by_pastTag(lins, current_step, const)

        # UPDATE GLOBAL VARIABLE
        # step1: Choose random lineage for likelihood function
        lins_RAND = select_random_lineages(lins)

        # step2: Maximum likelihood estimate
        print('Estimate mean fitness')
        glob.UPDATE_GLOBAL(current_step, const, lins_RAND, '2d')

        # run SModel_S for all lineages
        print('Compute Bayesian probabilities for ' + str(len(lins))+ ' lineages')
        run_dict = {'model_name': MODEL_NAME['SS'], 'lineage_name': mc.case_name}
        run_model_MCMCmultiprocessing(run_dict, lins, glob)

    # output result
    mr.output_global_parameters_BFM(const)
    mr.output_Selection_Coefficient_Bayes_v5()

if __name__ == '__main__':

    run_BASIL(start_step=1)

    # ---- (Optional) Play with different beta-threshold for lineage calling (default beta = [3.3])
    mr.output_Selection_Coefficient_Bayes_v5(beta=[3, 4, 5, 6])


