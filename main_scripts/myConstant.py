# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:35:02 2019

@author: Huanyu Kuo
MEE_Constant.py
"""
#
# FILE READING AND SETTINGS
#
InputFileDir = '../input/'
OutputFileDir = '../output/'
NUMBER_OF_PROCESSES = 12 # Multi-processing

#
# EXPERIMENTAL PARAMETERS
#
D = float(100)              # dilution factor
N = float(256*10**6)        # Carrying capacity: total number of cells in the flask before dilution (after growing)
cycle = float(2)            # number of cycle between data

#
# BAYESIAN PARAMETERS (default)
#
NUMBER_LINEAGE_MLE = 3000   # Number of reference lineages (randlomly selected) used to infer mean-fitness
epsilon = float(0.01)       # initial value of epsilon, default
beta = 3.3                  # criteria to make final lineage call (call adapted if a lineage's P(s) is mean >= beta*standard_deviation)

#
# USE ONLY FOR USING ANOTHER POSTERIOR FILE AS INITIALIZATION
#

# Define the starting "Time point" for this BASIL run. default = 1
FILE_START_TIME = 1
# The posterior file (generated from another BASIL run) as the initial states for this BASIL run. default = None
initializing_lineage_filename = 'posterior_Simulation_20220213_v7_SModel_S_T1.txt'#  None # 'posterior_Simulation_20220213_v7_SModel_S_T1.txt'


MODEL_NAME = {'N': 'NModel', 'SN': 'SModel_N', 'SS': 'SModel_S'}
LINEAGE_TAG = {'UNK': 'Unknown', 'NEU': 'Neutral', 'ADP': 'Adaptive'}

