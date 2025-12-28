# -*- coding: utf-8 -*-
"""
@author: Huanyu Kuo
"""
# ----- File I/O----------#
data = '../input/' + 'cheeseGeobarcodecounts.txt'   # barcode read count data
case_name = 'simulation_test'                       # naming this BASIL run
OutputFileDir = '../output/'                        # directory of output files

# ------ EXPERIMENTAL PARAMETERS in Barcode lineage tracking ------#
D = float(100)              # dilution factor
# N = float(256*10**6)        # Carrying capacity: total number of cells in the flask before dilution (after growing)
N = float(1*10**9)        # Carrying capacity: total number of cells in the flask before dilution (after growing)

# ------ COMPUTATIONAL PARAMETERS for BASIL performance-----#
NUMBER_OF_PROCESSES = 12    # Multiprocessing (suggested number is 12 to 40)
NUMBER_LINEAGE_MLE = 3000   # Number of reference lineages (randomly selected) used to infer mean-fitness

# ------ Use only for reading another posterior file as initialization of lineage (default=None)----#
# INITIAL_LINEAGES_FROM_FILE = None
INITIAL_LINEAGES_FROM_FILE =   '../input/'+ 'posterior_GeorestartingatT11_2025_11_25.txt'

# ------ Model Setting in BASIL algorithm (do not change) -----------------#
MODEL_NAME = {'N': 'NModel', 'SN': 'SModel_N', 'SS': 'SModel_S'}
LINEAGE_TAG = {'UNK': 'Unknown', 'NEU': 'Neutral', 'ADP': 'Adaptive'}

