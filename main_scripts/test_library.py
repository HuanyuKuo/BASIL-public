# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:15:38 2022

@author: huanyu

Test libraries
"""

import myConstant as mc
import myFunctions as mf
import myVariables as mv
import myReadfile as mr
import mylikelihood as llk


import multiprocessing as lib # multiprocessing is a standard library for python (Python version > 2.6)
print('Python version', lib.sys.version)

print('Library', 'version')

import numpy as lib
print(lib.__name__, lib.__version__)

import scipy as lib
print(lib.__name__, lib.__version__)

import matplotlib as lib
print(lib.__name__, lib.__version__)

import json as lib
print(lib.__name__, lib.__version__)

import pickle as lib # https://github.com/python/cpython/blob/3.10/Lib/pickle.py
print(lib.__name__, lib.format_version)

import noisyopt as lib # https://github.com/andim/noisyopt
print(lib.__name__, lib.__version__)

import pystan as lib # https://github.com/stan-dev/pystan
print(lib.__name__, lib.__version__)
