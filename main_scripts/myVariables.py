# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:35:08 2019

@author: Huanyu Kuo

MEE_Variables.py
"""

import myConstant as mc
#import MEE_Functions as mf
import numpy as np
#import test_llk_0110 as llk
import mylikelihood as llk

# A class store some constant values in experiment such like dilution ratio, total reads, etc.

LINEAGE_TAG = mc.LINEAGE_TAG
OutputFileDir = mc.OutputFileDir

class Constant():
    def __init__(self, total_reads, t_cycle):
        self.T = len(total_reads)                               # total time point
        self.T_cycle = t_cycle                                  # Time array in cycles
        self.Dt = [float(mc.D) for i in range(0,self.T)]        # dilution ratio
        self.Ct = ['']                                          # cycle between two reads
        self.Ct.extend( [t_cycle[i+1]-t_cycle[i] for i in range(len(t_cycle)-1)])
        self.Nt = [float(mc.N) for i in range(0,self.T)]        # total number of cells in the flask before dilution (after growing)
        self.Rt = total_reads
        #self.smin = mc.smin
        #self.smax = mc.smax
        self.eps = mc.epsilon
        
    def print_values(self):
        print("Total time points\t", self.T, "\t# const.T")
        print("Dilution\t", self.Dt, "\t# const.Dt")
        print("Cycle\t", self.Ct, "\t# const.Ct")
        print("Number of cell before dilution\t", self.Nt, "\t# const.Nt")
        print("Total Barcode read\t ", self.Rt)
             
class Global():
    def __init__(self, const):
        self.current_timepoint = 0
        self.epsilon = float(const.eps)  # measuring dispersion (0-1) (default) # epsilon=0, Measurement process is Poisson (VMR=1), if epsilon=1, Measurement process is NB with VMR=2. # VMR=vairance mean ratio
        self.meanfitness = float(0.0) # mean-fitness (default)
        self.R = const.Rt[self.current_timepoint]  # total number of reads at time t (default)
        self.D = const.Dt[self.current_timepoint]
        self.C = const.Ct[self.current_timepoint]
        self.N = const.Nt[self.current_timepoint]
        self.epsilon_const = float(const.eps)
                                   
    def print_values(self):
        print("Print value of Globals:")
        print("Current time point (integer)\t", self.current_timepoint)
        print("Measuring_Error\t", self.epsilon)
        print("Mean_Fitness(1/cycle)\t", self.meanfitness)
        print("Total_Read\t", self.R)
        print("Number_of_Cell_before_Dilution\t", self.N)
        #print("Number_of_Distinct_Barcode\t", self.NL)
        print("Dilution_Ratio\t", self.D)
        print("Number_of_Transition_Cycle_between_Time_point\t", self.C)
    
    def update_from_const(self, const, t):
        self.current_timepoint = t
        self.R = float(const.Rt[t])
        self.D = float(const.Dt[t])
        self.N = float(const.Nt[t])
        self.C = float(const.Ct[t])
        
    #def UPDATE_GLOBAL(self, t, const, lins, lineage_info):
    def UPDATE_GLOBAL(self, current_time, const, lineage_info, lins_1, dim):
        self.update_from_const(const, current_time)
        
        self.meanfitness = 0.0
        self.epsilon = self.epsilon_const#3.0#4.0
        
        if dim == '1d':
        #Optimize (ML) meanfitness with given epsilon
            sol1 = llk.maximum_likelihood_function_meanfitness(self, lins_1, const, current_time, self.epsilon)
            self.meanfitness = sol1.x[0]
            self.outputfile(lineage_info, sol1, len(lins_1))
        elif dim == '2d':
            '''
            #
            # Step 1: Optimize (ML) meanfitness with small epsilon
            sol1 = llk.maximum_likelihood_function_meanfitness(self, lins_1, const, current_time, self.epsilon)
            self.meanfitness = sol1.x[0]
            
            #
            # Step2: Optimize (ML) epsilon with fixed meanfitness
            sol2 = llk.maximum_likelihood_function_epsilon(self, lins_1, const, current_time, self.meanfitness)
            self.epsilon = 10**(sol2.x[0])
            self.outputfile2d(lineage_info, sol1, len(lins_1), sol2, len(lins_1))
            '''
            
            
            sol = llk.maximum_likelihood_function_global_variable2d(self, lins_1, const, current_time)
            self.meanfitness = sol.x[0]
            self.epsilon = 10**(sol.x[1])
            self.outputfile(lineage_info, sol, len(lins_1))

        
    #def outputfile(self, lineage_info, sol1, sol2, N_rand, OutputFileDir):
    def outputfile(self, lineage_info, sol, N_rand):
    #def  outputfile(self, lineage_info, sol1, n1, sol2, n2):
        outfilename = 'glob_'+lineage_info['lineage_name']+f"_T{self.current_timepoint}.txt"
        outputfiledirname = OutputFileDir+outfilename
        
        f = open(outputfiledirname, 'w') 
        f.write("===Print value of Globals===\n")
        f.write(f"Current time point (integer)\t{self.current_timepoint}\n")
        f.write(f"Measurment_dispersion_epsilon\t{self.epsilon}\n")
        f.write(f"Mean_Fitness\t{self.meanfitness}\n")
        f.write(f"Total_Read\t{self.R}\n")
        f.write(f"Number_of_Cell_before_Dilution\t{self.N}\n")
        f.write(f"Dilution_Ratio\t{self.D}\n")
        f.write(f"Number_of_Transition_Cycle_between_Time_point\t{self.C}\n")
        
        f.write(f"Number_of_lineage_to_fit_Global\t{N_rand}\n")
        f.write("\n===Fitting meanfitness and epsilon===\n\n")
        print(sol, file=f)
        
        '''
        f.write("\n===Fitting mean-fitness===\n\n")
        f.write(f"Number_of_lineage_to_fit\t{n1}\n")
        print(sol1, file=f)
        f.write("\n===Fitting epsilon===\n\n")
        f.write(f"Number_of_lineage_to_fit\t{n2}\n")
        print(sol2, file=f)
        '''
        f.close()
    '''
    #def outputfile2d(self, lineage_info, sol, N_rand):
    def  outputfile2d(self, lineage_info, sol1, n1, sol2):
        outfilename = 'glob_'+lineage_info['lineage_name']+f"_T{self.current_timepoint}.txt"
        outputfiledirname = OutputFileDir+outfilename
        f = open(outputfiledirname, 'w') 
        f.write("===Print value of Globals===\n")
        f.write(f"Current time point (integer)\t{self.current_timepoint}\n")
        f.write(f"Measurment_dispersion_epsilon\t{self.epsilon}\n")
        f.write(f"Mean_Fitness\t{self.meanfitness}\n")
        f.write(f"Total_Read\t{self.R}\n")
        f.write(f"Number_of_Cell_before_Dilution\t{self.N}\n")
        f.write(f"Dilution_Ratio\t{self.D}\n")
        f.write(f"Number_of_Transition_Cycle_between_Time_point\t{self.C}\n")
        
        f.write("\n===Fitting mean-fitness===\n\n")
        f.write(f"Number_of_lineage_to_fit\t{n1}\n")
        print(sol1, file=f)
        f.write("\n===Fitting epsilon===\n\n")
        f.write(f"Number_of_lineage_to_fit\t{n1}\n")
        print(sol2, file=f)
        
        f.close()
    '''
    def import_value_array(self, arr):
        # import global value from array
        self.meanfitness = arr[0]
        self.epsilon = arr[1]
        self.R = arr[2]
        self.D = arr[3]
        self.C = arr[4]
        self.N = arr[5]
        #self.NL = arr[6]
        self.current_timepoint = arr[6]

class Lineage():
    def __init__(self, reads, BCID):
        
        self.BCID = BCID                            # index of barcode 
        self.nm = NModel()                          # NModel for lineage
        self.sm = SModel()                          # SModel for lineage
        self.rt  = np.asarray(reads, dtype='float') # vector of reads = barcode read count over time
        
        self.T = ''                                 # total time point of reads
        self.r0 = ''                                # r0 = read of last time point
        self.r1 = ''                                # r1 = read of current time point
        self.TYPETAG = ''                           # Tag of lineage type
        #self.RUNTAG = ''                           # Tag of running lineage
        self.T_END = ''                             # Time to stop running a lineage (if extinct)
        self.SModel_N_smax = ''
        self.SModel_N_smin = ''
        #self.log10_BayesFactor_vector = ''
        #self.log10_BayesFactor_past = ''
        #self.log_prob_survive_past = ''
        self._initialize_t0()
        
    def _initialize_t0(self):
        self.T = np.size(self.rt)                         
        self.set_reads(0)                           # set r0, r1 (r1 = read of current time)
        #self.SModel_N_smax = mc.smax
        #self.SModel_N_smin = 0#mc.smin
        self._get_T_END()
        self._init_TAG()
        
        #self.log10_BayesFactor_vector = ['' for i in range(self.T_END)]
        
    # after running the initial-MCMC-model, initialize the lineage by adding the parameters of posterior P(n0)
    def INITIALIZE_POST_PARM(self, k, theta):
        self.nm.post_parm_Gamma_k = k
        self.nm.post_parm_Gamma_theta = theta
        self.sm.post_parm_Gamma_k = k
        self.sm.post_parm_Gamma_a = k*theta
        self.sm.post_parm_Gamma_b = 0.0
    
    # update read r0, r1 (r1 = read of current time)
    def set_reads(self, last_time):
        if (last_time < (self.T -1)) and (last_time >= 0):
            self.r0 = self.rt[last_time]
            self.r1 = self.rt[last_time+1]
        elif last_time < 0:
            self.r0 = self.rt[0]
            self.r1 = self.rt[1]
            print(f"Warning! in class Lineage set_reads for BCID {self.BCID}: input last_time ={last_time} must be integer >= 0  and < {self.T-1}")
        else:
            self.r0 = self.rt[-2]
            self.r1 = self.rt[-1]
            
    def reTAG(self, last_time):
        
        # Past TAG: update survival-probability
        #self._set_log_prob_survive_past(last_time)
        
        # Update the TAG
        self._reTAG(last_time)
        
    
    def _set_log_prob_survive_past(self, last_time):
        if last_time>0:
            if self.rt[last_time] > 0:
                self.log_prob_survive_past = 0
                
            elif (self.TYPETAG == LINEAGE_TAG['UNK']) or (self.TYPETAG == LINEAGE_TAG['NEU']): # pastTAG
                self.log_prob_survive_past = self.nm.log_prob_survive
                
            elif self.TYPETAG == LINEAGE_TAG['ADP']: # pastTAG
                self.log_prob_survive_past = self.sm.log_prob_survive
            else:
                self.log_prob_survive_past = -np.inf
        else:
            self.log_prob_survive_past = 0
    '''
    # compute the Bayes Factor
    def set_log10BF(self, last_time):
        if (last_time >= 0) and (last_time < self.T_END): 
            self.log10_BayesFactor_vector[last_time] = (self.sm.log_noramlization_const - self.nm.log_noramlization_const) /np.log(10)
            
        else:
            print(f"ERROR! in class Lineage set_log10BF for BCID {self.BCID}: input last_time ={last_time} must be integer >= 0  and < {self.T_END} to get log10 Bayes Factor")
    '''
    def log10_BayesFactor(self):
        return (self.sm.log_noramlization_const - self.nm.log_noramlization_const) /np.log(10)
    
    #def reTAG(self, critical_read, last_time):
    def _reTAG(self, last_time):
        self.TYPETAG = LINEAGE_TAG['ADP']
        '''
        critical_read = mc.rc_default#max(critical_read, rc_default)
        
        if last_time == 0:
            self._init_TAG()
        else:    
            if self.r0 < critical_read:
                newtag = LINEAGE_TAG['UNK']
            
            elif self.TYPETAG == LINEAGE_TAG['UNK']:
                newtag = LINEAGE_TAG['NEU']
            
            else:
                #log10_BayesFactor_past =  (self.sm.log_noramlization_const - self.nm.log_noramlization_const) /np.log(10)
                #self.log10_BayesFactor_past = log10_BayesFactor_past
                log10_BayesFactor_past = self.log10_BayesFactor()
                if (log10_BayesFactor_past > mc.log10BF_threshold) and (self.sm.post_parm_NormS_mean>0):
                    newtag = LINEAGE_TAG['ADP']
                else:
                    newtag = LINEAGE_TAG['NEU']
            
            self.TYPETAG = newtag
        '''
        
    def _get_T_END(self):
        self.T_END = self.T # by default, T_END is the end of time point
        time_zero = np.where(self.rt==0)[0] # get time point where read count is zero
        if len(time_zero) > 0:
            repeated_timezero_idx = list(np.where(np.diff(time_zero)==1)[0]) # if the difference of time point equals one, the zero read count is repeated for continuous two time point
            time_repeated_zero = [list(time_zero)[i+1] for i in repeated_timezero_idx] # find out the time point where the zero read count is repeated
            if len(time_repeated_zero) > 0:
                self.T_END = time_repeated_zero[0] # the first repeated time point as T_END
                
    def _init_TAG(self):
        self.TYPETAG = LINEAGE_TAG['ADP']
        '''
        if self.r0 <= mc.rc_default:
            self.TYPETAG = LINEAGE_TAG['UNK']
        else:
            self.TYPETAG = LINEAGE_TAG['NEU']
        '''
            
class NModel():
    def __init__(self):
        self.post_parm_Gamma_k = 1
        self.post_parm_Gamma_theta = 0.1
        self.log_noramlization_const = 0.0 # normalization constant for posterior
        self.log_prob_survive = 0.0
        
    def UPDATE_POST_PARM(self, k, theta, log_norm, log_prob_survive):
        self.post_parm_Gamma_k = k
        self.post_parm_Gamma_theta = theta
        self.log_noramlization_const = log_norm + log_prob_survive # conditional on survival
        self.log_prob_survive = log_prob_survive


class SModel():
    def __init__(self):
        self.post_parm_Gamma_k = 1
        self.post_parm_Gamma_a = 1
        self.post_parm_Gamma_b = 0
        self.post_parm_NormS_mean = 0
        self.post_parm_NormS_var = 0.5
        self.log_noramlization_const = 0.0 # normalization constant for posterior
        self.log_prob_survive = 0.0
    def UPDATE_POST_PARM(self, k, a, b, mean_s, var_s, log_norm, log_prob_survive):
        self.post_parm_Gamma_k = k
        self.post_parm_Gamma_a = a
        self.post_parm_Gamma_b = b
        self.post_parm_NormS_mean = mean_s
        self.post_parm_NormS_var = var_s
        self.log_noramlization_const = log_norm + log_prob_survive # conditional on survival
        self.log_prob_survive = log_prob_survive
    