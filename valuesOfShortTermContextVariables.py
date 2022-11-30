import sys
import os, glob
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import math

def strip_new_line_char(string):
    ""
    if "\n" in string:
        return string[:-1]
    return string
           
def findParametersCorrespondingToMinLikelihood(folder,subject,constraint):
    guess_WConstant = np.arange(0.25,0.53,0.03)
    guess_W1 = np.arange(0,1.1,0.1)
    guess_tau = 10**(np.arange(-1,0.7,0.15))
    params_array = np.zeros(shape=(4,1),dtype=float)
    for WC in range(len(guess_WConstant)):
        for W1 in range(len(guess_W1)):
            for tau in range(len(guess_tau)):
                params_array = np.append(params_array,
                                         np.expand_dims([guess_WConstant[WC], 
                                                         guess_W1[W1], 
                                                         guess_tau[tau],
                                                         guess_W1[W1]*np.exp(-1/guess_tau[tau])],1),axis=1)
    params_array = params_array[:,1:]
    arrayInds = np.arange(np.shape(params_array)[1])
    
    bestParameters = np.zeros((10,4))
    for iterations in range(0,300,30):
        file_dev = folder+'results_'+subject+'_'+str(iterations)+'.txt'
        with open(file_dev, 'r') as f:
            filelines = f.readlines()
            nll_values = np.array([float(x) for x in filelines])
        params_constrained = (params_array[3,:] < float(constraint) + 0.01)*(params_array[3,:] > float(constraint) - 0.01)
        #pdb.set_trace()
        indNLLConstrained = arrayInds[params_constrained]
        idxOfMinNll = np.argmin(nll_values)#[indNLLConstrained])
        bestParameters[int(iterations/30),:] = params_array[:,idxOfMinNll]#indNLLConstrained[idxOfMinNll]]
        print(bestParameters[int(iterations/30),:], np.around(np.min(nll_values),decimals=2), 
              np.around(np.min(nll_values[indNLLConstrained]),decimals=2))
    return np.mean(bestParameters,axis=0), np.std(bestParameters,axis=0)
            
if __name__ == '__main__':    
        
    """
    Obtaining data for a given subject
    """
    folder = sys.argv[1]
    subject = sys.argv[2]
    constraint = sys.argv[3]
    
    """
    Values of Parameters determining short-term context
    """
    meanParamValues, stdParamValues = findParametersCorrespondingToMinLikelihood(folder,subject,
                                                                                 constraint)
    print("Mean of WConstant, W1, tau, W1e^-(1/tau)",
          np.around(meanParamValues[:3],decimals=2),np.around(meanParamValues[3],decimals=3))
    print("Std of WConstant, W1, tau, W1e^-(1/tau)",
          np.around(stdParamValues[:3],decimals=2),np.around(stdParamValues[3],decimals=3))
    