import sys
import os, glob
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import math

def gaussian(x, mean, sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))

def Tones3dgrid(latentTones, sigma):    
    
    input_array_0 = np.expand_dims(gaussian(log_freq_percept, latentTones[0], sigma), axis = 1)
    input_array_1 = np.expand_dims(gaussian(log_freq_percept, latentTones[1], sigma), axis = 1)
    input_array_2 = np.expand_dims(gaussian(log_freq_percept, latentTones[2], sigma), axis = 1)
    s0 = 1/np.sum(input_array_0); s1 = 1/np.sum(input_array_1); s2 = 1/np.sum(input_array_2)
    input_array_0 *= s0; input_array_1 *= s1; input_array_2 *= s2; 
    
    input_array_mat = np.expand_dims(input_array_0@input_array_1.T,axis=2)@(input_array_2.T) #p(T1,T2..|H)     
    
    return input_array_mat

# define mle function
def MLE(params):
    [sigma_sensory,  
     Wconstant, W1, tau] = params # inputs are guesses for parameter values
    
    neg_ll = 0; 
    probability_high = np.zeros((len(trial_tones),1))
    for i_trial in range(1,len(trial_tones)):
        arePrevTrialsLow = 1-2*trial_corrans[:i_trial]
        prob_low = np.clip(Wconstant + W1*np.sum(np.flip(arePrevTrialsLow)*np.exp(-(np.arange(i_trial)+1)/tau)),
                           a_min=0,a_max=1)
        input_array_mat = Tones3dgrid(np.array([np.log10(trial_tones[i_trial][0]),
                                                np.log10(trial_tones[i_trial][1]),
                                                np.log10(trial_tones[i_trial][2])]),sigma=sigma_sensory)
        probability_high[i_trial] = np.sum(np.multiply((1-prob_low),input_array_mat))    
            
        if trial_behaviour[i_trial]:
            neg_ll += -np.log(probability_high[i_trial] + 0.0000001) # if high dist is chosen by observer
        else:
            neg_ll += -np.log(1 - probability_high[i_trial] + 0.0000001) # if low dist is chosen by observer
    return(neg_ll, probability_high)

def write_into_file(params, fresult):
    """
    New optimization algorithm: uses scipy.optimize.fmin. 
    Crude grid initially and then find minimum using the function.
    """

    stdSensory  = params
    guess_sensory_sigma = np.array([float(stdSensory)]);
    guess_WConstant = np.arange(0.46,0.8,0.03); 
    guess_W1 = np.arange(0,1.1,0.1); 
    guess_tau = 10**(np.arange(-1,0.7,0.15))

    neg_ll_array = np.zeros((len(guess_sensory_sigma), len(guess_WConstant), len(guess_W1), len(guess_tau)))
    for ss in range(len(guess_sensory_sigma)):
        for WC in range(len(guess_WConstant)):
            for W1 in range(len(guess_W1)):
                for tau in range(len(guess_tau)):
                    params = [guess_sensory_sigma[ss], guess_WConstant[WC], guess_W1[W1], guess_tau[tau]]
                    neg_ll_array[ss,WC,W1,tau],_ = MLE(params) 
                    fresult.write("%s\n" % neg_ll_array[ss,WC,W1,tau])
                    fresult.flush()
            
if __name__ == '__main__':
        
    """ 
    Obtaining data from a given expt
    """
    csv_test = pd.read_csv('/home/janakis/data/allTrials_longcontext_plow0pt70.csv')
    csv_data = pd.read_csv(sys.argv[1])
    
    """
    Get tones and values of keys pressed
    """    

    test_columns = list(csv_test.columns)
    test_tones_name = test_columns.index('Name')
    test_tones_col_idx = test_columns.index('Tones')
    df_names = (csv_test.iloc[0:800,test_tones_name]).values
    df_tones = (csv_test.iloc[0:800,test_tones_col_idx]).values

    n_tones = 3
    n_trials = csv_data.shape[0]-47
    
    tones_array_orig = np.zeros((n_trials,n_tones))
    tones_array_idxs_keep = []

    for i_wav in range(804):
        if isinstance(csv_data['Name'][i_wav+46],str):
            tones_array_orig[i_wav,:] = np.array(df_tones[np.where(csv_data['Name'][i_wav+46]\
                                                              ==df_names)[0]][0][1:-1].split(',')).astype(float)  
            tones_array_idxs_keep += [i_wav]


    df_tones = np.copy(tones_array_orig[tones_array_idxs_keep,:])
    df_corrans = np.copy(csv_data['corrAns'][46:csv_data.shape[0]])[tones_array_idxs_keep]
    df_keys = np.copy(csv_data['test_resp.keys'][46:csv_data.shape[0]])[tones_array_idxs_keep]
    
    """
    Find no response cases in the expt
    """
    no_response = np.intersect1d(np.where(df_keys!='h')[0],np.where(df_keys!='l')[0])
    print("Did not respond to: ",no_response)

    """
    Convert keys ['l','h'] to [0,1] and plot p(H|T)
    """
    corrans_num_orig = np.zeros_like(df_corrans)
    corrans_num_orig[df_corrans == 'h'] = 1

    keys_num_orig = np.zeros_like(df_keys)
    keys_num_orig[df_keys == 'h'] = 1

    corrans_num = corrans_num_orig[:800]
    keys_num = keys_num_orig[:800]
    tones_array = df_tones[:800,:]
    print(corrans_num.shape, keys_num.shape, tones_array.shape)
    print("Got correct: ", np.sum(keys_num==corrans_num)/len(tones_array))
    
    """
    Latent variables
    """
    expt_tones = np.arange(90,3000,1) #array of possible true tones
    log_freq_seq_array = np.arange(0.6,4.7,0.1)
    log_freq_percept = np.arange(0.6,4.7,0.1) # array of possible perceptual tones

    """
    Experimental data variables: tones and behaviour
    """
    idxs_with_response = np.delete(np.arange(len(tones_array)),no_response)
    trialTones = tones_array[idxs_with_response,:]
    trialBehaviour = keys_num[idxs_with_response]
    corrAns = corrans_num[idxs_with_response]

    print(log_freq_percept, log_freq_seq_array.shape)
    
    os.chdir("/home/janakis/results/longContextLowFromProlific/randomChoice/shortTermContext/")
        
    """
    Subsampling trials and behaviour
    """    
    for iteration in np.arange(0,300,30):
        if len(trialBehaviour)>iteration+500:
            trial_tones = trialTones[iteration:iteration+500,:]
            trial_behaviour = trialBehaviour[iteration:iteration+500]
            trial_corrans = corrAns[iteration:iteration+500]
            filename = sys.argv[2] + str(iteration) + ".txt"
            new_file = open(filename,"a+")   
            write_into_file(params=sys.argv[3],fresult = new_file)  
        
    new_file.close()
