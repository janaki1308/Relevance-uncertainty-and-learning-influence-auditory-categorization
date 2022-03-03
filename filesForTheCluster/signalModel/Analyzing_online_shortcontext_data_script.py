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

def posterior_array(freq_input, n_tones, p_back, p_low, log_prior):
    """
    Arguments: 
    freq_input - range of all possible frequencies (percepts?)
    p_back - prob of background
    p_low - prob of low condition
    log_prior - list of prior parameters
    """
    
    log_prior_low_mean = log_prior[0]; log_prior_low_sigma = log_prior[2];
    log_prior_high_mean = log_prior[1]; log_prior_high_sigma = log_prior[2];
    prior_low = gaussian(x=freq_input, mean=log_prior_low_mean, sigma=log_prior_low_sigma)
    prior_high = gaussian(x=freq_input, mean=log_prior_high_mean, sigma=log_prior_high_sigma)
    prior_dist_mixed_high = p_back*(1/len(freq_input)) + (1-p_back)*prior_high \
    #mixture model with p(T|B) = 1/no. of possible freqs
    prior_dist_mixed_high /= prior_dist_mixed_high.sum() #normalizing
    prior_dist_mixed_high = np.expand_dims(prior_dist_mixed_high, axis = 1)
    prior_dist_mixed_low = p_back*(1/len(freq_input)) + (1-p_back)*prior_low \
    #mixture model with p(T|B) = 1/no. of possible freqs
    prior_dist_mixed_low /= prior_dist_mixed_low.sum() #normalizing
    prior_dist_mixed_low = np.expand_dims(prior_dist_mixed_low, axis = 1)
        
    if n_tones == 3:
        prior_tones_low = np.expand_dims(prior_dist_mixed_low@np.transpose\
                                         (prior_dist_mixed_low),axis=2)@np.transpose(prior_dist_mixed_low) \
        #p(T1,T2..|L) 
        prior_tones_high = np.expand_dims(prior_dist_mixed_high@np.transpose\
                                          (prior_dist_mixed_high),axis=2)@np.transpose(prior_dist_mixed_high) \
        #p(T1,T2..|H) 
    elif n_tones == 1:
        prior_tones_low = prior_dist_mixed_low
        prior_tones_high = prior_dist_mixed_high
        
    normalizer = (1-p_low)*prior_tones_high + p_low*prior_tones_low #p(H)*p(T1,T2..|H) + p(L)*p(T1,T2..|L)
    posterior = prior_tones_high*(1-p_low)/normalizer
    # posterior /= np.sum(posterior)
    
    return prior_dist_mixed_high, prior_dist_mixed_low, prior_tones_high, prior_tones_low, normalizer, posterior                     
def behaviour(trial_tones, trial_behaviour, corrans_num, keys_num):
    """
    This is a different way of plotting the influence function, though very similar to the inital kind. 
    We also compare this to the short term context plots. 
    """

    nobias_tones = np.unique(trial_tones)
    dict_behaviour = {}
    bev = np.zeros((len(nobias_tones)))

    for i_tone in range(len(nobias_tones)):
        temp = np.array([],dtype=int)
        for itrial in range(len(trial_tones)):
            if nobias_tones[i_tone] in trial_tones[itrial]:
                temp = np.append(temp,int(itrial))
        dict_behaviour[i_tone] = temp.astype(int)        
        bev[i_tone] = np.mean(trial_behaviour[dict_behaviour[i_tone]])

    """
    Plot responses when the previous n responses are H
    """
    n=1
    idxPrevBHigh=[]; idxPrevBHighCurrHigh=[]; idxPrevBHighCurrLow=[];
    for ii in range(n,len(trial_tones)):
        if sum(corrans_num[ii-n:ii])==n and sum(keys_num[ii-n:ii])==n: 
            if keys_num[ii]==1:
                idxPrevBHighCurrHigh+=[ii]
            else:
                idxPrevBHighCurrLow+=[ii]
            idxPrevBHigh+=[ii]    
    print(len(idxPrevBHighCurrHigh),len(idxPrevBHighCurrLow))
    
    idxPrevBLow=[]; idxPrevBLowCurrLow=[]; idxPrevBLowCurrHigh=[];
    for ii in range(n,len(trial_tones)):
        if sum(corrans_num[ii-n:ii])==0 and sum(keys_num[ii-n:ii])==0: 
            if keys_num[ii]==1:
                idxPrevBLowCurrHigh+=[ii]
            else:
                idxPrevBLowCurrLow+=[ii]
            idxPrevBLow+=[ii]    
    print(len(idxPrevBLowCurrHigh),len(idxPrevBLowCurrLow))

    scPrevBHigh_trialtones = trial_tones[idxPrevBHigh,:]
    scPrevBHigh_trialbehaviour = trial_behaviour[idxPrevBHigh]

    scBHigh_tones = np.unique(scPrevBHigh_trialtones)
    bev_scBHigh= np.zeros((len(scBHigh_tones)))

    for i_tone in range(len(scBHigh_tones)):
        temp = np.array([],dtype=int)
        for itrial in range(len(scPrevBHigh_trialtones)):
            if scBHigh_tones[i_tone] in scPrevBHigh_trialtones[itrial]:
                temp = np.append(temp,int(itrial))
        dict_behaviour[i_tone] = temp.astype(int)        
        bev_scBHigh[i_tone] = np.mean(scPrevBHigh_trialbehaviour[dict_behaviour[i_tone]])
        
    return idxPrevBHigh, idxPrevBHighCurrHigh, idxPrevBHighCurrLow, \
idxPrevBLow, idxPrevBLowCurrHigh, idxPrevBLowCurrLow

# define mle function
def MLE(params):
    log_prior_low_mean, log_prior_high_mean, log_prior_sigma, sigma_sensory, prob_back, prob_low, alpha = \
    params[0], params[1], params[2], params[3], params[4], params[5], params[6] # inputs are parameters guesses
    
    scPrevBhigh_prob_low = alpha*prob_low 
    scPrevBlow_prob_low = alpha*prob_low + (1-alpha)
    
    _,_,LikelihoodLatentTonegivenHigh,LikelihoodLatentTonegivenLow,_,_ = \
    posterior_array(log_freq_seq_array, len(trial_tones[0]), p_back=prob_back, p_low=prob_low,\
                    log_prior=[log_prior_low_mean,log_prior_high_mean,log_prior_sigma])

    LikelihoodPerceptgivenHigh = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    LikelihoodPerceptgivenLow = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    
    for itrue1 in range(len(log_freq_seq_array)):
        for itrue2 in range(len(log_freq_seq_array)):
            for itrue3 in range(len(log_freq_seq_array)):
                probPerceptgivenLatentTones = Tones3dgrid([log_freq_seq_array[itrue1],\
                                                           log_freq_seq_array[itrue2],\
                                                           log_freq_seq_array[itrue3]],sigma=sigma_sensory)
                LikelihoodPerceptgivenHigh \
                += probPerceptgivenLatentTones * LikelihoodLatentTonegivenHigh[itrue1,itrue2,itrue3]
                LikelihoodPerceptgivenLow \
                += probPerceptgivenLatentTones * LikelihoodLatentTonegivenLow[itrue1,itrue2,itrue3]
    probHighgivenPercept = LikelihoodPerceptgivenHigh*(1-prob_low)/\
    (LikelihoodPerceptgivenHigh*(1-prob_low) + LikelihoodPerceptgivenLow*(prob_low))
    
    scPrevBHigh_probHighgivenPercept = LikelihoodPerceptgivenHigh*(1-scPrevBhigh_prob_low)/\
    (LikelihoodPerceptgivenHigh*(1-scPrevBhigh_prob_low) + LikelihoodPerceptgivenLow*(scPrevBhigh_prob_low))
    
    scPrevBLow_probHighgivenPercept = LikelihoodPerceptgivenHigh*(1-scPrevBlow_prob_low)/\
    (LikelihoodPerceptgivenHigh*(1-scPrevBlow_prob_low) + LikelihoodPerceptgivenLow*(scPrevBlow_prob_low))
    
    neg_ll = 0; 
    probability_high = np.zeros((len(trial_tones),1))
    for i_trial in range(len(trial_tones)):
        input_array_mat = Tones3dgrid(np.array([np.log10(trial_tones[i_trial][0]),\
                                               np.log10(trial_tones[i_trial][1]),
                                               np.log10(trial_tones[i_trial][2])]),sigma=sigma_sensory)
        if i_trial in idxPrevBHigh:
            probability_high0 = np.sum(np.multiply(scPrevBHigh_probHighgivenPercept>0.5,input_array_mat))
            probability_high[i_trial] = np.sum(np.multiply(scPrevBHigh_probHighgivenPercept>0.5,input_array_mat))
        elif i_trial in idxPrevBLow:
            probability_high0 = np.sum(np.multiply(scPrevBLow_probHighgivenPercept>0.5,input_array_mat))
            probability_high[i_trial] = np.sum(np.multiply(scPrevBLow_probHighgivenPercept>0.5,input_array_mat))
        else:
            probability_high0 = np.sum(np.multiply(probHighgivenPercept>0.5,input_array_mat))
            probability_high[i_trial] = np.sum(np.multiply(probHighgivenPercept>0.5,input_array_mat))    
            
        if trial_behaviour[i_trial]:
            if np.isnan(np.log(probability_high0 + 0.0000001)) \
            or np.isinf(np.log(probability_high0 + 0.0000001)) \
            or np.isnan(np.log(1-probability_high0 + 0.0000001)) \
            or np.isinf(np.log(1-probability_high0 + 0.0000001)):
                pdb.set_trace()
            neg_ll += -np.log(probability_high0 + 0.0000001) # if high dist is chosen by observer
        else:
            neg_ll += -np.log(1 - probability_high0 + 0.0000001) # if low dist is chosen by observer
    return(neg_ll, probability_high)

def write_into_file(lm_start, lm_end, hm_start, hm_end, fresult):
    """
    New optimization algorithm: uses scipy.optimize.fmin. 
    Crude grid initially and then find minimum using the function.
    """
    
    guess_low_mean = np.arange(lm_start,lm_end,0.15); guess_high_mean = np.arange(hm_start,hm_end,0.15); 
    guess_sigma = np.arange(0.05,1,0.2); guess_sensory_sigma = np.arange(0.05,1,0.2);
    guess_p_back = np.arange(0.05,1,0.2); guess_p_low = np.arange(0.05,1,0.2);
    guess_alpha = np.arange(0,1.1,0.2)

    # Constraining guesses of means of low and high distributions based on observed behaviour in figure shown above. 

    neg_ll_array = np.full((len(guess_low_mean), len(guess_high_mean), len(guess_sigma), 
                        len(guess_sensory_sigma), len(guess_p_back), len(guess_p_low), len(guess_alpha)),
                       fill_value=0., dtype=np.float64)
    for lm in range(len(guess_low_mean)):
        for hm in range(len(guess_high_mean)):
            for s in range(len(guess_sigma)):
                for ss in range(len(guess_sensory_sigma)):
                    for pb in range(len(guess_p_back)):
                        for pl in range(len(guess_p_low)):
                            for al in range(len(guess_alpha)):
                                params = [guess_low_mean[lm], guess_high_mean[hm], guess_sigma[s],
                                          guess_sensory_sigma[ss], guess_p_back[pb], guess_p_low[pl], guess_alpha[al]]
                                neg_ll_array[lm,hm,s,ss,pb,pl,al],_ = MLE(params) 
                                fresult.write("%s\n" % neg_ll_array[lm,hm,s,ss,pb,pl,al])
                                fresult.flush()
            
if __name__ == '__main__':
        
    """ 
    Obtaining data from a given expt
    """
    csv_test = pd.read_csv('/home/janakis/data/allTrials_nobias.csv')
    data_file = sys.argv[1]
    csv_data = pd.read_csv(data_file)
    
    """
    Get tones and values of keys pressed
    """
    test_columns = list(csv_test.columns)
    test_tones_name = test_columns.index('Name')
    test_tones_col_idx = test_columns.index('Tones')
    df_names = (csv_test.iloc[0:600,test_tones_name]).values
    df_tones = (csv_test.iloc[0:600,test_tones_col_idx]).values

    n_tones = 3
    n_trials = csv_data.shape[0]-45

    tones_array_orig = np.zeros((n_trials,n_tones))
    tones_array_idxs_keep = []

    for i_wav in range(603):
        if isinstance(csv_data['Name'][i_wav+44],str):
            tones_array_orig[i_wav,:] = np.array(df_tones[np.where(csv_data['Name'][i_wav+44]\
                                                              ==df_names)[0]][0][1:-1].split(',')).astype(float)  
            tones_array_idxs_keep += [i_wav]


    df_tones = np.copy(tones_array_orig[tones_array_idxs_keep,:])
    df_corrans = np.copy(csv_data['corrAns'][44:csv_data.shape[0]])[tones_array_idxs_keep]
    df_keys = np.copy(csv_data['test_resp.keys'][44:csv_data.shape[0]])[tones_array_idxs_keep]
    
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

    corrans_num_all = corrans_num_orig[:600]
    keys_num_all = keys_num_orig[:600]
    tones_array = df_tones[:600]
    
    """
    Latent variables
    """
    expt_tones = np.arange(90,3000,1) #array of possible true tones
    log_freq_seq_array = np.arange(0.6,4.7,0.1)
    log_freq_percept = np.arange(0.6,4.7,0.1) # array of possible perceptual tones
    
    """
    Experimental data variables: tones and behaviour
    """
    trial_tones = np.repeat(tones_array,1,axis = 0)
    trial_behaviour = np.reshape(keys_num_all,np.prod(keys_num_all.shape)) 
    # this has been changed to check how values change with observer responses    

    idxs_with_response = np.delete(np.arange(len(trial_tones)),no_response)
    trial_tones = trial_tones[idxs_with_response,:]
    trial_behaviour = trial_behaviour[idxs_with_response]
    corrans_num = corrans_num_all[idxs_with_response]
    keys_num = keys_num_all[idxs_with_response]
    print(log_freq_percept, log_freq_seq_array.shape)
    print(corrans_num.shape, keys_num.shape, trial_tones.shape)
    print("Got correct: ", np.sum(keys_num==corrans_num)/len(trial_tones))
    
    idxPrevBHigh, idxPrevBHighCurrHigh, idxPrevBHighCurrLow, \
    idxPrevBLow, idxPrevBLowCurrHigh, idxPrevBLowCurrLow = behaviour(trial_tones, trial_behaviour,
                                                                    corrans_num, keys_num)

    ratioHigh = len(idxPrevBHighCurrLow)/(len(idxPrevBHighCurrLow)+len(idxPrevBHighCurrHigh))
    ratioLow = len(idxPrevBLowCurrLow)/(len(idxPrevBLowCurrLow)+len(idxPrevBLowCurrHigh))
    print(ratioHigh, ratioLow)

    os.chdir("/home/janakis/results/shortterm")
    new_file = open(sys.argv[2],"a+")
         
    write_into_file(lm_start=float(sys.argv[3]), lm_end=float(sys.argv[4]), 
                    hm_start=float(sys.argv[5]), hm_end=float(sys.argv[6]),
                    fresult = new_file)  
        
    new_file.close()    
        