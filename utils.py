import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from tqdm.notebook import tqdm
import random
import pandas as pd
from scipy.io import savemat
import scipy
import matplotlib.cm as cm

"""
Analysis for new no context experiments. 
"""

log_freq_percept = np.arange(0.6,4.7,0.1)

def gaussian(x, mean, sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))

def Tones3dgrid(latentTones, sigma, freq_seq=log_freq_percept):
    
    input_array_0 = np.expand_dims(gaussian(freq_seq, latentTones[0], sigma), axis = 1)
    input_array_1 = np.expand_dims(gaussian(freq_seq, latentTones[1], sigma), axis = 1)
    input_array_2 = np.expand_dims(gaussian(freq_seq, latentTones[2], sigma), axis = 1)
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

def analysis2(csv_test,csv_data,total_number_possible_responses):

    n_tones = 3
    n_trials = csv_data.shape[0]-47

    """
    Get tones and values of keys pressed
    """
    test_columns = list(csv_test.columns)
    test_tones_name = test_columns.index('Name')
    test_tones_col_idx = test_columns.index('Tones')
    test_tones_cat_col_idx = test_columns.index('Tonekind')

    df_names = (csv_test.iloc[:,test_tones_name]).values
    df_tones = (csv_test.iloc[:,test_tones_col_idx]).values
    df_tone_cat = (csv_test.iloc[:,test_tones_cat_col_idx]).values

    tones_array_orig = np.zeros((n_trials,n_tones))
    tones_array_idxs_keep = []

    tones_cat_array_orig = np.zeros((n_trials,n_tones))
    tones_cat_array_idxs_keep = []

    for i_wav in range(n_trials):
        if isinstance(csv_data['Name'][i_wav+46],str):
            tones_array_orig[i_wav,:] = np.array(df_tones[np.where(csv_data['Name'][i_wav+46]\
                                                              ==df_names)[0]][0][1:-1].split(',')).astype(float)  
            tones_array_idxs_keep += [i_wav]

            tones_cat_array_orig[i_wav,:] = np.array(df_tone_cat[np.where(csv_data['Name'][i_wav+46]\
                                                              ==df_names)[0]][0][1:-1].split(',')).astype(float)  
            tones_cat_array_idxs_keep += [i_wav]


    df_tones = np.copy(tones_array_orig[tones_array_idxs_keep,:])
    df_tone_cat = np.copy(tones_cat_array_orig[tones_cat_array_idxs_keep,:])
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

    corrans_num = corrans_num_orig[:total_number_possible_responses]
    keys_num = keys_num_orig[:total_number_possible_responses]
    tones_array = df_tones[:total_number_possible_responses,:]
    tone_cat_array = df_tone_cat[:total_number_possible_responses,:]

    trial_tones = np.repeat(tones_array,1,axis = 0)
    trial_tone_cat = np.repeat(tone_cat_array,1,axis = 0)
    trial_behaviour = np.reshape(keys_num,np.prod(keys_num.shape)) 
    correct_answer = np.reshape(corrans_num,np.prod(corrans_num.shape)) 
    
    idxs_with_response = np.delete(np.arange(len(trial_tones)),no_response)
            
    trial_tones = trial_tones[idxs_with_response,:]
    trial_tone_cat = trial_tone_cat[idxs_with_response,:]
    trial_behaviour = trial_behaviour[idxs_with_response]
    correct_answer = correct_answer[idxs_with_response]
    
    return trial_tones, trial_behaviour, trial_tone_cat, correct_answer


def generate_behaviour(trial_tones, reps, n_tones, prob_back, prob_low, log_prior_params, sigma_sensory):
    """
    Trying two routes - 1. what if we have both sensory noise in that the perceived tones are from a gaussian 
    whose mean is the true tone and we have decision noise in that the at a particular perceived tone the observer 
    chooses high with probability p(H|T). So a trial is basically defined as [trial_tone, perceived_tone and 
    decision] 
    2. what if we only have sensory noise and the decision made is the best decision at a particular perceived 
    tone. 

    """    

    all_trial_tones = np.empty((len(trial_tones)*reps,n_tones))
    all_trial_behaviour = np.empty((len(trial_tones)*reps,1))
    prob_trial_behaviour = np.empty((len(trial_tones),1))
    probability_sim_high = np.zeros((len(trial_tones),1))

    _,_,LikelihoodLatentTonegivenHigh,LikelihoodLatentTonegivenLow,_,_ = \
    posterior_array(log_freq_percept, len(trial_tones[0]), p_back=prob_back, p_low=prob_low,
                    log_prior=log_prior_params)

    LikelihoodPerceptgivenHigh = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    LikelihoodPerceptgivenLow = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))

    for itrue1 in range(len(log_freq_percept)):
        for itrue2 in range(len(log_freq_percept)):
            for itrue3 in range(len(log_freq_percept)):
                probPerceptgivenLatentTones = Tones3dgrid([log_freq_percept[itrue1],
                                                           log_freq_percept[itrue2],
                                                           log_freq_percept[itrue3]],
                                                           freq_seq=log_freq_percept,
                                                           sigma=sigma_sensory)
                LikelihoodPerceptgivenHigh \
                += probPerceptgivenLatentTones * LikelihoodLatentTonegivenHigh[itrue1,itrue2,itrue3]
                LikelihoodPerceptgivenLow \
                += probPerceptgivenLatentTones * LikelihoodLatentTonegivenLow[itrue1,itrue2,itrue3]
    probHighgivenPercept = LikelihoodPerceptgivenHigh*(1-prob_low)/\
    (LikelihoodPerceptgivenHigh*(1-prob_low) + LikelihoodPerceptgivenLow*prob_low)

    for i_stim in range(len(trial_tones)):
        input_array = np.random.normal(loc=np.log10(trial_tones[i_stim]),scale=sigma_sensory,
                                       size=(reps,1,n_tones)) \
        #pick tones from the gaussian with mean as log(true_tone) and sensory sigma 0.1    
        for i_tperc in range(reps):
            perc_tone_idxs = np.zeros((n_tones,1),dtype=int)
            for i in range(n_tones):
                perc_tone_idxs[i] = np.argmin(np.abs(log_freq_percept-input_array[i_tperc][0][i]))
                # find relevant adjacent freq percepts   
            posterior_perc_tone = probHighgivenPercept[perc_tone_idxs[0],perc_tone_idxs[1],perc_tone_idxs[2]]
            # trial_behaviour = (np.random.random_sample() < np.squeeze(posterior_perc_tone)).astype(int)
            # this encodes decision noise
            trial_behaviour = np.squeeze(posterior_perc_tone) > 0.5
            # this makes the same choice for one tone percept every time that tone is perceived   
            all_trial_behaviour[i_stim*reps+i_tperc,:] = trial_behaviour
        all_trial_tones[i_stim*reps:(i_stim+1)*reps,:] = trial_tones[i_stim]    
        prob_trial_behaviour[i_stim] = np.mean(all_trial_behaviour[i_stim*reps:(i_stim+1)*reps])

        gaussian_array_mat = Tones3dgrid(np.array([np.log10(trial_tones[i_stim][0]),
                                                   np.log10(trial_tones[i_stim][1]),
                                                   np.log10(trial_tones[i_stim][2])]),
                                                   freq_seq=log_freq_percept,
                                                   sigma=sigma_sensory)         
        probability_sim_high[i_stim] = np.sum(np.multiply(probHighgivenPercept>0.5, gaussian_array_mat))

    """
    Shuffling the tones and the behaviour to simluate an experiment

    s = np.arange(all_trial_tones.shape[0])
    np.random.shuffle(s)
    all_trial_tones = all_trial_tones[s]
    all_trial_behaviour = all_trial_behaviour[s]
    """
    return all_trial_tones, probability_sim_high


def generate_behaviourVoting(trial_tones, reps, n_tones, decision_boundary, sigma_sensory):
    """
    Trying two routes - 1. what if we have both sensory noise in that the perceived tones are from a gaussian 
    whose mean is the true tone and we have decision noise in that the at a particular perceived tone the observer 
    chooses high with probability p(H|T). So a trial is basically defined as [trial_tone, perceived_tone and 
    decision] 
    2. what if we only have sensory noise and the decision made is the best decision at a particular perceived 
    tone. 

    """    

    all_trial_tones = np.empty((len(trial_tones)*reps,n_tones))
    all_trial_behaviour = np.empty((len(trial_tones)*reps,1))
    prob_trial_behaviour = np.empty((len(trial_tones),1))
    probability_sim_high = np.zeros((len(trial_tones),1))

    probHighgivenPercept = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    for ii in range(len(log_freq_percept)):
        for jj in range(len(log_freq_percept)):
            for kk in range(len(log_freq_percept)):
                if np.sum([log_freq_percept[ii],log_freq_percept[jj],log_freq_percept[kk]]>decision_boundary)>1:
                    probHighgivenPercept[ii,jj,kk] = 1

    for i_stim in range(len(trial_tones)):
        input_array = np.random.normal(loc=np.log10(trial_tones[i_stim]),scale=sigma_sensory,
                                       size=(reps,1,n_tones)) \
        #pick tones from the gaussian with mean as log(true_tone) and sensory sigma 0.1    
        for i_tperc in range(reps):
            perc_tone_idxs = np.zeros((n_tones,1),dtype=int)
            for i in range(n_tones):
                perc_tone_idxs[i] = np.argmin(np.abs(log_freq_percept-input_array[i_tperc][0][i]))
                # find relevant adjacent freq percepts   
            posterior_perc_tone = probHighgivenPercept[perc_tone_idxs[0],perc_tone_idxs[1],perc_tone_idxs[2]]
            trial_behaviour = np.squeeze(posterior_perc_tone) > 0.5
            # this makes the same choice for one tone percept every time that tone is perceived   
            all_trial_behaviour[i_stim*reps+i_tperc,:] = trial_behaviour
        all_trial_tones[i_stim*reps:(i_stim+1)*reps,:] = trial_tones[i_stim]    
        prob_trial_behaviour[i_stim] = np.mean(all_trial_behaviour[i_stim*reps:(i_stim+1)*reps])

        gaussian_array_mat = Tones3dgrid(np.array([np.log10(trial_tones[i_stim][0]),
                                                   np.log10(trial_tones[i_stim][1]),
                                                   np.log10(trial_tones[i_stim][2])]), freq_seq = log_freq_percept,
                                                   sigma=sigma_sensory)         
        probability_sim_high[i_stim] = np.sum(np.multiply(probHighgivenPercept, gaussian_array_mat))

    return all_trial_tones, probability_sim_high

# define mle function
def MLE_voting(params, trial_tones, trial_behaviour):
    decision_boundary, sigma_sensory = params[0], params[1] # inputs are guesses at our parameters  
    
    probHighgivenPercept = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    for ii in range(len(log_freq_percept)):
        for jj in range(len(log_freq_percept)):
            for kk in range(len(log_freq_percept)):
                if np.sum([log_freq_percept[ii],log_freq_percept[jj],log_freq_percept[kk]]>decision_boundary)>1:
                    probHighgivenPercept[ii,jj,kk] = 1
        
    neg_ll = np.zeros((len(trial_tones),1));
    probability_high = np.zeros((len(trial_tones),))
    for i_trial in range(len(trial_tones)):
        input_array_mat = Tones3dgrid(np.array([np.log10(trial_tones[i_trial][0]),\
                                               np.log10(trial_tones[i_trial][1]),
                                               np.log10(trial_tones[i_trial][2])]),
                                               freq_seq = log_freq_percept,
                                               sigma=sigma_sensory) 
        probability_high[i_trial] = np.sum(np.multiply(probHighgivenPercept,input_array_mat))
        #pdb.set_trace()

        if trial_behaviour[i_trial]:
            if np.isnan(np.log(probability_high[i_trial] + 0.0000001)) \
            or np.isinf(np.log(probability_high[i_trial] + 0.0000001)) \
            or np.isnan(np.log(1-probability_high[i_trial] + 0.0000001)) \
            or np.isinf(np.log(1-probability_high[i_trial] + 0.0000001)):
                pdb.set_trace()
            neg_ll[i_trial] += -np.log(probability_high[i_trial] + 0.0000001) # if high dist is chosen by observer
        else:
            neg_ll[i_trial] += -np.log(1 - probability_high[i_trial] + 0.0000001) # if low dist is chosen by observer
    return(neg_ll)

# define mle function
def MLE(params, trial_tones, trial_behaviour):
    #pdb.set_trace()
    
    log_prior_low_mean, log_prior_high_mean, log_prior_sigma, sigma_sensory, prob_back, prob_low = \
    params[0], params[1], params[2], params[3], params[4], params[5] # inputs are guesses at our parameters  
    
    _,_,LikelihoodLatentTonegivenHigh,LikelihoodLatentTonegivenLow,_,_ = \
    posterior_array(log_freq_percept, n_tones=3, p_back=prob_back, p_low=prob_low,\
                    log_prior=[log_prior_low_mean,log_prior_high_mean,log_prior_sigma])

    LikelihoodPerceptgivenHigh = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    LikelihoodPerceptgivenLow = np.zeros((len(log_freq_percept),len(log_freq_percept),len(log_freq_percept)))
    
    for itrue1 in range(len(log_freq_percept)):
        for itrue2 in range(len(log_freq_percept)):            
            for itrue3 in range(len(log_freq_percept)):
                probPerceptgivenLatentTones = Tones3dgrid([log_freq_percept[itrue1],\
                                                           log_freq_percept[itrue2],\
                                                           log_freq_percept[itrue3]],
                                                          freq_seq = log_freq_percept,
                                                           sigma=sigma_sensory)                                                            
                LikelihoodPerceptgivenHigh \
                += probPerceptgivenLatentTones * LikelihoodLatentTonegivenHigh[itrue1,itrue2,itrue3]
                LikelihoodPerceptgivenLow \
                += probPerceptgivenLatentTones * LikelihoodLatentTonegivenLow[itrue1,itrue2,itrue3]
    probHighgivenPercept = LikelihoodPerceptgivenHigh*(1-prob_low)/\
    (LikelihoodPerceptgivenHigh*(1-prob_low) + LikelihoodPerceptgivenLow*(prob_low))
    #pdb.set_trace()
    
    neg_ll = np.zeros((len(trial_tones),1));
    probability_high = np.zeros((len(trial_tones),1))
    for i_trial in range(len(trial_tones)):
        input_array_mat = Tones3dgrid(np.array([np.log10(trial_tones[i_trial][0]),\
                                               np.log10(trial_tones[i_trial][1]),
                                               np.log10(trial_tones[i_trial][2])]),
                                      freq_seq = log_freq_percept,
                                      sigma=sigma_sensory)
        probability_high0 = np.sum(np.multiply(probHighgivenPercept>0.5,input_array_mat))
        probability_high[i_trial] = np.sum(np.multiply(probHighgivenPercept>0.5,input_array_mat))
            
        if trial_behaviour[i_trial]:
            if np.isnan(np.log(probability_high0 + 0.0000001)) \
            or np.isinf(np.log(probability_high0 + 0.0000001)) \
            or np.isnan(np.log(1-probability_high0 + 0.0000001)) \
            or np.isinf(np.log(1-probability_high0 + 0.0000001)):
                pdb.set_trace()
            neg_ll[i_trial] += -np.log(probability_high0 + 0.0000001) # if high dist is chosen by observer
        else:
            neg_ll[i_trial] += -np.log(1 - probability_high0 + 0.0000001) # if low dist is chosen by observer
    #print(params, neg_ll)
    #pdb.set_trace()
    return(np.sum(neg_ll))

def psuedoRsquared(fitY,exptY,exptSem):
    """
    Error is based on deviance.
    """
    err = 1-np.sum((fitY-exptY)**2)/np.sum((exptY-np.mean(exptY))**2)
    return err

def plotting_pBHgivenT(subjectBehaviour, subjectTones, contextIndex):
    
    sampledSubjBhv = subjectBehaviour[contextIndex]
    sampledSubjTones = subjectTones[contextIndex,:]
    uniqueTonesSubj = np.unique(sampledSubjTones)
    tone1_prob_behaviour = np.zeros((len(uniqueTonesSubj)))
    tone2_prob_behaviour = np.zeros((len(uniqueTonesSubj)))
    tone3_prob_behaviour = np.zeros((len(uniqueTonesSubj)))

    for i_tone in range(len(uniqueTonesSubj)):
        tone1_prob_behaviour[i_tone] = np.mean(sampledSubjBhv[sampledSubjTones[:,0]==uniqueTonesSubj[i_tone]])
        tone2_prob_behaviour[i_tone] = np.mean(sampledSubjBhv[sampledSubjTones[:,1]==uniqueTonesSubj[i_tone]])
        tone3_prob_behaviour[i_tone] = np.mean(sampledSubjBhv[sampledSubjTones[:,2]==uniqueTonesSubj[i_tone]])   
        
    bhvSubj_mean = np.nanmean([tone1_prob_behaviour,tone2_prob_behaviour,tone3_prob_behaviour],axis=0)
    bhvSubj_std = np.nanstd([tone1_prob_behaviour,tone2_prob_behaviour,tone3_prob_behaviour],axis=0)     
    return uniqueTonesSubj, bhvSubj_mean, bhvSubj_std

def visualizeProbDistributions(sample_x, log_freq_low, log_freq_high):
    
    lowDistWeights = gaussian(mean = log_freq_low[0], 
                              sigma = log_freq_low[1], 
                              x = sample_x);
    lowDistWeights = lowDistWeights/sum(lowDistWeights);
    
    highDistWeights = gaussian(mean = log_freq_high[0], 
                               sigma = log_freq_high[1], 
                               x = sample_x);
    highDistWeights = highDistWeights/sum(highDistWeights);
    return lowDistWeights, highDistWeights