import numpy as np
import matplotlib.pyplot as plt
import pdb

def task(n_trials = 10, n_tones = 3):
    freq_back = [1,10]
    freq_low = [4,1]
    freq_high = [7,1]
    trial_tones = []
    
    for trial in range(n_trials):
        signal_rand = np.random.random()
        low_dist = signal_rand < 0.5
        tones = []
        for n_tone in range(n_tones):
            signal_back = np.random.random()
            background = signal_back < 0.2
            if background:
                tone = int(np.random.randint(freq_back[0],freq_back[1],1))
            else: 
                if low_dist:
                    tone = min(max(np.random.randn()*freq_low[1] + freq_low[0],1),10)
                else:
                    tone = min(max(np.random.randn()*freq_high[1] + freq_high[0],1),10)
            tones.append(tone)
        trial_tones.append(tones)        
    return trial_tones

def gaussian(x, mean, sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))

def posterior_array(freq_input, tones_idx, p_back, p_low, 
                    prior_back_min = 1, prior_back_max = 10,
                    prior_low_mean = 4, prior_low_sigma = 1,
                    prior_high_mean = 7, prior_high_sigma = 1):
    
    prior_low = gaussian(x=freq_input, mean=prior_low_mean, sigma=prior_low_sigma)
    prior_high = gaussian(x=freq_input, mean=prior_high_mean, sigma=prior_high_sigma)
    prior_dist_mixed_high = p_back*(1/len(range(prior_back_min, prior_back_max))) + (1-p_back)*prior_high
    prior_dist_mixed_high /= prior_dist_mixed_high.sum()
    prior_dist_mixed_low = p_back*(1/len(range(prior_back_min, prior_back_max))) + (1-p_back)*prior_low
    prior_dist_mixed_low /= prior_dist_mixed_low.sum()
    
    prior_tones_low = 1; prior_tones_high = 1
    for t in tones_idx:
        prior_tones_high *= prior_dist_mixed_high[t]
        prior_tones_low *= prior_dist_mixed_low[t]
    
    normalizer = (1-p_low)*prior_tones_high + p_low*prior_tones_low
    posterior = prior_tones_high*(1-p_low)/normalizer
    
    return prior_dist_mixed_high, prior_dist_mixed_low, posterior
