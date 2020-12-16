import numpy as np
import matplotlib.pyplot as plt
import pdb

def task(n_trials = 10):
    freq_back = [1,10]
    freq_low = [4,1]
    freq_high = [7,1]
    trial_tones = []
    
    for trial in range(n_trials):
        signal_rand = np.random.random()
        low_dist = signal_rand < 0.5
        tones = []
        for n_tone in range(3):
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
    
def prior_array(freq_input, p_back, p_low, 
                    prior_back_min = 1, prior_back_max = 10,
                    prior_low_mean = 4, prior_low_sigma = 1,
                    prior_high_mean = 7, prior_high_sigma = 1):
    
    prior_low = gaussian(x=freq_input, mean=prior_low_mean, sigma=prior_low_sigma)
    prior_high = gaussian(x=freq_input, mean=prior_high_mean, sigma=prior_high_sigma)
    prior_mixed = p_back*(1/len(range(prior_back_min, prior_back_max))) + (1-p_back)*(p_low*prior_low + (1-p_low)*prior_high)             
    prior_mixed /= prior_mixed.sum()
    return prior_mixed

def likelihood_array(freq_points, freq_input, sigma = 1):
    likelihood = np.zeros((len(freq_input),len(freq_points)))
    for i in range(len(freq_input)):
        likelihood[i,:] = gaussian(x=freq_points, mean=freq_input[i], sigma=sigma)
        likelihood[i,:] /= likelihood[i,:].sum()
    return likelihood

def posterior_array(freq_input, freq_points, prior, likelihood):
    posterior = np.zeros((len(freq_points), len(freq_input)))
    for i in range(len(freq_points)):
        posterior[i,:] = np.multiply(likelihood[:,i],prior)
    posterior /= np.sum(posterior, axis = 1, keepdims = True)    
    return posterior

def estimated_freq(freq_input, posterior):
    estimation = np.zeros_like(posterior)

    for i in range(len(posterior)):
        mean =  np.sum(freq_input*posterior[i])
        idx = np.argmin(np.abs(freq_input - mean))
        estimation[i, idx] = 1 
    return estimation

def sigmoid_response(freq_input, prior_low_mean, prior_low_sigma, 
                     prior_high_mean, prior_high_sigma):
    
    prior_low = gaussian(x=freq_input, mean=prior_low_mean, sigma=prior_low_sigma)
    prior_high = gaussian(x=freq_input, mean=prior_high_mean, sigma=prior_high_sigma)    
    intersect_freq = freq_input[abs(prior_low - prior_high) < 0.001]
    sigmoid_fn = 1./(1+np.exp(-(freq_input - intersect_freq)))
    sigmoid_fn[freq_input < prior_low_mean-2*prior_low_sigma] = 0
    sigmoid_fn[freq_input > prior_high_mean+2*prior_high_sigma] = 0
    return sigmoid_fn
    
def high_response(freq_input, estimated_freq, 
                  prior_low_mean, prior_low_sigma, 
                  prior_high_mean, prior_high_sigma):
    # function not used and needs to be checked
    response_function = np.zeros(len(estimated_freq))
    for i in range(len(estimated_freq)):
        reponse_function[i] = sigmoid(freq_input, freq_input(estimated_freq[i]==1),
                                      prior_low_mean, prior_low_sigma, 
                                      prior_high_mean, prior_high_sigma)
    return response_function    

def encoding_given_true_freq(freq_points, tone, sigma=1):
    encoding_of_true_freq = gaussian(freq_points, mean=tone, sigma=sigma)
    return encoding_of_true_freq/np.sum(encoding_of_true_freq)

def probability_estimated_freq(freq_estimation, encoding_of_true_freq):
    marginal = np.transpose(freq_estimation) @ encoding_of_true_freq    
    marginal /= np.sum(marginal)
    return marginal
    
def probability_high_response(sigmoid_weight, marginal):   
    return np.sum(sigmoid_weight @ marginal)

def main():    
    trial_tones = task()
    freq_input = np.linspace(1,10,100)
    freq_points = np.linspace(0,11,100)
    prior = prior_array(freq_input, p_back = 0.2, p_low = 0.5)
    likelihood = likelihood_array(freq_points, freq_input)
    posterior = posterior_array(freq_input, freq_points, prior, likelihood)
    freq_estimation = estimated_freq(freq_input, posterior)
    sigmoid_weight = sigmoid_response(freq_input, prior_low_mean=4, prior_low_sigma=1, 
                                     prior_high_mean=7, prior_high_sigma=1)
    probability_of_choosing_high = []
    for trial in trial_tones:
        choice_high = 1
        for tone in trial_tones[trial]:
            encoding_of_true_freq = encoding_given_true_freq(freq_points, tone, sigma=1)
            marginal = probability_estimated_freq(freq_estimation, encoding_of_true_freq)
            choice_high *= probability_high_response(sigmoid_weight, marginal)
        probability_of_choosing_high.append(choice_high)    
    
    