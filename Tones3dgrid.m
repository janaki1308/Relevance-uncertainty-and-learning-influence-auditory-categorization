function inputArrayMat = Tones3dgrid(latentTones, sigma) 
    
    inputArray0 = gaussian(log_freq_percept, latentTones(1), sigma);
    inputArray1 = gaussian(log_freq_percept, latentTones(2), sigma);
    inputArray2 = gaussian(log_freq_percept, latentTones(3), sigma);
    s0 = 1/sum(inputArray0); 
    s1 = 1/sum(inputArray1); 
    s2 = 1/sum(inputArray2);
    inputArray0 = inputArray0 * s0; 
    inputArray1 = inputArray1 * s1; 
    inputArray2 = inputArray2 * s2; 
    
    inputArrayMat = dot(inputArray0(:) * inputArray1(:).',inputArray2.');   
end
                                     
 

