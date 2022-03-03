function gaussFn = gaussian(x, mean, sigma)
    gaussFn = exp(-(x-mean)^2/(2*sigma^2));
end

