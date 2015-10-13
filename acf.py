"""Autocorrelation function for period measurement."""

import logging

import numpy as np
from scipy.signal import argrelextrema 


def acf(times,yvals):
    """ 
    computes the autocorrelation function for an evenly-sampled time-series 
    """
    cadence = np.median(np.diff(times))
    N = len(yvals)
    max_lag = N/2
    
    median_yval = np.median(yvals)
    norm_term = np.sum((yvals - median_yval)**2)
    lags = np.arange(max_lag)
    
    #print median_yval,norm_term,max_lag
      
    ACF0 = [np.sum((yvals[:N-j] - median_yval)*(yvals[j:] - median_yval)) for j in lags]
    ACF1 = ACF0/norm_term
    
    # smooth the ACF
    gauss_kernel = Gaussian1DKernel(18,x_size=55)
    ACF = ap_convolve(ACF1, gauss_kernel,boundary="extend")
    #ACF = ACF1
    
    periods = cadence*lags

    return periods,ACF



def find_prot(periods,ACF):
    """ 
    Determines the Prot from an ACF, using procedure in McQuillan et al. (2013)
    """

    # Find all local maxima in the ACF. If none, return -1

    max_loc = argrelextrema(ACF,np.greater,order=5)
    #print "max_loc",max_loc
    #print "edge",len(periods)
    if len(max_loc)==0:
        return -1
    max_per = periods[max_loc[0]]
    #print "max_per",max_per
    max_ACF = ACF[max_loc[0]]
    #print "max_acf",max_ACF

    # Find all local minima in the ACF.

    min_loc = argrelextrema(ACF,np.less,order=5)
    #print "min_loc",min_loc
    min_per = periods[min_loc[0]]
    #print "min_per",min_per
    min_ACF = ACF[min_loc[0]]
    #print "min_acf",min_ACF

    ### Find peak heights 
    ## Ignore first peak if it's close to 0
    if min_per[0]<1:
        peak_heights = np.zeros(len(max_per)-1)
        per_with_heights = max_per[1:]
        max_ACF_with_heights = max_ACF[1:]
    else:
        peak_heights = np.zeros(len(max_per))
        per_with_heights = max_per
        max_ACF_with_heights = max_ACF

    ## Ignore last peak if there are no minima to the right of it
    while len(np.where(min_per>per_with_heights[-1])[0])==0:
        peak_heights = peak_heights[:-1]
        per_with_heights = per_with_heights[:-1]
        max_ACF_with_heights = max_ACF_with_heights[:-1]
        if len(peak_heights)==0:
            print "No local minima to the right of any local maxima"
            return -1

    for i,max_p in enumerate(per_with_heights):
        # find the local minimum directly to the left of this maximum
        min_left = np.where(min_per<max_p)[0]
        min_loc_1 = min_left[-1]

        # find the local minimum directly to the right of this maximum
        min_right = np.where(min_per>max_p)[0]
        min_loc_2 = min_right[0]
        #print min_per[min_loc_1],max_p,min_per[min_loc_2]
        height1 = max_ACF_with_heights[i] - min_ACF[min_loc_1]
        height2 = max_ACF_with_heights[i] - min_ACF[min_loc_2]
        peak_heights[i] = (height1 + height2)/2.0
    #print peak_heights

    if (len(peak_heights)>1) and (peak_heights[1]>peak_heights[0]):
        # if the second peak is highest, the first peak is probably
        # a half-period alias, so take the second peak.
        best_period = per_with_heights[1]
    else:
        # if the first peak is highest, it's most likely the period
        best_period = per_with_heights[0]

    return best_period


def run_acf(times,yvals,input_period=None,plot=False):
    """ runs the acf function above """
    
    periods, ACF = acf(times,yvals)

    peak_loc = find_prot(periods,ACF)

    return peak_loc
