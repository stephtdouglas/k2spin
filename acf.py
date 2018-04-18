import logging

from astropy.convolution import convolve as ap_convolve
from astropy.convolution import Box1DKernel, Gaussian1DKernel
from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
import matplotlib.pyplot as plt
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

    max_loc = argrelextrema(ACF,np.greater,order=10)
    #print "max_loc",max_loc
    #print "edge",len(periods)
    if len(max_loc)==0:
        return np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
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
    min_allowed_p = periods[0]*2
    if len(min_per)==0:
        logging.warning("No ACF minima found")
        return np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
    elif len(max_per)==0:
        logging.warning("No ACF maxima found")
        return np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
    elif min_per[0]<min_allowed_p:
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
            logging.warning("No local minima to the right of any local maxima")
            return np.array([np.nan,np.nan,np.nan,np.nan,np.nan])

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
        best_height = peak_heights[1]
        which = 1
    else:
        # if the first peak is highest, it's most likely the period
        best_period = per_with_heights[0]
        best_height = peak_heights[0]
        which = 0

    return best_period, best_height, which, per_with_heights, peak_heights


def run_acf(times,yvals,input_period=None,plot=False):
    """ runs the acf function above, and plots the result """
    
    plot_ymin,plot_ymax = np.percentile(yvals,[1,99])

    periods, ACF = acf(times,yvals)
#    # find the maximum of the first peak
#    peak_locs = argrelextrema(ACF,np.greater,order=5)
#    #print periods[peak_locs[0]]
    find_out = find_prot(periods,ACF)
    peak_loc = find_out[0]
    print_period = "Prot = {:.2f}".format(peak_loc)

    if plot:
        plt.figure(figsize=(10,8))
    
        ax1 = plt.subplot(221)
        ax1.plot(times,yvals,'k-')
        ax1.set_ylabel("normalized flux",fontsize="large")
        ax1.set_xlabel("Time (d)",fontsize="large")
    
        ax2 = plt.subplot(222)
        ax2.plot(periods,ACF)
        ax2.set_xlabel(r"$\tau_K$",fontsize="x-large")
        ax2.set_ylabel("ACF",fontsize="large")
        plot2_ymin,plot2_ymax = ax2.get_ylim()
        ax2.set_ylim(plot2_ymin,plot2_ymax)
        if input_period:
            ax2.plot((input_period,input_period),(plot2_ymin,plot2_ymax),"g:",lw=2,label="Input={:.2f}".format(input_period))
    
        ax2.plot((peak_loc,peak_loc),(plot2_ymin,plot2_ymax),'r--',label=print_period)
        #ax2.plot(periods[peak_locs[0]],ACF[peak_locs[0]],'ro')
        ax2.legend()
        ax2.tick_params(labelleft=False,labelright=True)
    
        # Phase-fold the light-curve and plot the result
        phase = times/peak_loc - np.asarray((times/peak_loc),np.int)

        ax3 = plt.subplot(223)
        ax3.plot(phase,yvals,'k.')
        ax3.set_xlabel("Phase")
        ax3.set_ylabel("Flux")
        ax3.set_ylim(plot_ymin*0.98,plot_ymax)
        #print plot_ymin,plot_ymax
    
    return find_out
