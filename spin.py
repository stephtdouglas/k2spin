"""Run the actual lightcurve analysis."""

import logging

import numpy as np

def prep_lc(time, flux, unc_flux):
    """Trim, sigma-clip, and calculate stats on a lc."""

    # Trim the lightcurve, remove bad values

    # Run sigma-clipping if desired

    # Calculate statistics on lightcurve

    pass

def phase(time, flux, unc_flux, period):
    """Phase the input lightcurve by the input period."""
    # This should probably go in a different module, but I'm not sure where...

    pass

def init_test(time, flux, unc_flux, object_id):
    """Pre-whiten and detrend the lightcurve, and decide whether 
    to continue with more detailed processing.

    inputs
    ------


    outputs
    ------- 
    """

    # (ab)use the Pre-whitening routine to calculate the 
    # overall trend of the lightcurve (???)
    # Use the simple detrend function
    # Gives back the "detrended" lc and the "bulk_trend"

    # Run the periodogram (period_cleaner) on the original lc
    # (it runs a lomb-scargle function twice?)

    # Run on the detrended lightcurve

    # choose the LC that gives the most periodogram power, 
    # and define the threshold for a meaningful period. 
    # (the existing version of choose_LC looks like it takes a lot 
    #  of arguments that aren't being used anymore)

    # test if the period exceeds the power threshold defined above.

    # Plot the results from the raw and simplistically detrended lcs

    # Return the detrended_lc, bulk_trend, periodograms,
    # best periods, power threshold, 
    # and a flag for whether more processing is needed

    pass

def multi_search(time, flux, unc_flux, object_id, best_period,
                 best_threshold):
    """Search for multiple periods."""


        # start a while loop so we can look for multiple periods.
        # (only runs for one loop???)

            # pre-whiten according to the best-fit period

            # test the periodogram using the new residual lightcurve

            # If the new periodogram has greater power than the
            # threshold set by the first whitening, do this once:
        
                # Run period_cleaner on the (pre-whitened) lc
    
                # Test the new lightcurve, but it doesn't look like
                # it does anything with the test at this point.

    pass


def final_correction():
    """Run the final corrections on the lc."""

    # now loop through the light curve and find the 20 pixels 
    # closest in x and y to the pixel in question.
    # Loop through the lightcurve (0 to len-1)

       # Find the 20 closest pixels to the current point 
       # (note this number is completely arbitrary)

       # Divide the original flux at the current point
       # by the median of the closest pixels and then
       # by the final bulk trend at that point to get "corrected_LC"

       # Divide the original flux
       # by the bulk_trend*median(closest pixels) and then
       # by the final_bulk_trend to get "final_detrended"

       # Divide use_LC (either the pre-whitened_lc + 1 or
       # the simple detrended lc if no period was found)
       # by the median of the closest pixels to get "flattened_LC"

    pass

def run_one(time, flux, unc_flux, object_id):
    """Run one lightcurve through the analysis process."""

    # Initial processing

    # Run the initial search for periodicity
    # This should produce a plot comparing raw to the best period

    # If additional processing is needed, run the period search.

        # run the search for another period (multi_search)
        # This should also produce a plot
    
    # If the periodogram didn't pass the test above, keep raw lc

        # divide by the bulk trend and slice out the useful points


    # Run simple_detrend on the detrended lightcurve that results (???)
    # Returns the final_bulk_trend and final_detrended


    # And then more stats and plotting?

    # 

