"""Plot lightcurves, periodograms, and phased lightcurves."""

import logging
import itertools

import matplotlib.pyplot as plt


def setup_plots():
    """Set up a three-paneled figure."""

    pass


def plot_one(lightcurve, periodogram, power_threshold, best_period, 
             axes_list=None,**plotkwargs):
    """Plot input lightcurve, input periodogram, and input phased lightcurve
    """

    # If axes_list is None, create the figure and axes with setup_plots

    # Top panel: lightcurve

    # Middle panel: periodogram

    # Bottom panel: phased lightcurve


    pass

def compare_multiple(lightcurves, periodograms, best_periods, threshold,
                     kwarg_sets):
    """Plot multiple sets of lightcurves onto a three-paneled plot.

    There must be at least 2 lightcurves, periodograms, and best_periods,
    or this will fail. The less-processed lc should be first in each. 
    """

    # Pass each set of inputs to plot_one

    pass



# Other notes pulled from Kevin's .pro file, keeping for now for reference
    # plot the raw light curve, with the bulk trend overlaid 

    # plot the periodograms for both light curves as well.

    # plot the periodograms for both light curves as well.


            # Plot the original phased lightcurve

            # Overplot original - (whitened using best-fit period)
            # I think this is plotting the residual between the 
            # original and whitened lightcurve?
    
            # Plot the phased whitened lightcurve



                # Plot the (pre-whitened) light curve

                # Plot periodograms for both lightcurves


    # plot the raw light curve, with the bulk trend overlaid 
