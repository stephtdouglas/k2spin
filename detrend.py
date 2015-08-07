"""Remove trends from lightcurves."""

import logging

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import supersmoother
#from astroML import time_series
from gatspy.periodic import lomb_scargle_fast
from astropy import convolution

from k2spin import utils
from k2spin import evaluate

def pre_whiten(time, flux, unc_flux, period, kind="supersmoother",
               which="phased",phaser=None):
    """Phase a lightcurve and then smooth it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    period: float
        period to whiten by

    kind: string, optional
        type of smoothing to use. Defaults to "supersmoother."
        Other types are "boxcar", "linear" 

    which: string, optional
        whether to smooth the "phased" lightcurve (default) or the "full" 
        lightcurve. 

    phaser: Float, optional (default=None)
        if kind="boxcar", phaser is the Half-width of the smoothing window.
        if kind="supersmoother", phaser is alpha (the "bass enhancement").

    Outputs
    -------
    white_flux, white_unc, smoothed_flux: arrays

    """

    # phase the LC by the period
    if period is not None:
        phased_time = utils.phase(time, period)
    else:
        phased_time = time

    if kind.lower()=="supersmoother":

        if phaser is None:
            logging.warning("Phaser not set! "
                            "Set phaser=alpha (bass-enhancement value "
                            "for supersmoother) if desired.")

        if which.lower()=="phased":
            # Instantiate the supersmoother model object with the input period
            model = supersmoother.SuperSmoother(period=period)

            # Set up base arrays for phase for the fit
            x_vals = np.linspace(0,max(phased_time),1000)

            # Run a fit for the y phase
            y_fit = model.fit(phased_time, flux, unc_flux).predict(x_vals)

        elif which.lower()=="full":
            # Instantiate the supersmoother model object with the input period
            model = supersmoother.SuperSmoother(alpha=phaser)

            # Set up base arrays for time for the fit
            x_vals = np.linspace(min(time),max(time),1000)

            # run a fit for the y values
            y_fit = model.fit(time, flux, unc_flux).predict(x_vals)

        else:
            logging.warning("unknown which %s !!!",which)


    elif kind.lower()=="boxcar":

        if phaser is None:
            logging.warning("Phaser not set! "
                            "Set phaser to the width of the smoothing "
                            "box in pixels!")

        if which.lower()=="phased":
            # sort the phases
            sort_locs = np.argsort(phased_time)
            x_vals = phased_time[sort_locs]
            flux_to_fit = flux[sort_locs]

        elif which.lower()=="full":
            x_vals = time
            flux_to_fit = flux
        else:
            logging.warning("unknown which %s !!!",which)

        # Use astropy's convolution function!
        boxcar_kernel = convolution.Box1DKernel(width=phaser,
                                                mode="linear_interp")
        y_fit = convolution.convolve(flux_to_fit, boxcar_kernel,
                                     boundary="wrap")

    elif kind=="linear":

         if which!="full":
             logging.warning("Linear smoothing only allowed for full "
                             "lightcurve! Switching to full mode.")
             which = "full"

         # Fit a line to the data
         pars = np.polyfit(time, flux, deg=1)
         m, b = pars
         smoothed_flux = m * time + b

    else:
        logging.warning("unknown kind %s !!!",kind)

    if (kind=="supersmoother") or (kind=="boxcar"):
        # Interpolate back onto the original time array
        interp_func = interpolate.interp1d(x_vals, y_fit)

        if which.lower()=="phased":
            smoothed_flux = interp_func(phased_time)
        elif which.lower()=="full":
            smoothed_flux = interp_func(time)
        else:
            logging.warning("unknown which %s !!!",which)

    # Whiten the input flux by subtracting the smoothed version
    # The smoothed flux represents the "bulk trend"
    white_flux = flux - smoothed_flux
    white_unc = unc_flux

    return white_flux, white_unc, smoothed_flux

def simple_detrend(time, flux, unc_flux, kind="supersmoother",
                   phaser=None, to_plot=False):
    """Remove bulk trends from the LC

    Inputs
    ------
    time, flux, unc_flux: array_like

    kind: string, optional
        type of smoothing to use. Defaults to "supersmoother."
        Other types YET TO BE IMPLEMENTED are "boxcar"

    which: string, optional
        whether to smooth the "phased" lightcurve (default) or the "full" 
        lightcurve. 

    phaser: Float, optional (default=None)
        if kind="boxcar", phaser is the Half-width of the smoothing window.
        if kind="supersmoother", phaser is alpha (the "bass enhancement").

    Outputs
    -------
    detrended_flux, detrended_unc: arrays

    """

    # now (ab)use the Pre-whiten routine to calculate the overall trend
    # Maybe I should do something here to set alpha/the window size
    # for supersmoother
    w_flux, w_unc, bulk_trend = pre_whiten(time, flux, unc_flux, 
                                           period=None, kind=kind, 
                                           which="full",  
                                           phaser=phaser)

    # Actually detrend
    detrended_flux = flux / bulk_trend

    detrended_unc = unc_flux

    if to_plot:
        plt.figure(figsize=(10,8))
        ax1 = plt.subplot(311)
        ax1.plot(time, flux, 'k.')
        ax1.plot(time, bulk_trend, 'b-',lw=3)
        ax1.set_ylabel("Counts",fontsize="large")
        ax1.tick_params(labelleft=False, labelright=True)

        ax2 = plt.subplot(312)
        ax2.plot(time, w_flux, 'g.')
        ax2.set_ylabel("Flux - Bulk Trend",fontsize="large")
        ax2.tick_params(labelleft=False, labelright=True)

        ax3 = plt.subplot(313)
        ax3.plot(time, detrended_flux, 'r.')
        ax3.set_xlabel("Time (d)")
        ax3.set_ylabel("Flux / Bulk Trend",fontsize="large")
        ax3.tick_params(labelleft=False, labelright=True)

    # Return detrended
    return detrended_flux, detrended_unc, bulk_trend 
