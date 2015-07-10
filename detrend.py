"""Remove trends from lightcurves."""

import logging

import numpy as np
from scipy import interpolate
import supersmoother
from astroML import time_series

from k2spin import utils
from k2spin import clean

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
        Other types YET TO BE IMPLEMENTED are "boxcar"

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
        logging.warning("boxcar is not yet implemented.")
        # sort the phases

        # Stitch 3X the phased lightcurve together to avoid edge effects
        # I *think* I don't have to do this, and I can just force it to wrap

        # loop through and construct the moving average

    else:
        logging.warning("unknown kind %s !!!",kind)

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
    

def search_and_detrend(time, flux, unc_flux, kind="supersmoother",
                       which="phased",phase_window=None, 
                       prot_lims=None, num_prot=1000):
    """Test for a period and then pre-whiten with it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    kind: string, optional
        type of smoothing to use. Defaults to "supersmoother."
        Other types YET TO BE IMPLEMENTED are "boxcar"

    which: string, optional
        whether to smooth the "phased" lightcurve (default) or the "full" 
        lightcurve. 

    phase_window: Float, optional (default=None)
        Half-width of the boxcar smoothing window, must be specified if 
        kind="boxcar"

    prot_lims: list-like, length=2
        minimum and maximum rotation periods to search

    num_prot: integer
        How many rotation periods to search

    Outputs
    -------
    fund_period, periods_to_test, periodogram 
    white_flux, white_unc, smoothed_flux

    """

    # Define range of period space to search
    log_shortp = np.log10(prot_lims[0])
    log_longp = np.log10(prot_lims[1])
    delta_p = (log_longp - log_shortp) / (1. * num_prot-1)
    log_periods = np.arange(log_shortp, log_longp + delta_p, delta_p)
    periods_to_test = 10**log_periods
    omegas_to_test = 2.0 * np.pi / periods_to_test

    # Search for a period
    periodogram = time_series.lomb_scargle(time, flux, unc_flux, 
                                           omegas_to_test, 
                                           generalized=True)

    fund_loc = np.argmax(abs(periodogram))
    fund_period = periods_to_test[fund_loc]

    # Whiten on that period
    white_out = pre_whiten(time, flux, unc_flux, fund_period,  
                           kind=kind, which=which, 
                           phase_window=phase_window)

    white_flux, white_unc, smoothed_flux = white_out

    # Return the period, periodogram, and whitened lc
    return [fund_period, periods_to_test, periodogram, white_flux, white_unc, 
            smoothed_flux]

def period_cleaner(time, flux, unc_flux, 
                   prot_lims):
    """Test for a period and then pre-whiten with it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    prot_lims: list-like, length=2
        minimum and maximum rotation periods to search

    num_prot: integer
        How many rotation periods to search

    Outputs
    -------
    clean_time, clean_flux, clean_unc_flux: arrays

    """

    # Sigma clip the input light curve
    clip_time, clip_flux, clip_unc = clean.sigma_clip(time, flux, 
                                               unc_flux, clip_at=6)
    logging.debug("Finished sigma-clipping")

    # Perform 1-2 whitenings 
    # (this always whitens twice???)
    for iteration in range(2):

        # search and whiten once
        search_out = search_and_detrend(clip_time, clip_flux, clip_unc,
                                        prot_lims=prot_lims,
                                        num_prot=1000)

        fund_period, periods_to_test, periodogram = search_out[:3]
        white_flux, white_unc, smoothed_flux = search_out[-3:]

        # Stats on the pre-whitened lc
        white_med, white_std = utils.stats(white_flux, white_unc)

        # Sigma-clip the testing lc
        clip_time, clip_flux, clip_unc = clean.sigma_clip(time, 
                                                          white_flux, 
                                                          white_unc, 
                                                          clip_at=6)


    # Pre-whiten the full lc based on the final period
    white_out = pre_whiten(time, flux, unc_flux, fund_period)
    white_flux, white_unc, smoothed_flux = white_out

    # Why return this periodogram though, if I've already changed
    # the lightcurve???
    return clip_time, white_flux, white_unc, periods_to_test, periodogram


def simple_detrend(time, flux, unc_flux, kind="supersmoother",
                   phaser=None):
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
    detrended_flux = flux / bulk_trend - 1

    detrended_unc = unc_flux

    # Return detrended
    return detrended_flux, detrended_unc, bulk_trend 
