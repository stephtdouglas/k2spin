"""Remove trends from lightcurves."""

import logging

import numpy as np

def pre_whiten(time, flux, unc_flux, period, phase_window):
    """Phase a lightcurve and then smooth it.

    Inputs
    ------
    time, flux, unc_flux: array_like
    period:
    phase_window:

    Outputs
    -------

    """

    # phase the LC by the period

    # sort the phases

    # "Triple_orderly_phase/lc"???

    # loop through and construct the moving average

    pass


def period_cleaner(time, flux, unc_flux, min_prot, max_prot, 
                   num_prot):
    """Test for a period and then pre-whiten with it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    min_prot, max_prot:

    num_prot:

    Outputs
    -------
    clean_time, clean_flux, clean_unc_flux: arrays

    """

    # Define range of period space to search

    # Define the phase window to median filter over in pre-whitening

    # Sigma clip the input light curve

    # Perform 1-2 whitenings 
    # (this looks like it always whitens twice???)

        # Perform a first search for a period

        # Pre-whiten on that period

        # Stats on the pre-whitened lc

        # Sigma-clip the testing lc

        # Mark iteration as complete


    # Pre-whiten the full lc based on the final period


def simple_detrend(time, flux, unc_flux, period, phase_window):
    """Detrend the lightcurve based on the input period

    Inputs
    ------
    time, flux, unc_flux: array_like

    min_prot, max_prot:

    num_prot:

    Outputs
    -------
    clean_time, clean_flux, clean_unc_flux: arrays

    """

    # Whiten the lightcurve
    w_time, w_flux, w_unc = pre-whiten(time, flux, unc_flux, period, 
                                       phase_window)

    # Calculate the overall trend
    bulk_trend = flux - w_flux

    # Detrend
    detrended_flux = flux / bulk_trend

    # Return detrended
    return detrended_flux
