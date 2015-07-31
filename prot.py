"""Measure rotation periods."""


import logging

import numpy as np
from scipy import interpolate
import supersmoother
from astroML import time_series
from gatspy.periodic import lomb_scargle_fast

from k2spin import utils
from k2spin import clean
from k2spin import evaluate
from k2spin import detrend

def run_ls(time, flux, unc_flux, threshold, prot_lims=None, num_prot=1000,
           run_bootstrap=False):
    """Run a periodogram and return it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    prot_lims: list-like, length=2
        minimum and maximum rotation periods to search

    num_prot: integer
        How many rotation periods to search

    Outputs
    -------
    fund_period, fund_power, periods_to_test, periodogram, aliases

    sigmas
        only if run_bootstrap=True
    """

    logging.debug("run ls t %d f %d u %d", len(time), len(flux),
                  len(unc_flux))
    # Define range of period space to search
    # Using real frequencies not angular frequencies
    freq_term = 1.0 # 2.0 * np.pi

    set_f0 = freq_term / prot_lims[1]
    set_fmax = freq_term / prot_lims[0]
    n_freqs = 3e4
    set_df = (set_fmax - set_f0) / n_freqs
    freqs_to_test = set_f0 + set_df * np.arange(n_freqs)

    # Search for a period
    model = lomb_scargle_fast.LombScargleFast().fit(time, flux, unc_flux)
    periodogram = model.score_frequency_grid(f0=set_f0, df=set_df, N=n_freqs)
    logging.debug("pgram count %d", len(periodogram))
    periods_to_test = freq_term / freqs_to_test

    ls_out = evaluate.test_pgram(periods_to_test, periodogram, threshold)
    fund_period, fund_power, aliases, is_clean  = ls_out

    # Now bootstrap to find the typical height of the highest peak
    # (Use the same time points, but redraw the corresponding flux points
    # at random, allowing replacement)
    if run_bootstrap:
        N_bootstraps = 10
        n_points = len(flux)
        ind = np.random.randint(0, n_points, (N_bootstraps, n_points))
        bs_periods, bs_powers = np.zeros(N_bootstraps), np.zeros(N_bootstraps)
        for i, f_index in enumerate(ind):
            bs_model = lomb_scargle_fast.LombScargleFast().fit(time, 
                                               flux[f_index], unc_flux[f_index])
            bs_pgram = bs_model.score_frequency_grid(f0=set_f0,  
                                                     df=set_df, N=n_freqs)
            max_loc = np.argmax(bs_pgram)
            bs_periods[i] = periods_to_test[max_loc]
            bs_powers[i] = bs_pgram[max_loc]

        logging.debug("Periods and Powers")
        logging.debug(bs_periods)
        logging.debug(bs_powers)

        sigmas = np.percentile(bs_powers, [99, 95])
        logging.debug("Fund power: %f 99p %f 95p %f", 
                      fund_power, sigmas[0], sigmas[1])
    else:
        sigmas=None

    return (fund_period, fund_power, periods_to_test, periodogram, 
            aliases, sigmas)

def search_and_detrend(time, flux, unc_flux, kind="supersmoother",
                       which="phased",phaser=None, pgram_threshold=None,
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

    phaser: Float, optional (default=None)
        if kind="boxcar", phaser is the Half-width of the smoothing window.
        if kind="supersmoother", phaser is alpha (the "bass enhancement").

    pgram_threshold: float

    prot_lims: list-like, length=2
        minimum and maximum rotation periods to search for lomb-scargle

    num_prot: integer
        How many rotation periods to search

    Outputs
    -------
    fund_period, fund_power, periods_to_test, periodogram 
    white_flux, white_unc, smoothed_flux

    """

    logging.debug("S&T threshold %f",pgram_threshold)

    # Search for periodogram
    ls_out = run_ls(time, flux, unc_flux, pgram_threshold,  
                    prot_lims=prot_lims, num_prot=num_prot)
    fund_period, fund_power, periods_to_test, periodogram = ls_out[:4]
                                                       
                                                       

    # Whiten on that period
    white_out = detrend.pre_whiten(time, flux, unc_flux, fund_period,  
                                   kind=kind, which=which, 
                                   phaser=phaser)

    white_flux, white_unc, smoothed_flux = white_out

    # Return the period, periodogram, and whitened lc
    return [fund_period, fund_power, periods_to_test, periodogram, white_flux,
            white_unc, smoothed_flux]

def period_cleaner(time, flux, unc_flux, prot_lims, pgram_threshold):
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
    clip_time, white_flux, white_unc, smoothed_flux: arrays

    """

    # Sigma clip the input light curve
    clip_time, clip_flux, clip_unc, kept = clean.sigma_clip(time, flux, 
                                                            unc_flux, 
                                                            clip_at=6)
    logging.debug("Finished sigma-clipping")

    # Perform 1-2 whitenings 
    # (this always whitens twice???)
    for iteration in range(2):

        # search and whiten once
        search_out = search_and_detrend(clip_time, clip_flux, clip_unc,
                                        pgram_threshold=pgram_threshold,
                                        prot_lims=prot_lims,
                                        num_prot=1000)

        fund_period, fund_power, periods_to_test, periodogram = search_out[:4]
        white_flux, white_unc, smoothed_flux = search_out[-3:]

        # Stats on the pre-whitened lc
        white_med, white_std = utils.stats(white_flux, white_unc)

        # Sigma-clip the testing lc
        clip_time, clip_flux, clip_unc, kept = clean.sigma_clip(clip_time, 
                                                                white_flux, 
                                                                white_unc, 
                                                                clip_at=6)


    # Pre-whiten the full lc based on the final period
    white_out = detrend.pre_whiten(time, flux, unc_flux, fund_period)
    white_flux, white_unc, smoothed_flux = white_out

    # Return the newly whitened lightcurve and the bulk trend
    return time, white_flux, white_unc, smoothed_flux

