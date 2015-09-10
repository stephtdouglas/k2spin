"""Measure rotation periods."""


import logging

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import supersmoother
from astroML import time_series
from gatspy.periodic import lomb_scargle_fast

from k2spin.config import *
from k2spin import utils
from k2spin import clean
from k2spin import evaluate
from k2spin import detrend
from k2spin import plot

def run_ls(time, flux, unc_flux, threshold, prot_lims=None, 
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

def search_and_detrend(time, flux, unc_flux, prot_lims=None,
                       to_plot=False, **detrend_kwargs):
    """Test for a period and then pre-whiten with it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    kind: string, optional
        type of smoothing to use. Defaults to "supersmoother."
        Other types "boxcar", "linear"

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

    # Search for periodogram
    ls_out = run_ls(time, flux, unc_flux, 0.5,  
                    prot_lims=prot_lims)
    fund_period, fund_power, periods_to_test, periodogram = ls_out[:4]
                                                       
    # Whiten on that period
    white_out = detrend.pre_whiten(time, flux, unc_flux, fund_period,  
                                   which="phased", **detrend_kwargs)

    white_flux, white_unc, smoothed_flux = white_out

    detrended_flux = flux / smoothed_flux
    detrended_unc = unc_flux

    if to_plot==True:
        fig, ax_list = plot.plot_one([time, flux, unc_flux],
                                     [periods_to_test, periodogram],
                                     fund_period,
                                     power_threshold=0, data_label="Input")
        ax_list[0].plot(time, smoothed_flux, 'b.')

        ax_list[3].plot(time, white_flux, 'r.')
        ax_list[3].set_ylabel("Whitened Flux")
        ax_list[3].set_xlabel("Time (D)")
        plt.tight_layout()

    # Return the detrended flux
    return detrended_flux, detrended_unc, fund_period, fund_power

def detrend_for_correction(time, flux, unc_flux, prot_lims,
                           to_plot=False, detrend_kwargs=None):
    """Test for a period and then pre-whiten with it.

    Inputs
    ------
    time, flux, unc_flux: array_like

    prot_lims: list-like, length=2
        minimum and maximum rotation periods to search

    to_plot: bool (default=False)
        Whether to plot each step of the detrending process

    Outputs
    -------
    det_flux, det_unc: arrays

    """

    # Set up the plot
    if to_plot==True:
        filename = detrend_kwargs.get("filename",
                                      base_path+"unknown_detrending.pdf")
        junk = detrend_kwargs.pop("filename")
        if filename.endswith(".pdf")==False:
            filename = filename+".pdf"
        pp = PdfPages(filename)

    det_flux = np.copy(flux)
    det_unc = np.copy(unc_flux)

    # Whiten 4 times
    for iteration in range(6):

        # search and whiten once
        det_out = search_and_detrend(time, det_flux, det_unc, 
                                     prot_lims=prot_lims, to_plot=to_plot, 
                                     **detrend_kwargs)
        det_flux, det_unc, fund_period, fund_power = det_out
        if iteration==0:
            base_power = fund_power

        # Do not sigma-clip - I think we need the same points in the
        # input and output lcs for correction

        if to_plot:
            pp.savefig()
            plt.close()

        # Stop iterating if we've gotten to a small multiple of the 
        # 6-hour period
        fund_div = (fund_period / 0.25)
        fd_round = np.round(fund_div, 0)
        if (fund_period <=2.05) and (abs(fund_div - fd_round)<0.05):
            logging.warning("Stopped detrending; prot={0:.3f} fd={1:.4f} {2:.4f}".format(fund_period, fund_div, fd_round))
            break
        elif fund_power <= 0.1*base_power:
            logging.warning("Stopped detrending; base power {0:.3f} "
                            "fund power {1:.3f}".format(base_power, fund_power))
            break
        else:
            logging.warning("prot {0:.3f} power {1:.3f} fund_div {2:.3f} {3}".format(fund_period, fund_power, fund_div, fd_round))

    if to_plot:
        pp.close()


    # Return the newly detrended lightcurve and the bulk trend
    return det_flux, det_unc

