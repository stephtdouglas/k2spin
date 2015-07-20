"""Evaluate periodogram/ACF/etc. to pick best period."""

import logging

import numpy as np
import scipy.optimize as opt

from k2spin import utils

def test_pgram(periods, powers, threshold, n_aliases=3, 
               alias_with=0.25):
    """ID the most likely period and aliases.

    Inputs
    ------
    periods, powers: array-like

    threshold: float

    n_aliases: int

    alias_with: float
        aliases to search for. Defaults to 0.25 days (6 hrs), the time 
        between K2 thruster fires

    Outputs
    -------
    best_period: float

    best_power: float

    aliases: array

    is_clean: bool
    """
    logging.debug("Eval %d aliases with threshold %f",n_aliases, threshold)

    # Find the most likely period
    fund_loc = np.argmax(abs(powers))
    fund_period = periods[fund_loc]
    fund_power = powers[fund_loc]

    logging.info("Fundamental %d Prot=%f Power=%f", fund_loc, fund_period,
                  fund_power)

    # and aliases
    for_aliases = np.arange(1.0, n_aliases+1.0) / alias_with
    inverse_fundamental = 1. / fund_period
    pos_aliases = 1. / (inverse_fundamental + for_aliases)
    neg_aliases = 1. / abs(inverse_fundamental - for_aliases)

    aliases = np.append(pos_aliases, neg_aliases)
    tot_aliases = len(aliases)

    logging.debug("Aliases: {}".format(aliases))

    # percentages to use for finding values to clip
    fund_5percent = fund_period * 0.05
    fund_2percent = fund_period * 0.02

    # Clip the best peak out of the periodogram
    to_clip = np.where(abs(periods - fund_period)<=fund_5percent)[0]
    logging.debug(periods[to_clip])

    # Now clip out aliases
    for i, alias in enumerate(aliases):
        to_clip = np.union1d(to_clip, 
                             np.where(abs(periods - alias)<=fund_2percent)[0])

    clipped_periods = np.delete(periods, to_clip)
    clipped_powers = np.delete(powers, to_clip)


    # Find the maximum of the clipped periodogram
    clipped_max = np.argmax(clipped_powers)
    max_clip_power = clipped_powers[clipped_max]
    max_clip_period = clipped_periods[clipped_max]

    # Set clean = True if max of clipped periodogram < threshold*best_power
    if max_clip_power < (threshold * fund_power):
        is_clean = True
    else:
        is_clean = False
        logging.info("Max clipped power = %f", max_clip_power)

    # Return best_period, best_power, aliases, is_clean
    return fund_period, fund_power, aliases, is_clean


def fit_sine(time, flux, unc, period):
    """Fit a simple sine model fixed to the best-fit period."""

    def _sine_model(t, amp, yoffset, tshift):
        return amp * np.sin(2 * np.pi * t / period + tshift) + yoffset

    # Phase by the period, then extend the arrays 
    # To fit two cycles instead of one
    phased_time = utils.phase(time, period)
    fit_time = np.append(phased_time, phased_time + period)
    fit_flux = np.append(flux, flux)
    fit_unc = np.append(unc, unc)

    # initial amplitude and yoffset are stdev and median, respectively
    p0 = np.append(utils.stats(flux, unc), 0.0)

    popt, pcov = opt.curve_fit(_sine_model, fit_time, fit_flux, 
                                sigma=fit_unc, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    logging.debug("amplitude, yoffset, tshift")
    logging.debug(popt)
    logging.debug(perr)

    return phased_time, _sine_model(phased_time, *popt)
