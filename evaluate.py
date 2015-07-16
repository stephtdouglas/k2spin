"""Evaluate periodogram/ACF/etc. to pick best period."""

import logging

import numpy as np

def test_pgram(periods, powers, threshold, n_aliases=3):
    """ID the most likely period and aliases.

    Inputs
    ------
    periods, powers: array-like

    threshold: float

    n_aliases: int


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

    logging.debug("Fundamental %d Prot=%f Power=%f", fund_loc, fund_period,
                  fund_power)

    # and aliases
    for_aliases = np.arange(1.0, n_aliases+1.0)
    inverse_fundamental = 1. / fund_period
    pos_aliases = 1. / (inverse_fundamental + for_aliases)
    neg_aliases = 1. / abs(inverse_fundamental - for_aliases)

    aliases = np.append(pos_aliases, neg_aliases)
    tot_aliases = len(aliases)

    # Clip the best peak out of the periodogram
    to_clip = np.where(abs(periods - fund_period)<0.05)[0]

    # Now clip out aliases
    for i, alias in enumerate(aliases):
        to_clip = np.intersect1d(to_clip, 
                                 np.where(abs(periods - alias)<0.02)[0])
 
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

    # Return best_period, best_power, aliases, is_clean
    return fund_period, fund_power, aliases, is_clean
