"""Evaluate periodogram/ACF/etc. to pick best period."""

import logging

import numpy as np
import scipy.optimize as opt

from k2spin import utils

def clip_6hr(periods, powers):
    """Clip harmonics of the 6-hour period."""
    six_hr_alias = np.append(0.125, np.arange(0.25,2.1,0.25))
    two_percent = six_hr_alias * 0.02

    to_clip = np.array([])

    for i, alias in enumerate(six_hr_alias):
        to_clip = np.union1d(to_clip, 
                             np.where(abs(periods - alias)<=two_percent[i])[0])


    clipped_periods = np.delete(periods, to_clip)
    clipped_powers = np.delete(powers, to_clip)

    return clipped_periods, clipped_powers

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
    #logging.debug(periods[to_clip])

    # Now clip out aliases
    for i, alias in enumerate(aliases):
        to_clip = np.union1d(to_clip, 
                             np.where(abs(periods - alias)<=fund_2percent)[0])

    # Also clip aliases of the 6-hour period
    six_hr_alias = np.append(0.125, np.arange(0.25,2.1,0.25))
    for i, alias in enumerate(six_hr_alias):
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

def better_peak_finder(periods, powers, threshold, n_aliases=3, 
               alias_with=0.25):
    """
    Find likely periodogram peaks in a more sophisticated way.

    """
    six_hr_alias = np.append(0.125, np.arange(0.25,2.1,0.25))

    num_periods = len(periods)
    
    fund_loc = np.argmax(powers)
    fund_period = periods[fund_loc]
    fund_power = powers[fund_loc]
    logging.info("Fundamental %d Prot=%f Power=%f", fund_loc, 
                 fund_period, fund_power)
    print fund_loc, fund_period, fund_power

    max_period = periods[0]
    min_period = periods[-1]

    good_period = False
    clip_thrusters = False

    for iteration in np.arange(2):
        print iteration

        # If the period with the most power is the max available, 
        # reduce the maximum period a bit
        if fund_loc==0:
            good_period = False
            logging.info("Bad period: %.2f is max possible", fund_period)
            max_period = 0.8 * max_period
            print "new max period ", max_period

        # If the best period is near the top, but not quite there, drop the 
        # maximum by half (this is ok for me, because at 600 Myr 
        # periods > 35ish days are unlikely for bonafide cluster members)
        elif fund_period>(periods[0]*0.9):
            good_period = False
            logging.info("Bad period: %.2f in top 10\%", fund_period)
            max_period = 0.5 * max_period
            print "new max period ", max_period

        # If the fundamental period is very close to a six-hour alias,
        # Then clip the thruster fire aliases out of the periodogram 
        # and look again
        elif ((clip_thrusters==False) and 
              (abs(min(six_hr_alias - fund_period))<0.005)):
            good_period = False
            logging.info("Bad period: %.2f - thruster fire harmonic", 
                         fund_period)
            clip_thrusters = True

        # Else: hurray! (hopefully)
        else:
            good_period = True
            break
        
        if clip_thrusters==True:
            periods, powers = clip_6hr(periods, powers)


        print min_period, max_period, clip_thrusters
        new_loc = np.where((periods>=min_period) & (periods<=max_period))[0]
        powers = powers[new_loc]
        periods = periods[new_loc]

        fund_loc = np.argmax(abs(powers))
        fund_period = periods[fund_loc]
        fund_power = powers[fund_loc]

        logging.info("Fundamental %d Prot=%f Power=%f", fund_loc, 
                     fund_period, fund_power)
        print fund_loc, fund_period, fund_power

    return powers, periods
