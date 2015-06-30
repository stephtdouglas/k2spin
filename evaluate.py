"""Evaluate periodogram/ACF/etc. to pick best period."""

import logging

import numpy as np

def periodogram(periods, powers, threshold):
    """ID the most likely period and aliases.

    Inputs
    ------
    periods, powers: array-like


    Outputs
    -------
    best_period: float
    best_power: float
    is_clean: bool
    """

    # Find the most likely period and up to three aliases

    # Clip the best peak out of the periodogram

    # Now clip out aliases

    # Find the maximum of the clipped periodogram

    # Set clean = 1 if max_clipped < threshold*best_power

    # Return best_period, best_power, is_clean
