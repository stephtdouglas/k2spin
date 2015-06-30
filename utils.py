"""Compute lightcurve statistics."""

import logging

import numpy as np

def stats(flux, unc_flux):
    """Compute lightcurve statistics.

    Inputs
    ------
    time, flux, unc_flux: array_like

    Outputs
    -------
    lc_med, lc_std: floats

    """

    # Compute median and StDev
    lc_med = np.median(flux)
    lc_std = np.std(flux)

    return lc_med, lc_std


def phase(time, period):
    """Phase the input lightcurve by the input period."""
    # This should probably go in a different module, but I'm not sure where...

    phase = np.mod(time, period) / period

    return phase
