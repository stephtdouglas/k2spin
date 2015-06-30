"""Basic cleanup on lightcurves (trimming, sigma-clipping)."""

import logging

import numpy as np

def trim(time, flux, unc_flux):
    """Remove infs, NaNs, and negative flux values.

    Inputs
    ------
    time, flux, unc_flux: array_like

    Outputs
    -------
    trimmed_time, trimmed_flux, trimmed_unc: arrays

    """

    good = np.where((np.isfinite(flux)==True) & (flux>0) &
                    (np.isfinite(unc_flux)==True) & 
                    (np.isfinite(time)==True))[0]

    trimmed_time = time[good]
    trimmed_flux = flux[good]
    trimmed_unc = unc_flux[good]

    return trimmed_time, trimmed_flux, trimmed_unc

def sigma_clip(time, flux, unc_flux, sigma_clip):
    """Perform sigma-clipping on the lightcurve.


    Inputs
    ------
    time, flux, unc_flux: array_like

    Outputs
    -------
    clipped_time, clipped_flux, clipped_unc: arrays


    """

    # Compute statistics on the lightcurve

    # Sigma-clip the lightcurve

    # Return clipped lightcurve

    pass
