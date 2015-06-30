"""Basic cleanup on lightcurves (trimming, sigma-clipping)."""

import logging

import numpy as np

import k2spin.stats as stats

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

def sigma_clip(time, flux, unc_flux, clip_at=6):
    """Perform sigma-clipping on the lightcurve.

    Inputs
    ------
    time, flux, unc_flux: array_like

    clip_at: float (optional)
        how many sigma to clip at. Defaults to 6. 

    Outputs
    -------
    clipped_time, clipped_flux, clipped_unc: arrays
    """

    # Compute statistics on the lightcurve
    med, stdev  = stat.stats(flux, unc_flux)

    # Sigma-clip the lightcurve
    to_clip = np.where(abs(flux-med)>(stdev*clip_at))[0]
    clipped_time = np.delete(time, to_clip)
    clipped_flux = np.delete(flux, to_clip)
    clipped_unc = np.delete(unc_flux, to_clip)

    # Return clipped lightcurve
    return clipped_time, clipped_flux, clipped_unc


def prep_lc(time, flux, unc_flux, clip_at=6):
    """Trim, sigma-clip, and calculate stats on a lc.

    Inputs
    ------
    time, flux, unc_flux: array_like

    clip_at: float (optional)
        How many sigma to clip at. Defaults to 6. 
        Set to None for no sigma clipping

    Outputs
    -------
    clean_time, clean_flux, clean_unc: arrays
    """

    # Trim the lightcurve, remove bad values
    t_time, t_flux, t_unc = trim(time, flux, unc_flux)

    # Run sigma-clipping if desired
    if clip_at is not None:
        c_time, c_flux, c_unc = sigma_clip(t_time, t_flux, t_unc,
                                           clip_at=clip_at)
    else:
        c_time, c_flux, c_unc = t_time, t_flux, t_unc

    # Calculate statistics on lightcurve
    c_med, c_stdev = stat.stats(c_flux, c_unc)

    # Return cleaned lightcurve and statistics
    return c_time, c_flux, c_unc, c_med, c_stdev
