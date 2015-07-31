"""Basic cleanup on lightcurves (trimming, sigma-clipping)."""

import logging

import numpy as np
import matplotlib.pyplot as plt

import k2spin.utils as utils
from k2spin import detrend

def trim(time, flux, unc_flux):
    """Remove infs, NaNs, and negative flux values.

    Inputs
    ------
    time, flux, unc_flux: array_like

    Outputs
    -------
    trimmed_time, trimmed_flux, trimmed_unc: arrays
    good: boolean mask, locations that were kept

    """

    good = np.where((np.isfinite(flux)==True) & (flux>0) &
                    (np.isfinite(unc_flux)==True) & 
                    (np.isfinite(time)==True) & (time>2061.5))[0]

    trimmed_time = time[good]
    trimmed_flux = flux[good]
    trimmed_unc = unc_flux[good]

    return trimmed_time, trimmed_flux, trimmed_unc, good

def smooth_and_clip(time, flux, unc_flux, clip_at=3, to_plot=True):
    """Smooth the lightcurve, then clip based on residuals."""

    if to_plot:
        plt.figure(figsize=(8,4))
        ax = plt.subplot(111)
        ax.plot(time,flux,'k.',label="orig")

    # Simple sigma clipping first to get rid of really big outliers
    ct, cf, cu, to_keep = sigma_clip(time, flux, unc_flux, clip_at=clip_at)
    logging.debug("c len t %d f %d u %d tk %d", len(ct), len(cf),
                  len(cu), len(to_keep))
    if to_plot: ax.plot(ct, cf, '.',label="-1")

    # Smooth with supersmoother without much bass enhancement
    for i in range(3):
        logging.debug(i)
        det_out = detrend.simple_detrend(ct, cf, cu, phaser=0)
        detrended_flux, detrended_unc, bulk_trend = det_out

        # Take the difference, and find the standard deviation of the residuals
        logging.debug("flux, bulk trend, diff")
        logging.debug(cf[:5])
        logging.debug(bulk_trend[:5])
        f_diff = cf - bulk_trend
        logging.debug(f_diff[:5])
        diff_std = np.zeros(len(f_diff))
        diff_std[ct<=2102] = np.std(f_diff[ct<=2102])
        diff_std[ct>2102] = np.std(f_diff[ct>2102])
        logging.debug("std %f %f",diff_std[0], diff_std[-1])

        logging.debug("len tk %d diff %d", len(to_keep), len(f_diff))
        # Clip outliers based on residuals this time
        to_keep = to_keep[abs(f_diff)<=(diff_std*clip_at)]
        ct = time[to_keep]
        cf = flux[to_keep]
        cu = unc_flux[to_keep]
        if to_plot: ax.plot(ct, cf, '.',label=str(i))

    if to_plot: 
        ax.plot(ct, bulk_trend)
        ax.legend()

    clip_time = time[to_keep]
    clip_flux = flux[to_keep]
    clip_unc_flux = unc_flux[to_keep]

    return clip_time, clip_flux, clip_unc_flux, to_keep

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

    to_keep: boolean mask of locations that were kept
    """

    # Compute statistics on the lightcurve
    med, stdev  = utils.stats(flux, unc_flux)

    # Sigma-clip the lightcurve
    outliers = abs(flux-med)>(stdev*clip_at)
    to_clip = np.where(outliers==True)[0]
    to_keep = np.where(outliers==False)[0]
    logging.debug("Sigma-clipping")
    logging.debug(to_clip)
    clipped_time = np.delete(time, to_clip)
    clipped_flux = np.delete(flux, to_clip)
    clipped_unc = np.delete(unc_flux, to_clip)

    # Return clipped lightcurve
    return clipped_time, clipped_flux, clipped_unc, to_keep


def prep_lc(time, flux, unc_flux, clip_at=3):
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
    t_time, t_flux, t_unc, t_kept = trim(time, flux, unc_flux)

    # Run sigma-clipping if desired, repeat 2X
    if clip_at is not None:
        c_time, c_flux, c_unc, c_kept = smooth_and_clip(t_time, t_flux, t_unc,
                                                        clip_at=clip_at)
    else:
        c_time, c_flux, c_unc, c_kept = t_time, t_flux, t_unc, t_kept

    all_kept = t_kept[c_kept]

    # Calculate statistics on lightcurve
    c_med, c_stdev = utils.stats(c_flux, c_unc)

    # Return cleaned lightcurve and statistics
    return c_time, c_flux, c_unc, c_med, c_stdev, all_kept
