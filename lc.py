"""Class for lightcurves."""

import logging

import numpy as np

from k2spin import utils
from k2spin import clean
from k2spin import detrend

def LightCurve(object):
    """
    Class to contain lightcurves and run analysis. 

    """

    def __init__(time, flux, unc_flux):
        """Clean up the input data and sigma-clip it."""
        # Clean up the input lightcurve
        cleaned_out = clean.prep_lc(time, flux, unc_flux, clip_at=6.)
        self.time, self.flux, self.unc_flux, self.med, self.stdev = cleaned_out

        # Detrend the raw flux
        self._bulk_detrend()

    def fit_and_compare(self):
        """Search raw and detrended LCs for periods, and decide whether there's
        a period there.

        """
        # Run a fit on the raw lc

        # Run a fit on the detrended lc

        # Compare them

        # select the one with the highest peak in the periodogram

        pass

    def _bulk_detrend(self, alpha=8):
        """Smooth the rapid variations in the lightcurve and remove bulk trends.

        inputs
        ------
        alpha: float
            "bass enhancement" for supersmoother. 
            Defaults to 8 for now because Kevin was using that. 
        """

        det_out = detrend.simple_detrend(self.time, self.flux, self.unc_flux,
                                         kind="supersmoother", phaser=alpha)
        self.det_flux, self.det_unc, self.bulk_trend = det_out

    def _run_fit(self,lc_type):
        """Run a fit on a single lc, either "raw" or "detrended".
        """
        tt = self.time

        if lc_type=="raw":
            ff, uu = self.time, self.flux, self.unc
        elif lc_type=="detrended":
            ff, uu = self.det_flux, self.det_unc

        # Iteratively smooth, clip, and run a periodogram (period_cleaner)

        # Test the periodogram and pick the best period and power

    def _pick_lc(self):
        """Pick the raw or detrended lc to continue with."""
        pass

    def _multi_search(self,lc_type):
        """Search a lightcurve for a secondary signal."""
        pass

    
