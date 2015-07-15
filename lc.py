"""Class for lightcurves."""

import logging

import numpy as np
from astroML import time_series

from k2spin import utils
from k2spin import clean
from k2spin import detrend
from k2spin import plot

class LightCurve(object):
    """
    Class to contain lightcurves and run analysis. 

    """

    def __init__(self, time, flux, unc_flux, power_threshold=0.5):
        """Clean up the input data and sigma-clip it."""
        # Save the power threshold for later use
        self.power_threshold = power_threshold

        # Clean up the input lightcurve
        cleaned_out = clean.prep_lc(time, flux, unc_flux, clip_at=6.)
        self.time, self.flux, self.unc_flux, self.med, self.stdev = cleaned_out
        logging.debug("len init t %d f %d u %d", len(self.time), 
                      len(self.flux),len(self.unc_flux))

        # Detrend the raw flux
        self._bulk_detrend()

    def choose_initial(self):
        """Search raw and detrended LCs for periods, and decide whether there's
        a period there.

        """
        # Run a fit on the raw lc
        raw_fp, raw_power, raw_prots, raw_pgram = self._run_fit("raw")
        logging.debug("Ran raw fit")

        # Run a fit on the detrended lc
        det_fp, det_power, det_prots, det_pgram = self._run_fit("detrended")
        logging.debug("Ran detrended fit")

        # Compare them
        lc_to_use = self._pick_lc(raw_power, det_power)
        if lc_to_use<=1:
            logging.info("Using raw lightcurve")
            self.init_prot , self.init_power = raw_fp, raw_power
            self.init_periods_to_test, self.init_pgram = raw_prots, raw_pgram
        elif lc_to_use==2:
            logging.info("Using detrended lightcurve")
            self.init_prot , self.init_power = det_fp, det_power
            self.init_periods_to_test, self.init_pgram = det_prots, det_pgram

        logging.info("Initial Prot %f Power %f", self.init_prot, 
                     self.init_power)

        # Plot them up
        lcs = [[self.time, self.flux/self.med - 1, abs(self.unc_flux/self.med)],
               [self.time, self.det_flux, self.det_unc]]
        pgrams = [[raw_prots, raw_pgram], [det_prots, det_pgram]]
        best_periods = [raw_fp, det_fp]
        data_labels = ["Raw", "Detrended"]
        raw_det_fig = plot.compare_multiple(lcs, pgrams, best_periods, 
                                            self.power_threshold, data_labels,
                                            phase_by=self.init_prot)

        logging.debug("DONE!")


    def _bulk_detrend(self, alpha=8):
        """Smooth the rapid variations in the lightcurve and remove bulk trends.

        inputs
        ------
        alpha: float
            "bass enhancement" for supersmoother. 
            Defaults to 8 for now because Kevin was using that. 
        """

        logging.debug("Removing bulk trend...")
        det_out = detrend.simple_detrend(self.time, self.flux, self.unc_flux,
                                         kind="supersmoother", phaser=alpha)
        self.det_flux, self.det_unc, self.bulk_trend = det_out

        logging.debug("len detrended t %d f %d u %d", len(self.time), 
                      len(self.det_flux),len(self.det_unc))

    def _run_fit(self, use_lc, prot_lims=[0.3,70]):
        """Run a fit on a single lc, either "raw" or "detrended" 
        or a array/list of [time, flux, and unc]
        """
        tt = self.time

        if use_lc=="raw":
            logging.debug("fitting raw lc")
            tt, ff, uu = self.time, self.flux, self.unc_flux
        elif use_lc=="detrended":
            logging.debug("fitting detrended lc")
            tt, ff, uu = self.time, self.det_flux, self.det_unc
        else:
            logging.debug("fitting other lc")
            tt, ff, uu = use_lc

        # Iteratively smooth, clip, and run a periodogram (period_cleaner)
        cl_time, cl_flux, cl_unc, sm_flux = detrend.period_cleaner(tt, ff, uu,
                                                           prot_lims=prot_lims)

        logging.debug("Smoothed, now periodogram")
        logging.debug("Cleaned t %d f %d u %d", len(cl_time),
                      len(cl_flux), len(cl_unc))
        # Test the periodogram and pick the best period and power
        ls_out = detrend.run_ls(cl_time, cl_flux, cl_unc, prot_lims=prot_lims)
        fund_prot, fund_power, periods_to_test, periodogram = ls_out

        return fund_prot, fund_power, periods_to_test, periodogram

    def _pick_lc(self, fund_power1, fund_power2):
        """Pick the raw or detrended lc to continue with by 
        selecting the one with the highest peak in the periodogram
        (no consideration of the *locations* of those peaks)
        """
        # return a integer indicating which lightcurve to use (1 or 2)
        to_use = 0

        if fund_power1 > fund_power2:
            to_use = 1
        elif fund_power2 > fund_power1:
            to_use = 2
        else: # Either something's gone wrong, or they're exactly equal
            to_use = 0

        return to_use

    def _multi_search(self,lc_type):
        """Search a lightcurve for a secondary signal."""
        pass

    
