"""Class for lightcurves."""

import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt
from astroML import time_series

from k2spin import utils
from k2spin import clean
from k2spin import detrend
from k2spin import plot
from k2spin import prot
from k2spin import evaluate

class LightCurve(object):
    """
    Class to contain lightcurves and run analysis. 

    """

    def __init__(self, time, flux, unc_flux, x_pos, y_pos, name, 
                 power_threshold=0.5):
        """Clean up the input data and sigma-clip it."""
        # Save the power threshold for later use
        self.power_threshold = power_threshold
        self.name = name

        logging.debug(self.name)
        logging.debug("Threshold %f",self.power_threshold)

        # Clean up the input lightcurve
        cleaned_out = clean.prep_lc(time, flux, unc_flux, clip_at=3.)
        self.time, self.flux, self.unc_flux = cleaned_out[:3]
        self.med, self.stdev, all_kept = cleaned_out[3:]
        self.x_pos, self.y_pos = x_pos[all_kept], y_pos[all_kept]
        logging.debug("len init t %d f %d u %d", len(self.time), 
                      len(self.flux),len(self.unc_flux))

        # Detrend the raw flux
        self._bulk_detrend()

    def choose_initial(self):
        """Search raw and detrended LCs for periods, and decide whether there's
        a period there.

        """
        # Run a fit on the raw lc
        r_out = self._run_fit("raw")
        raw_fp, raw_power, raw_prots, raw_pgram, raw_alias, raw_sigma = r_out
        logging.debug("Ran raw fit")

        # Run a fit on the detrended lc
        d_out = self._run_fit("detrended")
        det_fp, det_power, det_prots, det_pgram, det_alias, det_sigma = d_out
        logging.debug("Ran detrended fit")

        # Compare them
        lc_to_use = self._pick_lc(raw_power, det_power)
        if lc_to_use<=1:
            logging.info("Using raw lightcurve")
            self.init_prot , self.init_power = raw_fp, raw_power
            self.init_periods_to_test, self.init_pgram = raw_prots, raw_pgram
            self.use_flux = self.flux / self.med
            self.use_unc = self.unc_flux / self.med
            self.init_sigmas = raw_sigma
            data_labels = ["Raw (Selected)", "Detrended"]
        elif lc_to_use==2:
            logging.info("Using detrended lightcurve")
            self.init_prot , self.init_power = det_fp, det_power
            self.init_periods_to_test, self.init_pgram = det_prots, det_pgram
            self.use_flux = self.det_flux 
            self.use_unc = self.unc_flux 
            self.init_sigmas = det_sigma
            data_labels = ["Raw", "Detrended (Selected)"]

        logging.info("Initial Prot %f Power %f", self.init_prot, 
                     self.init_power)

        # Get aliases for selected period
        eval_out = evaluate.test_pgram(self.init_periods_to_test, 
                                       self.init_pgram, self.power_threshold)
        if eval_out[-1]==False:
            logging.warning("Selected lightcurve is not clean")
        else:
            logging.debug("Selected lightcurve is clean")
        plot_aliases = [None, eval_out[2]]

        # Plot them up
        lcs = [[self.time, self.flux/self.med, abs(self.unc_flux/self.med)],
               [self.time, self.det_flux, self.det_unc]]
        pgrams = [[raw_prots, raw_pgram], [det_prots, det_pgram]]
        best_periods = [raw_fp, det_fp]
        sigmas = [raw_sigma, det_sigma]
        logging.debug(sigmas)
        rd_fig, rd_axes = plot.compare_multiple(lcs, pgrams, best_periods, 
                                                sigmas, 
                                                aliases=plot_aliases,
                                                data_labels=data_labels,  
                                                phase_by=self.init_prot)

        rd_fig.suptitle(self.name, fontsize="large", y=0.99)

        rd_fig.delaxes(rd_axes[3])

        plt.savefig("{}_raw_detrend.png".format(self.name))

    def correct_and_fit(self):
        """Position-correct and perform a fit."""
        logging.debug("Fitting corrected lightcurve")

        self._xy_correct()

        fit_out = self._run_fit([self.time, self.corrected_flux,
                                 self.corrected_unc])
        fund_prot, fund_power, periods_to_test, periodogram = fit_out[:4]
        aliases, sigmas = fit_out[4:]

        eval_out =  evaluate.test_pgram(periods_to_test, periodogram, 
                                        self.power_threshold)

        self.corrected_prot = fund_prot

        if eval_out[-1]==False:
            logging.warning("Corrected lightcurve is not clean")
        else:
            logging.debug("Corrected lightcurve is clean")
        plot_aliases = [None, eval_out[2]]

        # Plot them up
        lcs = [[self.time, self.use_flux, self.use_unc],
               [self.time, self.corrected_flux, self.corrected_unc]]
        pgrams = [[self.init_periods_to_test, self.init_pgram], 
                  [periods_to_test, periodogram]]
        best_periods = [self.init_prot, fund_prot]
        data_labels = ["Initial", "Corrected"]
        sigmas = [self.init_sigmas, sigmas]
        rd_fig, rd_axes = plot.compare_multiple(lcs, pgrams, best_periods, 
                                                sigmas, 
                                                aliases=plot_aliases,
                                                data_labels=data_labels,  
                                                phase_by=fund_prot)

        rd_fig.suptitle(self.name, fontsize="large", y=0.99)

        ptime, fsine = evaluate.fit_sine(self.time, self.corrected_flux,
                                         self.corrected_unc, 
                                         self.corrected_prot)

        plotx = np.argsort(ptime)
        rd_axes[2].plot(ptime[plotx], fsine[plotx], color="lightgrey", lw=2)
        rd_axes[2].set_ylim(min(fsine)*0.9, max(fsine)*1.1)
        
        use_residuals = self.use_flux - fsine
        cor_residuals = self.corrected_flux - fsine

        logging.debug("RESIDUALS")
        logging.debug(use_residuals[:10])
        logging.debug(cor_residuals[:10])

        rd_axes[3].errorbar(self.time % fund_prot, use_residuals, 
                            np.zeros_like(self.time), #self.use_unc,
                            fmt=plot.shape1, ms=2, capsize=0, 
                            ecolor=plot.color1, color=plot.color1,
                            mec=plot.color1)
        rd_axes[3].errorbar(self.time % fund_prot, cor_residuals, 
                            np.zeros_like(self.time), #self.corrected_unc,
                            fmt=plot.shape2, ms=2, capsize=0, 
                            ecolor=plot.color2, color=plot.color2,  
                            mec=plot.color2)
        rd_axes[3].set_xlim(0, fund_prot)

        plt.savefig("{}_corrected.png".format(self.name))
#        plt.show()
        plt.close("all")



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

    def _run_fit(self, use_lc, prot_lims=[0.1,70]):
        """Run a fit on a single lc, either "raw" or "detrended" 
        or a array/list of [time, flux, and unc]
        """

        if use_lc=="raw":
            logging.debug("fitting raw lc")
            tt, ff, uu = self.time, self.flux, self.unc_flux
        elif use_lc=="detrended":
            logging.debug("fitting detrended lc")
            tt, ff, uu = self.time, self.det_flux, self.det_unc
        else:
            logging.debug("fitting other lc")
            tt, ff, uu = use_lc

        logging.debug("_run_fit threshold %f", self.power_threshold)

        # Iteratively smooth, clip, and run a periodogram (period_cleaner)
        pc_out = prot.period_cleaner(tt, ff, uu, 
                                     pgram_threshold=self.power_threshold, 
                                     prot_lims=prot_lims)
        cl_time, cl_flux, cl_unc, sm_flux = pc_out

        logging.debug("Smoothed, now periodogram")
        logging.debug("Cleaned t %d f %d u %d", len(cl_time),
                      len(cl_flux), len(cl_unc))
        # Test the periodogram and pick the best period and power
        ls_out = prot.run_ls(cl_time, cl_flux, cl_unc, 
                             threshold=self.power_threshold,
                             prot_lims=prot_lims, run_bootstrap=True)
        #fund_prot, fund_power, periods_to_test, periodogram, aliases, sigmas

        return ls_out

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

    def _xy_correct(self, n_closest=21):
        """Correct for positional variations in the lightcurve once selected."""
        
        # Loop through the lightcurve and find the n closest pixels.
        # Then divide by the median flux from those pixels

        num_pts = len(self.use_flux)
        logging.debug("Pixel position correction %d", num_pts)

        self.corrected_flux = np.zeros(num_pts)
        self.corrected_unc = np.zeros(num_pts)

        first_half = self.time<=2102
        x_pos1 = self.x_pos[first_half==True] 
        y_pos1 = self.y_pos[first_half==True]
        x_pos2 = self.x_pos[first_half==False]
        y_pos2 = self.y_pos[first_half==False]

        for i, fval, xx, yy in itertools.izip(range(num_pts), self.use_flux,
                                              self.x_pos, self.y_pos):
            if first_half[i]:
                comp_x, comp_y = x_pos1, y_pos1
                comp_f = self.use_flux[first_half==True]
            else:
                comp_x, comp_y = x_pos2, y_pos2
                comp_f = self.use_flux[first_half==False]

#            comp_x, comp_y = self.x_pos, self.y_pos
#            comp_f = self.use_flux

            pix_sep = np.sqrt((xx - comp_x)**2 + (yy - comp_y)**2)
            min_ind = np.argpartition(pix_sep, n_closest)[:n_closest]
            logging.debug(np.median(pix_sep[min_ind]))

            median_nearest = np.median(comp_f[min_ind])
            logging.debug("This flux %f Median Nearest %f", 
                          fval, median_nearest)
            self.corrected_flux[i] = fval / median_nearest
            self.corrected_unc[i] = self.use_unc[i] / median_nearest

        logging.debug("Correction completed")

    def _plot_xy(self):
        """Plot some basic informational plots:
        Flux as a function of X-Y position
        Flux as a function of time
        """
        pass

    def _multi_search(self,lc_type):
        """Search a lightcurve for a secondary signal."""
        pass

    
