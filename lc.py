"""Class for lightcurves."""

import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt
from astroML import time_series

from k2spin.config import *
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
                 power_threshold=0.5, detrend_kwargs=None, to_plot=False):
        """Clean up the input data and sigma-clip it.

        input
        -----
        time, flux, unc_flux: array-like
            the lightcurve

        x_pos, y_pos: array-like
            centroid pixel positions

        name: string

        power_threshold: float (should remove...)

        detrend_kwargs: dict
            kind: string (default supersmoother)
                "supersmoother","boxcar", or "linear" 
            phaser: float, optional
                alpha, half-width of smoothing window, or None (respectively)

        """
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
        self._bulk_detrend(detrend_kwargs, to_plot)

    def choose_initial(self, to_plot=False):
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

        # Only consider peaks less than ~half the length of the lightcurve
#        max_peak_loc = 0.75 * (self.time[-1] - self.time[0])
        max_peak_loc = 40
        logging.info("Max Prot = %f", max_peak_loc)

        raw_loc2 = np.argmax(raw_pgram[raw_prots<max_peak_loc])
        raw_power2 = raw_pgram[raw_prots<max_peak_loc][raw_loc2]
        raw_prot2 = raw_prots[raw_prots<max_peak_loc][raw_loc2]
        logging.info("raw %d FP %f Power %f", raw_loc2, raw_prot2, raw_power2)

        det_loc2 = np.argmax(det_pgram[det_prots<max_peak_loc])
        det_power2 = det_pgram[det_prots<max_peak_loc][det_loc2]
        det_prot2 = det_prots[det_prots<max_peak_loc][det_loc2]
        logging.info("det %d FP %f Power %f", det_loc2, det_prot2, det_power2)

        # Compare them
        lc_to_use = self._pick_lc(raw_power2, det_power2)
        if lc_to_use<=1:
            logging.info("Using raw lightcurve")
            self.init_prot , self.init_power = raw_prot2, raw_power2
            self.init_periods_to_test, self.init_pgram = raw_prots, raw_pgram
            self.use_flux = self.flux / self.med
            self.use_unc = self.unc_flux / self.med
            self.init_sigmas = raw_sigma
            self.use = "raw"
            data_labels = ["Raw (Selected)", "Detrended"]
        elif lc_to_use==2:
            logging.info("Using detrended lightcurve")
            self.init_prot , self.init_power = det_prot2, det_power2
            self.init_periods_to_test, self.init_pgram = det_prots, det_pgram
            self.use_flux = self.det_flux 
            self.use_unc = self.unc_flux 
            self.init_sigmas = det_sigma
            self.use = "det"
            data_labels = ["Raw", "Detrended (Selected)"]

        logging.info("Initial Prot %f Power %f", self.init_prot, 
                     self.init_power)
        # get power at harmonics
        self.init_harmonics = self._harmonics(self.init_prot, 
                                              self.init_periods_to_test,
                                              self.init_pgram)

        # Get aliases for selected period
        eval_out = evaluate.test_pgram(self.init_periods_to_test, 
                                       self.init_pgram, self.power_threshold)

        # Get phase-folded, smoothed trend
        white_out2 = detrend.pre_whiten(self.time, self.use_flux, 
                                        self.use_unc, self.init_prot,
                                        which="phased")
        self.init_trend = white_out2[2]

 
        if eval_out[-1]==False:
            logging.warning("Selected lightcurve is not clean")
        else:
            logging.debug("Selected lightcurve is clean")
        plot_aliases = [None, eval_out[2]]

        if to_plot:
            # Plot them up
            lcs = [[self.time, self.flux/self.med, abs(self.unc_flux/self.med)],
                   [self.time, self.det_flux, self.det_unc]]
            pgrams = [[raw_prots, raw_pgram], [det_prots, det_pgram]]
            best_periods = [raw_prot2, det_prot2]
            sigmas = [raw_sigma, det_sigma]
            logging.debug(sigmas)
            rd_fig, rd_axes = plot.compare_multiple(lcs, pgrams, best_periods, 
                                                    sigmas, 
                                                    aliases=plot_aliases,
                                                    data_labels=data_labels,  
                                                    phase_by=self.init_prot)

            rd_fig.suptitle(self.name, fontsize="large", y=0.99)

            rd_fig.delaxes(rd_axes[3])

            plt.savefig("{0}plot_outputs/{1}_raw_detrend.png".format(base_path,
                                                                     self.name))
            plt.close("all")

    def correct_and_fit(self, to_plot=False, n_closest=21):
        """Position-correct and perform a fit."""
        logging.debug("Fitting corrected lightcurve")

        cl_flux, cl_unc = self._clean_it(self.use)

        self._xy_correct(correct_with=cl_flux, n_closest=n_closest)

        fit_out = self._run_fit([self.time, self.corrected_flux,
                                 self.corrected_unc])
        fund_prot, fund_power, periods_to_test, periodogram = fit_out[:4]
        aliases, sigmas = fit_out[4:]

        eval_out =  evaluate.test_pgram(periods_to_test, periodogram, 
                                        self.power_threshold)

        self.corr_prot = fund_prot
        self.corr_power = fund_power
        self.corr_sigmas = sigmas
        self.corr_periods = periods_to_test
        self.corr_pgram = periodogram

        self.corr_harmonics = self._harmonics(self.corr_prot,
                                              self.corr_periods,
                                              self.corr_pgram)

        if eval_out[-1]==False:
            logging.warning("Corrected lightcurve is not clean")
        else:
            logging.debug("Corrected lightcurve is clean")
        plot_aliases = [None, eval_out[2]]

        if to_plot:
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
                                             self.corr_prot)

            plotx = np.argsort(ptime)
            rd_axes[2].plot(ptime[plotx], fsine[plotx], color="lightgrey", lw=2)
#            rd_axes[2].set_ylim(min(fsine)*0.9, max(fsine)*1.1)
        
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

            plt.savefig("{0}plot_outputs/{1}_corrected.png".format(base_path, 
                                                                   self.name))
#            plt.show()
            plt.close("all")



    def _bulk_detrend(self, detrend_kwargs, to_plot=False):
        """Smooth the rapid variations in the lightcurve and remove bulk trends.

        inputs
        ------
        alpha: float
            "bass enhancement" for supersmoother. 
        """

        if detrend_kwargs is None:
            detrend_kwargs = dict()
        detrend_kwargs["kind"] = detrend_kwargs.get("kind", "supersmoother")
        detrend_kwargs["phaser"] = detrend_kwargs.get("phaser", None)

        logging.debug("Removing bulk trend...")
        det_out = detrend.simple_detrend(self.time, self.flux, self.unc_flux,
                                         to_plot=to_plot, **detrend_kwargs)
        self.det_flux, self.det_unc, self.bulk_trend = det_out

        logging.debug("len detrended t %d f %d u %d", len(self.time), 
                      len(self.det_flux),len(self.det_unc))
        if to_plot==True:
            fig = plt.gcf()
            fig.suptitle("{}; {} ({})".format(self.name,detrend_kwargs["kind"],
                         detrend_kwargs["phaser"]),fontsize="x-large")
            plt.savefig("{0}plot_outputs/{1}_detrend.png".format(base_path,
                                                                 self.name))

    def _run_fit(self, use_lc, prot_lims=[0.1,70]):
        """Run a fit on a single lc, either "raw" or "detrended" 
        or a array/list of [time, flux, and unc]
        """

        if use_lc=="raw":
            logging.debug("fitting raw lc")
            tt, ff, uu = self.time, self.flux, self.unc_flux
        elif (use_lc=="detrended") or (use_lc=="det"):
            logging.debug("fitting detrended lc")
            tt, ff, uu = self.time, self.det_flux, self.det_unc
        else:
            logging.debug("fitting other lc")
            tt, ff, uu = use_lc

        # Test the periodogram and pick the best period and power
        ls_out = prot.run_ls(tt, ff, uu, threshold=self.power_threshold,
                             prot_lims=prot_lims, run_bootstrap=True)
#        fund_prot, fund_power, periods_to_test, periodogram = ls_out[:4]

        return ls_out

    def _harmonics(self, fund_prot, periods, powers):
        """ Find 1/2 and 2x harmonics."""

        if fund_prot > (2 * min(periods)):
#            logging.debug("fund P %f half P %f", fund_prot, max(periods))
            half_fund = fund_prot / 2.0
            half_width = fund_prot * 0.01
            half_region = np.where(abs(half_fund - periods) < half_width)[0]
            half_peak = np.argmax(powers[half_region])
            half_per = periods[half_region][half_peak]
            half_pow = powers[half_region][half_peak]
        else:
            half_per, half_pow = np.nan, np.nan

        if fund_prot < (0.5 * max(periods)):
#            logging.debug("fund P %f max P %f", fund_prot, max(periods))
            twice_fund = fund_prot * 2.0
            twice_width = twice_fund * 0.01
            twice_region = np.where(abs(twice_fund - periods) < twice_width)[0]
            twice_peak = np.argmax(powers[twice_region])
            twice_per = periods[twice_region][twice_peak]
            twice_pow = powers[twice_region][twice_peak]
        else:
            twice_per, twice_pow = np.nan, np.nan

        return half_per, half_pow, twice_per, twice_pow

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

    def _clean_it(self, use_lc, prot_lims=[0.1,70]):
        """Clean all periodic signals from the lightcurve."""
        if use_lc=="raw":
            logging.debug("fitting raw lc")
            tt, ff, uu = self.time, self.flux, self.unc_flux
        elif (use_lc=="detrended") or (use_lc=="det"):
            logging.debug("fitting detrended lc")
            tt, ff, uu = self.time, self.det_flux, self.det_unc
        else:
            logging.debug("fitting other lc")
            tt, ff, uu = use_lc

        logging.debug("_run_fit threshold %f", self.power_threshold)

        # Iteratively smooth, clip, and run a periodogram (period_cleaner)
        dk = {"filename":"{0}plot_outputs/{1}_cleaning".format(base_path, 
                                                               self.name)}
        pc_out = prot.detrend_for_correction(tt, ff, uu,
                                             prot_lims=prot_lims,
                                             to_plot=False, 
                                             detrend_kwargs=dk)
        cl_flux, cl_unc = pc_out

        return cl_flux, cl_unc

    def _xy_correct(self, correct_with=None, n_closest=21):
        """Correct for positional variations in the lightcurve once selected."""
        
        # Loop through the lightcurve and find the n closest pixels.
        # Then divide by the median flux from those pixels

        num_pts = len(self.use_flux)
        logging.debug("Pixel position correction %d", num_pts)

        if correct_with is None:
            correct_with = self.use_flux

        self.corrected_flux = np.zeros(num_pts)
        self.corrected_unc = np.zeros(num_pts)
        self.median_flux = np.zeros(num_pts)

        first_half = self.time<=2264
        x_pos1 = self.x_pos[first_half==True] 
        y_pos1 = self.y_pos[first_half==True]
        x_pos2 = self.x_pos[first_half==False]
        y_pos2 = self.y_pos[first_half==False]

        for i, fval, xx, yy in itertools.izip(range(num_pts), self.use_flux,
                                              self.x_pos, self.y_pos):
            logging.debug(i)
            logging.debug(first_half[i])
            if first_half[i]:
                comp_x, comp_y = x_pos1, y_pos1
                comp_f = correct_with[first_half==True]
            else:
                comp_x, comp_y = x_pos2, y_pos2
                comp_f = correct_with[first_half==False]

#            comp_x, comp_y = self.x_pos, self.y_pos
#            comp_f = self.use_flux

            logging.debug(n_closest)
            pix_sep = np.sqrt((xx - comp_x)**2 + (yy - comp_y)**2)
            min_ind = np.argpartition(pix_sep, n_closest)[:n_closest]
            logging.debug(min_ind)
            logging.debug(np.median(pix_sep[min_ind]))

            median_nearest = np.median(comp_f[min_ind])
            #logging.debug("This flux %f Median Nearest %f", 
            #              fval, median_nearest)
            self.median_flux[i] = median_nearest
            self.corrected_flux[i] = fval / median_nearest
            self.corrected_unc[i] = self.use_unc[i] / median_nearest

        logging.debug("Correction completed")

    def _plot_xy(self):
        """Plot some basic informational plots:
        Flux as a function of X-Y position
        Flux as a function of time
        """
        pass

    def multi_search(self, to_plot=False):
        """Search a lightcurve for a secondary signal."""
        # Start with the corrected lightcurve and its associated period
        # Phase on that period and remove it
        white_out = detrend.pre_whiten(self.time, self.corrected_flux, 
                                       self.corrected_unc, self.corr_prot,
                                       which="phased")
        detrended_flux = self.corrected_flux / white_out[2]

        self.corr_trend = white_out[2]
        self.sec_flux = detrended_flux
        self.sec_unc = self.corrected_unc

        # Run lomb-scargle again and re-measure the period
        fit_out = self._run_fit([self.time, self.sec_flux, self.sec_unc])
        self.sec_prot = fit_out[0]
        self.sec_power = fit_out[1]
        self.sec_periods = fit_out[2]
        self.sec_pgram = fit_out[3]
        self.sec_sigmas = fit_out[5]

        eval_out =  evaluate.test_pgram(self.sec_periods, self.sec_pgram,
                                        self.power_threshold)
        plot_aliases = [None, eval_out[2]]

        white_out2 = detrend.pre_whiten(self.time, self.sec_flux, 
                                        self.sec_unc, self.sec_prot,
                                        which="phased")
        self.sec_trend = white_out2[2]

        # Plot!
        if to_plot:
            # Plot them up
            lcs = [[self.time, self.corrected_flux, self.corrected_unc],
                   [self.time, self.sec_flux, self.sec_unc]]
            pgrams = [[self.corr_periods, self.corr_pgram], 
                      [self.sec_periods, self.sec_pgram]]
            best_periods = [self.corr_prot, self.sec_prot]
            data_labels = ["Corrected", "Fund. Prot="
                           "{0:.2f}d Removed".format(self.corr_prot)]
            sigmas = [self.corr_sigmas, self.sec_sigmas]
            rd_fig, rd_axes = plot.compare_multiple(lcs, pgrams, best_periods, 
                                                    sigmas, 
                                                    aliases=plot_aliases,
                                                    data_labels=data_labels,  
                                                    phase_by=self.sec_prot)

            rd_fig.suptitle(self.name, fontsize="large", y=0.99)

            rd_fig.delaxes(rd_axes[3])

            rd_axes[0].plot(self.time, white_out[2], 'b-', lw=2)

            plt.savefig("{0}plot_outputs/{1}_second_period.png".format(
                        base_path,self.name))
        
