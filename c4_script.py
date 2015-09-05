
import logging, itertools, os
from datetime import date

import numpy as np
import astropy.io.ascii as at
import matplotlib.pyplot as plt

from k2spin import lc
from k2spin import k2io
from k2spin import plot

today = date.today().isoformat()

def std_ratio(y, cadence):

    y_use = y[np.isfinite(y)]
    six_hr = 0.25 / cadence

    six_hr_std = np.zeros_like(y_use)
    for i in range(6,len(y_use)-6):
        six_hr_std[i] = np.std(y_use[i-6:i+6])
    logging.debug("std y %f 6hr %f",np.std(y_use), np.median(six_hr_std))
    
    return np.std(y_use)/np.median(six_hr_std)

def choose_lc(lcs, filename, detrend_kwargs=None):

    time = lcs["t"]
    x_pos = lcs["x"]
    y_pos = lcs["y"]
    unc_flux = np.ones_like(time)

    ap_cols = []
    light_curves = []
    which = []
    periods = []
    powers = []
    for colname in lcs.dtype.names:
        good_flux = len(np.where(np.isfinite(lcs[colname]))[0])
        if (("flux" in colname) and (good_flux>2000) and
            (("2." in colname)==False)):
            ap_cols.append(colname)
            this_lc = lc.LightCurve(time, lcs[colname], unc_flux, x_pos, y_pos,
                                name=filename.split("/")[-1][:-4],
                                    detrend_kwargs=detrend_kwargs, to_plot=True)
            this_lc.choose_initial(to_plot=False)

            light_curves.append(this_lc)
            periods.append(this_lc.init_prot)
            powers.append(this_lc.init_power)
            which.append(this_lc.use)

    ap_cols = np.array(ap_cols)
    periods = np.array(periods)
    powers = np.array(powers)
    which = np.array(which)
    logging.info(ap_cols)
    logging.info(periods)
    logging.info(powers)
    logging.info(which)

    max_power = np.argmax(powers)
    best_col = ap_cols[max_power]
    logging.info("Using %s", best_col)
    return best_col, light_curves[max_power]

def run_one(filename,lc_dir="/home/stephanie/code/python/k2spin/lcs/", ap=None,
            detrend_kwargs=None, output_f=None, save_lcs=True):

    lcs = at.read(lc_dir+filename)

    time = lcs["t"]
    x_pos = lcs["x"]
    y_pos = lcs["y"]
    qual_flux = np.zeros_like(time)
    unc_flux = np.ones_like(time)

    if ap is None:
        best_col, light_curve = choose_lc(lcs, filename, detrend_kwargs)
        flux = lcs[best_col]
    else:
#    best_col = "flux_4.0"
        best_col = "flux_{0:.1f}".format(ap)
        flux = lcs[best_col]
        light_curve = lc.LightCurve(time, flux, unc_flux, x_pos, y_pos,
                                    name=filename.split("/")[-1][:-4],
                                    detrend_kwargs=detrend_kwargs)


    light_curve.choose_initial(to_plot=True)
    light_curve.correct_and_fit(to_plot=True)
    light_curve.multi_search(to_plot=True)

    plot.plot_xy(light_curve.x_pos, light_curve.y_pos, light_curve.time,
                 light_curve.flux, "Raw Flux")
    plt.suptitle(light_curve.name, fontsize="large")
    plt.savefig("plot_outputs/{}_xy_flux.png".format(light_curve.name))
#    plt.show()
    plt.close("all")

    epic = filename.split("/")[-1][4:13]
    if output_f is not None:
        output_f.write("\n{},{}".format(filename, epic))
        output_f.write(","+best_col.split("_")[-1])
        output_f.write(","+light_curve.use)
        # Initial lightcurve result
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.init_prot,
                                                 light_curve.init_power))
        s1, s2 = light_curve.init_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s1, s2))
        # Corrected lightcurve result
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.corr_prot,
                                                 light_curve.corr_power))
        s3, s4 = light_curve.corr_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s3,s4))
        # Second period search
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.sec_prot,
                                                 light_curve.sec_power))
        s5, s6 = light_curve.sec_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s5,s6))

    if save_lcs==True:
        # Write out the lightcurves themselves
        lc_out = open("output_lcs/{}_lcs.csv".format(
                      light_curve.name),"w")
        # Write out any relevant arguments and the date
        lc_out.write("# Generated on {0}".format(today))
        lc_out.write("# Best aperture: {0}".format(best_col))
        lc_out.write("# Detrend Kwargs: ")
        if detrend_kwargs is not None:
            for k in detrend_kwargs.keys():
                lc_out.write("{0} = {1} ".format(k, detrend_kwargs[k]))

        # Now write out the LCs
        lc_out.write("t,raw,det,corr,sec")
        for tt,rr,dd,cc,ss in itertools.izip(light_curve.time,
                                            light_curve.flux,
                                            light_curve.det_flux,
                                            light_curve.corrected_flux,
                                            light_curve.sec_flux):
            lc_out.write("\n{0:.6f},{1:.3f}".format(tt,rr))
            lc_out.write(",{0:.6f},{1:.6f}".format(dd,cc))
            lc_out.write(",{0:.6f}".format(ss))
        lc_out.close()

    # Write out periodograms
    pg_out = open("output_lcs/{}_pgram.csv".format(
                      light_curve.name),"w")
        # Write out any relevant arguments and the date
        pg_out.write("# Generated on {0}".format(today))
        pg_out.write("# Best aperture: {0}".format(best_col))
        pg_out.write("# Detrend Kwargs: ")
        if detrend_kwargs is not None:
            for k in detrend_kwargs.keys():
                pg_out.write("{0} = {1} ".format(k, detrend_kwargs[k]))

        # Now write out the LCs
        pg_out.write("{0}_period,{0}_power".format(light_curve.use))
        pg_out.write(",corr_period,corr_power,sec_period,sec_power")
        use_periods = light_curve.init_periods_to_test
        use_len = len(use_periods)
        corr_len = len(light_curve.corr_periods)
        sec_len = len(light_curve.sec_periods)
        max_len = max(use_len, corr_len, sec_len)
        for i in range(max_len):
            if i<use_len:
                pg_out.write("\n{0:.6f},{1:.6f}".format(use_periods[i],
                             light_curve.init_pgram[i]))
            else:
                pg_out.write("\nNaN,NaN")
            if i<corr_len:
                pg_out.write(",{0:.6f},{1:.6f}".format(
                             light_curve.corr_periods[i],
                             light_curve.corr_pgram[i]))
            else:
                pg_out.write(",NaN,NaN")
            if i<sec_len:
                pg_out.write(",{0:.6f},{1:.6f}".format(
                             light_curve.sec_periods[i],
                             light_curve.sec_pgram[i]))
            else:
                pg_out.write(",NaN,NaN")
        pg_out.close()



def run_list(listname, lc_dir, detrend_kwargs=None):
    
    lcs = at.read(listname)

    outfile = listname[:-4]+"_results_{}.csv".format(today)
    output_f = open(outfile,"w")
    output_f.write("filename,EPIC,ap,lc")
    output_f.write(",init_prot,init_power,init99,init95")
    output_f.write(",corr_prot,corr_power,corr99,corr95")
    output_f.write(",sec_prot,sec_power,sec99,sec95")

    for i, filename in enumerate(lcs["filename"]):
        logging.warning("starting %d %s",i,filename)
        if os.path.exists(filename):
            run_one(filename, lc_dir, ap=lcs["ap"][i],
                    detrend_kwargs=detrend_kwargs, output_f=output_f,
                    save_lcs=True)
        logging.warning("done %d %s",i,filename)

    output_f.close()

if __name__=="__main__":

    logging.basicConfig(level=logging.DEBUG)

    lc_dir = "/home/stephanie/code/python/k2phot/lcs/"

    lc_file = "ktwo210359769-c04.csv"
    ap = 6.5

#    run_one(lc_file, 
#            lc_dir="/home/stephanie/code/python/k2phot/lcs/", ap=ap,
#            detrend_kwargs={"kind":"supersmoother","phaser":10})

    run_list("../k2spin/c4_lcs_results_2015-09-02.csv", 
             lc_dir = "",
             detrend_kwargs={"kind":"supersmoother","phaser":10})
