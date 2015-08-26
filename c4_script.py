
import logging

import numpy as np
import astropy.io.ascii as at
import matplotlib.pyplot as plt

from k2spin import lc
from k2spin import k2io
from k2spin import plot

def std_ratio(y, cadence):

    y_use = y[np.isfinite(y)]
    six_hr = 0.25 / cadence

    six_hr_std = np.zeros_like(y_use)
    for i in range(6,len(y_use)-6):
        six_hr_std[i] = np.std(y_use[i-6:i+6])
    logging.debug("std y %f 6hr %f",np.std(y_use), np.median(six_hr_std))
    
    return np.std(y_use)/np.median(six_hr_std)

def choose_lc(lcs):
    cadence = np.median(np.diff(lcs["t"]))

    ap_cols = []
    std_rat = []
    for colname in lcs.dtype.names:
        if "flux" in colname:
            ap_cols.append(colname)
            std_rat.append(std_ratio(lcs[colname], cadence))
    ap_cols = np.array(ap_cols)
    std_rat = np.array(std_rat)
    logging.info(ap_cols)

    best_col = ap_cols[np.argmax(std_rat)]
    logging.info("Using %s", best_col)
    return best_col

def run_one(filename,lc_dir="/home/stephanie/code/python/k2spin/lcs/",
            detrend_kwargs=None):

    lcs = at.read(lc_dir+filename)

    time = lcs["t"]
    x_pos = lcs["x"]
    y_pos = lcs["y"]
    qual_flux = np.zeros_like(time)
    best_col = choose_lc(lcs)
    flux = lcs[best_col]
    unc_flux = np.ones_like(time)

    light_curve = lc.LightCurve(time, flux, unc_flux, x_pos, y_pos,
                                name=filename.split("/")[-1][:-4],
                                detrend_kwargs=detrend_kwargs)
    light_curve.choose_initial()
    light_curve.correct_and_fit()

    plot.plot_xy(light_curve.x_pos, light_curve.y_pos, light_curve.time,
                 light_curve.flux, "Raw Flux")
    plt.suptitle(light_curve.name, fontsize="large")
    plt.savefig(light_curve.name+"_xy_flux.png")
#    plt.show()
    plt.close("all")

def run_list(listname,lc_dir="/home/stephanie/code/python/k2spin/lcs/",
             num_apertures=2, detrend_kwargs=None):
    
    lcs = at.read(listname,names=["file"])

    for i, filename in enumerate(lcs["file"]):
        logging.info("%d %s",i,filename)
        run_one(filename, lc_dir, num_apertures=num_apertures,
                detrend_kwargs=detrend_kwargs)


if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)

    lc_dir = "/home/stephanie/code/python/k2spin/lcs/"

    lc_file = "ktwo210359769-c04.csv"

    run_one(lc_file, 
            lc_dir="/home/stephanie/code/python/k2phot/lcs/", 
            detrend_kwargs={"kind":"supersmoother","phaser":10})

#    run_list("c4_lcs.lst", num_apertures=2, 
#             detrend_kwargs={"kind":"supersmoother","phaser":10})
