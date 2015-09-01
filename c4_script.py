
import logging, itertools
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

def choose_lc(lcs):
    cadence = np.median(np.diff(lcs["t"]))

    ap_cols = []
    std_rat = []
    for colname in lcs.dtype.names:
        good_flux = len(np.where(np.isfinite(lcs[colname]))[0])
        if (("flux" in colname) and (good_flux>2000) and
            (("2." in colname)==False)):
            ap_cols.append(colname)
            std_rat.append(std_ratio(lcs[colname], cadence))
    ap_cols = np.array(ap_cols)
    std_rat = np.array(std_rat)
    logging.info(ap_cols)

    best_col = ap_cols[np.argmax(std_rat)]
    logging.info("Using %s", best_col)
    return best_col

def run_one(filename,lc_dir="/home/stephanie/code/python/k2spin/lcs/",
            detrend_kwargs=None, output_f=None, save_lcs=True):

    lcs = at.read(lc_dir+filename)

    time = lcs["t"]
    x_pos = lcs["x"]
    y_pos = lcs["y"]
    qual_flux = np.zeros_like(time)
#    best_col = choose_lc(lcs)
    best_col = "flux_4.0"
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
    plt.savefig("plot_outputs/{}_xy_flux.png".format(light_curve.name))
#    plt.show()
    plt.close("all")

    epic = filename.split("/")[-1][4:13]
    if output_f is not None:
        output_f.write("\n{},{}".format(filename, epic))
        output_f.write(","+best_col.split("_")[-1])
        output_f.write(","+light_curve.use)
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.init_prot,
                                                 light_curve.init_power))
        s1, s2 = light_curve.init_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s1, s2))
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.corr_prot,
                                                 light_curve.corr_power))
        s3, s4 = light_curve.corr_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s3,s4))

    if save_lcs==True:
        lc_out = open("output_lcs/{}_lcs.csv".format(
                      light_curve.name),"w")
        lc_out.write("t,raw,det,corr")
        for tt,rr,dd,cc in itertools.izip(light_curve.time,
                                          light_curve.flux,
                                          light_curve.det_flux,
                                          light_curve.corrected_flux):
            lc_out.write("\n{0:.6f},{1:.3f}".format(tt,rr))
            lc_out.write(",{0:.6f},{1:.6f}".format(dd,cc))
        lc_out.close()



def run_list(listname, lc_dir, detrend_kwargs=None):
    
    lcs = at.read(listname,names=["file"])

    outfile = listname[:-4]+"_results_{}.csv".format(today)
    output_f = open(outfile,"w")
    output_f.write("filename,EPIC,ap,lc,init_prot,init_power")
    output_f.write(",init99,init95,corr_prot,corr_power")
    output_f.write(",corr99,corr95")

    for i, filename in enumerate(lcs["file"]):
        logging.warning("starting %d %s",i,filename)
        run_one(filename, lc_dir, 
                detrend_kwargs=detrend_kwargs, output_f=output_f,save_lcs=True)
        logging.warning("done %d %s",i,filename)

    output_f.close()

if __name__=="__main__":

    logging.basicConfig(level=logging.DEBUG)

    lc_dir = "/home/stephanie/code/python/k2phot/lcs/"

    lc_file = "ktwo210359769-c04.csv"

#    run_one(lc_file, 
#            lc_dir="/home/stephanie/code/python/k2phot/lcs/", 
#            detrend_kwargs={"kind":"supersmoother","phaser":10})

    run_list("c4_lcs.lst", 
             lc_dir = "",
             detrend_kwargs={"kind":"supersmoother","phaser":10})
