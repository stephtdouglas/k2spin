
import logging, itertools, os, sys
from datetime import date

import numpy as np
import astropy.io.ascii as at
import matplotlib.pyplot as plt
from astroML import time_series

logging.basicConfig(level=logging.DEBUG)

#from k2spin.config import *
from k2spin import lc
from k2spin import k2io
from k2spin import plot
from k2spin import acf
from k2spin import fix_kepler

today = date.today().isoformat()

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
                                    detrend_kwargs=detrend_kwargs, to_plot=False)
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
    logging.warning("Using %s", best_col)
    return best_col, light_curves[max_power]

def run_one(filename,lc_dir, ap=None,
            detrend_kwargs=None, output_f=None, save_lcs=True):

    lcs = at.read(lc_dir+filename)

    time = lcs["t"]
    x_pos = lcs["x"]
    y_pos = lcs["y"]
    qual_flux = np.zeros_like(time)
    unc_flux = np.ones_like(time)

    if (ap is None) or (ap<1):
        best_col, light_curve = choose_lc(lcs, filename, detrend_kwargs)
        flux = lcs[best_col]
    else:
#    best_col = "flux_4.0"
        best_col = "flux_{0:.1f}".format(ap)
        flux = lcs[best_col]
        light_curve = lc.LightCurve(time, flux, unc_flux, x_pos, y_pos,
                                    name=filename.split("/")[-1][:-4],
                                    detrend_kwargs=detrend_kwargs)


    light_curve.choose_initial(to_plot=False)
    light_curve.correct_and_fit(to_plot=False, n_closest=21)
    light_curve.multi_search(to_plot=False)

#    plot.plot_xy(light_curve.x_pos, light_curve.y_pos, light_curve.time,
#                 light_curve.flux, "Raw Flux")
#    plt.suptitle(light_curve.name, fontsize="large")
#    plt.savefig(base_path+"plot_outputs/{}_xy_flux.png".format(light_curve.name))
##    plt.show()
#    plt.close("all")

    epic = filename.split("/")[-1][4:13]
    if output_f is not None:
        output_f.write("\n{},{}".format(filename, epic))
        output_f.write(","+best_col.split("_")[-1])
        output_f.write(","+light_curve.use)
        # Initial lightcurve result
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.init_prot,
                                                 light_curve.init_power))
        s1, s2 = light_curve.init_sigmas[:2]
        output_f.write(",{0:.4f},{1:.4f}".format(s1, s2))
        output_f.write(",{0:.4f},{1:.4f},{2:.4f},{3:.4f}".format(
                       *light_curve.init_harmonics[:]))
        # Corrected lightcurve result
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.corr_prot,
                                                 light_curve.corr_power))
        s3, s4 = light_curve.corr_sigmas[:2]
        output_f.write(",{0:.4f},{1:.4f}".format(s3,s4))
        output_f.write(",{0:.4f},{1:.4f},{2:.4f},{3:.4f}".format(
                       *light_curve.corr_harmonics[:]))
        # Second period search
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.sec_prot,
                                                 light_curve.sec_power))
        s5, s6 = light_curve.sec_sigmas[:2]
        output_f.write(",{0:.4f},{1:.4f}".format(s5,s6))

    if save_lcs==True:
        # Write out the lightcurves themselves
        lc_out = open("output_lcs/{0}_r{1}_lcs.csv".format(
                      light_curve.name,ap),"w")
        # Write out any relevant arguments and the date
        lc_out.write("# Generated on {0}".format(today))
        lc_out.write("\n# Aperture: {0}".format(best_col))
        lc_out.write("\n# Detrend Kwargs: ")
        if detrend_kwargs is not None:
            for k in detrend_kwargs.keys():
                lc_out.write("{0} = {1} ".format(k, detrend_kwargs[k]))
        
        lc_out.write("\nt,raw,bulk_trend,det,init_trend,med,corr,corr_trend,sec,sec_trend")
        for tt,rr,bb,dd,ii,mm,cc,ct,ss,st in itertools.izip(light_curve.time,
                    light_curve.flux, light_curve.bulk_trend,
                    light_curve.det_flux, light_curve.init_trend,
                    light_curve.median_flux,
                    light_curve.corrected_flux, light_curve.corr_trend,
                    light_curve.sec_flux, light_curve.sec_trend):
            lc_out.write("\n{0:.6f},{1:.3f}".format(tt,rr))
            lc_out.write(",{0:.6f},{1:.6f}".format(bb,dd))
            lc_out.write(",{0:.6f}".format(ii))
            lc_out.write(",{0:.6f},{1:.6f}".format(mm,cc))
            lc_out.write(",{0:.6f},{1:.6f}".format(ct,ss))
            lc_out.write(",{0:.6f}".format(st))
        lc_out.close()


def run_list(filenames, lc_dir, output_filename, ap=6, detrend_kwargs=None):
    
    outfile = "{0}_results_{1}.csv".format(output_filename,today)
    output_f = open("tables/{0}".format(outfile),"w")
    output_f.write("filename,EPIC,ap,lc")
    output_f.write(",init_prot,init_power,init99.9,init99")
    output_f.write(",init_0.5prot, init_0.5power, init_2prot, init_2power")
    output_f.write(",corr_prot,corr_power,corr99.9,corr99")
    output_f.write(",corr_0.5prot, corr_0.5power, corr_2prot, corr_2power")
    output_f.write(",sec_prot,sec_power,sec99.9,sec99")

    for i, filename in enumerate(filenames):
        if filename.endswith(".csv")==False:
            filename = filename+".csv"
        logging.warning("starting %d %s",i,filename)
        if os.path.exists(lc_dir+filename):
            try:
                run_one(filename, lc_dir, ap=ap,
                        detrend_kwargs=detrend_kwargs, output_f=output_f,
                        save_lcs=True)
            except:
                logging.warning("ERROR ON %s",i, filename)
                output_f.write("\n{0},".format(filename))
                output_f.write(",, ,,,, ,,,, ,,,, ,,,, ,,,,")
        else:
            logging.warning("SKIPPING %s",filename)
        logging.warning("done %d %s",i,filename)
#        break

    output_f.close()

if __name__=="__main__":

#    logging.basicConfig(level=logging.INFO)

    ap = int(sys.argv[1])

    arrayid = int(os.getenv("PBS_ARRAYID",0))

    use_date = "2016-01-23"
    input_file = "tables/instrumental_signals_{0}_phot_{1}.csv".format(arrayid, use_date)
    input_list = at.read(input_file)
    print(input_list.dtype)
    print(input_list["output_file"][0])

    output_filename = "instrumental_signals_{0}_spin_r{1}_{2}.csv".format(arrayid, ap, today)

    run_list(input_list["output_file"],
             lc_dir = "/vega/astro/users/sd2706/k2/c4/lcs/",
             output_filename = output_filename, ap=ap,
             detrend_kwargs={"kind":"supersmoother","phaser":10})


