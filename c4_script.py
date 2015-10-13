
import logging, itertools, os
from datetime import date

import numpy as np
import astropy.io.ascii as at
import matplotlib.pyplot as plt
from astroML import time_series

from k2spin.config import *
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


    light_curve.choose_initial(to_plot=True)
    light_curve.correct_and_fit(to_plot=True)
    light_curve.multi_search(to_plot=True)

    plot.plot_xy(light_curve.x_pos, light_curve.y_pos, light_curve.time,
                 light_curve.flux, "Raw Flux")
    plt.suptitle(light_curve.name, fontsize="large")
    plt.savefig(base_path+"plot_outputs/{}_xy_flux.png".format(light_curve.name))
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
        output_f.write(",{0:.4f},{1:.4f},{2:.4f},{3:.4f}".format(
                       *light_curve.init_harmonics[:]))
        # Corrected lightcurve result
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.corr_prot,
                                                 light_curve.corr_power))
        s3, s4 = light_curve.corr_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s3,s4))
        output_f.write(",{0:.4f},{1:.4f},{2:.4f},{3:.4f}".format(
                       *light_curve.corr_harmonics[:]))
        # Second period search
        output_f.write(",{0:.4f},{1:.4f}".format(light_curve.sec_prot,
                                                 light_curve.sec_power))
        s5, s6 = light_curve.sec_sigmas
        output_f.write(",{0:.4f},{1:.4f}".format(s5,s6))

    if save_lcs==True:
        # Write out the lightcurves themselves
        lc_out = open(base_path+"output_lcs/{}_lcs.csv".format(
                      light_curve.name),"w")
        # Write out any relevant arguments and the date
        lc_out.write("# Generated on {0}".format(today))
        lc_out.write("\n# Best aperture: {0}".format(best_col))
        lc_out.write("\n# Detrend Kwargs: ")
        if detrend_kwargs is not None:
            for k in detrend_kwargs.keys():
                lc_out.write("{0} = {1} ".format(k, detrend_kwargs[k]))

        # Now write out the LCs
#        names = ["raw","bulk_trend","det","corr","corr_trend","sec"]
#        lc_table = dict(zip(names,[light_curve.time,
#                                   light_curve.flux,
#                                   light_curve.bulk_trend,
#                                   light_curve.det_flux,
#                                   light_curve.corrected_flux,
#                                   light_curve.corr_trend,
#                                   light_curve.sec_flux]))
        
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

        # Write out periodograms
        pg_out = open("{0}output_lcs/{1}_pgram.csv".format(base_path,
                                                       light_curve.name),"w")
        # Write out any relevant arguments and the date
        pg_out.write("# Generated on {0}".format(today))
        pg_out.write("\n# Best aperture: {0}".format(best_col))
        pg_out.write("\n# Detrend Kwargs: ")
        if detrend_kwargs is not None:
            for k in detrend_kwargs.keys():
                pg_out.write("{0} = {1} ".format(k, detrend_kwargs[k]))

        # Now write out the periodograms
        pg_out.write("\n{0}_period,{0}_power".format(light_curve.use))
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

def acf_one(filename, lc_dir, ap=None, output_f=None):
    lcs = at.read(lc_dir+filename)
    epic = filename.split("/")[-1][4:13]
    plotname = "{0}acf_plots/{1}_acf.png".format(base_path,
                                                filename.split("/")[-1][:-4])

    print lcs.dtype

    time = lcs["t"]
    unc_flux = np.ones_like(time)

    #best_col = "flux_{0:.1f}".format(ap)
    best_col = "corr"
    flux = lcs[best_col]
    """
    # E-K ACF
    C_EK, C_EK_err, bins = time_series.ACF_EK(time, flux, unc_flux, 
                                              bins=np.linspace(0,70,400))
    t_EK = 0.5*(bins[1:] + bins[:-1])

    # Plot the results
    fig = plt.figure(figsize=(10, 8))

    # plot the input data
    ax = fig.add_subplot(211)
    #ax.errorbar(t, y, dy, fmt='.k', lw=1)
    ax.plot(t, y,'k.', lw=1)
    ax.set_xlabel('t (days)')
    ax.set_ylabel('observed flux')

    # plot the ACF
    ax = fig.add_subplot(212)
    #ax.errorbar(t_EK, C_EK, C_EK_err, fmt='.k', lw=1)
    ax.plot(t_EK, C_EK, 'k.', lw=1)
    ax.set_xlim(0, 20)
    #ax.set_ylim(-0.003, 0.003)

    ax.set_xlabel('t (days)')
    ax.set_ylabel('E-K ACF')
    """

    # Standard ACF with gap filling
    t, y, dy = fix_kepler.fill_gaps(time,flux,unc_flux)
    acf_out = acf.run_acf(t, y, plot=True)
    plt.suptitle("EPIC {0}".format(epic))
    plt.savefig(plotname)
    plt.close("all")
    best_period, best_height, which, periods, heights = acf_out

    if output_f is not None:
        output_f.write("\n{},{}".format(filename, epic))
        if ap is not None:
            output_f.write(",{:.1f}".format(ap))
        else:
            output_f.write(",0.0")

        # E-K result

        # gap-filling result
        output_f.write(",{0:.3f},{1:.3f},{2}".format(best_period, best_height, 
                                                     which))

def acf_list(listname, lc_dir):
    lcs = at.read(listname)

    outfile = listname.split("/")[-1][:-4]+"_acf_results_{}.csv".format(today)
    output_f = open("{0}tables/{1}".format(base_path,outfile),"w")
    output_f.write("filename,EPIC,ap,period,height,which")
    for i, filename in enumerate(lcs["filename"]):
        new_filename = base_path+"output_lcs/"+filename.split("/")[-1][:-4]+"_lcs.csv"
        logging.warning("starting %d %s",i,new_filename)
        if os.path.exists(new_filename):
            acf_one(new_filename, lc_dir, ap=lcs["ap"][i], output_f=output_f)
        else: 
            logging.warning("SKIPPING %s",new_filename)
        logging.warning("done %d %s",i,new_filename)

    output_f.close()

def run_list(listname, lc_dir, detrend_kwargs=None):
    
    lcs = at.read(listname)

    outfile = listname.split("/")[-1][:-4]+"_results_{}.csv".format(today)
    output_f = open("{0}tables/{1}".format(base_path,outfile),"w")
    output_f.write("filename,EPIC,ap,lc")
    output_f.write(",init_prot,init_power,init99,init95")
    output_f.write(",init_0.5prot, init_0.5power, init_2prot, init_2power")
    output_f.write(",corr_prot,corr_power,corr99,corr95")
    output_f.write(",corr_0.5prot, corr_0.5power, corr_2prot, corr_2power")
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

    logging.basicConfig(level=logging.DEBUG)#, 
#                        format="%(asctime)s - %(name) - %(message)s")

    # need to fix that need for replacement, probably
    lc_dir = base_path.replace("k2spin","k2phot")+"lcs/"

    lc_file = "ktwo210408563-c04.csv"
    epic = "210408563"
    ap = 5


#    run_one(lc_file, 
#            lc_dir=lc_dir, ap=ap,
#            detrend_kwargs={"kind":"supersmoother","phaser":10})

#    run_list(base_path+"c4_lcs_aps.csv", lc_dir = "",
#             detrend_kwargs={"kind":"supersmoother","phaser":10})

#    acf_list(base_path+"c4_lcs_aps.csv", lc_dir = "")


    res = at.read(base_path+"tables/c4_lcs_aps_results_2015-09-19_comments.csv")
    plot.paper_lcs(epic,res[4])
    plt.savefig("/home/stephanie/my_papers/hyadesk2/sample_lc.eps",
                bbox_inches="tight")
    plt.savefig("/home/stephanie/Dropbox/plots_for_sharing/sample_lc.png",
                bbox_inches="tight")
