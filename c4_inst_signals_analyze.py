
import logging, itertools, os, sys
from datetime import date

import numpy as np
import astropy.io.ascii as at
from astropy import table
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from astroML import time_series

logging.basicConfig(level=logging.WARNING)

today = date.today().isoformat()

def join_tables(base_tablename, n_arrayjobs, aps=[3,6], dates=None):
    """join a bunch of tables output by c4_inst_signals_run.py

    base_tablename: string, table name with _0_ instead of the relevant arrayid and r0 instead of the relevant aperture

    n_arrayjobs: integer, number of array jobs used

    """ 

    all_tables = []
    
    for arrayid in np.arange(36)+1:
        for i, ap in enumerate(aps):
            this_tablename = base_tablename.replace("_0_","_{0}_".format(
                                                    arrayid)).replace(
                                                    "r0","r{0}".format(ap))
            if dates is not None:
                this_tablename = this_tablename.replace(dates[0],dates[i])

            try:
                this_table = at.read(this_tablename, delimiter=",")
    
#                print(arrayid, ap)
    
                all_tables.append(this_table)

            except:
#                raise
                print("ERROR:", arrayid, ap)
                

    master_table = table.vstack(all_tables)
    print(len(master_table))
    print(master_table.dtype)

    return master_table



def analyze_output(master_table, aps):
    
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)

    logbins = np.logspace(-1,np.log10(70),100)

    prots = [master_table["init_prot"][master_table["init_power"]>master_table["init99"]],
             master_table["corr_prot"][master_table["corr_power"]>master_table["corr99"]],
             master_table["sec_prot"][master_table["sec_power"]>master_table["sec99"]]]
    powers = [master_table["init_power"][master_table["init_power"]>master_table["init99"]],
             master_table["corr_power"][master_table["corr_power"]>master_table["corr99"]],
             master_table["sec_power"][master_table["sec_power"]>master_table["sec99"]]]

    colors = ["k", "#ff0066", "#99ffff"]
    labels = ["init", "corr", "sec"]

    n, bins, patches = ax.hist(prots, bins=logbins, color=colors, 
                               histtype="bar",label=labels,lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Period (d)")
    ax.set_ylabel("N periods")
    ax.set_title("aps = {0}".format(aps))

    ax.axvline(0.25, color="Grey",zorder=-1, linestyle="--")
    for i in np.arange(1,6):
        ax.axvline(i, color="Grey", zorder=-11, linestyle="-.")
    ax.set_xlim(0.1,70)

    print(np.shape(n), np.shape(bins))
    maxy = max([max(n[i][(bins[:-1]>1) & (bins[:-1]<50)]) for i in range(3)])
    print(maxy)
    ax.set_ylim(0,maxy)

    plt.legend(loc="best")

    plt.savefig("plot_outputs/inst_signals_hist_{0}_{1}.pdf".format(today,str(aps)))
    plt.savefig("plot_outputs/inst_signals_hist_{0}_{1}.png".format(today,str(aps)))

    plt.clf()

    # Now plot periodogram power as well

    ax = plt.subplot(111)
    shapes = ["s","o","v"]
    for i in range(3):
        prot, power = prots[i], powers[i]
        shape = shapes[i]
        ax.scatter(prot, power, marker=shape, s=20, edgecolors="none",
                   c=colors[i],label=labels[i], alpha=0.1)

    ax.set_xscale("log")
    ax.set_xlabel("Period (d)")
    ax.set_ylabel("Periodogram Power")
    ax.set_title("aps = {0}".format(aps))

    ax.axvline(0.25, color="Grey",zorder=-1, linestyle="--")
    for i in np.arange(1,6):
        ax.axvline(i, color="Grey", zorder=-11, linestyle="-.")
    ax.set_xlim(0.1,70)
    ax.set_ylim(0,1)

    plt.legend(loc=2)

    plt.savefig("plot_outputs/inst_signals_powers_{0}_{1}.pdf".format(today,str(aps)))
    plt.savefig("plot_outputs/inst_signals_powers_{0}_{1}.png".format(today,str(aps)))

    plt.clf()

    # Now compare initial/corr Prot based on whether init=raw or init=det

    raw = np.where((master_table["lc"]=="raw") & 
                   (master_table["init_power"]>master_table["init99"]) &
                   (master_table["corr_power"]>master_table["corr99"]))[0]
    det = np.where((master_table["lc"]=="det") & 
                   (master_table["init_power"]>master_table["init99"]) &
                   (master_table["corr_power"]>master_table["corr99"]))[0]

    ax = plt.subplot(111)

    ax.plot(master_table["init_prot"][raw],master_table["corr_prot"][raw],
            "ko",alpha=0.2,label="init=raw")
    ax.plot(master_table["init_prot"][det],master_table["corr_prot"][det],
            "bs",alpha=0.2,mec="b",label="init=det")

    ax.legend(loc=2)

    ax.plot((0.1,70),(0.1,70),"r-")
    ax.set_xlim(0.1,70)
    ax.set_ylim(0.1,70)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Initial Prot")
    ax.set_ylabel("Corrected Prot")

    ax.set_title(str(aps))

    plt.savefig("plot_outputs/inst_signals_compare_{0}_{1}.pdf".format(today,str(aps)))
    plt.savefig("plot_outputs/inst_signals_compare_{0}_{1}.png".format(today,str(aps)))

    plt.close()

    for prot in prots:
        bad = np.where((prot>=20) & (prot<=25))[0]
        perc = len(bad)*1.0/len(prot)
        print("BAD {0}%".format(perc))


def plot_twenties(master_table, ap, min_power, max_power):

    print(ap, min_power, max_power)

    twenties = np.where((master_table["corr_power"]>master_table["corr99"])
                        & (master_table["corr_prot"]>20) & 
                        (master_table["corr_prot"]<30) & 
                        (master_table["corr_power"]>min_power) & 
                        (master_table["corr_power"]<=max_power))[0]

    tlen = len(twenties)
    nrand = 20

    rand_set = np.random.random_integers(0, len(twenties)-1, nrand)
    rand_twenties = np.unique(twenties[rand_set])


    plt.figure(figsize=(12,8))
    for i, ind in enumerate(rand_twenties):
        filename = "output_lcs/{0}_r{1}_lcs.csv".format(master_table["filename"][ind][:-4],ap)
        lcs = at.read(filename)

        period = master_table["corr_prot"][ind]
        phase = (lcs["t"] % period) / period

        ax = plt.subplot(5, 4, i+1)

        ax.plot(phase, lcs["corr"], "k.")
        ax.plot(phase, lcs["corr_trend"], ".", color="Grey")
#        ax.set_title("{0}, Power={1:.2f}".format(master_table["filename"][ind][4:13], master_table["corr_power"][ind]))
        ax.set_title("Prot={0:.2f}, Power={1:.2f}".format(master_table["corr_prot"][ind], master_table["corr_power"][ind]))

        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_yticklabels([])


    plt.tight_layout()
    plt.suptitle("ap = {0}".format(ap))

    plt.savefig("plot_outputs/phase_folded_{0}_{1}_P{2}-{3}.png".format(today,str(ap),min_power, max_power))

#    ax = plt.subplot(4, 5, 21)

    plt.close()
    


if __name__=="__main__":

    #    logging.basicConfig(level=logging.INFO)

    use_date = "2016-01-25"

    base_filename = "instrumental_signals_{0}_spin_r{1}_{2}.csv".format(0, 0, use_date)
    base_filename = "tables/{0}_results_{1}.csv".format(base_filename, use_date)  

    aps = [3,4,5,6]
    dates = ["2016-01-25", "2016-01-26", "2016-01-26", "2016-01-25"]

    master_table = join_tables(base_filename, 36, aps, dates)

    analyze_output(master_table, aps)

"""
    for ap, date in itertools.izip(aps, dates):

        print(ap,date)

#        if (ap<4) or (ap>5):
#            continue

        this_base = base_filename.replace(use_date, date)

        master_table = join_tables(this_base, 36, [ap])

        analyze_output(master_table, [ap])

#        for power in np.arange(0.2, 0.8, 0.1):
#            plot_twenties(master_table, ap, power, power+0.1)
"""
