"""Plot lightcurves, periodograms, and phased lightcurves."""

import logging
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import astropy.io.ascii as at

from k2spin import utils

color1 = "k"
color2 = "#FF4D4D"
shape1 = "o"
shape2 = "s"

def setup_plots():
    """Set up a four-paneled figure."""

    fig = plt.figure(figsize=(8,10))

    base_grid = (11,1)

    # Top axis - full light-curve
    ax1 = plt.subplot2grid(base_grid,(1,0),rowspan=3)
    ax1.set_xlabel("Time (d)")
    ax1.set_ylabel("Normalized Counts")

    # Second axis - periodogram (or whatever period-finding output)
    ax2 = plt.subplot2grid(base_grid,(4,0),rowspan=3)
    ax2.set_xlabel("Period (d)")
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_ylabel("Power (Max=1)")

    # Bottom axes should be squished against each other if possible
    # But that appears to be a major pain
    # Phased lightcurve and residuals
    ax3 = plt.subplot2grid(base_grid,(7,0),rowspan=2)
    #ax3.tick_params(labelbottom=False)
    ax3.set_xlabel("Phased time (d)")
    ax3.set_ylabel("Normalized Counts")

    ax4 = plt.subplot2grid(base_grid,(9,0),rowspan=2)
    ax4.set_xlabel("Phased time (d)")
    ax4.set_ylabel("Residuals")

    return fig, [ax1, ax2, ax3, ax4]


def plot_one(lightcurve, periodogram, best_period, power_threshold, data_label,
             residuals=None, aliases=None, axes_list=None,plot_title=None,
             phase_by=None, **plot_kwargs):
    """Plot input lightcurve, input periodogram, and input phased lightcurve
    INPUTS ARE DIFFERENT FROM REST OF CODE

    inputs
    ------
    lightcurve: arraylike (3, num_points)
        time, flux, unc_flux

    periodogram: arraylike (2, num_periods)
        periods_to_test, periodogram

    best_period: float

    power_threshold: float

    data_label: string
        name for the dataset being plotted

    residuals: array-like (default None)
        if provided, will be plotted in the bottom panel

    aliases: array-like (default None)
        if provided, will be plotted as vertical dotted lines on the 
        periodogram

    axes_list: optional, length=4
        if not provided, a new figure will be created.
        Axes should correspond to lightcurve, periodogram, phased lightcurve, 
        and phased residuals

    phase_by: float (default=None)
        if not None, the phased lightcurve will use this period 
        instead of best_period
    """

    # If axes_list is None, create the figure and axes with setup_plots
    if axes_list is None:
        fig, axes_list = setup_plots()
        plot_color, plot_marker = color1, shape1
    elif len(axes_list)!=4:
        logging.warning("Incorrect number of axes! Setting up new figure")
        fig, axes_list = setup_plots()
        plot_color, plot_marker = color1, shape1
    else:
        fig = plt.gcf()
        plot_color, plot_marker = color2, shape2

    # Top panel: lightcurve
    axes_list[0].plot(lightcurve[0], lightcurve[1], lw=0, 
                      marker=plot_marker,  mec=plot_color,
                      mfc=plot_color, ms=2, label=data_label)
    # Also plot vertical lines for the period if it's longer than 2 days
    # (shorter than that isn't really comprehensible)
    if best_period>=2.0:
        phase_mult = np.arange(min(lightcurve[0]), max(lightcurve[0]), 
                               best_period)
        for pm in phase_mult:
            axes_list[0].axvline(pm, color=plot_color, linestyle="--",
                                     zorder=-100, alpha=0.75)

    # Middle panel: periodogram
    logging.debug("plot periodograms")
    axes_list[1].plot(periodogram[0], periodogram[1], color=plot_color)
    axes_list[1].axvline(best_period, color=plot_color, linestyle="--")

    axes_list[1].set_xlim(xmin=0.1)

    # Plot the power threshold, if it would be visible
    ymax = axes_list[1].get_ylim()[1]
    logging.debug("ymax %f", ymax)
    if (((type(power_threshold)==float) or (type(power_threshold)==int))
        and power_threshold<ymax):
        # Only one threshold
        axes_list[1].axhline(power_threshold,color="Grey",ls="-.")
    elif (type(power_threshold)==float and 
          power_threshold>=ymax):
        logging.debug("peak lower than threshold")
    else:
        for threshold in power_threshold:
            logging.debug(threshold)
            axes_list[1].axhline(threshold,color=plot_color,ls="-.")

    # Plot aliases, if provided
    if aliases is not None:
        for alias in aliases:
            axes_list[1].axvline(alias, color=plot_color, linestyle=":")

    # Plot harmonics of the best period
    harmonics = np.array([0.5, 2])*best_period
    for harm in harmonics:
        axes_list[1].axvline(harm, color=plot_color, linestyle="--",
                             alpha=0.5)

    # Plot harmonics of the thruster firing time
    harmonics = np.append(0.125,np.arange(0.25,2.1,0.25))
    for harm in harmonics:
        axes_list[1].axvline(harm, color="LightGrey", linestyle="-",
                             alpha=0.5, zorder=-111)

    # Bottom panels: phased lightcurve
    if phase_by is None:
        phase_by = best_period
    phased = utils.phase(lightcurve[0], phase_by)
    axes_list[2].plot(phased, lightcurve[1], lw=0, marker=plot_marker, 
                      mec=plot_color, mfc=plot_color, ms=2)
    axes_list[2].set_xlim(0,phase_by)
    if residuals is not None:
        axes_list[3].plot(phased, residuals, lw=0, marker=plot_marker, 
                          mec=plot_color, mfc=plot_color, ms=2)
        axes_list[3].set_xlim(0,phase_by)

#    plt.tight_layout()

    return fig, axes_list

def compare_multiple(lightcurves, periodograms, best_periods, thresholds,
                     data_labels, aliases=None, phase_by=None, kwarg_sets=None):
    """Plot multiple sets of lightcurves onto a three-paneled plot.

    There must be at least 2 lightcurves, periodograms, and best_periods,
    or this will fail. The less-processed lc should be first in each. 
    """

    # Pass each set of inputs to plot_one
    if aliases is not None:
        al0 = aliases[0]
        al1 = aliases[1]
    else: 
        al0, al1 = None, None

    fig, ax_list = plot_one(lightcurves[0], periodograms[0], best_periods[0], 
                            thresholds[0], aliases=al0,
                            data_label=data_labels[0], phase_by=phase_by)
    #interp_lc = np.interp(lightcurves[1][0], lightcurves[0][0],  
    #                      lightcurves[0][1]
    #residuals = lightcurves[1][1] - lightcurves[0][1]
    fig, ax_list = plot_one(lightcurves[1], periodograms[1], best_periods[1], 
                            thresholds[1], aliases=al1,
                            data_label=data_labels[1], phase_by=phase_by,
                            axes_list=ax_list)

#    minf = min(min(lightcurves[0][1]), min(lightcurves[1][1]))
#    maxf = max(max(lightcurves[0][1]), max(lightcurves[1][1]))
#    ax_list[2].set_ylim(minf,maxf)

    ax_list[2].set_ylim(ax_list[0].get_ylim())

    leg = ax_list[0].legend(loc=3, ncol=2, mode="expand", numpoints=3, 
                            bbox_to_anchor=(0., 1.02, 1., .102))
    ltexts = leg.get_texts()
    ltexts[0].set_color("k")
    ltexts[1].set_color("#FF4D4D")

    plt.subplots_adjust(bottom=0.06, top=0.995, hspace=1.8)

    return fig, ax_list


# Other notes pulled from Kevin's .pro file, keeping for now for reference
    # plot the raw light curve, with the bulk trend overlaid 

    # plot the periodograms for both light curves as well.

    # plot the periodograms for both light curves as well.


            # Plot the original phased lightcurve

            # Overplot original - (whitened using best-fit period)
            # I think this is plotting the residual between the 
            # original and whitened lightcurve?
    
            # Plot the phased whitened lightcurve



                # Plot the (pre-whitened) light curve

                # Plot periodograms for both lightcurves


    # plot the raw light curve, with the bulk trend overlaid 


def plot_xy(xpix, ypix, time, color_by, color_label):
    """Plot the position as a function of time 
    and another value as a function of position."""

    fig = plt.figure(figsize=(8,10))

    base_grid = (10,8)

    # Set up the axes
    # Top axis - X position as a function of time
    ax1 = plt.subplot2grid(base_grid, (0, 0), rowspan=3, colspan=8)
    ax1.set_xlabel("Time (d)")
    ax1.set_ylabel("Centroid X")
    
    # Second axis - Y position as a function of time
    ax2 = plt.subplot2grid(base_grid, (3, 0), rowspan=3, colspan=8)
    ax2.set_xlabel("Time (d)")
    ax2.set_ylabel("Centroid Y")

    # Third axis - a square, showing flux as a function of X/Y
    ax3 = plt.subplot2grid(base_grid, (6, 0), rowspan=4, colspan=4)
    ax3.set_xlabel("Centroid X")
    ax3.set_ylabel("Centroid Y")
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)

    # Fourth axis - a square, showing color_by as a function of X/Y
    ax4 = plt.subplot2grid(base_grid, (6, 4), rowspan=4, colspan=4)
    ax4.set_xlabel("Centroid X")
    ax4.set_ylabel("Centroid Y")
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)

    axes_list = [ax1, ax2, ax3, ax4]

    # Now plot
    axes_list[0].plot(time, xpix, shape1, color=color1, ms=2)
    axes_list[1].plot(time, ypix, shape1, color=color1, ms=2)

    # Flux
    xyt = axes_list[2].scatter(xpix, ypix, c=time, edgecolor="none", 
                               alpha=0.5, vmin=np.percentile(time, 5), 
                               vmax=np.percentile(time, 95),
                               cmap="gnuplot")

    cbar_ticks = np.asarray(np.percentile(time,np.arange(10,100,20)),int)
    cbar1 = fig.colorbar(xyt, cax=cax3, ticks=cbar_ticks)
    cbar1.set_label("Time (d)")

    # color_by
    xyp = axes_list[3].scatter(xpix, ypix, c=color_by, edgecolor="none", 
                               alpha=0.5, vmin=np.percentile(color_by, 5), 
                               vmax=np.percentile(color_by, 95),
                               cmap="gnuplot")

    cbar_ticks = np.asarray(np.percentile(color_by,np.arange(10,100,20)),int)
    cbar2 = fig.colorbar(xyp, cax=cax4, ticks=cbar_ticks)
    cbar2.set_label(color_label)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

def paper_lcs(epic, output_row, campaign=4):
    """ Plot light curves and relevant periodograms for the paper."""

    logging.info("plot paper lcs %s", epic)

    init_lc = output_row["lc"]
    if init_lc=="raw":
        init_title = "Raw"
        initi=0
    else:
        init_title = "Detrended"
        initi=1
    logging.debug("init: %s %s %d",init_lc,init_title, initi)

#    final_lc = output_row["use"]
#    if final_lc=="corr":
#        final_title = "Corrected"
#    elif final_lc=="sec":
#        final_title = "Secondary"
#    else:
#        final_title = init_title

    lc_dir = "/home/stephanie/code/python/k2spin/output_lcs/"
    lcs = at.read("{0}ktwo{1}-c0{2}_lcs.csv".format(lc_dir, epic,
                                                    campaign))
    pgrams = at.read("{0}ktwo{1}-c0{2}_pgram.csv".format(lc_dir, 
                                                         epic, 
                                                         campaign))

    # portrait full-page plot (with a little room for a caption)
    fig = plt.figure(figsize=(7.5,9))
    split_grid = gridspec.GridSpec(3,1,height_ratios=[4,1,2])

    # Top half - four light curves spanning the entire width
    # and stacked on top of each other (share x-axis)
    # raw, detrended, corrected, second search
    lc_grid = gridspec.GridSpecFromSubplotSpec(4,1,
                                               subplot_spec=split_grid[0],
                                               hspace=0)
    # to access axes: ax = plt.subplot(grid[i(,j)])

    # Set up colormap to make ensure contrast/readability
    c = mcolors.ColorConverter().to_rgb
    cmap=plt.cm.cubehelix
    color_norm = Normalize(vmin=0,vmax=5)
    scalar_map = cm.ScalarMappable(norm=color_norm,cmap=cmap)
    mfcolors1 = scalar_map.to_rgba(np.arange(5))

    # Set up column names and colors (short names for labeling without overlap)
    cols = ["raw","det","corr","sec"]
    colors = mfcolors1[np.array([0,2,1,3])]#["k","r","b","g"]
    ctitles = ["Raw","Detrend","Correct","Second"]
    axes = []
    for i, colname in enumerate(cols):
        # Create subplot and plot appropriate light curve
        axes.append(plt.subplot(lc_grid[i]))
        axes[i].plot(lcs["t"],lcs[colname],".",color=colors[i],
                     label=ctitles[i])
        # Stacking them together, so don't need labels except for the last
        axes[i].tick_params(labelbottom=False, labelleft=False, labelright=True)
        axes[i].set_xlim(min(lcs["t"]))
        axes[i].get_yaxis().get_major_formatter().set_useOffset(False)
        plt.setp(axes[i].get_yticklabels()[::2], visible=False)
        if i>0:
            plt.setp(axes[i].get_yticklabels()[-1], visible=False)
        # yticklabels were taking up too much space, but probably do need smth
        #axes[i].set_yticklabels([])
#        leg = axes[i].legend(loc=2,numpoints=1,borderaxespad=0,
#                             frameon=False)
#        for text in leg.get_texts():
#            text.set_color(colors[i])
        axes[i].set_ylabel(ctitles[i],color=colors[i])

    # Plot the bulk trend used for initial detrending on the first panel
    axes[0].plot(lcs["t"],lcs["bulk_trend"],color=colors[1],lw=2)
    # Plot the median flux used to generate the corrected lightcurve
    #axes[initi].plot(lcs["t"],lcs["med"],color=colors[2],lw=2)
    # Plot the periodic trend removed to generate the secondary lightcurve
    axes[2].plot(lcs["t"],lcs["corr_trend"],color=colors[3],lw=2)

    # Add x-axis labels to the bottom panel
    axes[3].tick_params(labelbottom=True)
    axes[3].set_xlabel("Time (d)")
    #lc_grid.update(hspace=0)


    # Bottom half - three columns of three plots each
    # bottom 2 plots should be stacked together.
    # columns: raw/det (whichever selected), corrected, second search
    # rows: periodogram, phased lc, residuals
    pgram_grid = gridspec.GridSpecFromSubplotSpec(1,3,wspace=0,
                                        subplot_spec=split_grid[1])
    phased_grid = gridspec.GridSpecFromSubplotSpec(2,3,hspace=0,wspace=0,
                                        subplot_spec=split_grid[2])
    # pgram axes
    paxes = []
    # Phased axes
    haxes = []
    # Only plotting 3 columns here
    pcols = [init_lc, "corr","sec"]
    pcolors = [colors[initi],colors[2],colors[3]]
    ptitles = [init_title,"Corrected","Secondary"]
    # Want to set the same y-scaling for all 3 periodograms
    maxy = 0
    for i, colname in enumerate(pcols):
        # Initialize axis and plot appropriate periodogram
        paxes.append(plt.subplot(pgram_grid[i]))
        paxes[i].plot(pgrams[colname+"_period"],
                      pgrams[colname+"_power"],
                      "-",color=pcolors[i])
        paxes[i].set_xlabel(r"$P_{rot}$ (d)")
        paxes[i].set_xscale("log")
        paxes[i].set_xlim(0.1,70)
        paxes[i].set_xticklabels([0,0.1,1,10])
        paxes[i].set_title(ptitles[i],color=pcolors[i])
        # Only need yaxis on ends
        paxes[i].set_yticklabels([])

        # Retrieve appropriate period and 99% threshold from the output file:
        if i==0:
            colname2 = "init"
        else:
            colname2 = colname
        period = output_row[colname2+"_prot"]
        # Plot a triangle at the fundmantal period
        tri_y = output_row[colname2+"_power"]*1.1
        paxes[i].plot(period,tri_y,"v",mfc="DarkGrey")
        #paxes[i].axvline(period,ls="-", color="Grey")
        # And a horizonal line at the 99% significance threshold
        paxes[i].axhline(output_row[colname2+"99"],ls="--",color="Grey")

        maxy = max(maxy,paxes[i].get_ylim()[-1])

        # Now the axes for the phased light curve and residuals
        # Share axes across columns
        if i==0:
            s0,s1 = None,None
        else:
            s0,s1 = haxes[0][0],haxes[0][1]
        haxes.append([plt.subplot(phased_grid[0,i],sharey=s0),
                      plt.subplot(phased_grid[1,i],sharey=s1)])
        phased_t = ((lcs["t"]-lcs["t"][0]) % period) / period

        # Plot phased light curve
        if initi==0:
            haxes[i][0].plot(phased_t,lcs[colname]/np.median(lcs[colname]),
                             ".",color=pcolors[i])
        else:
            haxes[i][0].plot(phased_t,lcs[colname],".",color=pcolors[i])
        haxes[i][0].tick_params(labelbottom=False,labelleft=False)
        haxes[i][0].set_yticklabels([])
        haxes[i][0].set_title(r"$P_{rot}$=%.2f" % period,
                              color=pcolors[i])
        haxes[i][0].set_xlim(0,1)

        # Plot phased residuals (eventually...)
        haxes[i][1].set_xlim(0,1)
        haxes[i][1].set_xlabel("Phase")
        haxes[i][1].tick_params(labelleft=False)


    # Overplot smoothed periodic curve on initial lightcurve
    phased_t = ((lcs["t"]-lcs["t"][0]) % output_row["init_prot"]) / output_row["init_prot"]
    one_cycle = np.argsort(phased_t)
    haxes[0][0].plot(phased_t[one_cycle], 
                     lcs["init_trend"][one_cycle],color=pcolors[1],lw=2)
    if initi==0:
        residuals = lcs[init_lc]/np.median(lcs[init_lc]) - lcs["init_trend"]
    else:
        residuals = lcs[init_lc] - lcs["init_trend"]
    haxes[0][1].plot(phased_t[one_cycle], 
                     residuals[one_cycle],".",color=pcolors[0])
    haxes[0][1].axhline(0,color=pcolors[1],lw=2)

    # Overplot smoothed periodic curve on corrected lightcurve
    phased_t = ((lcs["t"]-lcs["t"][0]) % output_row["corr_prot"]) / output_row["corr_prot"] 
    one_cycle = np.argsort(phased_t)
    haxes[1][0].plot(phased_t[one_cycle], 
                     lcs["corr_trend"][one_cycle],color=pcolors[2],lw=2)
    residuals = lcs["corr"] - lcs["corr_trend"]
    haxes[1][1].plot(phased_t[one_cycle], 
                     residuals[one_cycle],".",color=pcolors[1])
    haxes[1][1].axhline(0,color=pcolors[2],lw=2)

    # Overplot smoothed periodic curve on secondary lightcurve
    phased_t = ((lcs["t"]-lcs["t"][0]) % output_row["sec_prot"]) / output_row["sec_prot"]
    one_cycle = np.argsort(phased_t)
    haxes[2][0].plot(phased_t[one_cycle], 
                     lcs["sec_trend"][one_cycle],color=mfcolors1[-1],lw=2)
    residuals = lcs["sec"] - lcs["sec_trend"]
    haxes[2][1].plot(phased_t[one_cycle], 
                     residuals[one_cycle],".",color=pcolors[2])
    haxes[2][1].axhline(0,color=mfcolors1[-1],lw=2)

    # Set tick/axis labels for periodograms
    paxes[0].set_ylabel("Power (max=1)")
    paxes[2].tick_params(labelleft=False,labelright=True)
    if maxy<=0.5:
        tick_step = 0.1
    else:
        tick_step = 0.2
    for ax in paxes:
        ax.set_ylim(0,maxy)
        ax.set_yticks(np.arange(0,maxy+0.01,0.1),minor=True)
        ax.set_yticks(np.arange(0,maxy+0.01,tick_step))
    paxes[2].set_yticklabels(np.arange(0,maxy+0.01,tick_step))

    # Set tick/axis labels for phased light curves
    haxes[0][0].set_ylabel("Flux")
    haxes[0][1].set_ylabel("Residuals")
    for i in (0,1):
        ticks = haxes[0][i].get_yticks()
        haxes[2][i].set_yticklabels(ticks)
        haxes[2][i].tick_params(labelright=True)
        plt.setp(haxes[2][i].get_yticklabels()[::2], visible=False)
    plt.setp(haxes[2][1].get_yticklabels()[-1], visible=False)

    plt.subplots_adjust(left=0.08,right=0.95,top=0.95,
                        hspace=0.4,wspace=0.2)
    plt.suptitle("EPIC {0}".format(epic),fontsize="x-large")
