
import logging

import numpy as np
import matplotlib.pyplot as plt

from k2spin import lc
from k2spin import k2io
from k2spin import plot

def run_one(filename)
    lc_out = k2io.read_double_aperture(lc_dir+lc_file)
    time, fluxes, unc_fluxes, x_pos, y_pos, qual_flux, apertures = lc_out
    
    light_curve = lc.LightCurve(time, fluxes[1], unc_fluxes[1], x_pos, y_pos,
                                name=lc_file[:-4])
    light_curve.choose_initial()
    light_curve.correct_and_fit()

    plot.plot_xy(light_curve.x_pos, light_curve.y_pos, light_curve.time,
                 light_curve.flux, "Raw Flux")
    plt.suptitle(light_curve.name, fontsize="large")
    plt.savefig(light_curve.name+"_xy_flux.png")
    plt.show()

def run_list(listname,lc_dir="/home/stephanie/code/python/k2spin/lcs/"):
    
    at.read("all_lcs.lst",names=["file"])

    for filename in at["file"]:
        run_one(lc_dir+filename)


if __name__=="__main__":

    logging.basicConfig(level=logging.DEBUG)

    lc_dir = "/home/stephanie/code/python/k2spin/lcs/"

#    lc_file = "EPIC_202533810_xy_ap5.0_3.0_fixbox.dat"
    lc_file = "EPIC_202521690_xy_ap5.0_3.0_fixbox.dat"

    run_one(lc_dir+lc_file)
