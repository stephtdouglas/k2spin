
import logging

import numpy as np
import astropy.io.ascii as at
import matplotlib.pyplot as plt

from k2spin import lc
from k2spin import k2io
from k2spin import plot

def run_one(filename,lc_dir="/home/stephanie/code/python/k2spin/lcs/",
            num_apertures=2):

    if num_apertures==2:
        lc_out = k2io.read_double_aperture(lc_dir+filename)
        time, fluxes, unc_fluxes, x_pos, y_pos, qual_flux, apertures = lc_out
        flux = fluxes[1]
        unc_flux = unc_fluxes[1]
    else:
        lc_out = k2io.read_single_aperture(lc_dir+filename)
        time, flux, unc_flux, x_pos, y_pos, qual_flux, aperture = lc_out
    
    light_curve = lc.LightCurve(time, flux, unc_flux, x_pos, y_pos,
                                name=filename.split("/")[-1][:-4],
                                detrend_kwargs={"kind":"supersmoother",
                                                "phaser":8})
    light_curve.choose_initial()
    light_curve.correct_and_fit()

    plot.plot_xy(light_curve.x_pos, light_curve.y_pos, light_curve.time,
                 light_curve.flux, "Raw Flux")
    plt.suptitle(light_curve.name, fontsize="large")
    plt.savefig(light_curve.name+"_xy_flux.png")
#    plt.show()
    plt.close("all")

def run_list(listname,lc_dir="/home/stephanie/code/python/k2spin/lcs/",
             num_apertures=2):
    
    lcs = at.read(listname,names=["file"])

    for i, filename in enumerate(lcs["file"]):
        logging.info("%d %s",i,filename)
        run_one(filename,lc_dir,num_apertures=num_apertures)


if __name__=="__main__":

    logging.basicConfig(level=logging.DEBUG)

    lc_dir = "/home/stephanie/code/python/k2spin/lcs/"

#    lc_file = "EPIC_202533810_xy_ap5.0_3.0_fixbox.dat"
    lc_file = "EPIC_202521690_xy_ap5.0_3.0_fixbox.dat"

    run_one(lc_file,num_apertures=2)

#    run_list("test_usco.lst",num_apertures=2)
#    run_list("test_M35.lst",num_apertures=1)
