
import logging

import numpy as np

from k2spin import lc
from k2spin import k2io

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)

    lc_dir = "/home/stephanie/code/python/k2spin/lcs/"

    lc_file = "EPIC_202533810_xy_ap5.0_3.0_fixbox.dat"

    lc_out = k2io.read_double_aperture(lc_dir+lc_file)
    time, fluxes, unc_fluxes, x_pos, y_pos, qual_flux, apertures = lc_out
    
    light_curve = lc.LightCurve(time, fluxes[1], unc_fluxes[1], lc_file[:-4])
    light_curve.choose_initial()
