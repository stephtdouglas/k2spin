"""Read in lightcurve files."""

import logging

import numpy as np
import astropy.io.ascii as at


def read_single_aperture(filename):
    """Read in one of AMC's K2 light curves, 

    inputs
    ------
    filename: string 
        should look like EPIC_205030103_xy_ap1.5_fixbox_cleaned.dat

    outputs
    -------
    time, flux, unc_flux, x_pos, y_pos, qual_flux: arrays

    aperture: float

    """
    # Read in the file
    lc = at.read(filename, delimiter=' ',data_start=1)
    split_filename = filename.split("/")[-1].split('_')
    logging.debug(split_filename)
    epicID = split_filename[1]
    aperture = split_filename[3]
    if aperture.startswith("ap"):
        aperture = aperture[2:]

    # Extract the useful columns
    time = lc["Dates"]
    flux = lc["Flux{}".format(aperture)]
    unc_flux = lc["Uncert{}".format(aperture)]
    x_pos = lc["Xpos"]
    y_pos = lc["Ypos"]
    qual_flux = lc["Quality"]

    aperture = float(aperture)

    # Return the columns
    return time, flux, unc_flux, x_pos, y_pos, qual_flux, aperture

def read_double_aperture(filename):
    """Read in one of AMC's K2 lc files with 2 aperture extractions.
    
    inputs
    ------
    filename: string 
        should look like EPIC_205030103_xy_ap#.#_#.#_fixbox.dat

    outputs
    -------
    time: array 

    flux, unc_flux: arrays, shape=(2, n_datapoints)
        A flux and uncertainty array for each aperture in the file

    x_pos, y_pos, qual_flux: arrays

    apertures: array, length=2
        The apertures contained in the file

    """
    # Read in the file
    lc = at.read(filename, delimiter=' ',data_start=1)
    split_filename = filename.split("/")[-1].split('_')
    logging.debug(split_filename)
    epicID = split_filename[1]

    # Extract the useful columns
    time = lc["Dates"]
    fluxes = np.array([lc["Flux5"], lc["Flux3"]])
    unc_fluxes = np.array([lc["Uncert5"], lc["Uncert3"]])
    apertures = np.array([5.,3.])
    x_pos = lc["Xpos"]
    y_pos = lc["Ypos"]
    qual_flux = lc["Quality"]

    # Return the columns
    return time, fluxes, unc_fluxes, x_pos, y_pos, qual_flux, apertures



def read_list(file_list):
    """Read in a list of lightcurve filenames."""
    pass

