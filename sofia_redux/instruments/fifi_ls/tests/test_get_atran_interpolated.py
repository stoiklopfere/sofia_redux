# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pytest
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

#from sofia_redux.instruments.fifi_ls.get_atran import get_atran_interpolated

sys.path.append("..")
from sofia_redux.instruments.fifi_ls.get_atran import get_atran_interpolated

from sofia_redux.instruments.fifi_ls.tests.resources import FIFITestCase, get_scm_files


def test_get_atran_interpolated():
    
    atran_dir = os.path.join(os.getcwd(), "data")

    #create fake header
    header = fits.Header()
    header['ZA_START'] = "41.5"
    header['ZA_END'] = "41.6"
    header['ALTI_STA'] = "43014.0"
    header['ALTI_END'] = "43008.0"
    header['WVZ_OBS'] = "3.5"
    header['G_WAVE_B'] = "51.819"
    header['G_WAVE_R'] = "157.741"
    header['CHANNEL'] = "RED"
    header['G_ORD_B'] = "2"
    
    # default: gets alt/za/resolution from header, no unsmoothed data
    default = get_atran_interpolated(header, use_wv=True, atran_dir=atran_dir, get_unsmoothed=True)
    
    filenames = [
        "atran_43K_40deg_3pwv_40-300mum.fits",
        "atran_43K_40deg_4pwv_40-300mum.fits",
        "atran_43K_45deg_3pwv_40-300mum.fits",
        "atran_43K_45deg_4pwv_40-300mum.fits"
    ]
    
    filenames = [os.path.join(atran_dir, path) for path in filenames]
    
    # plt.plot(default[1][1], default[1][0])
    # plt.plot(default[1][0], default[1][1])
    # plot_all_atran_files(filenames)
    # plt.show()
    
    
    assert default is not None

def plot_single_atran_file(filename):
    with fits.open(filename) as hdul:
        # hdul.info()
        wavelength = hdul[0].data[0,:]
        transmission = hdul[0].data[1,:]
        
        print("plotting file ", filename)
        plt.plot(wavelength, transmission, label=filename)
        
def plot_all_atran_files(filenames):
    [plot_single_atran_file(fn) for fn in filenames]

    plt.title("Transmission over Wavelength")
    plt.xlabel("Wavelength [$\mu$m]")
    plt.ylabel("Transmission")
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.xlim(63.,63.5)

    plt.show()


if __name__ == "__main__":
    test_get_atran_interpolated()