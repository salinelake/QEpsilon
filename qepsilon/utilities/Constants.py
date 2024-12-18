
import torch as th
import numpy as np


class Constants(object):
    """Class whose members are fundamental constants.
    Inner Unit:
        Length: um
        Time: us
        Energy: hbar \times Hz
    """
    ## energy units in hbar*MHz
    hbar_MHz = 1.0
    hbar_Hz = 1e-6 * hbar_MHz
    eV = hbar_Hz / 6.582119569e-16  
    Ry = 13.605693123 * eV 
    mRy = 1e-3 * Ry  
    Joule = 6.241509e18 * eV  
    amu_cc =  931.49410372e6 * eV 

    ## physical constants
    kb = 8.6173303e-5 * eV # hbar * MHz / K
    speed_of_light = 299792458 # um/us
    amu = amu_cc / speed_of_light**2 # hbar * MHz / (um/us)^2
    epsilon0 = 5.526349406e-3 / eV * 1e4  # e^2(hbar*MHz*um)^-1
    elementary_charge = 1.0 # electron charge

    ## length units in um
    mm = 1000
    um = 1.0
    nm = 1e-3
    Angstrom = 1e-4
    bohr_radius = 0.52917721092 * Angstrom 
    
    ## time units in us
    ms = 1000
    us = 1
    ns = 1e-3
    ps = 1e-6
     


    # ## magnetic units
    # muB = 1.0   # Bohr magneton
    # Tesla = 5.7883818060e-5 # eV/muB
    # electron_g_factor = 2.00231930436 # dimensionless
    # electron_gyro_ratio = electron_g_factor / hbar # muB/eV/ps
