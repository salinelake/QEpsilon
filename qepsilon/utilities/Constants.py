
import torch as th
import numpy as np


class Constants(object):
    """Class whose members are fundamental constants.
    Inner Unit:
        Length: um
        Time: us
        Energy: hbar \times MHz
    """
    ## energy units in hbar*MHz
    hbar_MHz = 1.0
    hbar_Hz = 1e-6 * hbar_MHz
    eV = hbar_Hz / 6.582119569e-16  
    meV = 1e-3 * eV
    Ry = 13.605693123 * eV 
    mRy = 1e-3 * Ry  
    Hartree = 27.211386245981 * eV
    Joule = 6.241509e18 * eV  
    amu_cc =  931.49410372e6 * eV 

    ## physical constants
    kb = 8.6173303e-5 * eV # hbar * MHz / K
    speed_of_light = 299792458 # um/us
    amu = amu_cc / speed_of_light**2 # hbar * MHz / (um/us)^2
    epsilon0 = 5.526349406e-3 / eV * 1e4  # e^2(hbar*MHz*um)^-1
    elementary_charge = 1.0 # electron charge

    ## length units in um
    cm = 10000
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
    fs = 1e-9
     


class Constants_Metal(object):
    """Class whose members are fundamental constants.
    Inner Unit:
        Length: pm
        Time: ps
        Energy: hbar \times THz
    """
    ## energy units in hbar*THz
    hbar = 1.0 # hbar = hbar_THz * ps = 1.0
    hbar_THz = 1.0
    hbar_MHz = 1e-6 * hbar_THz
    hbar_Hz = 1e-6 * hbar_MHz
    eV = hbar_Hz / 6.582119569e-16  
    meV = 1e-3 * eV
    Ry = 13.605693123 * eV 
    mRy = 1e-3 * Ry  
    Hartree = 27.211386245981 * eV
    Joule = 6.241509e18 * eV  
    amu_cc =  931.49410372e6 * eV 

    ## physical constants
    kb = 8.6173303e-5 * eV # hbar * THz / K
    speed_of_light = 299792458 # pm/ps
    amu = amu_cc / speed_of_light**2 # hbar * THz / (pm/ps)^2
    epsilon0 = None  # e^2(hbar*THz*pm)^-1
    elementary_charge = 1.0 # electron charge

    ## length units in pm
    cm = 1e10
    mm = 1e9
    um = 1e6
    nm = 1e3
    Angstrom = 100
    pm = 1
    bohr_radius = 0.52917721092 * Angstrom 
    
    ## time units in ps
    ms = 1e9
    us = 1e6
    ns = 1e3
    ps = 1
    fs = 1e-3
    As = 1e-6
    time_au = 2.4188843265864e-2 * fs
     