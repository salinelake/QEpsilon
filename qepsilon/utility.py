"""
This module contains some utility functions.
"""

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


def compose(op_sequence: list[th.Tensor]):
    """
    This function composes a sequence of operators.
    Args:
        op_sequence: a list of operators.
    """
    op = op_sequence[0]
    for i in range(1, len(op_sequence)):
        op = th.kron(op, op_sequence[i])
    return op

def ABAd(A: th.Tensor, B: th.Tensor):
    """
    This function computes A @ B @ A.conj().T.
    """
    if A.dim() != B.dim():
        raise ValueError("The dimensions of A and B do not match.")
    if A.dim() == 2:
        return th.matmul(A, th.matmul(B, A.conj().T))
    elif A.dim() == 3:
        return th.matmul(A, th.matmul(B, A.conj().permute(0, 2, 1)))
    else:
        raise ValueError("A and B must be 2D or 3D tensors.")

def bin2idx(config: th.Tensor):
    """
    This function converts a binary numb to an index.
    """
    if config.dtype != th.int64:
        raise ValueError("Config must be an integer tensor (th.int64).")
    if config.dim() != 1:
        raise ValueError("Config must be a 1D tensor.")
    nd = config.shape[0]
    multiplier = th.tensor([2**(nd-i-1) for i in range(nd)], dtype=th.int64, device=config.device)
    return th.matmul(config, multiplier)
