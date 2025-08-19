#!/usr/bin/python

import numpy as np


def combine_spectra(lcut, mcut, nspecies, ISOAP, divfac):
    """
    Python implementation of the combine_spectra Fortran subroutine.
    
    Combines power spectra for L=0 SOAP kernel computation.
    
    Parameters:
    -----------
    lcut : int
        Maximum angular momentum channel
    mcut : int
        Number of spherical harmonic components (2*lcut+1)
    nspecies : int
        Number of atomic species
    ISOAP : complex array, shape (nspecies, lcut+1, mcut, mcut)
        SOAP power spectrum data
    divfac : float array, shape (lcut+1,)
        Division factors (1/(2*l+1) for each l)
        
    Returns:
    --------
    PS : complex
        Combined power spectrum value
    """
    PS = 0.0 + 0.0j
    
    for l in range(lcut + 1):
        PSS = 0.0 + 0.0j
        
        for im in range(2 * l + 1):
            for ik in range(2 * l + 1):
                for ix in range(nspecies):
                    dcix = np.conj(ISOAP[ix, l, im, ik])
                    PKM = 0.0 + 0.0j
                    
                    for iy in range(nspecies):
                        PKM += ISOAP[iy, l, im, ik]
                    
                    PSS += PKM * dcix
        
        PS += PSS * divfac[l]
    
    return PS


def fill_spectra(lval, lcut, mcut, nspecies, ISOAP, CG2):
    """
    Python implementation of the fill_spectra Fortran subroutine.
    
    Fills power spectra for spherical tensor SOAP kernel computation (L > 0).
    
    Parameters:
    -----------
    lval : int
        Target angular momentum quantum number
    lcut : int
        Maximum angular momentum channel
    mcut : int
        Number of spherical harmonic components (2*lcut+1)
    nspecies : int
        Number of atomic species
    ISOAP : complex array, shape (nspecies, lcut+1, mcut, mcut)
        SOAP power spectrum data
    CG2 : float array, shape (lcut+1, lcut+1, mcut, mcut, 2*lval+1, 2*lval+1)
        Clebsch-Gordan coefficients
        
    Returns:
    --------
    PS : complex array, shape (2*lval+1, 2*lval+1)
        Filled power spectrum matrix
    """
    PS = np.zeros((2 * lval + 1, 2 * lval + 1), dtype=complex)
    
    for l1 in range(lcut + 1):
        for l in range(lcut + 1):
            for im in range(2 * l + 1):
                for ik in range(2 * l + 1):
                    for ix in range(nspecies):
                        dcix = np.conj(ISOAP[ix, l, im, ik])
                        
                        # Calculate valid ranges for iim and iik
                        iim_min = max(0, lval + im - l - l1)
                        iim_max = min(2 * lval, lval + im - l + l1)
                        iik_min = max(0, lval + ik - l - l1)
                        iik_max = min(2 * lval, lval + ik - l + l1)
                        
                        for iim in range(iim_min, iim_max + 1):
                            for iik in range(iik_min, iik_max + 1):
                                PKM = 0.0 + 0.0j
                                
                                # Calculate indices for ISOAP access
                                im_idx = im - l + l1 - iim + lval
                                ik_idx = ik - l + l1 - iik + lval
                                
                                # Check bounds to prevent array access errors
                                if (0 <= im_idx < mcut and 0 <= ik_idx < mcut):
                                    for iy in range(nspecies):
                                        PKM += ISOAP[iy, l1, im_idx, ik_idx]
                                    
                                    PS[iim, iik] += (PKM * 
                                                   CG2[l, l1, im, ik, iim, iik] * 
                                                   dcix)
    
    return PS
