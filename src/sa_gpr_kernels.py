#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import utils.kernels
import utils.kern_utils
import utils.parsing
import argparse
import sys
import numpy as np
from itertools import product

# THIS IS A WRAPPER THAT CALLS PYTHON SCRIPTS TO BUILD L-SOAP KERNELS.

def sagpr_kernel(ftrs=None, npoints=None, lval=None, sg=0.3, lc=6, rcut=3.0, 
               cweight=1.0, vrb=True, centers=None, nlist=None, atomic=False, 
               extrap=False, ntest=1):
    """
    Build SA-GPR kernels with specified parameters.
    
    This function computes symmetry-adapted SOAP kernels for machine learning
    of tensorial properties. It supports both standard and extrapolation modes,
    and can compute either global kernels or atomic environment kernels.
    
    Parameters:
    -----------
    ftrs : list or str
        List of ASE Atoms objects or path to features file containing 
        atomic coordinates and properties
    npoints : int, optional
        Number of data points to process. If None, uses all available data
    lval : int
        Order of the spherical tensor (angular momentum quantum number)
    sg : float, default=0.3
        Gaussian width (sigma) for SOAP kernels
    lc : int, default=6
        Angular cutoff (lcut) for SOAP kernels
    rcut : float, default=3.0
        Radial cutoff value for environment
    cweight : float, default=1.0
        Central atom weight in SOAP calculation
    vrb : bool, default=True
        Enable verbose output
    centers : list, optional
        List of atomic species to center environments on. If None, uses all species
    nlist : list, optional
        List of kernel powers for calculation. If None, defaults to [0]
    atomic : bool, default=False
        If True, compute kernels for individual atomic environments
    extrap : bool, default=False
        If True, use extrapolation mode for test/train split
    ntest : int, default=1
        Number of test points when using extrapolation mode
    
    Returns:
    --------
    None
        Function saves kernel files to disk with appropriate naming convention
    """
    
    # Initialize default values based on parsing.py defaults
    if nlist is None:
        nlist = [0]
    if centers is None:
        centers = []
    
    # Validate required parameters
    if lval is None:
        raise ValueError("lval (angular momentum quantum number) is required")
    if ftrs is None:
        raise ValueError("ftrs (features/coordinates) is required")
    
    # Handle string input for features file
    if isinstance(ftrs, str):
        from ase.io import read
        ftrs = read(ftrs, ':')
    
    # Set npoints if not provided
    if npoints is None:
        npoints = len(ftrs)

    # Print kernel computation information
    print("""
    NUMBER OF CONFIGURATIONS = {np}
    Building the symmetry adapted SOAP kernel for L = {lv}
    
    Kernel hyper-parameters:
    ---------------------------
    Gaussian width = {sg}
    Angular cutoff = {lc}
    Environment cutoff = {rcut}
    Central atom weight = {cw}
    """.format(np=npoints, lv=lval, sg=sg, lc=lc, rcut=rcut, cw=cweight))

    # Build kernels using the main kernel computation routine
    if extrap == False:
        # Standard mode: compute kernels for all data points
        [centers, atom_indexes, natmax, nat, kernels] = utils.kernels.build_kernels(
            lval, ftrs, npoints, sg, lc, rcut, cweight, vrb, centers, nlist)

        # Transformation matrices to convert from complex to real spherical harmonics
        CR = utils.kern_utils.complex_to_real_transformation([2*lval+1])[0]
        CC = np.conj(CR)  # Complex conjugate for transformation
        CT = np.transpose(CR)  # Transpose for back-transformation

        if atomic:
            # Compute atomic environment kernels
            # Transform local kernels from complex to real representation
            kloc = np.zeros((npoints, npoints, natmax, natmax,
                            2*lval+1, 2*lval+1), dtype=float)
            for i in range(npoints):
                for j in range(npoints):
                    for ii in range(nat[i]):
                        for jj in range(nat[j]):
                            # Apply complex-to-real transformation: CC * kernel * CT
                            kloc[i, j, ii, jj] = np.real(
                                np.dot(np.dot(CC, kernels[0][i, j, ii, jj]), CT))

            # Get indexes for atoms of the same type
            iat = 0
            atom_idx = {}
            for k in centers:
                atom_idx[k] = []
                for il in atom_indexes[0][k]:
                    atom_idx[k].append(iat)
                    iat += 1

            # Build kernels for identical atomic species
            katomic = {}
            natspe = {}
            ispe = 0
            for k in centers:
                natspe[ispe] = len(atom_idx[k])
                # Create kernel matrix for this atomic species
                katomic[ispe] = np.zeros(
                    (natspe[ispe]*npoints, natspe[ispe]*npoints, 2*lval+1, 2*lval+1), float)
                irow = 0
                for i in range(npoints):
                    for ii in atom_idx[k]:
                        icol = 0
                        for j in range(npoints):
                            for jj in atom_idx[k]:
                                katomic[ispe][irow, icol] = kloc[i, j, ii, jj]
                                icol += 1
                        irow += 1
                ispe += 1

            # Save atomic kernels to files
            envfile = []
            for k in centers:
                filename = "kernel{lv}_atom{k}_nconf{np}_sigma{sg}_lcut{lc}_cutoff{rc}_cweight{cw}.npy".format(
                    np=npoints, k=k, lv=lval, sg=sg, lc=lc, rcut=rcut, cw=cweight)
                envfile.append(open(filename, "w"))
            nspecies = len(centers)
            for ispe in range(nspecies):
                np.save(envfile[ispe], katomic[ispe])

        else:
            # Compute global kernels for different powers (n values)
            for n in range(len(nlist)):
                # Transform kernel from complex to real representation
                kernel = np.zeros(
                    (npoints, npoints, 2*lval+1, 2*lval+1), dtype=float)
                for i, j in product(range(npoints), range(npoints)):
                    # Apply complex-to-real transformation
                    kernel[i, j] = np.real(
                        np.dot(np.dot(CC, kernels[1+n][i, j]), CT))
                
                # Save kernel to file with descriptive filename
                kernel_file = "kernel{lv}_{np}_sigma{sg}_lcut{lc}_cutoff{rcut}_cweight{cw}_n{n}.npy".format(
                    lv=lval, np=npoints, sg=sg, lc=lc, rcut=rcut, cw=cweight, n=nlist[n])
                np.save(kernel_file, kernel)

    else:
        # Extrapolation mode: separate test and training data
        [centers, atom_indexes, natmax, nat, kernels] = utils.extra_kernels.build_kernels(
            lval, ftrs, npoints, sg, lc, rcut, cweight, vrb, centers, nlist, ntest)

        # Transformation matrices to convert from complex to real spherical harmonics
        CR = utils.kern_utils.complex_to_real_transformation([2*lval+1])[0]
        CC = np.conj(CR)
        CT = np.transpose(CR)

        if atomic:
            # Compute atomic environment kernels for extrapolation
            # Transform local kernels from complex to real representation
            kloc = np.zeros((ntest, npoints-ntest, natmax,
                            natmax, 2*lval+1, 2*lval+1), dtype=float)
            for i in range(ntest):
                for j in range(npoints-ntest):
                    for ii in range(nat[i]):
                        for jj in range(nat[j]):
                            # Apply complex-to-real transformation
                            kloc[i, j, ii, jj] = np.real(
                                np.dot(np.dot(CC, kernels[0][i, j, ii, jj]), CT))

            # Get indexes for atoms of the same type
            # Test molecules are at the beginning of the list
            iat = 0
            atom_idx_test = {}
            for k in centers:
                atom_idx_test[k] = []
                for il in atom_indexes[0][k]:
                    atom_idx_test[k].append(iat)
                    iat += 1

            # Training molecules are at the end of the list
            iat = 0
            atom_idx_train = {}
            for k in centers:
                atom_idx_train[k] = []
                for il in atom_indexes[-1][k]:
                    atom_idx_train[k].append(iat)
                    iat += 1

            # Build kernels for identical atomic species in extrapolation mode
            katomic = {}
            natspe_train = {}
            natspe_test = {}
            ispe = 0
            for k in centers:
                natspe_train[ispe] = len(atom_idx_train[k])
                natspe_test[ispe] = len(atom_idx_test[k])
                # Create kernel matrix between test and training environments
                katomic[ispe] = np.zeros(
                    (natspe_test[ispe]*ntest, natspe_train[ispe]*(npoints-ntest), 2*lval+1, 2*lval+1), float)
                irow = 0
                for i in range(ntest):
                    for ii in atom_idx_test[k]:
                        icol = 0
                        for j in range(npoints-ntest):
                            for jj in atom_idx_train[k]:
                                katomic[ispe][irow, icol] = kloc[i, j, ii, jj]
                                icol += 1
                        irow += 1
                ispe += 1

            # Save extrapolation atomic kernels to files
            envfile = []
            for k in centers:
                filename = "kernel{lv}_atom{k}_ntest{nt}_ntrain{ntr}_sigma{sg}_lcut{lc}_cutoff{rc}_cweight{cw}.npy".format(
                    nt=ntest, ntr=(npoints-ntest), nk=k, lv=lval, sg=sg, lc=lc, rcut=rcut, cw=cweight)
                envfile.append(open(filename, "w"))
            nspecies = len(centers)
            for ispe in range(nspecies):
                np.save(envfile[ispe], katomic[ispe])

        else:
            # Compute global kernels for extrapolation mode
            for n in range(len(nlist)):
                # Transform kernel from complex to real representation
                kernel = np.zeros(
                    (ntest, npoints-ntest, 2*lval+1, 2*lval+1), dtype=float)
                for i, j in product(range(ntest), range(npoints-ntest)):
                    # Apply complex-to-real transformation
                    kernel[i, j] = np.real(
                        np.dot(np.dot(CC, kernels[1+n][i, j]), CT))
                
                # Save extrapolation kernel to file
                kernel_file = "kernel{lv}_ntest{nt}_ntrain{ntr}_sigma{sg}_lcut{lc}_cutoff{rcut}_cweight{cw}.npy".format(
                    lv=lval, nt=ntest, ntr=(npoints-ntest), sg=sg, lc=lc, rcut=rcut, cw=cweight)
                np.save(kernel_file, kernel)


if __name__ == "__main__":
    # Command-line interface - maintains compatibility with existing scripts
    # Get command-line arguments using the parsing utility
    args = utils.parsing.add_command_line_arguments_tenskernel("Tensorial kernel")
    [ftrs, npoints, lval, sg, lc, rcut, cweight, vrb, centers, nlist, atomic,
        extrap, ntest] = utils.parsing.set_variable_values_tenskernel(args)

    # Call the main kernel computation function with parsed arguments
    sagpr_kernel(ftrs, npoints, lval, sg, lc, rcut, cweight, vrb, centers, nlist, atomic,
               extrap, ntest)