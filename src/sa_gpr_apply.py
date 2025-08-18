#!/usr/bin/env python3
from __future__ import print_function
from builtins import range
import utils.kernels
import utils.parsing
import utils.kern_utils
import scipy.linalg
import sys
import numpy as np

###############################################################################################################################

def do_sagpr(lvals, lm, fractrain, tens, kernel_flatten, sel, rdm, rank, ncycles, nat, peratom):
    """
    Perform the core SA-GPR (Symmetry-Adapted Gaussian Process Regression) calculation.
    
    This function implements the main regression algorithm, including data partitioning,
    kernel matrix operations, training, prediction, and error analysis.
    
    Parameters:
    -----------
    lvals : list
        List of angular momentum quantum numbers (L values)
    lm : list
        List of regularization parameters for each L value
    fractrain : float
        Fraction of data used for training
    tens : list
        Tensor property values
    kernel_flatten : list
        List of flattened kernel matrices
    sel : list
        Selection indices for training set
    rdm : int
        Number of random training points
    rank : int
        Rank of tensor being learned
    ncycles : int
        Number of regression cycles
    nat : list
        Number of atoms per configuration
    peratom : bool
        Whether to use per-atom scaling
    """

    # initialize regression
    degen = [(2*l+1) for l in lvals]
    intrins_dev   = np.zeros(len(lvals),dtype=float)
    intrins_error = np.zeros(len(lvals),dtype=float)
    abs_error     = np.zeros(len(lvals),dtype=float)

    if ncycles > 1:
         print("Results averaged over {} cycles".format(ncycles))

    for ic in range(ncycles):

        # Get a list of members of the training and testing sets
        ndata = len(tens)
        [ns,nt,ntmax,trrange,terange] = utils.kern_utils.shuffle_data(ndata,sel,rdm,fractrain)

        # Build kernel matrices
        kernel = [utils.kern_utils.unflatten_kernel(ndata,degen[i],kernel_flatten[i]) for i in range(len(lvals))]

        # Partition properties and kernel for training and testing
        [vtrain,vtest,ktr,kte,nattrain,nattest] = utils.kern_utils.partition_kernels_properties(tens,kernel,trrange,terange,nat)

        # Extract the non-equivalent tensor components; include degeneracy
        [tenstrain,tenstest,mask1,mask2] = utils.kern_utils.get_non_equivalent_components(vtrain,vtest)

        # Unitary transormation matrix from Cartesian to spherical, Condon-Shortley convention
        CS = utils.kern_utils.get_CS_matrix(rank,mask1,mask2)

        # Transformation matrix from complex to real spherical harmonics
        CR = utils.kern_utils.complex_to_real_transformation(degen)

        # Extract the real spherical components of the tensors
        [ vtrain_part,vtest_part ] = utils.kern_utils.partition_spherical_components(tenstrain,tenstest,CS,CR,degen,ns,nt)

        # Subtract the mean if L=0
        meantrain = np.zeros(len(degen),dtype=float)
        for i in range(len(degen)):
            if degen[i]==1:
                vtrain_part[i]  = np.real(vtrain_part[i]).astype(float)
                meantrain[i]    = np.mean(vtrain_part[i])
                vtrain_part[i] -= meantrain[i]
                vtest_part[i]   = np.real(vtest_part[i]).astype(float)

        # Build training kernels
        ktrain_all_pred = [utils.kern_utils.build_training_kernel(nt,degen[i],ktr[i],lm[i]) for i in range(len(degen))]
        ktrain     = [ktrain_all_pred[i][0] for i in range(len(degen))]
        ktrainpred = [ktrain_all_pred[i][1] for i in range(len(degen))]

        # Invert training kernels
        invktrvec = [scipy.linalg.solve(ktrain[i],vtrain_part[i]) for i in range(len(degen))]

        # Build testing kernels
        ktest = [utils.kern_utils.build_testing_kernel(ns,nt,degen[i],kte[i]) for i in range(len(degen))]

        # Predict on test data set
        outvec = [np.dot(ktest[i],invktrvec[i]) for i in range(len(degen))]
        for i in range(len(degen)):
            if degen[i]==1:
                outvec[i] += meantrain[i]

        # Accumulate errors
        for i in range(len(degen)):
            intrins_dev[i] += np.std(vtest_part[i])**2
            abs_error[i] += np.sum((outvec[i]-vtest_part[i])**2)/(degen[i]*ns)

        # Convert the predicted full tensor back to Cartesian coordinates
        predcart = utils.kern_utils.spherical_to_cartesian(outvec,degen,ns,CR,CS,mask1,mask2)
        testcart = np.real(np.concatenate(vtest)).astype(float)

        if peratom:
            corrfile = "prediction.csv"
            with open(corrfile, 'w') as f:
                f.write("sample_id,true_value,predicted_value,num_atoms\n")  # CSV header
                for i in range(ns):
                    true_val = np.split(testcart,ns)[i]*nattest[i]
                    pred_val = np.split(predcart,ns)[i]*nattest[i]
                    # For tensor components, write each component as a separate row
                    for j in range(len(true_val)):
                        f.write("{},{},{},{}\n".format(i, true_val[j], pred_val[j], nattest[i]))
        else:
            corrfile = "prediction.csv"
            with open(corrfile, 'w') as f:
                f.write("sample_id,true_value,predicted_value\n")  # CSV header
                for i in range(ns):
                    true_val = np.split(testcart,ns)[i]
                    pred_val = np.split(predcart,ns)[i]
                    # For tensor components, write each component as a separate row
                    for j in range(len(true_val)):
                        f.write("{},{},{}\n".format(i, true_val[j], pred_val[j]))



    # Find average error
    for i in range(len(degen)):
        intrins_dev[i] = np.sqrt(intrins_dev[i]/float(ncycles))
        abs_error[i] = np.sqrt(abs_error[i]/float(ncycles))
        intrins_error[i] = 100*np.sqrt(abs_error[i]**2/intrins_dev[i]**2)

    # Print out errors
    print("\ntesting data points: {}".format(ns))
    print("training data points: {}".format(nt))
    for i in range(len(degen)):
        print("--------------------------------")
        print("RESULTS FOR L=%i MODULI (lambda=%f)"%(lvals[i],lm[i]))
        print("-----------------------------------------------------")
        print("STD {}".format(intrins_dev[i]))
        print("ABS RSME {}".format(abs_error[i]))
        print("RMSE = {:.4f}".format(intrins_error[i]))

###############################################################################################################################

def sapgr_apply(lvals=None, lm=None, fractrain=1.0, tens=None, kernels=None, 
                sel=None, rdm=0, rank=None, ncycles=1, nat=None, peratom=False):
    """
    Apply Symmetry-Adapted Gaussian Process Regression (SA-GPR) for learning tensorial properties.
    
    This function performs machine learning on pre-computed kernels to predict tensorial properties
    of molecular systems. It supports different tensor ranks and can handle both per-atom and 
    total property predictions.
    
    Parameters:
    -----------
    lvals : list, optional
        List of angular momentum quantum numbers (L values) corresponding to spherical tensor components.
        If None, will be computed based on rank parameter.
        For even rank: [0, 2, 4, ..., rank]
        For odd rank: [1, 3, 5, ..., rank]
    lm : list
        List of regularization parameters (lambda values) for Kernel Ridge Regression.
        Must have the same length as lvals. Required parameter.
    fractrain : float, default=1.0
        Fraction of data points used for training (remaining fraction used for testing).
        Value between 0 and 1.
    tens : list
        List of tensor property values for each configuration. Required parameter.
        Format depends on tensor rank and peratom setting.
    kernels : list
        List of file paths containing pre-computed kernel matrices. Required parameter.
        Should correspond to the L values in lvals.
    sel : list, optional
        Selection range for training set as [start, end] indices. 
        If None or empty, uses random or fraction-based selection.
    rdm : int, default=0
        Number of random training points to select. If 0, uses fractrain instead.
    rank : int
        Rank of the tensor to be learned (0=scalar, 1=vector, 2=matrix, etc.).
        Required parameter.
    ncycles : int, default=1
        Number of regression cycles with different random training/test splits.
        Results are averaged over all cycles.
    nat : list
        List of number of atoms in each configuration. Required for peratom scaling.
    peratom : bool, default=False
        If True, scale properties by number of atoms in each configuration.
    
    Returns:
    --------
    None
        Function performs regression and prints results. Saves predictions to 'prediction.csv'.
    
    Raises:
    -------
    ValueError
        If required parameters (lm, tens, kernels, rank) are not provided.
    """
    
    # Validate required parameters
    if lm is None:
        raise ValueError("lm (regularization parameters) is required")
    if tens is None:
        raise ValueError("tens (tensor properties) is required")
    if kernels is None:
        raise ValueError("kernels (kernel file paths) is required")
    if rank is None:
        raise ValueError("rank (tensor rank) is required")
    
    # Set default values based on parsing.py defaults
    if sel is None:
        sel = []
    if nat is None:
        nat = []
    
    # Compute lvals if not provided, based on tensor rank
    if lvals is None:
        if rank % 2 == 0:
            # Even rank: include L=0, 2, 4, ..., rank
            lvals = [l for l in range(0, rank + 1, 2)]
        else:
            # Odd rank: include L=1, 3, 5, ..., rank
            lvals = [l for l in range(1, rank + 1, 2)]
    
    # Validate that lm and lvals have matching lengths
    if len(lm) != len(lvals):
        raise ValueError("Number of regularization parameters must equal number of L values!")
    
    # Read-in kernel matrices from files
    print("Loading kernel matrices...")

    kernel = []
    for k in range(len(kernels)):
        # Load kernel matrix and flatten it for processing
        kr = np.load(kernels[k])
        kr = np.reshape(kr, np.size(kr))
        kernel.append(kr)

    print("...Kernels loaded.")
    
    # Perform the actual SA-GPR calculation
    do_sagpr(lvals, lm, fractrain, tens, kernel, sel, rdm, rank, ncycles, nat, peratom)

if __name__ == "__main__":
    # This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.

    # Parse input arguments
    args = utils.parsing.add_command_line_arguments_learn("SA-GPR")
    [lvals,lm,fractrain,tens,kernels,sel,rdm,rank,ncycles,nat,peratom] = utils.parsing.set_variable_values_learn(args)
    sapgr_apply(lvals,lm,fractrain,tens,kernels,sel,rdm,rank,ncycles,nat,peratom)
