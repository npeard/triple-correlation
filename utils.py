#!/usr/bin/env python

import numpy as np
import h5py
from tqdm import tqdm


def append_to_h5file(cosPhi, Phi, phase, file_path="data.h5"):
    """Appends training data consisting of an image stack, the associated
    marginalized cosPhi, and the structure phase to a file.

    Parameters
    ----------
    cosPhi : (N, M) array
        The marginalized cosPhi data
    Phi : (N, M) array
        The marginalized Phi data
    phase : (M,) array
        The target phase for the structure
    file_path : str, optional
        The output HDF5 file

    Returns
    -------
    None"""

    f = h5py.File(file_path, 'a')
    # Create datasets if they don't exist, otherwise append data

    if cosPhi is not None:
        if "cosPhi" in f.keys():
            f["cosPhi"].resize(
                (f["cosPhi"].shape[0] + 1), axis=0)
            new_data = np.expand_dims(cosPhi, axis=0)
            f["cosPhi"][-1:] = new_data
        else:
            f.create_dataset("cosPhi",
                             data=np.expand_dims(cosPhi, axis=0),
                             maxshape=(None, cosPhi.shape[0],
                                       cosPhi.shape[1]),
                             chunks=True)
    if Phi is not None:
        if "Phi" in f.keys():
            f["Phi"].resize(
                (f["Phi"].shape[0] + 1), axis=0)
            new_data = np.expand_dims(Phi, axis=0)
            f["Phi"][-1:] = new_data
        else:
            f.create_dataset("Phi",
                             data=np.expand_dims(Phi, axis=0),
                             maxshape=(None, Phi.shape[0],
                                       Phi.shape[1]),
                             chunks=True)
    
    if "phase" in f.keys():
        f["phase"].resize((f["phase"].shape[0] + 1),
                          axis=0)
        new_data = np.expand_dims(phase, axis=0)
        f["phase"][-1:] = new_data
    else:
        f.create_dataset("phase",
                         data=np.expand_dims(phase, axis=0),
                         maxshape=(None, phase.shape[0]),
                         chunks=True)
