#!/usr/bin/env python

import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torch import FloatTensor


class PhiDataset(Dataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase",
                 input_op=None, target_op=None):
        self.h5_file = h5_file
        self.input_key = input_key
        self.target_key = target_key
        self.input_op = input_op
        self.target_op = target_op
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['phase'])
        self.opened_flag = False
        
    @staticmethod
    def sign_to_binary(x):
        # Map (-1, 1) to (0, 1) so that we can use BCELoss
        return (x + 1) / 2
    
    def open_hdf5(self):
        # solves issue where hdf5 file opened in __init__ prevents multiple
        # workers: https://github.com/pytorch/pytorch/issues/11929
        read_file = h5py.File(self.h5_file, 'r')
        
        self.inputs = read_file[self.input_key]
        self.targets = read_file[self.target_key]
        
        # Perform additional data operations on inputs
        if self.input_op == "abs":
            self.inputs = np.abs(self.inputs)
        elif self.input_op == "cos":
            self.inputs = np.cos(self.inputs)
        elif self.input_op is not None:
            raise ValueError("Invalid input operation")
        
        # Perform additional data operations on targets
        if self.target_op == "sign_to_binary":
            self.targets = self.sign_to_binary(self.targets)
        elif self.target_op is not None:
            raise ValueError("Invalid target operation")
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.opened_flag:  # not hasattr(self, 'h5_file'):
            self.open_hdf5()
            self.opened_flag = True
            # print("open_hdf5 finished")
        return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])

    
def get_custom_dataloader(h5_file, batch_size=128, shuffle=True,
                          linear_only=False, sign_only=False):
    if linear_only:
        dataset = PhiDataset(h5_file)
        if sign_only:
            dataset = PhiDataset(h5_file, target_key="Phi",
                                 input_op="abs", target_op="sign_to_binary")
    else:
        dataset = PhiDataset(h5_file, input_op="cos")

    # We can use DataLoader to get batches of data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True)

    return dataloader