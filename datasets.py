#!/usr/bin/env python

import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torch import FloatTensor, abs


class PhiDataset(Dataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase"):
        self.h5_file = h5_file
        self.input_key = input_key
        self.target_key = target_key
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
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.opened_flag:  # not hasattr(self, 'h5_file'):
            self.open_hdf5()
            self.opened_flag = True
            # print("open_hdf5 finished")
        # inputs, targets == Phi (with sign), phase
        return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])
    
    
class AbsPhiDataset(PhiDataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase"):
        super().__init__(h5_file, input_key=input_key, target_key=target_key)
    
    def __getitem__(self, idx):
        # inputs, targets == abs(Phi), phase
        inputs, targets = super().__getitem__(idx)
        return abs(inputs), targets
    
    
class SignPhiDataset(PhiDataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase"):
        super().__init__(h5_file, input_key=input_key, target_key=target_key)
    
    def __getitem__(self, idx):
        # inputs, targets == abs(Phi), sign_to_binary(Phi)
        inputs, targets = super().__getitem__(idx)
        return abs(inputs), self.sign_to_binary(inputs)
    
    
class MultiTaskPhiDataset(PhiDataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase"):
        super().__init__(h5_file, input_key=input_key, target_key=target_key)
        
    def __getitem__(self, idx):
        inputs, targets = super().__getitem__(idx)
        # input1, target1, target2 == abs(Phi), sign_to_binary(Phi), phase
        return abs(inputs), self.sign_to_binary(inputs), targets

    
def get_custom_dataloader(h5_file, batch_size=128, shuffle=True,
                          absPhi=False, signPhi=False):
    if not absPhi:
        dataset = PhiDataset(h5_file)
        if signPhi:
            dataset = SignPhiDataset(h5_file)
    else:
        dataset = AbsPhiDataset(h5_file)

    # We can use DataLoader to get batches of data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True)

    return dataloader