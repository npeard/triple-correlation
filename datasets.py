#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
from speckle1d import Fluorescence1D
import utils

class PhiDataset(Dataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase", unpack_diagonals=False):
        self.h5_file = h5_file
        self.input_key = input_key
        self.target_key = target_key
        self.unpack_diagonals = unpack_diagonals
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['phase'])
        self.opened_flag = False
        
    @staticmethod
    def sign_to_binary(x):
        # Map (-1, 1) to (0, 1) so that we can use BCELoss
        x = np.sign(x)
        return (x + 1) / 2
    
    @staticmethod
    def unpack_by_diagonals(x):
        """Unpack a 2D tensor by diagonals from top-right to bottom-left.
        
        Args:
            x (torch.Tensor): Input 2D tensor of shape (n, n)
            
        Returns:
            torch.Tensor: Flattened tensor containing all diagonals concatenated
        """
        # # cut off zero element edges
        # x = x[1:, 1:]

        # First flip left-right
        x = torch.fliplr(x)
        
        # Get dimensions
        n = x.size(0)
        assert x.size(0) == x.size(1), "Input tensor must be square"
        
        # List to store all diagonals
        diagonals = []
        
        # Extract diagonals from offset n-1 to -(n-1)
        for offset in range(n-1, -(n), -1):
            diagonal = torch.diagonal(x, offset=offset)
            diagonals.append(diagonal)
            
        # Concatenate all diagonals into single tensor
        return torch.cat(diagonals)

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
        input_tensor = torch.FloatTensor(self.inputs[idx])
        # cut off zero element edges
        input_tensor = input_tensor[1:, 1:]

        if self.unpack_diagonals:
            input_tensor = self.unpack_by_diagonals(input_tensor)
        else:
            input_tensor = input_tensor.reshape(input_tensor.size(0)**2)
        return input_tensor, torch.FloatTensor(self.targets[idx])
    
    
class AbsPhiDataset(PhiDataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase", unpack_diagonals=False):
        super().__init__(h5_file, input_key=input_key, target_key=target_key, unpack_diagonals=unpack_diagonals)
    
    def __getitem__(self, idx):
        # inputs, targets == abs(Phi), phase
        inputs, targets = super().__getitem__(idx)
        return torch.abs(inputs), targets
    
    
class SignPhiDataset(PhiDataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase"):
        super().__init__(h5_file, input_key=input_key, target_key=target_key)
    
    def __getitem__(self, idx):
        # inputs, targets == abs(Phi), sign_to_binary(Phi)
        inputs, targets = super().__getitem__(idx)
        return torch.abs(inputs), self.sign_to_binary(inputs)
    
    
class MultiTaskPhiDataset(PhiDataset):
    def __init__(self, h5_file, input_key="Phi", target_key="phase"):
        super().__init__(h5_file, input_key=input_key, target_key=target_key)
        
    def __getitem__(self, idx):
        inputs, targets = super().__getitem__(idx)
        # input1, target1, target2 == abs(Phi), sign_to_binary(Phi), phase
        return torch.abs(inputs), self.sign_to_binary(inputs), targets

    
def get_custom_dataloader(h5_file, batch_size=128, shuffle=True,
                          absPhi=False, signPhi=False, multiTask=False, unpack_diagonals=False):
    if not absPhi:
        dataset = PhiDataset(h5_file, unpack_diagonals=unpack_diagonals)
    else:
        dataset = AbsPhiDataset(h5_file, unpack_diagonals=unpack_diagonals)
        if signPhi:
            dataset = SignPhiDataset(h5_file, unpack_diagonals=unpack_diagonals)
    if multiTask:
        dataset = MultiTaskPhiDataset(h5_file, unpack_diagonals=unpack_diagonals)

    # We can use DataLoader to get batches of data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=7,
        persistent_workers=True,
        pin_memory=True)

    return dataloader


def generate_pretraining_data(num_pix, num_samples, file_path):
    # Generate pretraining data
    # num_pix: number of pixels in each sample
    
    for _ in tqdm(range(num_samples)):
        phase = np.random.uniform(-np.pi, np.pi, num_pix // 2)
        phase = np.concatenate((-phase, np.zeros(1), np.flip(phase)))
        
        # These lines work in PyTorch
        # phase = torch.FloatTensor(phase)
        # Phi = BaseDecoder.encode(phase.unsqueeze(0))
        
        # Use this Numpy module for JIT compilation speeds and compatibility
        # with h5py concatenation
        Phi = Fluorescence1D.compute_Phi_from_phase(phase[num_pix // 2:])
        
        utils.append_to_h5file(Phi, phase, file_path)

if __name__ == '__main__':
    file_path = f"./data/pretest_numpix21_1e+04_samples.h5"
    
    # Test both regular and diagonal-unpacked versions
    dataset = PhiDataset(file_path)
    dataset_diag = PhiDataset(file_path, unpack_diagonals=True)
    
    print("Original shape:", dataset[0][0].shape)
    print("Unpacked shape:", dataset_diag[0][0].shape)
    
    # Verify the unpacking manually
    x = dataset[0][0]
    print("\nManual unpacking test:")
    print("Original:\n", x)
    print("Flipped:\n", torch.fliplr(x))
    print("First diagonal:", torch.diagonal(torch.fliplr(x), offset=x.size(0)-1))
    print("Unpacked:\n", dataset_diag[0][0])

    # dataset = PhiDataset(file_path)
    # print(len(dataset))
    # print(dataset[0][0])
    # print(np.fliplr(dataset[0][0]))
    # print(np.diagonal(np.fliplr(dataset[0][0]), offset=1))
    # print(dataset[0][0].view(-1, 36))
    # print(np.diag(dataset[0][0]))