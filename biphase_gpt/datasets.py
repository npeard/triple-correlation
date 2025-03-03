#!/usr/bin/env python

from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


class BaseH5Dataset(Dataset):
    """Base class for HDF5 datasets.
    
    Args:
        file_path: Path to HDF5 file
        input_key: Key for input data in HDF5 file
        target_key: Key for target data in HDF5 file
        transform: Optional transform to apply to inputs
        target_transform: Optional transform to apply to targets
        cache_size: Number of items to cache in memory (0 for no caching)
    """
    
    def __init__(
        self,
        file_path: str,
        input_key: str = "Phi",
        target_key: str = "phase",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_size: int = 0
    ):
        self.file_path = file_path
        self.input_key = input_key
        self.target_key = target_key
        self.transform = transform
        self.target_transform = target_transform
        self.cache_size = cache_size
        
        # Initialize cache
        self._cache = {}
        self._cache_keys = []
        
        # Validate file and get dataset info
        with h5py.File(self.file_path, 'r') as f:
            self._validate_file_structure(f)
            self.length = self._get_dataset_length(f)
        
        self.opened_flag = False
    
    def _validate_file_structure(self, f: h5py.File) -> None:
        """Validate the HDF5 file has required datasets."""
        if self.input_key not in f:
            raise ValueError(f"Input key '{self.input_key}' not found in file")
        if self.target_key not in f:
            raise ValueError(f"Target key '{self.target_key}' not found in file")
    
    def _get_dataset_length(self, f: h5py.File) -> int:
        """Get the length of the dataset."""
        return len(f[self.target_key])
    
    def open_hdf5(self):
        """Open HDF5 file for reading.
        
        This is done lazily to support multiprocessing in DataLoader.
        """
        if not self.opened_flag:
            self.h5_file = h5py.File(self.file_path, 'r')
            self.inputs = self.h5_file[self.input_key]
            self.targets = self.h5_file[self.target_key]
            self.opened_flag = True
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        # Check cache first
        if idx in self._cache:
            inputs, targets = self._cache[idx]
        else:
            # Lazy loading of HDF5 file
            self.open_hdf5()
            
            # Load data
            inputs = self.inputs[idx]
            targets = self.targets[idx]
            
            # Apply transforms if specified
            if self.transform is not None:
                inputs = self.transform(inputs)
            if self.target_transform is not None:
                targets = self.target_transform(targets)
            
            # Add to cache
            if self.cache_size > 0:
                self._add_to_cache(idx, (inputs, targets))
        
        return torch.FloatTensor(inputs), torch.FloatTensor(targets)
    
    def _add_to_cache(self, key: int, value: Tuple[np.ndarray, np.ndarray]) -> None:
        """Add an item to the cache, maintaining cache size limit."""
        if self.cache_size == 0:
            return
            
        if len(self._cache) >= self.cache_size:
            # Remove oldest item if cache is full
            oldest_key = self._cache_keys[0]
            del self._cache[oldest_key]
            self._cache_keys.pop(0)
        
        self._cache[key] = value
        self._cache_keys.append(key)


class PhiDataset(BaseH5Dataset):
    """Dataset for Phi matrices with optional diagonal unpacking."""
    
    def __init__(
        self,
        file_path: str,
        input_key: str = "Phi",
        target_key: str = "phase",
        unpack_diagonals: bool = False,
        **kwargs
    ):
        super().__init__(file_path, input_key, target_key, **kwargs)
        self.unpack_diagonals = unpack_diagonals
    
    @staticmethod
    def unpack_by_diagonals(x: torch.Tensor) -> torch.Tensor:
        """Unpack a 2D tensor by diagonals from top-right to bottom-left."""
        # First flip left-right
        x = torch.fliplr(x)
        
        # Get dimensions
        n = x.size(0)
        assert x.size(0) == x.size(1), "Input tensor must be square"
        
        # Extract diagonals from offset n-1 to -(n-1)
        diagonals = [torch.diagonal(x, offset=offset) for offset in range(n-1, -(n), -1)]
        
        # Concatenate all diagonals into single tensor
        return torch.cat(diagonals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = super().__getitem__(idx)
        
        # Cut off zero element edges
        inputs = inputs[1:, 1:]
        
        if self.unpack_diagonals:
            inputs = self.unpack_by_diagonals(inputs)
        else:
            inputs = inputs.reshape(-1)  # Flatten
        
        return inputs, targets


class AbsPhiDataset(PhiDataset):
    """Dataset that returns absolute values of Phi matrices."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = super().__getitem__(idx)
        return torch.abs(inputs), targets

def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int = 4,
    **dataset_kwargs: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing.
    
    Args:
        train_path: Path to training data HDF5 file
        val_path: Path to validation data HDF5 file
        test_path: Path to test data HDF5 file
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments to pass to the dataset class
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PhiDataset(train_path, **dataset_kwargs)
    val_dataset = PhiDataset(val_path, **dataset_kwargs)
    test_dataset = PhiDataset(test_path, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    file_path = f"./data/pretest_numpix21_1e+04_samples.h5"
    
    dataset = PhiDataset(file_path)
    dataset_diag = PhiDataset(file_path, unpack_diagonals=True)
    
    print("Original shape:", dataset[0][0].shape)
    print("Unpacked shape:", dataset_diag[0][0].shape)
    
    x = dataset[0][0]
    print("\nManual unpacking test:")
    print("Original:\n", x)
    print("Flipped:\n", torch.fliplr(x))
    print("First diagonal:", torch.diagonal(torch.fliplr(x), offset=x.size(0)-1))
    print("Unpacked:\n", dataset_diag[0][0])