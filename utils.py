#!/usr/bin/env python

import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def append_to_h5file(Phi, phase, file_path):
    """Append Phi matrix and phase to HDF5 file.
    
    Args:
        Phi: Phi matrix to append
        phase: Phase array to append
        file_path: Path to HDF5 file
    """
    with h5py.File(file_path, 'a') as f:
        if 'Phi' not in f:
            f.create_dataset('Phi',
                         data=np.expand_dims(Phi, axis=0),
                         maxshape=(None, *Phi.shape),
                         chunks=True)
            f.create_dataset('phase',
                         data=np.expand_dims(phase, axis=0),
                         maxshape=(None,) + phase.shape,
                         chunks=True)
        else:
            f['Phi'].resize((f['Phi'].shape[0] + 1), axis=0)
            f['Phi'][-1] = Phi
            f['phase'].resize((f['phase'].shape[0] + 1), axis=0)
            f['phase'][-1] = phase


def generate_pretraining_data(
    file_path: str,
    num_pix: int,
    num_samples: int,
    chunk_size: Optional[int] = None
) -> None:
    """Generate pretraining data and save to HDF5 file.
    
    Args:
        file_path: Path to save the HDF5 file
        num_pix: Number of pixels in each sample
        num_samples: Number of samples to generate
        chunk_size: Optional chunk size for HDF5 dataset compression
    """
    # Create parent directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Default chunk size to balance between compression and access speed
    if chunk_size is None:
        chunk_size = min(100, num_samples)
    
    with h5py.File(file_path, 'w') as f:
        # Store generation parameters as attributes
        f.attrs['num_samples'] = num_samples
        f.attrs['num_pix'] = num_pix
        
        # Create datasets with chunking and compression
        f.create_dataset(
            'Phi',
            shape=(num_samples, num_pix, num_pix),
            dtype='float32',
            chunks=(chunk_size, num_pix, num_pix),
            compression='gzip',
            compression_opts=4
        )
        
        f.create_dataset(
            'phase',
            shape=(num_samples, num_pix),
            dtype='float32',
            chunks=(chunk_size, num_pix),
            compression='gzip',
            compression_opts=4
        )
        
        # Generate data
        print(f"\nGenerating {num_samples} samples...")
        for i in tqdm(range(num_samples)):
            # Generate random phase
            phase = np.random.uniform(-np.pi, np.pi, num_pix // 2)
            phase = np.concatenate((-phase, np.zeros(1), np.flip(phase)))
            
            # Compute Phi matrix
            Phi = Fluorescence1D.compute_Phi_from_phase(phase[num_pix // 2:])
            
            # Store in dataset
            f['Phi'][i] = Phi
            f['phase'][i] = phase


def create_train_val_test_datasets(
    output_dir: str,
    num_pix: int = 21,
    train_samples: int = int(1e6),
    val_samples: int = int(1e4),
    test_samples: int = int(1e4),
    **kwargs
) -> None:
    """Create train, validation and test datasets for pretraining.
    
    Args:
        output_dir: Directory to save the datasets
        num_pix: Number of pixels in each sample
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        **kwargs: Additional arguments passed to generate_pretraining_data
    """
    print("Generating training dataset...")
    generate_pretraining_data(
        file_path=str(Path(output_dir) / f'pretrain_numpix{num_pix}_{train_samples:.0e}_samples.h5'),
        num_pix=num_pix,
        num_samples=train_samples,
        **kwargs
    )
    
    print("\nGenerating validation dataset...")
    generate_pretraining_data(
        file_path=str(Path(output_dir) / f'prevalid_numpix{num_pix}_{val_samples:.0e}_samples.h5'),
        num_pix=num_pix,
        num_samples=val_samples,
        **kwargs
    )
    
    print("\nGenerating test dataset...")
    generate_pretraining_data(
        file_path=str(Path(output_dir) / f'pretest_numpix{num_pix}_{test_samples:.0e}_samples.h5'),
        num_pix=num_pix,
        num_samples=test_samples,
        **kwargs
    )


def visualize_pretraining_dataset(
    file_path: str,
    num_samples: int = 4,
    random_seed: Optional[int] = None,
    save_path: Optional[str] = None
) -> None:
    """Visualize random samples from a pretraining dataset.
    
    Args:
        file_path: Path to HDF5 file
        num_samples: Number of samples to visualize
        random_seed: Optional random seed for reproducibility
        save_path: Optional path to save the visualization
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    with h5py.File(file_path, 'r') as f:
        # Get dataset info
        total_samples = f['Phi'].shape[0]
        indices = np.random.choice(total_samples, num_samples, replace=False)
        indices.sort()  # Sort indices for HDF5 compatibility
        
        # Load selected samples
        Phi_samples = f['Phi'][indices]
        phase_samples = f['phase'][indices]
        
        # Get metadata
        metadata = dict(f.attrs)
        
        # Create figure with 2 rows (Phi and phase) and num_samples columns
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        
        # Plot samples
        for i in range(num_samples):
            # Plot Phi matrix
            im_phi = axes[0, i].imshow(Phi_samples[i], cmap='viridis')
            axes[0, i].set_title(f'Phi Matrix {i+1}')
            axes[0, i].axis('off')
            plt.colorbar(im_phi, ax=axes[0, i])
            
            # Plot phase
            im_phase = axes[1, i].plot(phase_samples[i])
            axes[1, i].set_title(f'Phase {i+1}')
            axes[1, i].set_ylim(-np.pi, np.pi)
            axes[1, i].grid(True)
        
        # Add overall title
        plt.suptitle(f"Samples from {Path(file_path).name}\n" + 
                    f"Total Samples: {total_samples}, Pixels: {metadata['num_pix']}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def inspect_pretraining_dataset(file_path: str) -> dict:
    """Calculate and return statistics about the pretraining dataset.
    
    Args:
        file_path: Path to HDF5 file
    
    Returns:
        Dictionary containing dataset statistics
    """
    with h5py.File(file_path, 'r') as f:
        Phi = f['Phi'][:]
        phase = f['phase'][:]
        metadata = dict(f.attrs)
        
        stats = {
            'num_samples': Phi.shape[0],
            'num_pix': metadata['num_pix'],
            'Phi_stats': {
                'min': float(Phi.min()),
                'max': float(Phi.max()),
                'mean': float(Phi.mean()),
                'std': float(Phi.std())
            },
            'phase_stats': {
                'min': float(phase.min()),
                'max': float(phase.max()),
                'mean': float(phase.mean()),
                'std': float(phase.std())
            }
        }
    return stats