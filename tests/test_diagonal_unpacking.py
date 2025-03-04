import pytest
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biphase_gpt.datasets import AbsPhiDataset
from biphase_gpt.lightning_config import GPTDecoder


def test_diagonal_unpacking_equivalence():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create a batch of random square matrices
    batch_size = 3
    n = 5  # matrix size
    x = torch.randn(batch_size, n, n)
    
    # Get results from both methods
    dataset_method_results = []
    for i in range(batch_size):
        # Apply original method from AbsPhiDataset to each matrix individually
        single_matrix = x[i]
        unpacked = AbsPhiDataset.unpack_by_diagonals(single_matrix)
        dataset_method_results.append(unpacked)
    
    # Stack results from original method
    dataset_method_results = torch.stack(dataset_method_results)
    
    # Apply batched method from GPTDecoder
    batched_method_results = GPTDecoder._unpack_by_diagonals_batched(x)
    
    # Compare results
    np.testing.assert_allclose(
        dataset_method_results.numpy(),
        batched_method_results.numpy(),
        rtol=1e-5,
        atol=1e-5
    )
    
    # Test with batch size of 1 to ensure it works in that case too
    single_batch = x[0:1]  # Keep the batch dimension
    single_dataset_result = AbsPhiDataset.unpack_by_diagonals(single_batch.squeeze())
    single_batched_result = GPTDecoder._unpack_by_diagonals_batched(single_batch)
    
    np.testing.assert_allclose(
        single_dataset_result.numpy(),
        single_batched_result.squeeze().numpy(),
        rtol=1e-5,
        atol=1e-5
    )


if __name__ == "__main__":
    test_diagonal_unpacking_equivalence()
