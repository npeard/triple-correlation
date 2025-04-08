import pytest
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biphase_gpt.lightning_config import GPTDecoder
from fluo import Fluorescence1D


def test_encode_equivalence():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create antisymmetric phase sequence
    n = 33 # sequence length must be odd
    half_n = n // 2

    # Generate random positive half (not including center)
    pos_half = np.random.randn(half_n)
    phase = np.concatenate(([0], pos_half))

    # Convert to torch tensor and add batch dimension
    phase_tensor = torch.from_numpy(phase).float().unsqueeze(0)

    # Get GPTDecoder output
    gpt_output = GPTDecoder._encode(phase_tensor[:, :])

    # Get Fluorescence1D output
    # Use only positive half for compute_Phi_from_phase
    fluo_output = Fluorescence1D.compute_Phi_from_phase(phase)

    # Take absolute value and remove first row and column to match GPTDecoder output
    fluo_output = np.abs(fluo_output[1:, 1:])

    # Convert GPT output to numpy for comparison
    gpt_output = gpt_output.squeeze().numpy()

    # Compare outputs
    np.testing.assert_allclose(gpt_output, fluo_output, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_encode_equivalence()
