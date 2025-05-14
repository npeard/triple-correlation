import numpy as np
import torch

from biphase_gpt.lightning_config import GPTDecoder
from fluo import Fluorescence1D
from fluo.speckle2d import Fluorescence2D


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

def test_encode_2D_equivalence():
    """Test that _encode_2D produces the same output as compute_Phi_from_phase from Fluorescence2D"""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create a 2D phase array with random values
    n = 11  # Small odd size for faster testing
    phase_np = np.random.uniform(-np.pi, np.pi, (n, n))

    # Set origin to zero as required
    phase_np[0, 0] = 0

    # Convert to torch tensor and add batch dimension
    phase_torch = torch.from_numpy(phase_np).float().unsqueeze(0)

    # Get output from Fluorescence2D.compute_Phi_from_phase
    fluo_output = Fluorescence2D.compute_Phi_from_phase(phase_np)

    # Get output from our PyTorch implementation
    gpt_output = GPTDecoder._encode_2D(phase_torch)

    # Convert PyTorch output to numpy for comparison
    gpt_output_np = gpt_output.squeeze().numpy()

    # Compare outputs
    np.testing.assert_allclose(
        gpt_output_np, np.abs(fluo_output),
        rtol=1e-5, atol=1e-5,
        err_msg="2D encoding outputs differ"
    )

    print("All 2D encoding tests passed!")
