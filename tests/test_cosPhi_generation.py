import numpy as np
import pytest
from fluo import Fluorescence1D, Fluorescence2D


def test_cosPhi_1d():
    """Test that cosPhi_from_phase and cosPhi_from_data produce similar results in 1D."""
    # Initialize model with small parameters for faster testing
    fluo = Fluorescence1D(kmax=3, num_pix=21, num_atoms=5)

    # Get cosPhi from both methods
    cosPhi_from_data = fluo.cosPhi_from_data(num_shots=10000)
    cosPhi_from_data = cosPhi_from_data[
        fluo.num_pix - 1:3 * fluo.num_pix // 2,
        fluo.num_pix - 1:3 * fluo.num_pix // 2
    ]

    cosPhi_from_phase = fluo.cosPhi_from_phase()

    # Verify shapes match
    assert cosPhi_from_data.shape == cosPhi_from_phase.shape, \
        "Shapes do not match for 1D model"

    # Check that values are reasonably close
    # We use a relatively high tolerance since these are statistical estimates
    np.testing.assert_allclose(
        cosPhi_from_data,
        cosPhi_from_phase,
        rtol=0.3,  # 30% relative tolerance
        atol=0.3,  # 0.3 absolute tolerance
        err_msg="1D cosPhi values differ too much between methods"
    )


def test_cosPhi_2d():
    """Test that cosPhi_from_phase and cosPhi_from_data produce similar results in 2D."""
    # Initialize model with small parameters for faster testing
    fluo = Fluorescence2D(kmax=3, num_pix=11, num_atoms=5)

    # Get cosPhi from both methods
    cosPhi_from_phase = fluo.cosPhi_from_phase()

    cosPhi_from_data = fluo.cosPhi_from_data(num_shots=10000)
    cosPhi_from_data = cosPhi_from_data[
        fluo.num_pix - 1:3 * fluo.num_pix // 2,
        fluo.num_pix - 1:3 * fluo.num_pix // 2,
        fluo.num_pix - 1:3 * fluo.num_pix // 2,
        fluo.num_pix - 1:3 * fluo.num_pix // 2
    ]

    # Verify shapes match
    assert cosPhi_from_data.shape == cosPhi_from_phase.shape, \
        "Shapes do not match for 2D model"

    # Take a 2D slice for comparison (same as in plot_cosPhi)
    cosPhi_from_data = cosPhi_from_data[1, :, 1, :]
    cosPhi_from_phase = cosPhi_from_phase[1, :, 1, :]

    # Check that values are reasonably close
    # We use a relatively high tolerance since these are statistical estimates
    np.testing.assert_allclose(
        cosPhi_from_data,
        cosPhi_from_phase,
        rtol=0.3,  # 30% relative tolerance
        atol=0.3,  # 0.3 absolute tolerance
        err_msg="2D cosPhi values differ too much between methods"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
