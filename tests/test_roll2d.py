import numpy as np
import torch
from fluo.speckle2d import roll2d
from biphase_gpt.lightning_config import roll2d_torch

def test_roll2d_matches_numpy():
    # Create a test array
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # Test cases: (shift_x, shift_y)
    test_shifts = [
        (0, 0),    # No shift
        (1, 0),    # Shift right
        (0, 1),    # Shift down
        (-1, -1),  # Shift up-left
        (2, 1),    # Larger shift
    ]

    for shift_x, shift_y in test_shifts:
        # Compare our roll2d with numpy's roll
        result = roll2d(arr, shift_x, shift_y)
        expected = np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)

        np.testing.assert_array_equal(
            result,
            expected,
            err_msg=f"Failed for shifts: x={shift_x}, y={shift_y}"
        )

def test_roll2d_different_shapes():
    # Test with non-square array
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8]])

    # Test shifts
    shift_x, shift_y = 1, 2

    result = roll2d(arr, shift_x, shift_y)
    expected = np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)

    np.testing.assert_array_equal(
        result,
        expected,
        err_msg="Failed for non-square array"
    )

def test_roll2d_large_shifts():
    # Test with shifts larger than array dimensions
    arr = np.array([[1, 2],
                    [3, 4]])

    # Shifts larger than dimensions should wrap around
    shift_x, shift_y = 3, 4  # 3 % 2 = 1, 4 % 2 = 0

    result = roll2d(arr, shift_x, shift_y)
    expected = np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)

    np.testing.assert_array_equal(
        result,
        expected,
        err_msg="Failed for large shifts"
    )

def test_roll2d_equivalence():
    """Test that roll2d_torch produces the same output as roll2d from speckle2d.py"""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create a 2D array with random values
    n = 11  # Small odd size for faster testing
    arr_np = np.random.randn(n, n)

    # Convert to torch tensor and add batch dimension
    arr_torch = torch.from_numpy(arr_np).float().unsqueeze(0)

    # Test various shift combinations
    shift_combinations = [
        (0, 0),    # No shift
        (1, 0),    # Shift in x only
        (0, 1),    # Shift in y only
        (3, 2),    # Positive shifts in both
        (-2, -3),  # Negative shifts in both
        (5, -4)    # Mixed positive and negative
    ]

    for shift_x, shift_y in shift_combinations:
        # Get output from original roll2d function
        np_output = roll2d(arr_np, shift_x, shift_y)

        # Get output from our PyTorch implementation
        torch_output = roll2d_torch(arr_torch, shift_x, shift_y)

        # Convert PyTorch output to numpy for comparison
        torch_output_np = torch_output.squeeze().numpy()

        # Compare outputs
        np.testing.assert_allclose(
            torch_output_np, np_output,
            rtol=1e-5, atol=1e-5,
            err_msg=f"Outputs differ for shifts ({shift_x}, {shift_y})"
        )

    print("All roll2d tests passed!")
