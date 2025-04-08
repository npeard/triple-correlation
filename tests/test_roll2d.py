import numpy as np
import pytest
from fluo.speckle2d import roll2d

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
