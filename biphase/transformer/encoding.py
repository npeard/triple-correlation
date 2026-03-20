#!/usr/bin/env python

"""Encoding utilities for phase retrieval from triple correlations.

Provides functions to encode phase arrays into absolute Phi matrices,
used for computing encoding loss during training and by the differentiable solver.
"""

import torch
from torch import nn


@torch.jit.script
def roll2d_torch(arr: torch.Tensor, shift_x: int, shift_y: int) -> torch.Tensor:
    """Torch-compatible implementation of np.roll for 2D arrays.

    Args:
        arr: 2D input tensor of shape (batch_size, nx, ny)
        shift_x: Number of places to shift along first spatial dimension
        shift_y: Number of places to shift along second spatial dimension

    Returns:
        torch.Tensor: Shifted tensor of same shape as input
    """
    out = torch.roll(arr, shifts=shift_x, dims=1)
    return torch.roll(out, shifts=shift_y, dims=2)


@torch.jit.script
def encode_1d(phase: torch.Tensor) -> torch.Tensor:
    """Re-encode abs(Phi) matrix from predicted 1D phase.

    Computes only the required output elements without creating
    the full-size intermediate tensor, saving memory and improving performance.

    Args:
        phase: Predicted phase tensor of shape (batch_size, phase_length)

    Returns:
        torch.Tensor: Re-encoded abs(Phi) matrix
    """
    batch_size = phase.size(0)
    phase_len = phase.size(1)
    Phi_dim = phase_len // 2 + 1

    out_dim = Phi_dim - 1

    encoded = torch.zeros((batch_size, out_dim, out_dim), device=phase.device)

    for i in range(1, Phi_dim):
        rolled_phase = torch.roll(phase, -i, dims=-1)
        phase_diff = (
            rolled_phase[:, 1:Phi_dim] - phase[:, 1:Phi_dim] - phase[:, i].unsqueeze(-1)
        )
        encoded[:, i - 1, :] = torch.abs(phase_diff)

    return encoded


@torch.jit.script
def encode_2d(phase: torch.Tensor) -> torch.Tensor:
    """Re-encode abs(Phi) matrix from predicted 2D phase.

    Computes only the required output elements without creating
    the full-size intermediate Phi tensor, saving memory and improving
    performance for large arrays.

    Args:
        phase: Predicted phase tensor of shape (batch_size, num_pix, num_pix)

    Returns:
        torch.Tensor: Re-encoded abs(Phi) 4D tensor of shape
            (batch_size, half_nx-1, half_ny-1, half_nx-1, half_ny-1)
    """
    batch_size, nx, ny = phase.shape
    assert nx == ny, 'Phase must be square'

    half_nx = nx // 2 + 1
    half_ny = ny // 2 + 1
    out_nx = half_nx - 1
    out_ny = half_ny - 1

    Phi_abs = torch.zeros(
        (batch_size, out_nx, out_ny, out_nx, out_ny), device=phase.device
    )

    for nx_shift in range(1, half_nx):
        for ny_shift in range(1, half_ny):
            shifted_phase = roll2d_torch(phase, -nx_shift, -ny_shift)
            phase_at_shift = phase[:, nx_shift, ny_shift].unsqueeze(1).unsqueeze(2)
            phase_diff = (
                shifted_phase[:, 1:half_nx, 1:half_ny]
                - phase[:, 1:half_nx, 1:half_ny]
                - phase_at_shift
            )
            Phi_abs[:, nx_shift - 1, ny_shift - 1, :, :] = torch.abs(phase_diff)

    return Phi_abs


def encoding_loss(
    phase: torch.Tensor, absPhi: torch.Tensor, num_pix: int | tuple
) -> torch.Tensor:
    """Compute loss between re-encoded abs(Phi) matrix and input abs(Phi) matrix.

    Args:
        phase: Predicted phase tensor
        absPhi: Input absolute Phi matrix sequence (flattened)
        num_pix: Number of pixels (int for 1D, tuple for 2D)

    Returns:
        torch.Tensor: MSE loss between re-encoded and input abs(Phi)
    """
    if isinstance(num_pix, tuple):
        phase = phase.view(-1, num_pix[0], num_pix[1])
        encoded = encode_2d(phase).flatten(start_dim=1)
    elif isinstance(num_pix, int):
        encoded = encode_1d(phase).flatten(start_dim=1)
    else:
        raise TypeError('num_pix must be int or tuple')

    return nn.MSELoss()(encoded, absPhi)
