#!/usr/bin/env python

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class TCN2DConfig:
    """Configuration for TCN2D model."""

    in_seq_len: int
    out_seq_len: int
    input_dim: int
    output_dim: int
    dropout: float
    bias: bool
    embd_tcn_kernel_size: int
    embd_tcn_num_channels: list[int]
    embd_tcn_dilation_base: int
    embd_tcn_stride: int
    embd_tcn_activation: str
    embd_tcn_dropout: float


class Chomp2d(nn.Module):
    """Remove padding at the end of the sequence for 2D convolutions."""

    def __init__(self, chomp_size_h: int, chomp_size_w: int):
        super().__init__()
        self.chomp_size_h = chomp_size_h
        self.chomp_size_w = chomp_size_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding from height and width dimensions.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Chomped tensor with padding removed
        """
        if self.chomp_size_h > 0 and self.chomp_size_w > 0:
            return x[:, :, : -self.chomp_size_h, : -self.chomp_size_w].contiguous()
        elif self.chomp_size_h > 0:
            return x[:, :, : -self.chomp_size_h, :].contiguous()
        elif self.chomp_size_w > 0:
            return x[:, :, :, : -self.chomp_size_w].contiguous()
        else:
            return x


class TemporalBlock2D(nn.Module):
    """Single block of 2D temporal convolutions with dilation and causal masking."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int],
        dropout: float = 0.2,
        activation_fn: nn.Module = None,
    ):
        super().__init__()

        self.conv1 = nn.utils.weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp2d(padding[0], padding[1])
        self.activation1 = activation_fn
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp2d(padding[0], padding[1])
        self.activation2 = activation_fn
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp2,
            self.activation1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation2,
            self.dropout2,
        )

        # 1x1 convolution for residual connection if input and output channels differ
        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.final_activation = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor with residual connection applied
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # Ensure residual and output have the same spatial dimensions
        if res.shape != out.shape:
            # Crop residual to match output size if needed
            res = res[:, :, : out.shape[2], : out.shape[3]]

        return self.final_activation(out + res)


class TCN2D(nn.Module):
    """2D Temporal Convolutional Network that respects causality along both dimensions.

    Takes 2D input and applies 2D convolutions with causal masking, then fans out
    to a 1D sequence for compatibility with existing training schemes.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        # Get activation function
        act_fn_map = {
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU(),
        }
        try:
            activation_fn = act_fn_map.get(config.embd_tcn_activation)
        except (AttributeError, KeyError):
            raise KeyError(
                f'Unsupported activation function: {config.embd_tcn_activation}'
            ) from None

        # Build TCN layers
        layers = []
        num_levels = len(config.embd_tcn_num_channels)

        # For 2D, we start with a single input channel (the encoded 2D matrix)
        input_channels = 1
        for i in range(num_levels):
            dilation_h = config.embd_tcn_dilation_base**i
            dilation_w = config.embd_tcn_dilation_base**i

            in_channels = (
                input_channels if i == 0 else config.embd_tcn_num_channels[i - 1]
            )
            out_channels = config.embd_tcn_num_channels[i]

            # 2D kernel size - use same size for both dimensions
            kernel_size = (config.embd_tcn_kernel_size, config.embd_tcn_kernel_size)
            stride = (config.embd_tcn_stride, config.embd_tcn_stride)
            dilation = (dilation_h, dilation_w)

            # Calculate padding to maintain sequence length for causal convolution
            # For causal masking, we pad only on the left/top
            padding_h = (config.embd_tcn_kernel_size - 1) * dilation_h
            padding_w = (config.embd_tcn_kernel_size - 1) * dilation_w
            padding = (padding_h, padding_w)

            layers.append(
                TemporalBlock2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    dilation,
                    padding,
                    dropout=config.embd_tcn_dropout,
                    activation_fn=activation_fn,
                )
            )

        self.tcn_network = nn.Sequential(*layers)

        # Calculate the spatial size after TCN processing
        # This depends on the input size and the network architecture
        # We'll compute this dynamically in the first forward pass
        self._output_projection = None
        self._spatial_pooling = None

    def _create_output_layers(self):
        """Create output projection layers based on spatial size."""
        final_channels = self.config.embd_tcn_num_channels[-1]

        # Global average pooling to reduce spatial dimensions
        self._spatial_pooling = nn.AdaptiveAvgPool2d(1)

        # Final projection to desired output size
        final_size = self.config.output_dim * self.config.out_seq_len
        self._output_projection = nn.Sequential(
            nn.Linear(final_channels, final_size), nn.Linear(final_size, final_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2D TCN.

        Args:
            x: Input tensor from GPTDecoder._encode() with 2D structure
               Expected shape: (batch_size, height, width) where height=width

        Returns:
            1D output tensor compatible with GPTDecoder targets when num_pix is int
        """
        batch_size = x.size(0)

        # Ensure input has the right shape for 2D convolution
        if len(x.shape) == 3:
            # Add channel dimension: (batch, height, width) -> (batch, 1, height, width)
            x = x.unsqueeze(1)
        else:
            raise ValueError(f'Input must be 3D (batch, height, width), got {x.shape}')

        assert x.shape[2] == x.shape[3], 'Input must be square'

        # Process through TCN network
        features = self.tcn_network(x)

        # Create output layers if not already created
        if self._output_projection is None:
            self._create_output_layers()
            # Move to same device as input
            self._spatial_pooling = self._spatial_pooling.to(x.device)
            self._output_projection = self._output_projection.to(x.device)

        # Apply global average pooling to reduce spatial dimensions
        pooled = self._spatial_pooling(features)  # (batch, channels, 1, 1)

        # Flatten and apply final projection
        flattened = pooled.view(batch_size, -1)  # (batch, channels)
        return self._output_projection(flattened)
