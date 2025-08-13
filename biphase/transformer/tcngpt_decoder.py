#!/usr/bin/env python

from typing import override

import torch
from torch import nn

from biphase.transformer.gpt_decoder import GPTDecoder
from biphase.transformer.multidim_tcn import TCN2D, TCN2DConfig
from biphase.transformer.nano_gpt import GPT, GPTConfig


class TCNGPTDecoder(GPTDecoder):
    """Lightning Module combining TCN2D for 2D input processing with GPT.

    This module uses TCN2D to extract features from 2D correlation matrices and feeds
    the extracted 1D sequence to a GPT model for final phase prediction.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model_hparams: Combined hyperparameters for both TCN2D and GPT models
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function.
        """
        super().__init__(
            model_hparams=model_hparams,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _create_tcn2d_config(self) -> TCN2DConfig:
        """Create TCN2DConfig from model configuration."""
        num_pix = self.model_hparams['num_pix']
        if isinstance(num_pix, str):
            num_pix = eval(num_pix)

        # Determine sequence lengths based on num_pix
        if isinstance(num_pix, tuple):
            num_pix = num_pix[0]
            Phi_dim = num_pix // 2 + 1
            Phi_dim -= 1  # for removal of zero-valued row/column with no information
            in_seq_len = Phi_dim**4
            out_seq_len = num_pix**2
        elif isinstance(num_pix, int):
            Phi_dim = num_pix // 2 + 1
            Phi_dim -= 1  # for removal of zero-valued row/column with no information
            in_seq_len = Phi_dim**2
            out_seq_len = num_pix
        else:
            raise TypeError(
                f'Unsupported type for num_pix in _create_tcn2d_config: {type(num_pix)}'
            )

        # Filter parameters for TCN2D config
        import inspect

        tcn2d_config_fields = set(inspect.signature(TCN2DConfig).parameters.keys())
        tcn_params = {
            k: v
            for k, v in self.model_hparams.items()
            if k in tcn2d_config_fields
            and k.startswith(
                ('embd_tcn_', 'input_dim', 'output_dim', 'dropout', 'bias')
            )
        }

        return TCN2DConfig(in_seq_len=in_seq_len, out_seq_len=out_seq_len, **tcn_params)

    def _create_gpt_config(self) -> GPTConfig:
        """Create GPTConfig from model configuration for the GPT component."""
        # GPT input will be the TCN2D output projected to a sequence
        # We need to determine the appropriate sequence length for GPT
        gpt_seq_len = self.model_hparams.get('gpt_seq_len', 256)

        num_pix = self.model_hparams['num_pix']
        if isinstance(num_pix, str):
            num_pix = eval(num_pix)

        # Output sequence length for final prediction
        if isinstance(num_pix, tuple):
            out_seq_len = num_pix[0] ** 2
        elif isinstance(num_pix, int):
            out_seq_len = num_pix
        else:
            raise TypeError(f'Unsupported type for num_pix: {type(num_pix)}')

        # Filter parameters for GPT config
        import inspect

        gpt_config_fields = set(inspect.signature(GPTConfig).parameters.keys())
        gpt_params = {
            k: v
            for k, v in self.model_hparams.items()
            if k in gpt_config_fields and not k.startswith('embd_tcn_')
        }

        return GPTConfig(in_seq_len=gpt_seq_len, out_seq_len=out_seq_len, **gpt_params)

    @override
    def create_model(self) -> 'TCNGPTModel':
        """Create combined TCN2D + GPT model instance."""
        tcn_config = self._create_tcn2d_config()
        gpt_config = self._create_gpt_config()
        return TCNGPTModel(tcn_config, gpt_config)

    @override
    def compute_inputs_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute input matrices from phase data, keeping 2D shape for TCN2D.

        Args:
            phases: Phase data tensor of shape (batch_size, phase_length)

        Returns:
            torch.Tensor: Computed input matrices (2D shape preserved for TCN2D)
        """
        # Disable gradient tracking for input computation to save memory
        with torch.no_grad():
            # Process based on dimensionality
            if isinstance(self.num_pix, int):
                # 1D case - reshape if needed
                if len(phases.shape) == 1:
                    phases = phases.unsqueeze(0)  # Add batch dimension if missing

                # Use the optimized _encode method for 1D
                inputs = self._encode(phases)
                # Keep 2D structure - don't flatten for TCN2D

            elif isinstance(self.num_pix, tuple):
                raise NotImplementedError(
                    '2D phase prediction not implemented for TCNGPTDecoder'
                )

            else:
                raise TypeError(f'Unsupported num_pix type: {type(self.num_pix)}')

        # Create a new tensor that requires grad for the model
        return inputs.clone().detach()

    @override
    def loss_function(
        self,
        y_hat: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        """Custom loss function for TCNGPT training.

        Args:
            y_hat: Model predictions
            targets: Ground truth targets
            x: Additional input tensor for encoding loss (2D from TCN2D)
            *args, **kwargs: Additional arguments
        """
        # For encoding loss computation, we need to flatten x if it's 2D
        # so it matches the expected format in GPTDecoder.loss_function
        x_flattened = (
            x.flatten(start_dim=1) if x is not None and len(x.shape) == 3 else x
        )

        # Call parent loss function with flattened inputs
        return super().loss_function(y_hat, targets, x_flattened, *_args, **_kwargs)


class TCNGPTModel(nn.Module):
    """Combined model using TCN2D to process 2D inputs and GPT for sequence modeling."""

    def __init__(self, tcn_config: TCN2DConfig, gpt_config: GPTConfig):
        super().__init__()
        self.tcn_config = tcn_config
        self.gpt_config = gpt_config

        # Create TCN2D for 2D feature extraction
        self.tcn2d = TCN2D(tcn_config)

        # Create GPT for sequence modeling
        self.gpt = GPT(gpt_config)

        # Projection layer to convert TCN2D output to GPT input format
        # TCN2D outputs features of size (output_dim * out_seq_len)
        # GPT expects input of size (seq_len, input_dim)
        tcn_output_size = tcn_config.output_dim * tcn_config.out_seq_len
        gpt_input_size = gpt_config.in_seq_len * gpt_config.input_dim

        self.tcn_to_gpt_projection = nn.Sequential(
            nn.Linear(tcn_output_size, gpt_input_size),
            nn.LayerNorm(gpt_input_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through combined TCN2D + GPT model.

        Args:
            x: Input tensor from compute_inputs_from_phases with 2D structure
               Expected shape: (batch_size, height, width)

        Returns:
            Tensor: Phase predictions from GPT model
        """
        batch_size = x.size(0)

        # Pass through TCN2D to extract features from 2D input
        tcn_features = self.tcn2d(x)  # Shape: (batch_size, tcn_output_size)

        # Project TCN features to GPT input format
        gpt_input = self.tcn_to_gpt_projection(tcn_features)
        # Shape: (batch_size, gpt_input_size)

        # Reshape for GPT: (batch_size, seq_len, input_dim)
        gpt_input = gpt_input.view(batch_size, self.gpt_config.in_seq_len)

        # Pass through GPT for final prediction
        return self.gpt(gpt_input)
