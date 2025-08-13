#!/usr/bin/env python

from typing import override

import numpy as np
import torch

from biphase.transformer.gpt_decoder import GPTDecoder
from biphase.transformer.multidim_tcn import TCN2D, TCN2DConfig


class TCNDecoder(GPTDecoder):
    """Lightning Module for training TCN2D models.

    Inherits from GPTDecoder but uses TCN2D instead of GPT and overrides
    input computation to skip flattening for 2D inputs.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model_hparams: Hyperparameters for the TCN2D model
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

        # Determine whether we are training a 1D or 2D phase prediction model
        # and set the sequence lengths accordingly
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

        # Dynamically get TCN2DConfig field names and filter params
        import inspect

        tcn2d_config_fields = set(inspect.signature(TCN2DConfig).parameters.keys())
        filtered_params = {
            k: v for k, v in self.model_hparams.items() if k in tcn2d_config_fields
        }

        return TCN2DConfig(
            in_seq_len=in_seq_len, out_seq_len=out_seq_len, **filtered_params
        )

    @override
    def create_model(self) -> TCN2D:
        """Create TCN2D model instance based on config."""
        return TCN2D(self._create_tcn2d_config())

    @override
    def compute_inputs_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute input matrices from phase data, keeping 2D shape for TCN2D.

        Overrides GPTDecoder method to skip flattening step when generating inputs,
        allowing the TCN2D to work with 2D structured data.

        Args:
            phases: Phase data tensor of shape (batch_size, phase_length)

        Returns:
            torch.Tensor: Computed input matrices (2D shape preserved for TCN2D)
        """
        # Disable gradient tracking for input computation to save memory
        with torch.no_grad():
            # Process based on dimensionality
            if isinstance(self.num_pix, int | np.int64):
                # 1D case - reshape if needed
                # batch_size = phases.shape[0]
                # For 1D, we need to ensure phases has the right shape for _encode
                # _encode expects (batch_size, phase_length)
                if len(phases.shape) == 1:
                    phases = phases.unsqueeze(0)  # Add batch dimension if missing

                # Use the optimized _encode method for 1D
                inputs = self._encode(phases)
                # Keep 2D structure - don't flatten for TCN2D
                # inputs shape: (batch_size, height, width)

            elif isinstance(self.num_pix, tuple):
                raise NotImplementedError(
                    '2D phase prediction not implemented for TCNDecoder'
                )

            else:
                raise TypeError(f'Unsupported num_pix type: {type(self.num_pix)}')

        # Create a new tensor that requires grad for the model
        # This detaches from the computation graph while preserving device and dtype
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
        """Custom loss function for TCN2D training.

        Overrides GPTDecoder loss to flatten inputs to correct shape for
        the parent loss function when needed.

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
