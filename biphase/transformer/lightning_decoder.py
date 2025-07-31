#!/usr/bin/env python

from typing import Any, override

import lightning as L
import numpy as np
import torch
from torch import nn, optim

from biphase.transformer.nano_gpt import GPT, GPTConfig


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
    # Get dimensions
    batch_size, nx, ny = arr.shape

    # Use torch.roll for each dimension separately
    out = torch.roll(arr, shifts=shift_x, dims=1)  # Roll along x dimension
    return torch.roll(out, shifts=shift_y, dims=2)  # Roll along y dimension


class BaseLightningModule(L.LightningModule):
    """Base Lightning Module for all models."""

    def __init__(
        self,
        model_hparams: dict,
        optimizer_hparams: dict,
        scheduler_hparams: dict,
        loss_hparams: dict,
    ):
        """Args:
        model: PyTorch model to train
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function.
        """
        super().__init__()
        self.model_hparams = model_hparams

        # Set default optimizer hyperparameters if none provided
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.save_hyperparameters(ignore=['model'])
        self.model = self.create_model()

        torch.set_float32_matmul_precision('high')

    def _create_gpt_config(self) -> GPTConfig:
        """Create GPTConfig from model configuration."""
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
                f'Unsupported type for num_pix in _create_gpt_config: {type(num_pix)}'
            )

        return GPTConfig(
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            n_layer=self.model_hparams['n_layer'],
            n_head=self.model_hparams['n_head'],
            n_embd=self.model_hparams['n_embd'],
            dropout=self.model_hparams.get('dropout', 0.1),
            bias=self.model_hparams.get('bias', False),
            is_causal=self.model_hparams.get('is_causal', True),
        )

    def create_model(self) -> GPT:
        """Create model instance based on config."""
        model_type = self.model_hparams.pop('type')
        if model_type == 'GPT':
            return GPT(self._create_gpt_config())
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Configure optimizer
        optimizer_name = self.optimizer_hparams.pop('name')
        if optimizer_name == 'Adam':
            # Discard momentum hyperparameter for Adam optimizer
            _ = self.optimizer_hparams.pop('momentum', None)
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        # Configure multi-stage scheduler: linear warmup then cosine annealing
        warmup_epochs = self.scheduler_hparams.get('warmup_epochs', 0)
        cosine_epochs = self.scheduler_hparams.get('cosine_epochs', 0)
        eta_min = self.scheduler_hparams.get('eta_min', 0)
        T_max = self.scheduler_hparams.get('T_max', cosine_epochs)

        # LambdaLR for linear warmup
        def warmup_lambda(epoch: int) -> float:
            if warmup_epochs == 0:
                return 1.0
            return float(epoch + 1) / float(warmup_epochs)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    # TODO: what is this, where is it used?
    def _get_progress_bar_dict(self) -> dict[str, Any]:
        """Modify progress bar display."""
        items = super()._get_progress_bar_dict()
        items.pop('v_num', None)
        return items


class GPTDecoder(BaseLightningModule):
    """Lightning Module for training GPT models."""

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model_hparams: Hyperparameters for the GPT model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        num_pix: Number of pixels in the phase array, used to determine how to
        do the encoding loss, 1D vs 2D.
        """
        # Number of pixels in the phase array, used to determine how to
        # do the encoding loss, 1D vs 2D
        self.num_pix = eval(model_hparams['num_pix'])

        super().__init__(
            model_hparams=model_hparams,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )

    def loss_function(
        self,
        y_hat: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        """Custom loss function for GPT training.

        Args:
            y_hat: Model predictions
            targets: Ground truth targets
            x: Additional input tensor for encoding loss
            *args, **kwargs: Additional arguments
        """
        # Base loss (MSE)
        # I've found that comparing the absolute value trains better
        # because there is an overall global +/-1 sign ambiguity
        loss = nn.MSELoss()(torch.abs(y_hat), torch.abs(targets))
        # But we should also try to infer the overall sign
        loss += nn.MSELoss()(y_hat, targets)

        # Add encoding loss if enabled and x is provided
        if self.loss_hparams.get('encoding_weight', 0) > 0 and x is not None:
            encoding_weight = self.loss_hparams['encoding_weight']
            loss += encoding_weight * self.encoding_loss(y_hat, x)

        return loss

    def encoding_loss(self, phase: torch.Tensor, absPhi: torch.Tensor) -> torch.Tensor:
        """Compute loss between re-encoded abs(Phi) matrix and input abs(Phi) matrix.

        Args:
            phase: Predicted phase, single quadrant
            absPhi: Input absolute Phi matrix sequence, has length
            ((n//2+1)//2+1) - 1)**2 where n is the length of the predicted
            phase (nanoGPT is configured to predict the full antisymmetric
            output internally), and then we remove the zero value row and
            column with no information due to symmetry.

        Returns:
            torch.Tensor: MSE loss between re-encoded and input abs(Phi)
        """
        if isinstance(self.num_pix, tuple):
            # 2D phase has been flattened, reshape for _encode_2D compatibility
            # should have shape (batch_size, num_pix, num_pix)
            phase = phase.view(-1, self.num_pix[0], self.num_pix[1])
            encoded = self._encode_2D(phase).flatten(start_dim=1)
        elif isinstance(self.num_pix, int):
            encoded = self._encode(phase).flatten(start_dim=1)
        else:
            raise TypeError('num_pix must be int or tuple')

        return nn.MSELoss()(encoded, absPhi)

    @staticmethod
    @torch.jit.script
    def _encode(phase: torch.Tensor) -> torch.Tensor:
        """Memory-efficient re-encode abs(Phi) matrix from predicted phase.

        This version computes only the required output elements without creating
        the full-size intermediate tensor, saving memory and improving performance.

        Args:
            phase: Predicted phase, single quadrant included
        Returns:
            torch.Tensor: Re-encoded abs(Phi) matrix
        """
        batch_size = phase.size(0)
        phase_len = phase.size(1)
        Phi_dim = phase_len // 2 + 1

        # Calculate output dimensions (after trimming and removing zero row/column)
        out_dim = Phi_dim - 1  # Remove zero row and column

        # Initialize output tensor with the final size
        encoded = torch.zeros((batch_size, out_dim, out_dim), device=phase.device)

        # Compute only the required elements directly
        for i in range(1, Phi_dim):  # Start from 1 to skip zero row
            # Compute the rolled phase and phase differences for the output region only
            rolled_phase = torch.roll(phase, -i, dims=-1)
            phase_diff = (
                rolled_phase[:, 1:Phi_dim]
                - phase[:, 1:Phi_dim]
                - phase[:, i].unsqueeze(-1)
            )

            # Store the absolute value directly
            encoded[:, i - 1, :] = torch.abs(phase_diff)

        return encoded

    @staticmethod
    @torch.jit.script
    def _encode_2D(phase: torch.Tensor) -> torch.Tensor:
        """Memory-efficient re-encode abs(Phi) matrix from predicted 2D phase.

        This version computes only the required output elements without creating
        the full-size intermediate Phi tensor, saving memory and improving
        performance for large arrays.

        Args:
            phase: Predicted phase tensor of shape (batch_size, num_pix, num_pix)

        Returns:
            torch.Tensor: Re-encoded abs(Phi) 4D tensor of shape
                (batch_size, half_nx-1, half_ny-1, half_nx-1, half_ny-1)
        """
        # Get dimensions
        batch_size, nx, ny = phase.shape
        assert nx == ny, 'Phase must be square'

        # Calculate output dimensions (after trimming and removing zero row/column)
        half_nx = nx // 2 + 1
        half_ny = ny // 2 + 1
        out_nx = half_nx - 1  # Remove zero row
        out_ny = half_ny - 1  # Remove zero column

        # Initialize output tensor with the final size
        Phi_abs = torch.zeros(
            (batch_size, out_nx, out_ny, out_nx, out_ny), device=phase.device
        )

        # Compute only the required elements directly
        for nx_shift in range(1, half_nx):  # Start from 1 to skip zero row
            for ny_shift in range(1, half_ny):  # Start from 1 to skip zero column
                # Shift the phase array
                shifted_phase = roll2d_torch(phase, -nx_shift, -ny_shift)

                # Get the phase at the shift position for broadcasting
                phase_at_shift = phase[:, nx_shift, ny_shift].unsqueeze(1).unsqueeze(2)

                # Compute phase difference for the output region only
                phase_diff = (
                    shifted_phase[:, 1:half_nx, 1:half_ny]
                    - phase[:, 1:half_nx, 1:half_ny]
                    - phase_at_shift
                )

                # Store the absolute value directly
                Phi_abs[:, nx_shift - 1, ny_shift - 1, :, :] = torch.abs(phase_diff)

        return Phi_abs

    def compute_inputs_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute input Phi matrices from phase data using optimized PyTorch methods.
        Disables gradient tracking during computation to save memory.

        Args:
            phases: Phase data tensor of shape (batch_size, phase_length)

        Returns:
            torch.Tensor: Computed input Phi matrices (flattened)
        """
        # Disable gradient tracking for input computation to save memory
        with torch.no_grad():
            # Process based on dimensionality
            if isinstance(self.num_pix, int | np.int64):
                # 1D case - reshape if needed
                batch_size = phases.shape[0]
                # For 1D, we need to ensure phases has the right shape for _encode
                # _encode expects (batch_size, phase_length)
                if len(phases.shape) == 1:
                    phases = phases.unsqueeze(0)  # Add batch dimension if missing

                # Use the optimized _encode method for 1D
                inputs = self._encode(phases)

            elif isinstance(self.num_pix, tuple):
                # 2D case - reshape flattened phase
                batch_size = phases.shape[0]
                # Reshape to (batch_size, height, width) for _encode_2D
                phases_reshaped = phases.view(
                    batch_size, self.num_pix[0], self.num_pix[1]
                )

                # Use the optimized _encode_2D method
                inputs = self._encode_2D(phases_reshaped)

            else:
                raise TypeError(f'Unsupported num_pix type: {type(self.num_pix)}')

            # Flatten the inputs for the model
            inputs_flat = inputs.flatten(start_dim=1)

        # Create a new tensor that requires grad for the model
        # This detaches from the computation graph while preserving device and dtype
        return inputs_flat.clone().detach()

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step for GPT model.

        Args:
            batch: Phase data tensor
            batch_idx: Index of current batch
        """
        # Now batch is just the phase data
        phases = batch

        # Compute inputs on the fly
        inputs = self.compute_inputs_from_phases(phases)

        # Forward pass with computed inputs
        y_hat = self(inputs)

        # Compute loss
        loss = self.loss_function(y_hat, phases, inputs)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # noqa: ARG002
        """Validation step for GPT model.

        Args:
            batch: Phase data tensor
            batch_idx: Index of current batch
        """
        # Now batch is just the phase data
        phases = batch

        # Compute inputs on the fly
        inputs = self.compute_inputs_from_phases(phases)

        # Forward pass with computed inputs
        y_hat = self(inputs)

        # Compute loss
        loss = self.loss_function(y_hat, phases, inputs)

        # Verify encoding/unpacking order during sanity check (first validation)
        if self.trainer.sanity_checking:
            # Re-encode the targets and compare with input, they should be identical
            encoding_loss = self.encoding_loss(phases, inputs)
            assert encoding_loss < 1e-6, (
                f'Encoding verification failed! Loss: {encoding_loss:.2e}'
            )

        self.log('val_loss', loss, prog_bar=True)

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Test step for GPT model.

        Args:
            batch: Phase data tensor
            batch_idx: Index of current batch
        """
        # Now batch is just the phase data
        phases = batch

        # Compute inputs on the fly
        inputs = self.compute_inputs_from_phases(phases)

        # Forward pass with computed inputs
        y_hat = self(inputs)

        # Compute loss
        loss = self.loss_function(y_hat, phases, inputs)

        self.log('test_loss', loss)

    @override
    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Prediction step for GPT model. Return all relevant quantities for plotting.

        Args:
            batch: Phase data tensor
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader
        """
        # Now batch is just the phase data
        phases = batch

        # Compute inputs on the fly
        inputs = self.compute_inputs_from_phases(phases)

        # Forward pass with computed inputs
        predictions = self.model(inputs)

        if isinstance(self.num_pix, tuple):
            # 2D phase has been flattened, reshape for _encode_2D compatibility
            # should have shape (batch_size, num_pix, num_pix)
            predictions = predictions.view(-1, self.num_pix[0], self.num_pix[1])
            encoded = self._encode_2D(predictions)
            inputs_reshaped = inputs.view_as(encoded)
            # Get a 2D slice of the 4D inputs and re-encoded outputs
            inputs_reshaped = inputs_reshaped[:, 1, 1, :, :]
            encoded = encoded[:, 1, 1, :, :]
            # TODO: implement antisymmetry of the predictions and targets for plotting
            full_predictions = predictions
            full_targets = phases.view_as(predictions)
        elif isinstance(self.num_pix, int):
            encoded = self._encode(predictions)
            inputs_reshaped = inputs.view_as(encoded)
            # Implement antisymmetry of the predictions and targets for plotting
            full_predictions = torch.concat(
                [-torch.fliplr(predictions), predictions[:, 1:]], dim=1
            )
            full_targets = torch.concat([-torch.fliplr(phases), phases[:, 1:]], dim=1)
        else:
            raise TypeError('num_pix must be int or tuple')

        return full_predictions, full_targets, encoded, inputs_reshaped
