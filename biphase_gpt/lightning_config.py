#!/usr/bin/env python

from typing import Any

import lightning as L
import torch
import numpy as np
from torch import nn, optim

from biphase_gpt.nano_gpt import GPT, GPTConfig


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
    out = torch.roll(out, shifts=shift_y, dims=2)  # Roll along y dimension

    return out


class BaseLightningModule(L.LightningModule):
    """Base Lightning Module for all models"""

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
        loss_hparams: Hyperparameters for the loss function
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
        """Create GPTConfig from model configuration"""
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
            raise ValueError(
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
        """Create model instance based on config"""
        model_type = self.model_hparams.pop('type')
        if model_type == 'GPT':
            print('Creating GPT model...')
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
        def warmup_lambda(epoch):
            if warmup_epochs == 0:
                return 1.0
            return float(epoch + 1) / float(warmup_epochs)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
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
        """Modify progress bar display"""
        items = super()._get_progress_bar_dict()
        items.pop('v_num', None)
        return items


class GPTDecoder(BaseLightningModule):
    """Lightning Module for training GPT models"""

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
        do the encoding loss, 1D vs 2D
        """

        # Number of pixels in the phase array, used to determine how to
        # do the encoding loss, 1D vs 2D
        self.num_pix = eval(model_hparams['num_pix'])
        print("self.num_pix", self.num_pix)

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
        *args,
        **kwargs,
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
        if self.loss_hparams.get('use_encoding_loss', False) and x is not None:
            encoding_weight = self.loss_hparams.get('encoding_weight', 1.0)
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
            # 2D phase has been flattened, so we need to reshape it to be compatible with _encode_2D
            # should have shape (batch_size, num_pix, num_pix)
            phase = phase.view(-1, self.num_pix[0], self.num_pix[1])
            encoded = self._encode_2D(phase).flatten(start_dim=1)
        elif isinstance(self.num_pix, int):
            encoded = self._encode(phase).flatten(start_dim=1)
        else:
            raise ValueError('num_pix must be int or tuple')

        loss = nn.MSELoss()(encoded, absPhi)

        return loss

    @staticmethod
    @torch.jit.script
    def _encode(phase: torch.Tensor) -> torch.Tensor:
        """Re-encode abs(Phi) matrix from predicted phase.

        Args:
            phase: Predicted phase, single quadrant included
        Returns:
            torch.Tensor: Re-encoded abs(Phi) matrix
        """
        batch_size = phase.size(0)
        Phi_dim = phase.size(1) // 2 + 1

        # Re-encode abs(Phi) matrix from predicted phase
        encoded = torch.zeros(
            (batch_size, phase.size(1), phase.size(1)), device=phase.device
        )

        for i in range(phase.size(1)):
            encoded[:, i, :] = (
                torch.roll(phase, -i, dims=-1) - phase - phase[:, i].unsqueeze(-1)
            )

        encoded = encoded[:, :Phi_dim, :Phi_dim]
        # Remove zero row and column
        encoded = encoded[:, 1:, 1:]
        # absolute value
        encoded = torch.abs(encoded)
        return encoded

    @staticmethod
    @torch.jit.script
    def _encode_2D(phase: torch.Tensor) -> torch.Tensor:
        """Re-encode abs(Phi) matrix from predicted 2D phase.

        This is a 2D version of _encode that works with 2D phase data, similar to
        Fluorescence2D.compute_Phi_from_phase but implemented in PyTorch.

        Args:
            phase: Predicted phase tensor of shape (batch_size, num_pix, num_pix)

        Returns:
            torch.Tensor: Re-encoded abs(Phi) 4D tensor of shape (batch_size, half_nx, half_ny, half_nx, half_ny)
        """
        # Get dimensions
        batch_size, nx, ny = phase.shape
        assert nx == ny, 'Phase must be square'

        # Initialize output tensor
        Phi = torch.zeros((batch_size, nx, ny, nx, ny), device=phase.device)

        # Compute Phi using nested loops (similar to the numba implementation)
        for nx_shift in range(nx):
            for ny_shift in range(ny):
                # Shift the phase array
                shifted_phase = roll2d_torch(phase, -nx_shift, -ny_shift)

                # Compute the phase difference
                # For each batch, subtract the original phase and the phase at the shift position
                phase_at_shift = (
                    phase[:, nx_shift, ny_shift].unsqueeze(1).unsqueeze(2)
                )  # Add dimensions for broadcasting
                Phi[:, nx_shift, ny_shift, :, :] = (
                    shifted_phase - phase - phase_at_shift
                )

        # Trim to match the expected output dimensions
        half_nx = nx // 2 + 1
        half_ny = ny // 2 + 1
        Phi = Phi[:, :half_nx, :half_ny, :half_nx, :half_ny]

        # Remove zero row and column
        Phi = Phi[:, 1:, 1:, 1:, 1:]

        # Take absolute value for the final output
        Phi_abs = torch.abs(Phi)

        return Phi_abs

    def compute_inputs_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute input Phi matrices from phase data on the fly using optimized PyTorch methods.
        Disables gradient tracking during computation to save memory.
        
        Args:
            phases: Phase data tensor of shape (batch_size, phase_length)
            
        Returns:
            torch.Tensor: Computed input Phi matrices (flattened)
        """
        # Disable gradient tracking for input computation to save memory
        with torch.no_grad():
            # Process based on dimensionality
            if isinstance(self.num_pix, int) or isinstance(self.num_pix, np.int64):
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
                phases_reshaped = phases.view(batch_size, self.num_pix[0], self.num_pix[1])
                
                # Use the optimized _encode_2D method
                inputs = self._encode_2D(phases_reshaped)
                
            else:
                raise ValueError(f"Unsupported num_pix type: {type(self.num_pix)}")
            
            # Flatten the inputs for the model
            inputs_flat = inputs.flatten(start_dim=1)
            
        # Create a new tensor that requires grad for the model
        # This detaches from the computation graph while preserving device and dtype
        return inputs_flat.clone().detach()
    
    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
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

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> None:
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
            print(f'✓ Encoding verification passed (loss: {encoding_loss:.2e})')

        self.log('val_loss', loss, prog_bar=True)

    def test_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> None:
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
            # 2D phase has been flattened, so we need to reshape it to be compatible with _encode_2D
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
            raise ValueError('num_pix must be int or tuple')

        return full_predictions, full_targets, encoded, inputs_reshaped
