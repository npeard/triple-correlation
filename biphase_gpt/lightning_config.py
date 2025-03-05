#!/usr/bin/env python

from typing import Any, Dict, Optional, Tuple
import torch
from torch import optim, nn
import lightning as L
from torch.nn import functional as F
from biphase_gpt.datasets import AbsPhiDataset
from biphase_gpt.nano_gpt import GPT, GPTConfig


class BaseLightningModule(L.LightningModule):
    """Base Lightning Module for all models"""
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer_name: str = "Adam",
        optimizer_hparams: Optional[Dict] = None,
        scheduler_hparams: Optional[Dict] = None,
        loss_hparams: Optional[Dict] = None,
    ):
        """
        Args:
            model: PyTorch model to train
            optimizer_name: Name of the optimizer to use
            optimizer_hparams: Hyperparameters for the optimizer
            scheduler_hparams: Hyperparameters for the learning rate scheduler
            loss_hparams: Hyperparameters for the loss function
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        # Set default optimizer hyperparameters if none provided
        self.optimizer_hparams = optimizer_hparams or {"lr": 1e-3, "weight_decay": 1e-5}
        self.scheduler_hparams = scheduler_hparams or {
            "milestones": [250, 450],
            "gamma": 0.1
        }
        self.loss_hparams = loss_hparams or {}
        
        torch.set_float32_matmul_precision('high')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Configure optimizer
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")
        
        # Configure scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_hparams)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }
    
    def _get_progress_bar_dict(self) -> Dict[str, Any]:
        """Modify progress bar display"""
        items = super()._get_progress_bar_dict()
        items.pop("v_num", None)
        return items


class GPTDecoder(BaseLightningModule):
    """Lightning Module for training GPT models"""
    
    def __init__(
        self,
        model_type: str = "GPT",
        model_hparams: Optional[Dict] = None,
        optimizer_name: str = "Adam",
        optimizer_hparams: Optional[Dict] = None,
        scheduler_hparams: Optional[Dict] = None,
        loss_hparams: Optional[Dict] = None,
    ):
        """
        Args:
            model_type: Name of the model (should be "GPT")
            model_hparams: Hyperparameters for the GPT model
            optimizer_name: Name of the optimizer to use
            optimizer_hparams: Hyperparameters for the optimizer
            scheduler_hparams: Hyperparameters for the learning rate scheduler
            loss_hparams: Hyperparameters for the loss function
        """
        # Create GPT model
        if model_type != 'GPT':
            raise ValueError('model_type must be \'GPT\' for GPTDecoder')
        
        model_hparams = model_hparams or {}
        gpt_config = GPTConfig(**model_hparams)
        model = GPT(gpt_config)
        
        super().__init__(
            model=model,
            optimizer_name=optimizer_name,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )
    
    def loss_function(self, y_hat: torch.Tensor, targets: torch.Tensor, x: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
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
        
        # Add encoding loss if enabled and x is provided
        if self.loss_hparams.get("use_encoding_loss", False) and x is not None:
            encoding_weight = self.loss_hparams.get("encoding_weight", 1.0)
            loss += encoding_weight * self.encoding_loss(y_hat, x)
        
        return loss
    
    def encoding_loss(self, phase: torch.Tensor, absPhi: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between re-encoded abs(Phi) matrix and input abs(Phi) matrix.
        
        Args:
            phase: Predicted phase, both quadrants included
            absPhi: Input absolute Phi matrix sequence, has length 
            ((n//2+1)//2+1) - 1)**2 where n is the length of the predicted 
            phase (nanoGPT is configured to predict the full antisymmetric 
            output internally), and then we remove the zero value row and 
            column with no information due to symmetry.
        Returns:
            torch.Tensor: MSE loss between re-encoded and input abs(Phi)
        """
        encoded = self._encode(phase)

        if self.loss_hparams.get("unpack_diagonals", False):
            encoded = self._unpack_by_diagonals_batched(encoded)
        else:
            encoded = encoded.flatten(start_dim=1)  # Flatten
        
        # Compare with input. Both should be flat and unpacked 
        # from square in the same order.
        loss = nn.MSELoss()(encoded, absPhi)
        return loss

    @staticmethod
    def _encode(phase: torch.Tensor) -> torch.Tensor:
        """Re-encode abs(Phi) matrix from predicted phase.
        
        Args:
            phase: Predicted phase, both quadrants included
        Returns:
            torch.Tensor: Re-encoded abs(Phi) matrix
        """
        batch_size = phase.size(0)
        phase_dim = phase.size(1)//2 + 1
        encoded_dim = (phase_dim//2 + 1)

        # Phase in positive quadrant
        phase = torch.flip(phase[:, :phase_dim], dims=[1])
        
        # Re-encode abs(Phi) matrix from predicted phase
        encoded = torch.zeros((batch_size, phase_dim, phase_dim), device=phase.device)
        
        for i in range(phase_dim):
            encoded[:, i, :] = (torch.roll(phase, -i, dims=-1) - phase - phase[:, i].unsqueeze(-1))

        encoded = encoded[:, :encoded_dim, :encoded_dim]
        # Remove zero row and column
        encoded = encoded[:, 1:, 1:]
        # absolute value
        encoded = torch.abs(encoded)
        return encoded
    
    @staticmethod
    def _unpack_by_diagonals_batched(x: torch.Tensor) -> torch.Tensor:
        """
        Unpack a batch of square matrices by diagonals, similar to AbsPhiDataset.unpack_by_diagonals
        but handles batched input.
        
        Args:
            x: Input tensor of shape (batch_size, n, n)
            
        Returns:
            torch.Tensor: Flattened tensor containing diagonals in order from top-right to bottom-left
        """
        # First flip left-right for each matrix in batch
        x = torch.flip(x, dims=[-1])
        
        # Get dimensions
        batch_size, n, _ = x.shape
        assert x.size(-1) == x.size(-2), "Input tensors must be square"
        
        # Extract diagonals from offset n-1 to -(n-1) for each matrix in batch
        # and concatenate them directly without padding
        all_diagonals = []
        for b in range(batch_size):
            batch_diagonals = []
            for offset in range(n-1, -(n), -1):
                diag = torch.diagonal(x[b], offset=offset)
                batch_diagonals.append(diag)
            # Concatenate all diagonals for this batch item
            all_diagonals.append(torch.cat(batch_diagonals))
        
        # Stack results from each batch
        return torch.stack(all_diagonals)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for GPT model.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, x)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step for GPT model.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, x)
        
        # Verify encoding/unpacking order during sanity check (first validation)
        if self.trainer.sanity_checking:
            # Re-encode the targets and compare with input
            encoded = self._encode(y)
            if self.loss_hparams.get("unpack_diagonals", False):
                encoded = self._unpack_by_diagonals_batched(encoded)
            else:
                encoded = encoded.flatten(start_dim=1)
            
            encoding_loss = nn.MSELoss()(encoded, x)
            assert encoding_loss < 1e-6, f"Encoding verification failed! Loss: {encoding_loss:.2e}"
            print(f"✓ Encoding verification passed (loss: {encoding_loss:.2e})")
        
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step for GPT model.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, x)
        
        self.log('test_loss', loss)
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for GPT model. Return all relevant quantities for plotting.
        
        Args:
            batch: Input tensor
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader
        """
        x, y = batch
        predictions = self.model(x)
        encoded = self._encode(predictions)
        # reshape x to square
        x = x.view_as(encoded)
        # Use the same unpacking logic as in AbsPhiDataset
        if self.loss_hparams.get("unpack_diagonals", False):
            print("Unpacking diagonals...")
            encoded = self._unpack_by_diagonals_batched(encoded)
        else:
            print("Unpacking flat...")
            encoded = encoded.flatten(start_dim=1)  # Flatten
        # reshape encoded to square to match x
        encoded = encoded.view_as(x)
        return predictions, y, encoded, x
