#!/usr/bin/env python

from typing import Any, Dict, Optional, Tuple
import torch
from torch import optim, nn
import lightning as L
from torch.nn import functional as F
import numpy as np
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
        task_name: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch model to train
            optimizer_name: Name of the optimizer to use
            optimizer_hparams: Hyperparameters for the optimizer
            scheduler_hparams: Hyperparameters for the learning rate scheduler
            loss_hparams: Hyperparameters for the loss function
            task_name: Name of the task (e.g. "phase_regression")
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
        self.task_name = task_name
        
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
                "monitor": self.scheduler_hparams.get("monitor", "val_loss")
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
        model_name: str = "GPT",
        model_hparams: Optional[Dict] = None,
        optimizer_name: str = "Adam",
        optimizer_hparams: Optional[Dict] = None,
        scheduler_hparams: Optional[Dict] = None,
        loss_hparams: Optional[Dict] = None,
        task_name: Optional[str] = None
    ):
        """
        Args:
            model_name: Name of the model (should be "GPT")
            model_hparams: Hyperparameters for the GPT model
            optimizer_name: Name of the optimizer to use
            optimizer_hparams: Hyperparameters for the optimizer
            scheduler_hparams: Hyperparameters for the learning rate scheduler
            loss_hparams: Hyperparameters for the loss function
            task_name: Name of the task
        """
        # Create GPT model
        if model_name != "GPT":
            raise ValueError("model_name must be 'GPT' for GPTDecoder")
        
        model_hparams = model_hparams or {}
        gpt_config = GPTConfig(**model_hparams)
        model = GPT(gpt_config)
        
        super().__init__(
            model=model,
            optimizer_name=optimizer_name,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
            task_name=task_name
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
        loss = nn.MSELoss()(y_hat, targets)
        
        # Add antisymmetry loss if enabled
        if self.loss_hparams.get("use_antisymmetry_loss", False):
            antisym_weight = self.loss_hparams.get("antisymmetry_weight", 1.0)
            loss += antisym_weight * self.antisymmetry_loss(y_hat)
        
        # Add encoding loss if enabled and x is provided
        if self.loss_hparams.get("use_encoding_loss", False) and x is not None:
            encoding_weight = self.loss_hparams.get("encoding_weight", 1.0)
            loss += encoding_weight * self.encoding_loss(y_hat, x)
        
        return loss
    
    @staticmethod
    def antisymmetry_loss(outputs: torch.Tensor) -> torch.Tensor:
        """Compute antisymmetry loss to enforce phase antisymmetry.
        
        Args:
            outputs: Model outputs to enforce antisymmetry on
        """
        positive_x = outputs[:, outputs.size(1) // 2 + 1:]
        negative_x = torch.flip(outputs[:, :outputs.size(1) // 2], [1])
        
        loss = torch.sum(torch.add(positive_x, negative_x)**2) + torch.sum(
            outputs[:, outputs.size(1) // 2]**2)
        return loss
    
    def encoding_loss(self, phase: torch.Tensor, absPhi: torch.Tensor) -> torch.Tensor:
        """Compute encoding loss to ensure phase can reconstruct input.
        
        Args:
            phase: Predicted phase
            absPhi: Input absolute Phi matrix
        """
        # Get dimensions
        batch_size = phase.size(0)
        n = phase.size(1)
        
        # Construct Phi matrix from phase
        Phi_recon = torch.zeros((batch_size, n, n), device=phase.device)
        
        # Fill in the matrix
        for i in range(n):
            for j in range(i + 1, n):
                phase_diff = phase[:, j] - phase[:, i]
                Phi_recon[:, i, j] = torch.cos(phase_diff)
                Phi_recon[:, j, i] = torch.cos(phase_diff)
        
        # Compare with input
        loss = nn.MSELoss()(torch.abs(Phi_recon), absPhi)
        return loss
    
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
        """Prediction step for GPT model.
        
        Args:
            batch: Input tensor
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader
        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x)
