#!/usr/bin/env python

"""Unified Lightning module for phase retrieval training."""

from typing import Any

import lightning as L
import numpy as np
import torch
from torch import nn, optim

from biphase.transformer.encoding import encode_1d, encode_2d, encoding_loss
from biphase.transformer.models.factory import create_model


class LitModule(L.LightningModule):
    """Lightning Module for training phase retrieval models."""

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Initialize LitModule.

        Args:
            model_hparams: Hyperparameters for the model
            optimizer_hparams: Hyperparameters for the optimizer
            scheduler_hparams: Hyperparameters for the learning rate scheduler
            loss_hparams: Hyperparameters for the loss function
        """
        super().__init__()
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.save_hyperparameters(ignore=['model'])

        torch.set_float32_matmul_precision('high')

        # Parse num_pix for encoding dimensions
        self.num_pix = eval(model_hparams['num_pix'])

        # Create model via factory
        model_type = model_hparams.get('type', 'GPT')
        self.model = create_model(model_type, model_hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer_name = self.optimizer_hparams.pop('name')
        if optimizer_name == 'Adam':
            _ = self.optimizer_hparams.pop('momentum', None)
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        warmup_epochs = self.scheduler_hparams['warmup_epochs']
        eta_min = self.scheduler_hparams['eta_min']
        T_max = self.scheduler_hparams['T_max']

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

    def loss_function(
        self,
        y_hat: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor | None = None,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        """Custom loss function for training.

        Args:
            y_hat: Model predictions
            targets: Ground truth targets
            x: Additional input tensor for encoding loss
        """
        abs_loss = nn.MSELoss()(torch.abs(y_hat), torch.abs(targets))
        sign_loss = nn.MSELoss()(y_hat, targets)

        enc_loss = 0
        encoding_weight = self.loss_hparams['encoding_weight']
        if encoding_weight > 0 and x is not None:
            enc_loss += encoding_loss(y_hat, x, self.num_pix)

        return {
            'abs': abs_loss,
            'sign': sign_loss,
            'encoding': enc_loss,
            'total': encoding_weight * enc_loss + abs_loss + sign_loss,
        }

    def compute_inputs_from_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute input Phi matrices from phase data.

        Args:
            phases: Phase data tensor of shape (batch_size, phase_length)

        Returns:
            torch.Tensor: Computed input Phi matrices (flattened)
        """
        with torch.no_grad():
            if isinstance(self.num_pix, int | np.int64):
                if len(phases.shape) == 1:
                    phases = phases.unsqueeze(0)
                inputs = encode_1d(phases)

            elif isinstance(self.num_pix, tuple):
                batch_size = phases.shape[0]
                phases_reshaped = phases.view(
                    batch_size, self.num_pix[0], self.num_pix[1]
                )
                inputs = encode_2d(phases_reshaped)

            else:
                raise TypeError(f'Unsupported num_pix type: {type(self.num_pix)}')

            inputs_flat = inputs.flatten(start_dim=1)

        return inputs_flat.clone().detach()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        phases = batch
        inputs = self.compute_inputs_from_phases(phases)
        y_hat = self(inputs)
        loss_dict = self.loss_function(y_hat, phases, inputs)

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'train_{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),
                on_epoch=True,
            )

        return loss_dict['total']

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # noqa: ARG002
        phases = batch
        inputs = self.compute_inputs_from_phases(phases)
        y_hat = self(inputs)
        loss_dict = self.loss_function(y_hat, phases, inputs)

        if self.trainer.sanity_checking:
            loss_dict = self.loss_function(phases, phases, inputs)
            assert loss_dict['encoding'] < 1e-6, (
                f'Encoding verification failed! Loss: {loss_dict["encoding"]:.2e}'
            )

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'val_{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),
                sync_dist=True,
            )

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # noqa: ARG002
        phases = batch
        inputs = self.compute_inputs_from_phases(phases)
        y_hat = self(inputs)
        loss_dict = self.loss_function(y_hat, phases, inputs)

        for loss_name, loss_value in loss_dict.items():
            self.log(f'test_{loss_name}_loss', loss_value, sync_dist=True)

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> torch.Tensor:
        phases = batch
        inputs = self.compute_inputs_from_phases(phases)
        predictions = self.model(inputs)

        if isinstance(self.num_pix, tuple):
            predictions = predictions.view(-1, self.num_pix[0], self.num_pix[1])
            encoded = encode_2d(predictions)
            inputs_reshaped = inputs.view_as(encoded)
            inputs_reshaped = inputs_reshaped[:, 1, 1, :, :]
            encoded = encoded[:, 1, 1, :, :]
            full_predictions = predictions
            full_targets = phases.view_as(predictions)
        elif isinstance(self.num_pix, int):
            encoded = encode_1d(predictions)
            inputs_reshaped = inputs.view_as(encoded)
            full_predictions = torch.concat(
                [-torch.fliplr(predictions), predictions[:, 1:]], dim=1
            )
            full_targets = torch.concat([-torch.fliplr(phases), phases[:, 1:]], dim=1)
        else:
            raise TypeError('num_pix must be int or tuple')

        return full_predictions, full_targets, encoded, inputs_reshaped
