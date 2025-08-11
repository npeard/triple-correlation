#!/usr/bin/env python

import logging

import numpy as np
import torch
from torch import nn, optim

from biphase.transformer.lightning_decoder import GPTDecoder


class TrainablePhase(nn.Module):
    """A trainable phase array that can be optimized with PyTorch."""

    def __init__(self, shape: int | tuple):
        """Initialize trainable phase parameters.

        Args:
            shape: Shape of the phase array
                  - For 1D: int (number of pixels)
                  - For 2D: tuple (nx, ny)
        """
        super().__init__()

        if isinstance(shape, int):
            self.phase = nn.Parameter(torch.randn(shape, dtype=torch.float32))
            self.is_1d = True
            self.num_pix = shape
        elif isinstance(shape, tuple):
            self.phase = nn.Parameter(torch.randn(shape, dtype=torch.float32))
            self.is_1d = False
            self.num_pix = shape
        else:
            raise ValueError('Shape must be int (1D) or tuple (2D)')

    def forward(self) -> torch.Tensor:
        """Return the current phase parameters."""
        return self.phase


class DifferentiableSolver:
    """Differentiable phase retrieval solver using PyTorch optimization.

    This solver takes cosPhi correlation data and optimizes a phase array
    to minimize the MSE between the input arccos(cosPhi) and the re-encoded
    phase using GPTDecoder encoding methods.
    """

    def __init__(
        self,
        cosPhi: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.01,
        log_level: str = 'WARNING',
    ):
        """Initialize the differentiable solver.

        Args:
            cosPhi: Correlation array
                   - For 1D: 2D array of shape (2*num_pix-1, 2*num_pix-1)
                   - For 2D: 4D array of shape (2*num_pix-1, 2*num_pix-1,
                             2*num_pix-1, 2*num_pix-1)
            epochs: Total number of training epochs
            lr: Learning rate for Adam optimizer
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.cosPhi = torch.tensor(cosPhi, dtype=torch.float32)
        self.epochs = epochs
        self.lr = lr
        self.ndim = len(cosPhi.shape)

        # Setup logging
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self.logger.setLevel(getattr(logging, log_level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Determine dimensionality and setup
        if self.ndim == 2:
            self.is_1d = True
            self.num_pix = int((cosPhi.shape[0] + 1) / 2)
            self.phase_shape = self.num_pix
            self.logger.debug(
                f'Initialized 1D differentiable solver with {self.num_pix} pixels'
            )
        elif self.ndim == 4:
            self.is_1d = False
            self.num_pix = int((cosPhi.shape[0] + 1) / 2)
            self.phase_shape = (self.num_pix, self.num_pix)
            self.logger.debug(
                f'Initialized 2D differentiable solver with {self.num_pix}x{self.num_pix} pixels'
            )
        else:
            raise ValueError(
                f'cosPhi must be 2D (for 1D solve) or 4D (for 2D solve), '
                f'got {self.ndim}D'
            )

        # Prepare target tensor (arccos of cosPhi)
        self.target = self._prepare_target()
        self.logger.debug('Target shape: %s', self.target.shape)

        # Initialize trainable phase
        self.trainable_phase = TrainablePhase(self.phase_shape)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainable_phase.to(self.device)
        self.target = self.target.to(self.device)
        self.cosPhi = self.cosPhi.to(self.device)

        self.logger.debug(f'Using device: {self.device}')

    def _prepare_target(self) -> torch.Tensor:
        """Prepare the target tensor by symmetrizing cosPhi and taking arccos.

        Returns:
            Target tensor for optimization
        """
        if self.is_1d:
            # 1D symmetrization - combine symmetric regions and average
            cosPhi_sym = (
                self.cosPhi[
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                ]
                + torch.flip(
                    self.cosPhi[
                        self.num_pix // 2 : self.num_pix,
                        self.num_pix // 2 : self.num_pix,
                    ],
                    dims=[0, 1],
                )
            ) / 2
        else:
            # 2D symmetrization - combine symmetric 4D regions and average
            cosPhi_sym = (
                self.cosPhi[
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                ]
                + torch.flip(
                    self.cosPhi[
                        self.num_pix // 2 : self.num_pix,
                        self.num_pix // 2 : self.num_pix,
                        self.num_pix // 2 : self.num_pix,
                        self.num_pix // 2 : self.num_pix,
                    ],
                    dims=[0, 1, 2, 3],
                )
            ) / 2

        # Take arccos and handle numerical issues
        cosPhi_sym = torch.clamp(cosPhi_sym, -1.0, 1.0)
        target_Phi = torch.arccos(cosPhi_sym)

        if self.is_1d:
            # For 1D, need to trim to remove zero row/column with no information
            Phi_dim = self.num_pix
            out_dim = Phi_dim - 1
            target_trimmed = target_Phi[1:Phi_dim, 1:Phi_dim]
        else:
            # For 2D, need to trim to remove zero rows/columns
            half_nx = self.num_pix
            half_ny = self.num_pix
            out_nx = half_nx - 1
            out_ny = half_ny - 1
            target_trimmed = target_Phi[1:half_nx, 1:half_ny, 1:half_nx, 1:half_ny]

        return target_trimmed.flatten()

    def _compute_loss(self, phase: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between target and re-encoded phase.

        Args:
            phase: Current phase estimate

        Returns:
            MSE loss
        """
        if self.is_1d:
            # Add batch dimension before using _encode from GPTDecoder
            encoded = GPTDecoder._encode(phase.unsqueeze(0))
        else:
            # Add batch dimension before using _encode_2D from GPTDecoder
            encoded = GPTDecoder._encode_2D(phase.unsqueeze(0))

        encoded_flat = encoded.flatten()
        zero_loss = torch.abs(phase[0]) if self.is_1d else torch.abs(phase[0, 0])
        return nn.MSELoss()(encoded_flat, self.target) + zero_loss

    def solve(self) -> tuple[torch.Tensor, list]:
        """Solve for the phase using PyTorch optimization.

        Returns:
            Tuple of (optimized_phase, loss_history)
        """
        self.logger.info(
            f'Starting differentiable phase retrieval for {self.num_pix} pixels'
        )
        self.logger.info(f'Training for {self.epochs} epochs with lr={self.lr}')

        # Setup optimizer
        optimizer = optim.AdamW(self.trainable_phase.parameters(), lr=self.lr)

        # Setup schedulers
        warmup_epochs = max(1, self.epochs // 10)  # 10% warmup
        cosine_epochs = self.epochs - warmup_epochs

        # Linear warmup scheduler
        def warmup_lambda(epoch: int) -> float:
            if warmup_epochs == 0:
                return 1.0
            return float(epoch + 1) / float(warmup_epochs)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )

        # Cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=0.0
        )

        # Sequential scheduler combining warmup and cosine annealing
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        loss_history = []

        # Training loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass
            phase = self.trainable_phase()
            loss = self._compute_loss(phase)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Logging
            if epoch % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
                current_lr = scheduler.get_last_lr()[0]
                self.logger.info(
                    f'Epoch {epoch:4d}/{self.epochs}: Loss = {loss_val:.6f}, LR = {current_lr:.6f}'
                )
            elif epoch < 10:
                current_lr = scheduler.get_last_lr()[0]
                self.logger.debug(
                    f'Epoch {epoch:4d}/{self.epochs}: Loss = {loss_val:.6f}, LR = {current_lr:.6f}'
                )

        final_phase = self.trainable_phase().detach()
        self.logger.info(f'Optimization completed. Final loss: {loss_history[-1]:.6f}')

        return final_phase, loss_history
