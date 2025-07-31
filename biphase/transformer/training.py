#!/usr/bin/env python

import contextlib
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Union

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from biphase.transformer.datasets import get_data_loaders
from biphase.transformer.lightning_decoder import BaseLightningModule, GPTDecoder


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    model_config: dict[str, Any]
    training_config: dict[str, Any]
    data_config: dict[str, Any]
    loss_config: dict[str, Any]
    is_hyperparameter_search: bool = False
    search_space: dict[str, list[Any]] | None = None

    def __post_init__(self):
        """Set default values for running checkpoints. Check points do not
        need all config variables.
        """
        if self.model_config == {}:
            print('TrainingConfig in checkpoint mode...')  # noqa: T201
            self._set_checkpoint_defaults()
        else:
            print('TrainingConfig in training mode...')  # noqa: T201

    def _set_checkpoint_defaults(self):
        """Set default values for running checkpoints. Check points do not
        need all config variables.
        """
        # Training defaults
        self.training_config.setdefault('batch_size', 64)

        # Data defaults
        self.data_config.setdefault('data_dir', './biphase_gpt/data')
        self.data_config.setdefault('train_file', 'train.h5')
        self.data_config.setdefault('val_file', 'val.h5')
        self.data_config.setdefault('test_file', 'test.h5')
        self.data_config.setdefault('num_workers', 4)
        # self.data_config.setdefault("dataset_params", {
        #     "train_samples": 10000,
        #     "val_samples": 1000,
        #     "test_samples": 1000,
        #     "num_pix": 21
        # })

    @classmethod
    def from_yaml(
        cls, config_path: str
    ) -> Union['TrainingConfig', list['TrainingConfig']]:
        """Load configuration from YAML file.

        If the file is a hyperparameter search config, returns a list of configs.
        Otherwise, returns a single config.

        Args:
            config_path: Path to YAML configuration file
        """
        with Path(config_path).open() as f:
            config_dict = yaml.safe_load(f)

        # Check if this is a hyperparameter search config
        if (
            any(isinstance(v, list) for v in config_dict['model'].values())
            or any(isinstance(v, list) for v in config_dict['training'].values())
            or any(isinstance(v, list) for v in config_dict['loss'].values())
        ):
            return cls._create_search_configs(config_dict)

        return cls(
            model_config=config_dict['model'],
            training_config=config_dict['training'],
            loss_config=config_dict['loss'],
            data_config=config_dict['data'],
        )

    @classmethod
    def _create_search_configs(
        cls, config_dict: dict[str, Any]
    ) -> list['TrainingConfig']:
        """Create multiple configurations for hyperparameter search."""
        # Separate list and non-list parameters
        model_lists = {
            k: v for k, v in config_dict['model'].items() if isinstance(v, list)
        }
        model_fixed = {
            k: v for k, v in config_dict['model'].items() if not isinstance(v, list)
        }

        training_lists = {
            k: v for k, v in config_dict['training'].items() if isinstance(v, list)
        }
        training_fixed = {
            k: v for k, v in config_dict['training'].items() if not isinstance(v, list)
        }

        loss_lists = {
            k: v for k, v in config_dict['loss'].items() if isinstance(v, list)
        }
        loss_fixed = {
            k: v for k, v in config_dict['loss'].items() if not isinstance(v, list)
        }

        # Generate all combinations
        model_keys = list(model_lists.keys())
        model_values = list(model_lists.values())

        training_keys = list(training_lists.keys())
        training_values = list(training_lists.values())

        loss_keys = list(loss_lists.keys())
        loss_values = list(loss_lists.values())

        configs = []

        # Generate model combinations
        model_combinations = list(product(*model_values)) if model_values else [()]
        training_combinations = (
            list(product(*training_values)) if training_values else [()]
        )
        loss_combinations = list(product(*loss_values)) if loss_values else [()]

        for model_combo in model_combinations:
            model_config = model_fixed.copy()
            model_config.update(dict(zip(model_keys, model_combo, strict=False)))

            for training_combo in training_combinations:
                training_config = training_fixed.copy()
                training_config.update(
                    dict(zip(training_keys, training_combo, strict=False))
                )

                for loss_combo in loss_combinations:
                    loss_config = loss_fixed.copy()
                    loss_config.update(dict(zip(loss_keys, loss_combo, strict=False)))

                    configs.append(
                        cls(
                            model_config=model_config,
                            training_config=training_config,
                            loss_config=loss_config,
                            data_config=config_dict['data'],
                            is_hyperparameter_search=True,
                            search_space={
                                'model': model_lists,
                                'training': training_lists,
                                'loss': loss_lists,
                            },
                        )
                    )

        # Randomly shuffle configurations
        random.shuffle(configs)
        return configs


class ModelTrainer:
    """Main trainer class for managing model training."""

    def __init__(
        self,
        config: TrainingConfig,
        experiment_name: str | None = None,
        checkpoint_dir: str | None = None,
    ):
        """Args:
        config: Training configuration
        experiment_name: Name for logging and checkpointing
        checkpoint_dir: Directory for saving checkpoints.
        """
        self.config = config
        self.experiment_name = experiment_name or config.model_config['type']
        self.checkpoint_dir = checkpoint_dir or './checkpoints'

        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if experiment_name != 'checkpoint_eval':
            # Setup data
            self.setup_data()

            # Create Lightning module
            self.lightning_module = self.create_lightning_module()

            # Setup training
            self.trainer = self.setup_trainer()

        # Check what version of PyTorch is installed
        print(f'PyTorch version: {torch.__version__}')  # noqa: T201

        # Check the current CUDA version being used
        print(f'Current CUDA version: {torch.version.cuda}')  # noqa: T201

        if torch.version.cuda is not None:
            # Check if CUDA is available and if so, print the device name
            print(f'GPU device name: {torch.cuda.get_device_name(0)}')  # noqa: T201

            # Check if FlashAttention is available
            print('FlashAttention available:', torch.backends.cuda.flash_sdp_enabled())  # noqa: T201

    def setup_data(self):
        """Setup data loaders."""
        # Convert data_dir to absolute path
        base_dir = Path(__file__).parent.parent  # Go up two levels from training.py

        def resolve_path(data_dir: str, filename: str | None = None) -> str:
            """Resolve path relative to project root, joining with filename."""
            # Remove leading './' if present
            data_dir = str(data_dir).lstrip('./')
            abs_dir = base_dir / data_dir

            # Create directory if it doesn't exist
            Path(abs_dir).mkdir(parents=True, exist_ok=True)

            # If filename is provided, join it with the directory
            return str(abs_dir / filename) if filename else str(abs_dir)

        # Get absolute data directory
        data_dir = self.config.data_config['data_dir']

        # Resolve paths for data files
        train_path = resolve_path(data_dir, self.config.data_config.get('train_file'))
        val_path = resolve_path(data_dir, self.config.data_config.get('val_file'))
        test_path = resolve_path(data_dir, self.config.data_config.get('test_file'))

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=self.config.training_config['batch_size'],
            num_workers=self.config.data_config['num_workers'],
        )

    def create_lightning_module(self) -> BaseLightningModule:
        """Create lightning module based on model type."""
        num_pix = self.config.data_config.get('dataset_params', {}).get('num_pix', 21)
        if isinstance(num_pix, str):
            num_pix = eval(num_pix)

        optimizer_hparams = {
            'name': self.config.training_config.get('optimizer', 'Adam'),
            # TODO: why is this loaded as a string?
            'lr': eval(self.config.training_config.get('learning_rate', 5e-4)),
            'momentum': self.config.training_config.get('momentum', 0.9),
        }

        # Common scheduler hyperparameters
        max_epochs = self.config.training_config.get('max_epochs', 500)
        warmup_epochs = int(0.1 * max_epochs)
        cosine_epochs = max_epochs - warmup_epochs
        target_lr = eval(self.config.training_config.get('learning_rate', 1e-3))

        scheduler_hparams = {
            'warmup_epochs': warmup_epochs,
            'cosine_epochs': cosine_epochs,
            'target_lr': target_lr,
            'T_max': cosine_epochs,  # For CosineAnnealingLR
            'eta_min': self.config.training_config.get('eta_min', 0),
        }

        if self.config.model_config['type'] == 'GPT':
            return GPTDecoder(
                model_hparams=self.config.model_config,
                optimizer_hparams=optimizer_hparams,
                scheduler_hparams=scheduler_hparams,
                loss_hparams=self.config.loss_config,
            )
        else:
            raise ValueError("Unknown model type, can't initialize Lightning.")

    def setup_trainer(self) -> L.Trainer:
        """Setup Lightning trainer with callbacks and loggers."""
        # Callbacks
        callbacks = []
        # Add WandB logger if configured
        if self.config.training_config.get('use_logging', False):
            loggers = [
                WandbLogger(
                    project=self.config.training_config.get(
                        'wandb_project', 'ml-template'
                    ),
                    name=self.experiment_name,
                    save_dir=self.checkpoint_dir,
                )
            ]
            callbacks.append(LearningRateMonitor())
            callbacks.append(
                ModelCheckpoint(
                    dirpath=Path(self.checkpoint_dir) / self.experiment_name,
                    filename=str(loggers[0].experiment.id) + '_{epoch}-{val_loss:.4f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                )
            )
        else:
            loggers = []

        # Get accelerator and device settings from config
        accelerator = self.config.training_config.get('accelerator', 'auto')
        devices = self.config.training_config.get('devices', 1)

        # Convert devices to proper type if it's a string
        if isinstance(devices, str):
            with contextlib.suppress(ValueError):
                devices = int(devices)

        return L.Trainer(
            max_epochs=self.config.training_config['max_epochs'],
            callbacks=callbacks,
            logger=loggers,
            check_val_every_n_epoch=7,
            accelerator=accelerator,
            devices=devices,
        )

    def train(self):
        """Train the model."""
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        # if self.config.training_config.get('use_logging', False):
        #     self.trainer.loggers[0].experiment.finish()

    def test(self):
        """Test the model."""
        if hasattr(self, 'test_loader'):
            self.trainer.test(self.lightning_module, dataloaders=self.test_loader)

    def plot_predictions_from_checkpoint(self, checkpoint_path: str):  # noqa: PLR0915
        """Plot predictions from a checkpoint.

        Handles both 1D and 2D cases:
        - 1D case: Shows 3 subplots (Inputs, Predictions/Targets, Encoded)
        - 2D case: Shows 4 subplots (Inputs, Predictions, Targets, Encoded)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        model = GPTDecoder.load_from_checkpoint(checkpoint_path)
        trainer = L.Trainer(accelerator='gpu', devices=1, logger=[])

        # setup dataloaders
        self.setup_data()

        # get num_pix from test dataset for prediction output organization
        num_pix = self.test_loader.dataset.num_pix
        # Cast np.int64 to int so that predict_step doesn't throw an error
        if isinstance(num_pix, np.int64):
            num_pix = int(num_pix)
        model.num_pix = num_pix

        predictions = trainer.predict(model, self.test_loader)

        # y[batch_idx][return_idx], return_idx 0...3:
        # 0: Predictions, 1: Targets, 2: Encoded, 3: Inputs
        # Ensure all tensors are on CPU before conversion to numpy
        y_hat = predictions[0][0].cpu()
        y = predictions[0][1].cpu()
        encoded = predictions[0][2].cpu()
        inputs = predictions[0][3].cpu()

        # Get batch length
        batch_len = len(y_hat)

        # Print shapes for debugging

        # Determine if we're dealing with 1D or 2D data
        is_2d = len(y_hat.shape) > 2 and y_hat.shape[-1] > 1 and y_hat.shape[-2] > 1

        for i in range(batch_len):
            # Calculate individual loss components for this sample
            # Extract single sample tensors and ensure correct shape
            y_hat_sample = y_hat[i : i + 1]
            y_sample = y[i : i + 1]
            inputs_sample = inputs[i : i + 1]

            # Print shapes for debugging this specific sample

            # Calculate loss components using MSE
            mse = torch.nn.MSELoss()
            abs_target_loss = mse(torch.abs(y_hat_sample), torch.abs(y_sample)).item()
            direct_target_loss = mse(y_hat_sample, y_sample).item()

            # Calculate encoding loss using encoded output from predict_step
            # Avoids reshaping issues since predict_step computed the encoding
            encoded_sample = encoded[i : i + 1]
            encoding_loss = mse(encoded_sample, inputs_sample).item()

            # Apply encoding weight if available
            encoding_weight = (
                model.loss_hparams.get('encoding_weight', 1.0)
                if hasattr(model, 'loss_hparams')
                else 1.0
            )
            weighted_encoding_loss = encoding_weight * encoding_loss

            # Calculate total loss
            total_loss = abs_target_loss + direct_target_loss + weighted_encoding_loss

            # Format loss values for display with 3 significant figures
            loss_title = (
                f'Total Loss: {total_loss:.3g}\n'
                f'Abs Target Loss: {abs_target_loss:.3g}\n'
                f'Direct Target Loss: {direct_target_loss:.3g}\n'
                f'Encoding Loss: {encoding_loss:.3g}'
            )

            # Convert tensors to numpy for plotting
            inputs_np = inputs[i].detach().numpy()
            y_hat_np = y_hat[i].detach().numpy()
            y_np = y[i].detach().numpy()
            encoded_np = encoded[i].detach().numpy()

            if is_2d:
                # 2D case: 4 subplots (Inputs, Predictions, Targets, Encoded)
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                (ax1, ax2), (ax3, ax4) = axes

                # Plot inputs
                im1 = ax1.imshow(inputs_np, origin='lower')
                ax1.set_title('Inputs')
                plt.colorbar(im1, ax=ax1)

                # Plot predictions
                im2 = ax2.imshow(y_hat_np, origin='lower')
                ax2.set_title('Predictions')
                plt.colorbar(im2, ax=ax2)

                # Plot targets
                im3 = ax3.imshow(y_np, origin='lower')
                ax3.set_title('Targets')
                plt.colorbar(im3, ax=ax3)

                # Plot encoded
                im4 = ax4.imshow(encoded_np, origin='lower')
                ax4.set_title('Encoded')
                plt.colorbar(im4, ax=ax4)
            else:
                # 1D case: 3 subplots (Inputs, Predictions/Targets, Encoded)
                fig = plt.figure(figsize=(10, 8))
                gs = plt.GridSpec(2, 2, figure=fig)

                # Create 3 subplots: top-left, top-right, and bottom span
                ax1 = fig.add_subplot(gs[0, 0])  # Inputs
                ax2 = fig.add_subplot(gs[0, 1])  # Predictions/Targets
                ax3 = fig.add_subplot(gs[1, 0])  # Encoded

                # Plot inputs
                im1 = ax1.imshow(inputs_np, origin='lower')
                ax1.set_title('Inputs')
                plt.colorbar(im1, ax=ax1)

                # Plot predictions and targets
                num_pix = (y_np.shape[0] + 1) / 2 - 1
                x_range = np.arange(-num_pix, num_pix + 1, 1)
                ax2.plot(x_range, y_np, label='Targets')
                ax2.plot(x_range, y_hat_np, label='Predictions')
                ax2.set_xlabel('Pixels')
                ax2.set_ylabel('Phase')
                ax2.legend()

                # Plot encoded
                im3 = ax3.imshow(encoded_np, origin='lower')
                ax3.set_title('Encoded')
                plt.colorbar(im3, ax=ax3)

            # Add sample number as main title and loss info as subtitle
            fig.suptitle(f'Sample {i + 1}/{batch_len}', fontsize=14, fontweight='bold')

            # Add loss information as a smaller subtitle
            plt.figtext(
                0.5,
                0.01,
                loss_title,
                ha='center',
                fontsize=10,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
            )

            plt.tight_layout(
                rect=[0, 0.1, 1, 0.95]
            )  # Adjust layout to make room for the subtitle
            plt.show()
