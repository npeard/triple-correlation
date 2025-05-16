#!/usr/bin/env python

import os
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Union

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from biphase_gpt.datasets import get_data_loaders
from biphase_gpt.lightning_config import BaseLightningModule, GPTDecoder
from biphase_gpt.nano_gpt import GPT, GPTConfig


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""

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
            print('TrainingConfig in checkpoint mode...')
            self._set_checkpoint_defaults()
        else:
            print('TrainingConfig in training mode...')
            print('Creating GPTConfig...')
            self.gpt_config = self._create_gpt_config()

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

    def _create_gpt_config(self) -> GPTConfig:
        """Create GPTConfig from model configuration"""
        num_pix = self.data_config.get('dataset_params', {}).get('num_pix', 21)
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
            n_layer=self.model_config.get('n_layer', 1),
            n_head=self.model_config.get('n_head', 4),
            n_embd=self.model_config.get('n_embd', 128),
            dropout=self.model_config.get('dropout', 0.1),
            bias=self.model_config.get('bias', False),
            is_causal=self.model_config.get('is_causal', True),
        )

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
        with open(config_path) as f:
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
        """Create multiple configurations for hyperparameter search"""
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
                training_config.update(dict(zip(training_keys, training_combo, strict=False)))

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
    """Main trainer class for managing model training"""

    def __init__(
        self,
        config: TrainingConfig,
        experiment_name: str | None = None,
        checkpoint_dir: str | None = None,
    ):
        """Args:
        config: Training configuration
        experiment_name: Name for logging and checkpointing
        checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.experiment_name = experiment_name or config.model_config['type']
        self.checkpoint_dir = checkpoint_dir or './checkpoints'

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if experiment_name != 'checkpoint_eval':
            # Setup data
            self.setup_data()

            # Create model and lightning module
            self.model = self.create_model()
            self.lightning_module = self.create_lightning_module()

            # Setup training
            self.trainer = self.setup_trainer()

        # Check what version of PyTorch is installed
        print(torch.__version__)

        # Check the current CUDA version being used
        print('CUDA Version: ', torch.version.cuda)

        if torch.version.cuda is not None:
            # Check if CUDA is available and if so, print the device name
            print('Device name:', torch.cuda.get_device_properties('cuda').name)

            # Check if FlashAttention is available
            print('FlashAttention available:', torch.backends.cuda.flash_sdp_enabled())

    def setup_data(self):
        """Setup data loaders"""
        # Convert data_dir to absolute path
        base_dir = Path(__file__).parent.parent  # Go up two levels from training.py

        def resolve_path(data_dir: str, filename: str = None) -> str:
            """Resolve path relative to project root, optionally joining with filename"""
            # Remove leading './' if present
            data_dir = str(data_dir).lstrip('./')
            abs_dir = base_dir / data_dir

            # Create directory if it doesn't exist
            os.makedirs(abs_dir, exist_ok=True)

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

    def create_model(self) -> BaseLightningModule:
        """Create model instance based on config"""
        model_type = self.config.model_config.get('type')
        if model_type == 'GPT':
            print('Creating GPT model...')
            return GPT(self.config.gpt_config)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    def create_lightning_module(self) -> BaseLightningModule:
        """Create lightning module based on model type"""
        num_pix = self.config.data_config.get('dataset_params', {}).get('num_pix', 21)
        if isinstance(num_pix, str):
            num_pix = eval(num_pix)
        if isinstance(self.model, GPT):
            return GPTDecoder(
                model_type='GPT',
                model_hparams=asdict(self.config.gpt_config),
                optimizer_name=self.config.training_config['optimizer'],
                optimizer_hparams={
                    # TODO: why is this loaded as a string?
                    'lr': eval(self.config.training_config['learning_rate'])
                },
                scheduler_hparams={
                    'T_max': self.config.training_config['T_max'],
                    'eta_min': self.config.training_config['eta_min'],
                },
                loss_hparams=self.config.loss_config,
                num_pix=num_pix,
            )
        else:
            raise ValueError("Unknown model type, can't initialize Lightning.")

    def setup_trainer(self) -> L.Trainer:
        """Setup Lightning trainer with callbacks and loggers"""
        # Callbacks
        callbacks = [
            # ModelCheckpoint(
            #     dirpath=os.path.join(self.checkpoint_dir, self.experiment_name),
            #     filename='{epoch}-{val_loss:.4f}',
            #     monitor='val_loss',
            #     mode='min',
            #     save_top_k=-1,
            # ),
            # EarlyStopping(
            #     monitor='val_loss',
            #     patience=10,
            #     mode='min'
            # ),
        ]

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
                    dirpath=os.path.join(self.checkpoint_dir, self.experiment_name),
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
            try:
                devices = int(devices)
            except ValueError:
                # If it can't be converted to int, keep as string (e.g. for specific GPU like '0')
                pass

        return L.Trainer(
            max_epochs=self.config.training_config['max_epochs'],
            callbacks=callbacks,
            logger=loggers,
            check_val_every_n_epoch=7,
            accelerator=accelerator,
            devices=devices,
        )

    def train(self):
        """Train the model"""
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        # if self.config.training_config.get('use_logging', False):
        #     self.trainer.loggers[0].experiment.finish()

    def test(self):
        """Test the model"""
        if hasattr(self, 'test_loader'):
            self.trainer.test(self.lightning_module, dataloaders=self.test_loader)

    def plot_predictions_from_checkpoint(self, checkpoint_path: str):
        """Plot predictions from a checkpoint

        Handles both 1D and 2D cases:
        - 1D case: Shows 3 subplots (Inputs, Predictions/Targets, Encoded)
        - 2D case: Shows 4 subplots (Inputs, Predictions, Targets, Encoded)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        model = GPTDecoder.load_from_checkpoint(checkpoint_path)
        trainer = L.Trainer(accelerator='cpu', logger=[])

        # setup dataloaders
        self.setup_data()

        # get num_pix from the test dataset so we can properly organize the prediction output
        num_pix = self.test_loader.dataset.num_pix
        # Cast np.int64 to int so that predict_step doesn't throw an error
        if isinstance(num_pix, np.int64):
            num_pix = int(num_pix)
        model.num_pix = num_pix

        predictions = trainer.predict(model, self.test_loader)

        # y[batch_idx][return_idx], return_idx 0...3:
        # 0: Predictions, 1: Targets, 2: Encoded, 3: Inputs
        batch_len = len(predictions[0][0].numpy())
        y_hat = predictions[0][0].numpy()
        y = predictions[0][1].numpy()
        encoded = predictions[0][2].numpy()
        inputs = predictions[0][3].numpy()

        # Print shapes for debugging
        print(f'Predictions shape: {y_hat.shape}')
        print(f'Targets shape: {y.shape}')
        print(f'Encoded shape: {encoded.shape}')
        print(f'Inputs shape: {inputs.shape}')

        # Determine if we're dealing with 1D or 2D data
        is_2d = len(y_hat.shape) > 2 and y_hat.shape[-1] > 1 and y_hat.shape[-2] > 1

        for i in range(batch_len):
            if is_2d:
                # 2D case: 4 subplots (Inputs, Predictions, Targets, Encoded)
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                (ax1, ax2), (ax3, ax4) = axes

                # Plot inputs
                im1 = ax1.imshow(inputs[i], origin='lower')
                ax1.set_title('Inputs')
                plt.colorbar(im1, ax=ax1)

                # Plot predictions
                im2 = ax2.imshow(y_hat[i], origin='lower')
                ax2.set_title('Predictions')
                plt.colorbar(im2, ax=ax2)

                # Plot targets
                im3 = ax3.imshow(y[i], origin='lower')
                ax3.set_title('Targets')
                plt.colorbar(im3, ax=ax3)

                # Plot encoded
                im4 = ax4.imshow(encoded[i], origin='lower')
                ax4.set_title('Encoded')
                plt.colorbar(im4, ax=ax4)
            else:
                # 1D case: 3 subplots (Inputs, Predictions/Targets, Encoded)
                fig = plt.figure(figsize=(10, 8))
                gs = plt.GridSpec(2, 2, figure=fig)

                # Create 3 subplots: top-left, top-right, and bottom spanning both columns
                ax1 = fig.add_subplot(gs[0, 0])  # Inputs
                ax2 = fig.add_subplot(gs[0, 1])  # Predictions/Targets
                ax3 = fig.add_subplot(gs[1, 0])  # Encoded (spans both columns)

                # Plot inputs
                im1 = ax1.imshow(inputs[i], origin='lower')
                ax1.set_title('Inputs')
                plt.colorbar(im1, ax=ax1)

                # Plot predictions and targets
                num_pix = (y[i].shape[0] + 1) / 2 - 1
                x_range = np.arange(-num_pix, num_pix + 1, 1)
                ax2.plot(x_range, y[i], label='Targets')
                ax2.plot(x_range, y_hat[i], label='Predictions')
                ax2.set_xlabel('Pixels')
                ax2.set_ylabel('Phase')
                ax2.legend()

                # Plot encoded
                im3 = ax3.imshow(encoded[i], origin='lower')
                ax3.set_title('Encoded')
                plt.colorbar(im3, ax=ax3)

            # Add some config info as title
            plt.suptitle(f'Sample {i + 1}/{batch_len}')
            plt.tight_layout()
            plt.show()
