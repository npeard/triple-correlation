#!/usr/bin/env python

from typing import Optional, Dict, Any, Union, List, Tuple
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from lightning.pytorch.loggers import WandbLogger
from dataclasses import dataclass, asdict
import random
from itertools import product

from biphase_gpt.lightning_config import BaseLightningModule, GPTDecoder
from biphase_gpt.datasets import create_data_loaders
from biphase_gpt.nano_gpt import GPTConfig, GPT


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    loss_config: Dict[str, Any]
    is_hyperparameter_search: bool = False
    search_space: Optional[Dict[str, List[Any]]] = None
    
    def __post_init__(self):
        """Set default values and handle GPT config if present"""
        self._set_defaults()
        if self.model_config.get("type") == "GPT":
            self.gpt_config = self._create_gpt_config()
    
    def _set_defaults(self):
        """Set default values for configs if not provided"""
        # Model defaults
        self.model_config.setdefault("type", "GPT")
        self.gpt_config = self._create_gpt_config()
        
        # Training defaults
        self.training_config.setdefault("max_epochs", 50)
        self.training_config.setdefault("batch_size", 512)
        self.training_config.setdefault("optimizer", "Adam")
        self.training_config.setdefault("learning_rate", 3e-4)
        self.training_config.setdefault("accelerator", "cpu")
        self.training_config.setdefault("devices", "0")
        self.training_config.setdefault("use_logging", False)
        self.training_config.setdefault("wandb_project", "triple_correlation")
        self.training_config.setdefault("experiment_name", "biphase_gpt")
        self.training_config.setdefault("checkpoint_dir", "./biphase_gpt/checkpoints")
        self.training_config.setdefault("random_seed", 42)
        
        # Data defaults
        self.data_config.setdefault("data_dir", "./biphase_gpt/data")
        self.data_config.setdefault("train_file", "train.h5")
        self.data_config.setdefault("val_file", "val.h5")
        self.data_config.setdefault("test_file", "test.h5")
        self.data_config.setdefault("num_workers", 4)
        self.data_config.setdefault("dataset_params", {
            "train_samples": 10000,
            "val_samples": 1000,
            "test_samples": 1000,
            "num_pix": 21
        })
    
    def _create_gpt_config(self) -> GPTConfig:
        """Create GPTConfig from model configuration"""
        num_pix = self.data_config.get("num_pix", 21)
        Phi_dim = (num_pix // 2 + 1)//2 + 1
        Phi_dim -= 1 # for removal of zero-valued row/column with no information
        return GPTConfig(
            in_seq_len=Phi_dim**2,
            out_seq_len=num_pix//2 + 1,
            n_layer=self.model_config.get("n_layer", 1),
            n_head=self.model_config.get("n_head", 4),
            n_embd=self.model_config.get("n_embd", 128),
            dropout=self.model_config.get("dropout", 0.1),
            bias=self.model_config.get("bias", False)
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> Union['TrainingConfig', List['TrainingConfig']]:
        """Load configuration from YAML file.
        
        If the file is a hyperparameter search config, returns a list of configs.
        Otherwise, returns a single config.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Check if this is a hyperparameter search config
        if any(isinstance(v, list) for v in config_dict["model"].values()) or \
           any(isinstance(v, list) for v in config_dict["training"].values()) or \
           any(isinstance(v, list) for v in config_dict["loss"].values()):
            return cls._create_search_configs(config_dict)
        
        return cls(
            model_config=config_dict["model"],
            training_config=config_dict["training"],
            loss_config=config_dict["loss"],
            data_config=config_dict["data"]
        )
    
    @classmethod
    def _create_search_configs(cls, config_dict: Dict[str, Any]) -> List['TrainingConfig']:
        """Create multiple configurations for hyperparameter search"""
        # Separate list and non-list parameters
        model_lists = {k: v for k, v in config_dict["model"].items() if isinstance(v, list)}
        model_fixed = {k: v for k, v in config_dict["model"].items() if not isinstance(v, list)}
        
        training_lists = {k: v for k, v in config_dict["training"].items() if isinstance(v, list)}
        training_fixed = {k: v for k, v in config_dict["training"].items() if not isinstance(v, list)}

        loss_lists = {k: v for k, v in config_dict["loss"].items() if isinstance(v, list)}
        loss_fixed = {k: v for k, v in config_dict["loss"].items() if not isinstance(v, list)}
        
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
        training_combinations = list(product(*training_values)) if training_values else [()]
        loss_combinations = list(product(*loss_values)) if loss_values else [()]
        
        for model_combo in model_combinations:
            model_config = model_fixed.copy()
            model_config.update(dict(zip(model_keys, model_combo)))
            
            for training_combo in training_combinations:
                training_config = training_fixed.copy()
                training_config.update(dict(zip(training_keys, training_combo)))
                
                configs.append(cls(
                    model_config=model_config,
                    training_config=training_config,
                    loss_config=loss_config,
                    data_config=config_dict["data"],
                    is_hyperparameter_search=True,
                    search_space={
                        "model": model_lists,
                        "training": training_lists,
                        "loss": loss_lists
                    }
                ))
        
        # Randomly shuffle configurations
        random.shuffle(configs)
        return configs


class ModelTrainer:
    """Main trainer class for managing model training"""
    
    def __init__(
        self,
        config: TrainingConfig,
        experiment_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            config: Training configuration
            experiment_name: Name for logging and checkpointing
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.experiment_name = experiment_name or config.model_config['type']
        self.checkpoint_dir = checkpoint_dir or "./checkpoints"
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
        print("CUDA Version: ", torch.version.cuda)

        if torch.version.cuda is not None:
            # Check if CUDA is available and if so, print the device name
            print("Device name:", torch.cuda.get_device_properties("cuda").name)

            # Check if FlashAttention is available
            print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

    
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

        # unpack diagonally or square
        unpack_diagonals = self.config.data_config.get('unpack_diagonals', False)
        
        if test_path:
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                batch_size=self.config.training_config['batch_size'],
                num_workers=self.config.data_config['num_workers'],
                unpack_diagonals=unpack_diagonals
            )
        else:
            self.test_loader = None
            self.train_loader, self.val_loader, _ = create_data_loaders(
                train_path=train_path,
                val_path=val_path,
                test_path=None,
                batch_size=self.config.training_config['batch_size'],
                num_workers=self.config.data_config['num_workers'],
                unpack_diagonals=unpack_diagonals
            )
    
    def create_model(self) -> BaseLightningModule:
        """Create model instance based on config"""
        model_type = self.config.model_config.pop('type')
        if model_type == 'GPT':
            print('Creating GPT model...')
            return GPT(self.config.gpt_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_lightning_module(self) -> BaseLightningModule:
        """Create lightning module based on model type"""
        if isinstance(self.model, GPT):
            return GPTDecoder(
                model_type='GPT',
                model_hparams=asdict(self.config.gpt_config),
                optimizer_name=self.config.training_config['optimizer'],
                optimizer_hparams={
                    # TODO: why is this loaded as a string?
                    'lr': eval(self.config.training_config['learning_rate']),
                },
                # I need the Lightning module to know how the data is being unpacked
                loss_hparams=self.config.loss_config
            )
        else:
            raise ValueError(f"Unknown model type, can't initialize Lightning.")
    
    def setup_trainer(self) -> L.Trainer:
        """Setup Lightning trainer with callbacks and loggers"""
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(self.checkpoint_dir, self.experiment_name),
                filename='{epoch}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            #LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Add WandB logger if configured
        if self.config.training_config.get('use_logging', False):
            loggers = [
                WandbLogger(
                    project=self.config.training_config.get('wandb_project', 'ml-template'),
                    name=self.experiment_name,
                    save_dir=os.path.join(self.checkpoint_dir, 'wandb')
                )
            ]
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
            accelerator=accelerator,
            devices=devices
        )
    
    def train(self):
        """Train the model"""
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
    
    def test(self):
        """Test the model"""
        if hasattr(self, 'test_loader'):
            self.trainer.test(
                self.lightning_module,
                dataloaders=self.test_loader
            )
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        config: TrainingConfig
    ) -> 'ModelTrainer':
        """Load model from checkpoint"""
        trainer = cls(config)
        trainer.lightning_module = trainer.lightning_module.load_from_checkpoint(
            checkpoint_path,
            model=trainer.model
        )
        return trainer
