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
from dataclasses import dataclass, field
import random
from itertools import product

from lightning_config import BaseLightningModule, GPTDecoder
from datasets import create_data_loaders
from nanogpt import GPTConfig


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
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
        self.model_config.setdefault("type", "CNNAutoencoder")
        self.model_config.setdefault("in_channels", 3)
        self.model_config.setdefault("latent_dim", 128)
        
        # Training defaults
        self.training_config.setdefault("max_epochs", 100)
        self.training_config.setdefault("batch_size", 32)
        self.training_config.setdefault("learning_rate", 1e-3)
        self.training_config.setdefault("weight_decay", 1e-4)
        self.training_config.setdefault("accelerator", "auto")  # Options: 'cpu', 'gpu', 'auto'
        self.training_config.setdefault("devices", 1)  # Can be int or str
        
        # Data defaults
        self.data_config.setdefault("num_workers", 4)
    
    def _create_gpt_config(self) -> GPTConfig:
        """Create GPTConfig from model configuration"""
        return GPTConfig(
            n_layer=self.model_config.get("n_layer", 4),
            n_head=self.model_config.get("n_head", 4),
            n_embd=self.model_config.get("n_embd", 128),
            dropout=self.model_config.get("dropout", 0.1),
            bias=self.model_config.get("bias", True)
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
           any(isinstance(v, list) for v in config_dict["training"].values()):
            return cls._create_search_configs(config_dict)
        
        return cls(
            model_config=config_dict["model"],
            training_config=config_dict["training"],
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
        
        # Generate all combinations
        model_keys = list(model_lists.keys())
        model_values = list(model_lists.values())
        
        training_keys = list(training_lists.keys())
        training_values = list(training_lists.values())
        
        configs = []
        
        # Generate model combinations
        model_combinations = list(product(*model_values)) if model_values else [()]
        training_combinations = list(product(*training_values)) if training_values else [()]
        
        for model_combo in model_combinations:
            model_config = model_fixed.copy()
            model_config.update(dict(zip(model_keys, model_combo)))
            
            for training_combo in training_combinations:
                training_config = training_fixed.copy()
                training_config.update(dict(zip(training_keys, training_combo)))
                
                configs.append(cls(
                    model_config=model_config,
                    training_config=training_config,
                    data_config=config_dict["data"],
                    is_hyperparameter_search=True,
                    search_space={
                        "model": model_lists,
                        "training": training_lists
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

        # Check if CUDA is available and if so, print the device name
        print("Device name:", torch.cuda.get_device_properties("cuda").name)

        # Check if FlashAttention is available
        print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

    
    def setup_data(self):
        """Setup data loaders"""
        if self.config.data_config['test_path']:
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                train_path=self.config.data_config['train_path'],
                val_path=self.config.data_config['val_path'],
                test_path=self.config.data_config['test_path'],
                batch_size=self.config.training_config['batch_size'],
                num_workers=self.config.data_config['num_workers']
            )
        else:
            self.test_loader = None
            self.train_loader, self.val_loader, _  = create_data_loaders(
                train_path=self.config.data_config['train_path'],
                val_path=self.config.data_config['val_path'],
                test_path=None,
                batch_size=self.config.training_config['batch_size'],
                num_workers=self.config.data_config['num_workers']
            )
    
    def create_model(self) -> BaseModel:
        """Create model instance based on config"""
        model_type = self.config.model_config.pop('type')
        if model_type == "GPT":
            return GPT(**self.config.model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_lightning_module(self) -> BaseLightningModule:
        """Create lightning module based on model type"""
        if isinstance(self.model, GPT):
            return GPTDecoder(
                model=self.model,
                optimizer_name=self.config.training_config['optimizer'],
                optimizer_hparams={
                    'lr': self.config.training_config['learning_rate'],
                    'weight_decay': self.config.training_config['weight_decay']
                }
            )
        else:
            return BaseLightningModule(
                model=self.model,
                optimizer_name=self.config.training_config['optimizer'],
                optimizer_hparams={
                    'lr': self.config.training_config['learning_rate'],
                    'weight_decay': self.config.training_config['weight_decay']
                }
            )
    
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
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Add WandB logger if configured
        if self.config.training_config.get('use_wandb', False):
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
