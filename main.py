#!/usr/bin/env python

import argparse
import random
import numpy as np
from pathlib import Path
from biphase_gpt.training import TrainingConfig, ModelTrainer
from biphase_gpt.datasets import create_train_val_test_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Train a new model or test from checkpoint')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file. Required for training, optional for testing.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for testing. If provided, will run in test mode.'
    )
    parser.add_argument(
        '--regenerate_datasets',
        action='store_true',
        help='Regenerate training, validation, and test datasets'
    )
    return parser.parse_args()

def setup_random_seed(seed=None):
    """Set random seed for reproducibility."""
    if seed is None:
        # Generate a random seed between 0 and 2^32 - 1
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    return seed

def main():
    args = parse_args()
    
    # For testing mode (checkpoint provided), config is not necessary
    if args.checkpoint and not args.config:
        print("Loading from checkpoint for quick plotting...")
        print(args.checkpoint)
        model_trainer = ModelTrainer(TrainingConfig({},{},{},{}), experiment_name="checkpoint_eval")
        model_trainer.plot_predictions_from_checkpoint(checkpoint_path=args.checkpoint)
        return
    
    # For training mode, load config
    if not args.config:
        raise ValueError("Config file is required for training mode")
    
    config = TrainingConfig.from_yaml(args.config)
    
    # Set random seed from config
    seed = setup_random_seed(config.training_config.get('random_seed'))
    
    # Regenerate datasets if requested
    if args.regenerate_datasets:
        print(f"\nRegenerating datasets with random seed: {seed}")
        create_train_val_test_datasets(
            output_dir=config.data_config['data_dir'],
            **config.data_config.get('dataset_params', {})
        )
        print("Dataset regeneration complete!\n")
    
    # Create trainer
    trainer = ModelTrainer(
        config=config,
        experiment_name=config.training_config.get('experiment_name'),
        checkpoint_dir=config.training_config.get('checkpoint_dir')
    )

    # Start training
    trainer.train()

if __name__ == '__main__':
# Training new model:
# python main.py --config path/to/config.yaml

# Testing from checkpoint:
# python main.py --checkpoint path/to/checkpoint.ckpt

# Fine-tuning from checkpoint:
# python main.py --config path/to/config.yaml --checkpoint path/to/checkpoint.ckpt

# Regenerating datasets:
# python main.py --config path/to/config.yaml --regenerate_datasets

    main()