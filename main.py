#!/usr/bin/env python

import os
import argparse
import random
import numpy as np
from pathlib import Path
from training_new import TrainingConfig, ModelTrainer
from utils import create_train_val_test_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Run mode: train, test, or both'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for testing or resuming training'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Name for the experiment (used in logging and checkpoints)'
    )
    parser.add_argument(
        '--regenerate_datasets',
        action='store_true',
        help='Regenerate training, validation, and test datasets with a new random seed'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        help='Random seed for dataset generation. If not provided, a random one will be used'
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
    # Parse command line arguments
    args = parse_args()
    
    # Load config
    config = TrainingConfig.from_yaml(args.config)
    
    # Regenerate datasets if requested
    if args.regenerate_datasets:
        seed = setup_random_seed(args.random_seed)
        print(f"\nRegenerating datasets with random seed: {seed}")
        
        # Get dataset parameters from config
        dataset_params = config.dataset_params if hasattr(config, 'dataset_params') else {}
        
        # Generate new datasets
        create_train_val_test_datasets(
            output_dir=config.data_dir,
            **dataset_params
        )
        print("Dataset regeneration complete!\n")
    
    # Create trainer
    trainer = ModelTrainer(
        config=config,
        experiment_name=args.experiment_name
    )
    
    # Load from checkpoint if provided
    if args.checkpoint:
        trainer = ModelTrainer.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            config=config
        )
    
    # Run training and/or testing
    if args.mode in ['train', 'both']:
        trainer.train()
    
    if args.mode in ['test', 'both']:
        trainer.test()

if __name__ == '__main__':
    main()


# Example code, how to run training:
# python main.py --config configs/autoencoder_config.yaml --mode train --experiment_name my_experiment
# How to test a trained model:
# python main.py --config configs/autoencoder_config.yaml --mode test --checkpoint path/to/checkpoint.ckpt
# How to regenerate datasets and train:
# python main.py --config configs/autoencoder_config.yaml --mode train --regenerate_datasets --random_seed 42