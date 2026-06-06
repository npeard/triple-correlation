#!/usr/bin/env python

import argparse
import logging
import random
import time

import numpy as np

from biphase.transformer.training import ModelTrainer, TrainingConfig


def setup_logging(verbosity: str) -> None:
    """Setup logging configuration based on verbosity level."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }

    level = level_map.get(verbosity.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a new model or test from checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./biphase/transformer/configs/nanogpt_config.yaml',
        help='Path to YAML config file. Required for training, optional for testing.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for testing. If provided, will run in test mode.',
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging verbosity level (default: INFO)',
    )
    return parser.parse_args()


def setup_random_seed(seed: int | None = None) -> int:
    """Set random seed for reproducibility."""
    if seed is None:
        # Generate a random seed between 0 and 2^32 - 1
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.default_rng(seed)
    return seed


def main():
    args = parse_args()

    # Setup logging based on verbosity argument
    setup_logging(args.verbosity)
    logger = logging.getLogger(__name__)

    # For testing mode (checkpoint provided), config is not necessary
    if args.checkpoint:
        logger.info('Loading from checkpoint for quick plotting...')
        logger.info(f'Checkpoint path: {args.checkpoint}')
        model_trainer = ModelTrainer(
            TrainingConfig({}, {}, {}), experiment_name='checkpoint_eval'
        )
        model_trainer.plot_predictions_from_checkpoint(checkpoint_path=args.checkpoint)
        return

    # For training mode, load config
    if not args.config:
        raise ValueError('Config file is required for training mode')

    # Load config
    logger.info(f'Loading config from {args.config}')
    config = TrainingConfig.from_yaml(args.config)

    # Set random seed from time
    setup_random_seed(int(time.time()))

    # Convert single config to list for unified processing
    configs = config if isinstance(config, list) else [config]

    # Train with each configuration
    for idx, train_config in enumerate(configs):
        logger.info(f'Starting training run {idx + 1}/{len(configs)}')
        trainer = ModelTrainer(
            config=train_config,
            experiment_name=train_config.training_config.get('experiment_name'),
            checkpoint_dir=train_config.training_config.get('checkpoint_dir'),
        )

        logger.debug('Beginning model training...')
        trainer.train()
        logger.debug('Training completed, starting testing...')
        trainer.test()
        logger.info(f'Training run {idx + 1} completed successfully')
        if trainer.config.training_config.get('use_logging', False):
            trainer.trainer.loggers[0].experiment.finish()


if __name__ == '__main__':
    # Training new model:
    # python main.py --config path/to/config.yaml

    # Testing from checkpoint:
    # python main.py --checkpoint path/to/checkpoint.ckpt

    main()
