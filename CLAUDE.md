# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements phase retrieval from triple correlations of fluorescence intensity, as described in [this paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-15-25082&id=532719). The project consists of:

- **Fluorescence simulation** (`fluo/` directory): Core simulation code for generating fluorescence intensity patterns and computing correlations
- **Phase retrieval algorithms** (`biphase_gpt/` directory): GPT-based neural network models for phase retrieval from triple correlations  
- **Plotting and visualization** (`plots/` directory): Paper figures and data visualization
- **Tests** (`tests/` directory): Unit tests for core functionality

## Architecture

The codebase follows a modular structure:

### Core Simulation (`fluo/`)
- `speckle1d.py` / `speckle2d.py`: Main fluorescence simulation classes with methods for computing intensity patterns, correlations, and phase retrieval
- `triphase1d.py` / `triphase2d.py`: Triple correlation and phase retrieval algorithms
- `xtal.py`: Crystal structure utilities

### Neural Network Training (`biphase_gpt/`)
- `training.py`: Main training orchestration with `ModelTrainer` class and `TrainingConfig` for hyperparameter management
- `lightning_config.py`: PyTorch Lightning modules including `BaseLightningModule` and `GPTDecoder`
- `nano_gpt.py`: GPT model architecture implementation
- `datasets.py`: Data loading and preprocessing for neural network training
- `configs/nanogpt_config.yaml`: Training configuration with hyperparameter grids for model search

### Entry Points
- `main.py`: Main training script that supports:
  - Training new models: `python3 main.py --config path/to/config.yaml`
  - Testing from checkpoint: `python3 main.py --checkpoint path/to/checkpoint.ckpt`
  - Dataset regeneration: `python3 main.py --config path/to/config.yaml --regenerate_datasets`

## Development Commands

The project uses `taskipy` for task automation. All commands should use `python3`:

```bash
# Run tests
task test

# Lint code 
task lint

# Format code
task format

# Check spelling
task spell

# Run pre-commit hooks
task precommit

# Run format, lint and test in sequence
task all
```

### Direct command equivalents:
```bash
# Testing
python3 -m pytest

# Linting
python3 -m ruff check .

# Formatting  
python3 -m ruff format . && python3 -m ruff check --fix .

# Spell checking
codespell .
```

## Dependencies

Key dependencies include:
- **PyTorch + Lightning**: Neural network training framework
- **NumPy + Numba**: Numerical computing with JIT compilation for performance
- **h5py**: HDF5 data storage for training datasets
- **matplotlib**: Plotting and visualization
- **wandb**: Experiment tracking (when `use_logging: true` in config)

## Configuration

Training configurations are in YAML format (see `biphase_gpt/configs/nanogpt_config.yaml`). The config supports:
- Hyperparameter grid search with lists (e.g., `n_layer: [1,2,4]`)
- Model architecture parameters (`n_layer`, `n_head`, `n_embd`, `num_pix`)
- Training settings (optimizer, learning rate, batch size)
- Data paths and dataset generation parameters

## Data Flow

1. **Dataset Generation**: `create_pretraining_datasets()` generates synthetic fluorescence data stored in HDF5 format
2. **Training**: `ModelTrainer` loads data via `get_data_loaders()` and trains GPT models using PyTorch Lightning
3. **Evaluation**: Models can be tested from checkpoints with `plot_predictions_from_checkpoint()`

## Testing

Tests are located in `tests/` and cover:
- Core simulation functionality (`test_cosPhi_generation.py`, `test_encoding.py`)
- Numerical utilities (`test_roll2d.py`)

Run with: `task test` or `python3 -m pytest`