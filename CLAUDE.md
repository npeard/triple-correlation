# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `~/.claude/CLAUDE.md` (from `~/Documents/Projects/claude-config`) for
cross-project workflow, verification, and design preferences -- this file
covers only what's specific to triple-correlation.

## Project Overview

This repository implements phase retrieval from triple correlations of fluorescence intensity, as described in [this paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-15-25082&id=532719). The project consists of:

- **Fluorescence simulation** (`fluo/` directory): Core simulation code for generating fluorescence intensity patterns and computing correlations
- **Phase retrieval via neural networks** (`biphase/transformer/` directory): GPT-based models for phase retrieval from triple correlations, with on-the-fly synthetic data generation
- **Differentiable solver** (`biphase/solver/` directory): Gradient-based phase retrieval solver
- **Plotting and visualization** (`plots/` directory): Paper figures and data visualization
- **Tests** (`tests/` directory): Unit tests for core functionality

## Architecture

The codebase follows a modular structure:

### Core Simulation (`fluo/`)
- `speckle1d.py` / `speckle2d.py`: Main fluorescence simulation classes with methods for computing intensity patterns, correlations, and phase retrieval
- `triphase1d.py` / `triphase2d.py`: Triple correlation and phase retrieval algorithms
- `xtal.py`: Crystal structure utilities

### Neural Network Training (`biphase/transformer/`)
- `lit_module.py`: Unified PyTorch Lightning module (`LitModule`) that handles model creation, on-the-fly data generation, training/validation/test steps, and loss computation
- `training.py`: Training orchestration with `ModelTrainer` class and `TrainingConfig` for hyperparameter management
- `encoding.py`: JIT-compiled encoding functions (`encode_1d`, `encode_2d`) for computing Phi matrices from phase arrays
- `models/`: Neural network architectures accessed via `factory.py`'s `create_model()` factory method
  - `gpt.py`: GPT model architecture (transformer-based)
  - `factory.py`: Model factory -- sole access point for creating `nn.Module` models
- `configs/nanogpt_config.yaml`: Training configuration with hyperparameter grids for model search

### Entry Points
- `main.py`: Main training script that supports:
  - Training new models: `python3 main.py --config path/to/config.yaml`
  - Testing from checkpoint: `python3 main.py --checkpoint path/to/checkpoint.ckpt`

## Development Commands

Use [Pixi](https://pixi.prefix.dev) for all development tasks. Pixi manages an
isolated Python interpreter and lockfile-backed dependencies under `.pixi/`, so
always invoke Python and tools through `pixi run` (or from inside `pixi shell`)
-- never use system Python or a separate venv.

- `pixi run all` - Run full pipeline (format, lint, ascii, typecheck, test)
- `pixi run format` - Format code with ruff and auto-fix issues
- `pixi run lint` - Check code style with ruff
- `pixi run ascii` - Fail on any non-ASCII character in `*.py`/`*.md`
- `pixi run typecheck` - Type-check with ty
- `pixi run test` - Run pytest test suite
- `pixi run spell` - Run codespell for spell checking
- `pixi run precommit` - Run all pre-commit hooks
- `pixi run -e test pytest` - Run pytest in the minimal `test` environment (CI-equivalent)
- `pixi run -e dev preflight` - Step 0: is this repo fit to work in right now?
  (pre-commit installed and current, clean tree, expected branch, tests
  detected). Add `--with-tests` to actually run the suite:
  `pixi run -e dev python scripts/preflight.py --with-tests`

For installation and setup:

```bash
pixi install                  # Solve and materialize the default environment (editable install)
pixi run pre-commit install   # One-time: set up git hooks
```

Tasks are defined in `[tool.pixi.tasks]` in `pyproject.toml`; dependencies live
in `[tool.pixi.*]` feature/environment tables. The lockfile `pixi.lock` is
committed for reproducibility.

Environments: `default` (library + test feature), `dev` (dev + test; tooling),
`test` (CI-equivalent).

### Environment constraints (easy to break, so stated here)

**Platforms are `win-64` and `linux-64`.** The macOS entries were dropped
deliberately: nobody develops or runs this on a Mac, and carrying them makes
the solver find a torch/numpy/numba set consistent across Intel and
Apple-silicon macOS too, which is what historically forced backdated pins.
Re-add them only alongside someone who actually tests there.

**`ruff` and `codespell` come from conda in the *default* env**, not from the
`dev` feature. `format`, `lint` and `spell` are defined in
`[tool.pixi.tasks]`, which runs in the default env, so a dev-only ruff makes
those three tasks broken by construction. Same reason the default env pulls in
the `test` feature: `test` is defined there too.

**Tasks invoke tools as `python -m <tool>`, not as a bare executable.** A bare
`pytest` or `codespell` resolves against PATH rather than the environment,
which is the failure mode the pixi setup exists to prevent.

**One OpenMP runtime, so the whole numeric stack comes from PyPI.** `numpy`,
`numba` and `torch` all resolve from `[project.dependencies]`, *not* conda.
Conda-forge's numeric packages link `libomp` while the PyPI torch wheel
bundles Intel's `libiomp5md.dll`; loading both aborts the process with
"OMP: Error #15". Do not move `numba` back to `[tool.pixi.dependencies]`, and
do not reach for `KMP_DUPLICATE_LIB_OK` -- that flag's own error text warns it
can silently produce incorrect results. numba 0.67.0, numpy 2.5.3 and torch
2.14.0 were confirmed to import together in one process with no abort.

**Torch currently resolves to a CPU-only wheel** (`2.14.0+cpu` on win-64),
because that is PyPI's default win-64 build. Training GPT models in
`biphase/transformer/` on it is unusably slow. Fixing it means pinning a CUDA
index in `[tool.pixi.pypi-dependencies]`; that is a recorded task, not done
yet.

**Pre-commit hooks all route through `pixi run -e dev`.** They must not name
`.pixi/envs/dev/bin/...` -- that path only exists on Unix, so on win-64 every
hook silently failed. `ruff` runs as a local pixi hook rather than via
`ruff-pre-commit` so the hook and `pixi run lint` cannot disagree about what
passes.

**Important**: Always run `pixi run format` first when encountering linting/formatting issues before making manual edits. This auto-fixes most formatting problems and saves time.

## Dependencies

Key dependencies include:
- **PyTorch + Lightning**: Neural network training framework
- **NumPy + Numba**: Numerical computing with JIT compilation for performance
- **matplotlib**: Plotting and visualization
- **wandb**: Experiment tracking (when `use_logging: true` in config)

## Configuration

Training configurations are in YAML format (see `biphase/transformer/configs/nanogpt_config.yaml`). The config supports:
- Hyperparameter grid search with lists (e.g., `n_layer: [1,2,4]`)
- Model architecture parameters (`n_layer`, `n_head`, `n_embd`, `num_pix`)
- Training settings (optimizer, learning rate, batch size, steps_per_epoch)
- Loss settings (encoding_weight for static loss weighting)

## Data Flow

1. **Phase generation**: `LitModule` generates random synthetic phases on-the-fly on GPU each training step (no stored datasets)
2. **Encoding**: Phases are encoded into absolute Phi matrices via JIT-compiled `encode_1d`/`encode_2d`
3. **Training**: The model learns to predict phases from Phi matrices; loss combines abs, sign, and encoding components
4. **Evaluation**: Models can be tested from checkpoints with `plot_predictions_from_checkpoint()`

## Testing

Tests are located in `tests/` and cover:
- Core simulation functionality (`test_cosPhi_generation.py`, `test_encoding.py`)
- Numerical utilities (`test_roll2d.py`)

Run with: `pixi run test` or `pixi run -e test pytest`

## Code Quality

- Ruff formatting and linting with Google-style docstrings
- Type hints required (Python 3.12+)
- Pre-commit hooks: ruff, ty, ascii-only, codespell, nbstripout, standard checks

### ASCII-only source convention

**Never use Unicode characters in source files (this file included).** Math
notation in docstrings, comments, and string literals must be written in
**plain text or LaTeX**, not Unicode glyphs. Use `rho`, `psi`, `tau`,
`<psi|k>`, `->`, `<=`, `+/-`, `d^2`, `A^T` instead of the corresponding
Greek letters, angle brackets, arrows, and relation glyphs.

Enforcement: `scripts/check_ascii.py` (run via `pixi run ascii`, part of
`pixi run all`, and the `ascii-only` pre-commit hook) fails on any non-ASCII
codepoint in `*.py`/`*.md`.
