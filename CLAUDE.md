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
- **The paper itself** (`report/` directory): the published article as a Typst
  document built by a Snakemake DAG (`pixi run report`)
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
- `pixi run report` - Build `report/build/paper.pdf` via the Snakemake DAG
- `pixi run report-dry` - Show what `report` would rebuild, without building
- `pixi run report-clean` - Delete `report/build/`

For installation and setup:

```bash
pixi install                  # Solve and materialize the default environment (editable install)
pixi run pre-commit install   # One-time: set up git hooks
```

Tasks are defined in `[tool.pixi.tasks]` in `pyproject.toml`; dependencies live
in `[tool.pixi.*]` feature/environment tables. The lockfile `pixi.lock` is
committed for reproducibility.

Environments: `default` (library + test feature), `dev` (dev + test; tooling),
`test` (CI-equivalent), `report` (typst + snakemake; document build only).

### Environment constraints (easy to break, so stated here)

**Python is 3.14, and numba is what sets the ceiling.** Not 3.15: numba ships
no cp315 wheel, while numpy, torch and matplotlib all do, and numba is
load-bearing here (the JIT-compiled `encode_1d`/`encode_2d` on the training
path). 3.15 was also still a release candidate on conda-forge when this was
set. Before bumping again, check numba's wheel tags rather than Python's
release notes -- `curl -s https://pypi.org/pypi/numba/json | grep -oE
'cp3[0-9]+'` -- and expect five sites to move together:
`requires-python`, the classifier, `[tool.pixi.dependencies].python`,
`[tool.ty.environment].python-version`, and the CI matrix.

Bumping the interpreter does **not** rebuild the PyPI wheels. `pixi install`
swaps the conda interpreter and reuses the cached wheels, which are
ABI-tagged, so the environment lands with a 3.14 interpreter and `cp312`
extension modules and numpy fails to import at all. Delete `.pixi/envs/<env>`
and reinstall for each environment.

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

**Torch is pinned to the cu126 wheel index**, not left to PyPI's default
win-64 build -- that default is CPU-only, which made training GPT models in
`biphase/transformer/` unusably slow. The pin lives in
`[tool.pixi.pypi-dependencies]` and resolves to `2.14.0+cu126`, with
`torch.cuda.is_available()` true on both GPUs in this box. cu126 rather than
the cu124 that `smi` pins: the cu124 index stopped at torch 2.6.0, and cu126
is the oldest index still carrying 2.14.0, the version the numba/numpy
co-installability below was verified against. Changing this index means
re-running the OpenMP check in `[tool.pixi.dependencies]`, since each CUDA
wheel bundles its own OpenMP runtime.

**Line endings are fully normalized to LF, and should stay that way.** Seven
files once carried CRLF in the index, left from before `.gitattributes` set
`* text=auto`. That mattered because the first edit to such a file
renormalizes all of it, so a one-line change arrives as a whole-file diff --
cosmetic, but it buries the real change. Four were normalized in `5b36fe5`,
two normalized as a side effect of being edited, and `fluo/fluo1d.py` in
`4206078`; `git ls-files --eol | grep 'i/crlf'` is now empty.

If a whole-file diff ever appears again, this is why. Do not fight it --
normalize that file in its own commit with `git add --renormalize`, so the
substantive change stays reviewable. Binaries and empty files show `-text` or
`none` and are fine.

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

## Reports (`report/`)

The published paper, "Ab Initio Spatial Phase Retrieval via Intensity Triple
Correlations", as a Typst document built through a Snakemake DAG. Typst is a
conda package, so unlike a TeX engine the whole toolchain is
lockfile-reproducible with no manual install.

- `paper.typ` - the paper: main body plus the supplement, which is roughly
  half the document. **The content is frozen.** It is a markup conversion of
  the published LaTeX, and the prose, math, claims and captions are the
  published ones. Converting markup is fair game; changing what it says is a
  separate, deliberate act.
- `lib/theme.typ` - styling plus the math helpers mirroring the LaTeX
  `\newcommand`s. Single-column US letter, not the two-column OSA class.
- `assets/figure_1.png` .. `figure_11.png` - the published renders, committed
  as static binaries.
- `figures/` - empty, for figures generated by the DAG. See its `.gitkeep`.
- `biblio.bib` - kept byte-identical to the exported original, so re-exporting
  it is an overwrite rather than a merge. Excluded from the
  trailing-whitespace and end-of-file pre-commit hooks for that reason.
- `build/` - gitignored: the compiled PDF.

Adding a generated figure takes three coordinated edits: drop
`figures/plot_<name>.py` on disk, add `<name>` to the document's entry in
`DOCUMENT_FIGURES` in the `Snakefile` (which is what `FIGURES` is derived
from), and call `#fig("<name>")` in the document.
`tests/test_report_build.py` fails if those drift apart in either direction.
Adding a *static* figure is two edits: the PNG in `assets/` and an
`#asset("<name>")` call; the same test checks both directions there too.

**Before editing a `.typ` file, invoke the `typst` skill.** It carries the
silent traps -- the ones that compile at exit 0 and render wrong -- verified
against Typst 0.15.1: numbering patterns that break references, `vec()`
versus `arrow()`, doubled reference nouns, subequations, and what to check in
the render rather than the exit code. It is general Typst knowledge and is
not repeated here.

What the skill cannot know is what *this paper* decided. Four conventions,
each a deliberate choice rather than a default:

- **`\vec{q}` goes through the `vv` helper** in `lib/theme.typ`, not a bare
  `arrow()` at 431 sites. The helper's name has no other meaning, so reaching
  for Typst's `vec()` -- which builds a column vector -- is not available to
  muscle memory.
- **`bra` and `ket` are defined in `lib/theme.typ`.** Typst has no builtin
  for either, and the paper uses `\ket` 23 times.
- **`set ref(supplement: none)` is global**, because the published LaTeX
  writes the noun itself at nearly every one of 57 sites ("Section \ref{}",
  "Fig.~\ref{}"). Suppressing it globally reproduces the published wording
  everywhere instead of leaving 57 sites to judge one at a time.
- **The supplement's counters reset and re-prefix at the boundary**
  (`paper.typ`, just before `= Supplement`). Five main-body cross-references
  resolve against those numbers, so the reset is load-bearing. Only two of
  the five have live references; the other three are referenced solely from
  LaTeX blocks that were commented out before publication, which the
  conversion correctly dropped.

Three more that are about the build, not the markup:

- **Figures must be deterministic** once they are generated rather than
  committed. Seed any RNG: the DAG's caching is only trustworthy if rebuilding
  unchanged inputs is a no-op.
- **Snakemake 9 tracks input *content*, not mtime.** `touch`ing a file will
  not force a rebuild, so a run that "should" have rebuilt and reported
  nothing to do is correct behaviour, not a broken DAG. Use
  `--forcerun <rule>` (or `--forceall`) when you actually need a rebuild.
- **Assets copied in from elsewhere can carry future mtimes**, which
  snakemake correctly refuses to build against as clock skew. The
  `report/assets/` PNGs arrived from `scratch/` stamped ~5 hours ahead and had
  to be normalized. If a copy-in produces an unexplained snakemake error,
  check the mtimes before the rules.

Build invariants: `pixi run report` from a clean `build/` produces
`report/build/paper.pdf`, and an immediate second run reports nothing to do.
Editing `paper.typ`, the theme, `biblio.bib` or any asset PNG rebuilds the
PDF; nothing else does. A clean compile emits zero Typst warnings, so any
warning is a regression worth reading.

**The scaffold is duplicated from `smi/report/` on purpose.** The Snakemake
`figure`/`document`/`clean` rules, the `report` pixi tasks and the `figpath`
helper are substantially the same in both repos, and an audit will keep
noticing it. That is a deliberate choice, not drift: the two documents want
to diverge -- this one is a single-column journal article with an abstract
and back-matter, that one is a slide deck plus a standalone report -- and a
shared template would couple two paper repos so that a change for one has to
be safe for the other. Copy and adapt when starting a third; do not factor
these two together.

Snakemake comes from PyPI: it has no conda-forge win-64 build. Local
`--cores` execution is what is exercised here; the cluster executors are the
part with real Windows gaps.

## Testing

Tests are located in `tests/` and cover:
- Core simulation functionality (`test_cosPhi_generation.py`, `test_encoding.py`)
- Numerical utilities (`test_roll2d.py`)
- Report build invariants (`test_report_build.py`): asset and citation
  references resolve in both directions, `report/*.typ` is ASCII-only (the
  `ascii` task covers `*.py`/`*.md` only), and the figure registry matches
  disk

Run with: `pixi run test` or `pixi run -e test pytest`

## Code Quality

- Ruff formatting and linting with Google-style docstrings
- Type hints required (Python 3.14+)
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
