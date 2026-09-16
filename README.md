# Ab Initio Spatial Phase Retrieval via Intensity Triple Correlations

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOE.495920-blue)](https://doi.org/10.1364/OE.495920)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

## Overview

This repository implements ab initio phase retrieval from intensity triple correlations of incoherent light sources. It resolves the sign ambiguity in closure phases recovered from the triple correlation, which is what permits a complete Fourier inversion and a direct image reconstruction of an arbitrary array of independent emitters from far-field intensity correlations alone.

## The phase problem

Coherent diffractive imaging loses the phase of the scattered field when a photodetector measures intensity. Second-order intensity correlations (the Hanbury Brown-Twiss effect) recover Fourier amplitudes, but recovering the phase from triple correlations well enough to reconstruct an image has remained open since closure phases were discovered in the 1960s. The obstacle is that each measured closure phase is determined only up to a sign.

### Applications

- **Astronomy**: Imaging star clusters with radio telescope arrays
- **X-ray Science**: Fluorescence imaging without lenses
- **Microscopy**: High-resolution imaging of fluorescent molecules and atoms
- **Quantum Optics**: Many-body correlations in ultracold atomic gases

### What the method contributes

- A numerical algorithm that resolves the closure-phase sign ambiguity using redundant closure information
- An error-minimization step for noisy data
- Spatial resolution beyond the physical extent of the detector
- Memory-efficient computation for large detector arrays

## Key Results

Simulations in the paper demonstrate:

- Accurate phase retrieval for clusters of several atoms with only 10^4 shots
- Complete image reconstruction without coherent diffraction data
- Phase information retrieval beyond physical detector boundaries

The published article is built from source in this repository as a Typst
document; see [The paper](#the-paper) below.

## The Science Behind It

The method uses **third-order intensity correlations** (bispectrum) to extract closure phases:

$$\Phi(\vec{m},\vec{n}) = \pm[\phi(\vec{m}+\vec{n}) - \phi(\vec{m}) - \phi(\vec{n})]$$

Redundant closure information constrains the sign ambiguity (+/-), and the phases are then recovered through:

1. **Intersection Algorithm**: Finding common values among multiple sign-ambiguous solutions
2. **Iterative Refinement**: Correcting early pixel errors to maintain global consistency

## The paper

The published article lives in `report/` as a Typst document with its
bibliography and figures, built by a Snakemake DAG:

```bash
pixi install -e report
pixi run report        # writes report/build/paper.pdf
pixi run report-dry    # show what would rebuild, without building
pixi run report-clean  # delete report/build/
```

`report/paper.typ` is the paper body and supplement, `report/lib/theme.typ` the
shared styling, `report/assets/` the committed figure PNGs, and
`report/biblio.bib` the references. See the `report/` section of `CLAUDE.md`
for the build invariants and for the procedure to add a figure.

## Quick Start for Contributors

This project uses [Pixi](https://pixi.prefix.dev) for environment and task
management (lockfile-backed, reproducible). Pixi installs its own Python
interpreter and every dependency under `.pixi/`, so do not create a venv or
use the system `pip` -- always go through `pixi`.

1. Clone the repository:
   ```bash
   git clone https://github.com/npeard/triple-correlation.git
   cd triple-correlation
   ```

2. Install the environments you need:
   ```bash
   pixi install             # default: library + pytest
   pixi install -e dev      # adds ty, pre-commit, bandit
   pixi install -e report   # adds typst + snakemake for the paper build
   ```

3. Install the git hooks (one-time):
   ```bash
   pixi run pre-commit install
   ```

4. Create a feature branch, make changes, and run the tooling through Pixi:

   | Command | What it does |
   | --- | --- |
   | `pixi run -e dev preflight` | Is this repo fit to work in right now? Hooks installed and current, clean tree, expected branch |
   | `pixi run format` | `ruff format` plus `ruff check --fix` |
   | `pixi run lint` | `ruff check` (no fix) |
   | `pixi run ascii` | Fail on any non-ASCII character in `*.py`/`*.md` |
   | `pixi run -e dev typecheck` | Type-check with `ty` |
   | `pixi run test` | Run the pytest suite |
   | `pixi run spell` | Run codespell |
   | `pixi run precommit` | Run all pre-commit hooks |
   | `pixi run -e dev all` | `format` -> `lint` -> `ascii` -> `typecheck` -> `test` |
   | `pixi run report` | Build `report/build/paper.pdf` |
   | `pixi run report-dry` | Show what `report` would rebuild |
   | `pixi run report-clean` | Delete `report/build/` |

   `typecheck`, `all` and `preflight` are defined on the `dev` feature and need
   `-e dev`; the `report` tasks are on the `report` feature and pixi selects
   that environment for them automatically. To run pytest in the minimal `test`
   environment, matching CI exactly: `pixi run -e test pytest`.

5. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```

6. Open a Pull Request on GitHub.

## Running the training script

`main.py` trains the phase-retrieval models and evaluates checkpoints:

```bash
pixi run python main.py --config biphase/transformer/configs/nanogpt_config.yaml
pixi run python main.py --checkpoint path/to/checkpoint.ckpt
```

Supported platforms are `win-64` and `linux-64`.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{Peard2023,
  title={Ab initio spatial phase retrieval via intensity triple correlations},
  author={Peard, Nolan and Ayyer, Kartik and Chapman, Henry N.},
  journal={Optics Express},
  volume={31},
  number={15},
  pages={25082--25092},
  year={2023},
  publisher={Optica Publishing Group},
  doi={10.1364/OE.495920}
}
```
