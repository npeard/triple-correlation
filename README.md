# Ab Initio Spatial Phase Retrieval via Intensity Triple Correlations

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOE.495920-blue)](https://doi.org/10.1364/OE.495920)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

## Overview

This repository contains the implementation of a breakthrough method for **ab initio phase retrieval** from intensity triple correlations of incoherent light sources. For the first time, this work solves the long-standing **sign problem** in closure phase determination, enabling complete Fourier inversion and direct image reconstruction of arbitrary arrays of independent emitters using only far-field intensity correlations.

## The Phase Problem Revolution

Traditional coherent diffractive imaging faces a fundamental challenge: phase information is lost when measuring intensities with photodetectors. While second-order intensity correlations (Hanbury Brown-Twiss effect) can reveal Fourier amplitudes, retrieving the phase from the triple correlations to enable complete image reconstruction has remained elusive for **over 60 years** since the discovery of closure phases in the 1960s.

**This work finally cracks the code.**

### 🌟 **Applications Across Fields**
- **Astronomy**: Imaging star clusters with radio telescope arrays
- **X-ray Science**: Fluorescence imaging without lenses
- **Microscopy**: High-resolution imaging of fluorescent molecules and atoms
- **Quantum Optics**: Many-body correlations in ultracold atomic gases

### 🚀 **Technical Breakthroughs**
- Novel numerical algorithm using redundant closure phase information
- Sophisticated error minimization approach for noisy data
- Enhanced spatial resolution beyond detector physical limits
- Memory-efficient computation for large detector arrays

## Key Results

Our simulations demonstrate:
- ✅ Accurate phase retrieval for clusters of several atoms with only 10⁴ shots
- ✅ Complete image reconstruction without coherent diffraction data
- ✅ Phase information retrieval beyond physical detector boundaries

## The Science Behind It

The method leverages **third-order intensity correlations** (bispectrum) to extract closure phases:

$$\Phi(\vec{m},\vec{n}) = \pm[\phi(\vec{m}+\vec{n}) - \phi(\vec{m}) - \phi(\vec{n})]$$

The breakthrough lies in using redundant information to constrain the sign ambiguity (±), enabling complete phase recovery through:

1. **Intersection Algorithm**: Finding common values among multiple sign-ambiguous solutions
2. **Iterative Refinement**: Correcting early pixel errors to maintain global consistency

## Quick Start for Contributors

1. Clone the repository:
   ```bash
   git clone https://github.com/username/your-project.git
   cd your-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pip install pre-commit nbstripout
   pre-commit install
   ```

5. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

6. Make your changes and run tasks:
   ```bash
   # Run tests
   task test

   # Lint your code
   task lint

   # Format your code
   task format

   # Check spelling
   task spell

   # Run pre-commit hooks manually
   task precommit

   # Run format, lint and test in sequence
   task all
   ```

7. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```

8. Open a Pull Request on GitHub

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
