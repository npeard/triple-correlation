# Phase Retrieval from Triple Correlations of Fluorescence Intensity
Simulation of fluorescence or speckle intensity correlations and phase retrieval from the triple correlations as described in [this paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-15-25082&id=532719).

The core simulation code to generate fluorescence intensity patterns and compute their correlations is in the "fluo" directory. Phase retrieval algorithms are in the "biphase" files.

Plots of simulation data used in the paper are in the "Plots" directory. These are not yet up to date with ongoing work.

If you use my code, please send me a note (I'd love to hear about what you are working on) and cite the paper.

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
