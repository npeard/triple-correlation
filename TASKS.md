# Triphase ML Tasks

## Auto-research setup

- [ ] **Generate the paper figures through the Snakemake DAG** -- replace `report/assets/` static PNGs with figures generated through the Snakemake DAG, by revitalizing `plots/PaperFigures.py` into `report/figures/plot_<name>.py` scripts. Each script takes `--output <path>` and writes exactly that path, must be deterministic (seed any RNG, since the DAG's caching is only trustworthy if rebuilding unchanged inputs is a no-op), and shares style via `report/figures/_figcommon.py`. Adding one takes three coordinated edits: the script, the name in the Snakefile's `FIGURES`, and the entry in `DOCUMENT_FIGURES`; `tests/test_report_build.py` fails if those drift.
- [x] **Make the two cosPhi tests deterministic** -- done. `Fluorescence1D` and `Fluorescence2D` now take `rng: np.random.Generator | int | None`, normalized with `np.random.default_rng(rng)` and used at both draw sites in each class (atom array, per-shot emitter phases); unseeded behaviour is unchanged. `tests/test_cosPhi_generation.py` seeds both tests and raises num_shots (1D 50000 -> 200000, 2D 10000 -> 30000). The original `rtol=0.2`/`atol=0.1` band was **kept, not widened** -- see the note below on why the old failures were undersampling rather than a bad tolerance.
- [ ] **Unblock the `ty` pre-commit hook, which fails every Python commit** -- the hook runs with `pass_filenames: false`, so it type-checks the whole repo regardless of what is staged, and the repo has 20 pre-existing diagnostics. The only way through is `git commit --no-verify`, which skips all 14 hooks including the ones that pass; a hook that is always red trains everyone to bypass the entire gate, so the status quo is worse than either fixing or scoping. Measured breakdown: 18 in `biphase/` (solver + transformer), 2 in `plots/`, 6 inside numpy's own `.pyi` stubs under `.pixi/` (which should not be checked at all); `fluo/`, `tests/`, `scripts/` and `main.py` are clean. By code: 7 `call-non-callable`, 6 `no-matching-overload`, 2 `invalid-return-type`, 2 `invalid-parameter-default`, 2 `invalid-assignment`, 1 `unresolved-import`. Three candidate fixes, in preference order: (1) fix the 20 -- most are third-party stub noise, `call-non-callable` on `nn.ModuleDict`/`ModuleList` access typing as `Tensor | Module` and `no-matching-overload` on `np.ufunc.outer` / `np.argpartition`; (2) make the `.pixi` exclude actually cover import-following -- `[tool.ty.src] exclude` already lists `.pixi` yet 6 diagnostics still come from numpy's vendored stubs, so 6 of 20 are free; (3) scope the hook to the subtrees that are clean today and let it grow, so the gate protects new code instead of being bypassed wholesale. Do **not** simply add the failing codes to `[tool.ty.rules]`: this repo and `smi` already share an identical rules block, so these 20 are categories `smi` does not hit, and blanket-ignoring them is suppression rather than fixing.
- [ ] **Investigate two possible real bugs hiding among the `ty` diagnostics** -- separate from the stub noise, and nobody has looked at either: `biphase/solver/biphase1d.py:198` and `biphase/solver/biphase2d.py:455` each assign an ndarray over the name `logThetaError`, which is a function -- either deliberate shadowing or a genuine bug. And `plots/PaperFigures.py:7` imports `harminv`, a dependency declared nowhere in `pyproject.toml`.
- [ ] **Get a CUDA build of torch** -- `pixi run python -c "import torch"` reports `2.14.0+cpu` on win-64, because PyPI's default win-64 wheel is CPU-only by design; this repo trains GPT models in `biphase/transformer/`, so CPU-only torch makes that path unusably slow. `smi` pins an explicit CUDA index in `[tool.pixi.pypi-dependencies]`: `torch = { version = ">=2.5", index = "https://download.pytorch.org/whl/cu124" }`. Do it as its own change, since it re-solves the lockfile and changes what `pixi run test` exercises, and verify `torch.cuda.is_available()` is true afterward. This box has two GPUs (RTX 3090 Ti 24 GB, Quadro RTX 4000 8 GB), so pin the device explicitly in any benchmark. Re-run the OpenMP import check (numba + numpy + torch in one process) afterward too: a CUDA wheel changes which OpenMP runtime is bundled, and the current no-abort result was measured against the CPU wheel.
- [ ] **Clear 12 ruff findings in existing code** -- in `biphase/solver/biphase2d.py`, `biphase/transformer/models/gpt.py`, `biphase/transformer/training.py`, `plots/PaperFigures.py`, `plots/plot1d.py` and `plots/test_differentiable_solver.py`; mostly `RUF059` (unused unpacked variables) and `PLC0415` (imports not at top of file). The deferred imports are often deliberate (heavy or optional dependencies), so this needs per-site judgment rather than a blanket hoist to module scope.
- [ ] **Record the two build-system gotchas before someone debugs a non-problem** -- Snakemake 9 tracks input *content*, not mtime, so touching a file will not force a rebuild; use `--forcerun` when you need one. And the `report/assets/` figure PNGs arrived from `scratch/` with mtimes ~5 hours in the future, which snakemake correctly refuses to build against as clock skew; those were normalized, but re-check if more assets are copied in from elsewhere.

---

## Performance & Scalability

- [ ] **4D `g3` via Numba** -- write a function that computes the 4D `g3` directly using Numba, avoiding storage of the full 6D tensor; also speeds up fine-tuning data generation
- [ ] **Debug memory usage in `GPTDecoder` input generation** -- try running with no encoding loss; prefer element-wise computation over large intermediate matrices; understand PyTorch batch parallelization and optimizer memory overhead
- [ ] **Diagnose `n_heads=8` CUDA OOM** -- `n_heads=8` causes out-of-memory errors while `n_heads=4` does not; identify the root cause (attention matrix size? intermediate allocations?)
- [ ] **PyTorch profiler audit** -- run the profiler to identify memory bottlenecks end-to-end: [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [ ] **Model scaling: k-space tiling** -- design a tiling strategy to generate the bimagnitude for small k-space regions at a time; train a CNN on small tiles and extrapolate to full k-space
- [ ] **Model scaling: bimagnitude compression** -- investigate positional embedding + deep-set-style aggregation of bimagnitude cross-diagonals as a compression scheme before feeding the full array to the model

---

## Data, Encoding & Distributions

- [ ] **Check encoding accuracy under discretization** -- evaluate how accurate the encoding is when `kmax` is large and `num_pix` is small; is there a meaningful resolution floor?
- [ ] **Pre-training data distribution analysis** -- finish characterizing the 1D and 2D pre-training data distributions to inform upcoming experiments; see [distribution of transformed RVs reference](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/03%3A_Distributions/3.07%3A_Transformations_of_Random_Variables)
- [ ] **cos/arccos input transformation** -- think through how applying cos/arccos (since `g3` gives `cos(Phi)`) shifts the dataset distribution for fine-tuning; does this argue for using `cos(Phi)` directly as model input instead of `|Phi|`?
- [ ] **Scaling law plot** -- for fixed hyperparameters, characterize test-loss decrease as a function of training set size and MLP width (and/or depth)

---

## Model Architecture & Experiments

- [ ] **Input-independent baseline** -- implement a baseline that ignores the input entirely (predicts the training-set mean); establishes a lower bound on meaningful model performance -- [Karpathy: Recipe for Training NNs](https://karpathy.github.io/2019/04/25/recipe/)
- [ ] **Collocation baseline: `phi(m) = MLP(m)`** -- try direct optimization of a small MLP over k-space as a baseline; this maximally exploits information in a single `g3` and allows natural k-space interpolation if the network is trained for `phi(k)`; relates to neural operator approaches
- [ ] **Matrix factorization for Transformers** -- read and understand low-rank / factored attention approaches: [arXiv:2406.11307](https://arxiv.org/pdf/2406.11307)

---

## Novel Directions: Bispectral ML

- [ ] **Bispectral neural operators** -- concept: contrastive learning of the phase with a discretization-free ansatz and group-invariant architecture
- [ ] **Bispectral attention mechanism** -- design a custom attention kernel based on triple correlations (vs. the standard pairwise / double-correlation attention); explore what a "bispectral attention" module looks like
- [ ] **Self-supervised contrastive learning on input images** -- apply orbit-separation contrastive loss directly on input images; test whether the model can learn filters relating to object structure without supervised phase labels
- [ ] **Geometric Machine Learning for biphase** -- study RNN, GNN, and Transformer architectures through the lens of geometric/equivariant ML and assess applicability to the biphase problem: [arXiv:2407.09468](https://arxiv.org/pdf/2407.09468) - [arXiv:2104.13478](https://arxiv.org/pdf/2104.13478)

---

## Code Quality & Testing

- [ ] **`setup_data()` dimension validation in `ModelTrainer`** -- compare dataset input dimensions to model expectations at setup time and raise a clear error on mismatch; avoids confusing failures at inference
- [ ] **Consider a variance-weighted assertion for `test_cosPhi`** -- `tests/test_cosPhi_generation.py` is deterministic and green as of the seeding work above, so this is an accuracy-of-the-assertion item, not a flakiness one. The residual is strongly non-uniform across the array for a known reason: `cosPhi_from_data` divides the measured closure by the triple product `g1[n] * g1[m] * g1[n + m]`, so at pixels where the coherent structure factor has a near-zero the shot noise is amplified by ~`1/g1`. Measured in 1D at seed 7: `|g1| = 0.048` at one pixel (smallest of 41) and the row through it carried ~7x the mean error of every other row. The estimator is unbiased -- errors fall off with num_shots (seed 7 worst element 0.49 at 50000 shots -> 0.16 at 800000 -> 0.10 at 3200000) -- so a flat `atol` is simply the wrong shape for the residual: it is slack where `g1` is large and tight where `g1` is small. A per-element tolerance scaled by the predicted `1/(g1[n] g1[m] g1[n+m])` variance would test the same claim at a fraction of the shots, which matters because the 2D test costs ~5 min at 30000 shots. Not done here: reverse-fitting tolerance to observed noise is how the old flaky band was created, and deriving the variance properly is real work, not a test tweak.

---

## Literature Review

- [ ] **Bispectral signal/pulse recovery and multimode fibers** -- review papers on ML-based bispectral inversion and fiber imaging:
  - [arXiv:2103.01551](https://arxiv.org/pdf/2103.01551)
  - [Sciencedirect: bispectrum signal recovery](https://www.sciencedirect.com/science/article/pii/S1296214701012793)
  - [Deep learning of ultrafast pulses with multimode fiber](https://pubs.aip.org/aip/app/article/5/9/096106/1024553/Deep-learning-of-ultrafast-pulses-with-a-multimode)
  - [Resolution limits of fiber imaging (Stanford)](https://ee.stanford.edu/~jmk/pubs/res.limits.fiber.imaging.OpEx.1-13.pdf)
- [ ] **Review paper** -- [Science Advances (2023)](https://www.science.org/doi/10.1126/sciadv.adh5396)

---

## Background Concepts

- [ ] **Dynamic programming** -- study Bellman-Ford algorithm and dynamic programming fundamentals as potential tools for phase-unwrapping or sign-ambiguity resolution
