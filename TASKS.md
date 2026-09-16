# Triphase ML Tasks

## Auto-research setup

- [ ] **Generate the paper figures through the Snakemake DAG** -- replace `report/assets/` static PNGs with figures generated through the Snakemake DAG, by revitalizing `plots/PaperFigures.py` into `report/figures/plot_<name>.py` scripts. Each script takes `--output <path>` and writes exactly that path, must be deterministic (seed any RNG, since the DAG's caching is only trustworthy if rebuilding unchanged inputs is a no-op), and shares style via `report/figures/_figcommon.py`. Adding one takes three coordinated edits: the script, the name in the Snakefile's `FIGURES`, and the entry in `DOCUMENT_FIGURES`; `tests/test_report_build.py` fails if those drift.
- [ ] **Split the slow cosPhi tests behind a `slow` marker** -- `pixi run test` now takes ~313s, up from ~95s, because making the closure estimator converge required raising shot counts (1D 50000 -> 200000, 2D 10000 -> 30000) in `tests/test_cosPhi_generation.py`, which is essentially all of that time. A suite that slow stops being run before every commit, which defeats the point of having just made it deterministic. Concrete fix: register a `slow` marker in `[tool.pytest.ini_options]`, mark the two Monte Carlo tests `@pytest.mark.slow`, change the `test` task to `-m "not slow"`, and add a `test-slow` task running the full set for pre-merge and CI. Overlaps with the variance-weighted assertion item under Code Quality & Testing: a per-element tolerance that holds at far fewer shots would make this split unnecessary, so do one or the other, not both.

- [ ] **Deduplicate the report scaffold shared with `smi`** -- *raised from `agent-config` on 2026-09-15.* `report/Snakefile` and `report/lib/theme.typ` were built by copying from `smi/report/`, and measured today they still share 54 of 101 non-blank Snakefile lines and 55 of 190 theme lines, including a byte-identical `figpath` helper (`theme.typ:163` here, `:193` there). The divergent halves -- this repo's `abstract`, `bmsection` and author marks against that repo's `slides-theme` and `takeaway` -- are correctly per-project and should stay. What drifts silently is the shared quartet: the Snakemake `figure`/`document`/`clean` rules, the `report` pixi tasks, `figpath`, and the ASCII test over `*.typ`. Fixing a bug in one copy leaves the other wrong. Decide whether it becomes a small shared template or stays duplicated deliberately; either is defensible as a decision, neither as an accident. Overlaps with the Snakemake-DAG figure item above, so do that one first and settle this while the Snakefile is already open.

- [ ] **Move the Typst gotchas out of `paper.typ` comments** -- *raised from `agent-config` on 2026-09-15.* The hard-won Typst knowledge in this repo lives only as inline comments: `paper.typ:600-607` explains why heading numbering must be a function rather than the pattern string `"S1.1"` (with a pattern, a reference to a level-2 heading silently drops the `S`), and `lib/theme.typ:58-66` explains why `set ref(supplement: none)` is set globally across 57 call sites. `CLAUDE.md` mentions Typst zero times, so an agent reading the project instructions learns none of it and rediscovers it by hitting the same silent failures. `agent-config` now carries a general `typst` skill with these verified against 0.15.1, so the durable fix here is a short pointer in `CLAUDE.md` naming the skill and the two project-specific conventions, not a copy of the content.

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
- [ ] **Consider a variance-weighted assertion for `test_cosPhi`** -- `tests/test_cosPhi_generation.py` is deterministic and green now that both classes take a seeded `rng`, so this is an accuracy-of-the-assertion item, not a flakiness one. The residual is strongly non-uniform across the array for a known reason: `cosPhi_from_data` divides the measured closure by the triple product `g1[n] * g1[m] * g1[n + m]`, so at pixels where the coherent structure factor has a near-zero the shot noise is amplified by ~`1/g1`. Measured in 1D at seed 7: `|g1| = 0.048` at one pixel (smallest of 41) and the row through it carried ~7x the mean error of every other row. The estimator is unbiased -- errors fall off with num_shots (seed 7 worst element 0.49 at 50000 shots -> 0.16 at 800000 -> 0.10 at 3200000) -- so a flat `atol` is simply the wrong shape for the residual: it is slack where `g1` is large and tight where `g1` is small. A per-element tolerance scaled by the predicted `1/(g1[n] g1[m] g1[n+m])` variance would test the same claim at a fraction of the shots, which matters because the 2D test costs ~5 min at 30000 shots. Not done here: reverse-fitting tolerance to observed noise is how the old flaky band was created, and deriving the variance properly is real work, not a test tweak.

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
