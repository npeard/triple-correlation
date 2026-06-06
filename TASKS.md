# Triphase ML Tasks

## Performance & Scalability

- [ ] **4D `g3` via Numba** — write a function that computes the 4D `g3` directly using Numba, avoiding storage of the full 6D tensor; also speeds up fine-tuning data generation
- [ ] **Debug memory usage in `GPTDecoder` input generation** — try running with no encoding loss; prefer element-wise computation over large intermediate matrices; understand PyTorch batch parallelization and optimizer memory overhead
- [ ] **Diagnose `n_heads=8` CUDA OOM** — `n_heads=8` causes out-of-memory errors while `n_heads=4` does not; identify the root cause (attention matrix size? intermediate allocations?)
- [ ] **PyTorch profiler audit** — run the profiler to identify memory bottlenecks end-to-end: [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [ ] **Model scaling: k-space tiling** — design a tiling strategy to generate the bimagnitude for small k-space regions at a time; train a CNN on small tiles and extrapolate to full k-space
- [ ] **Model scaling: bimagnitude compression** — investigate positional embedding + deep-set-style aggregation of bimagnitude cross-diagonals as a compression scheme before feeding the full array to the model

---

## Data, Encoding & Distributions

- [ ] **Check encoding accuracy under discretization** — evaluate how accurate the encoding is when `kmax` is large and `num_pix` is small; is there a meaningful resolution floor?
- [ ] **Pre-training data distribution analysis** — finish characterizing the 1D and 2D pre-training data distributions to inform upcoming experiments; see [distribution of transformed RVs reference](https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/03%3A_Distributions/3.07%3A_Transformations_of_Random_Variables)
- [ ] **cos/arccos input transformation** — think through how applying cos/arccos (since `g3` gives `cos(Φ)`) shifts the dataset distribution for fine-tuning; does this argue for using `cos(Φ)` directly as model input instead of `|Φ|`?
- [ ] **Scaling law plot** — for fixed hyperparameters, characterize test-loss decrease as a function of training set size and MLP width (and/or depth)

---

## Model Architecture & Experiments

- [ ] **Input-independent baseline** — implement a baseline that ignores the input entirely (predicts the training-set mean); establishes a lower bound on meaningful model performance — [Karpathy: Recipe for Training NNs](https://karpathy.github.io/2019/04/25/recipe/)
- [ ] **Collocation baseline: `φ(m) = MLP(m)`** — try direct optimization of a small MLP over k-space as a baseline; this maximally exploits information in a single `g3` and allows natural k-space interpolation if the network is trained for `φ(k)`; relates to neural operator approaches
- [ ] **Matrix factorization for Transformers** — read and understand low-rank / factored attention approaches: [arXiv:2406.11307](https://arxiv.org/pdf/2406.11307)

---

## Novel Directions: Bispectral ML

- [ ] **Bispectral neural operators** — concept: contrastive learning of the phase with a discretization-free ansatz and group-invariant architecture
- [ ] **Bispectral attention mechanism** — design a custom attention kernel based on triple correlations (vs. the standard pairwise / double-correlation attention); explore what a "bispectral attention" module looks like
- [ ] **Self-supervised contrastive learning on input images** — apply orbit-separation contrastive loss directly on input images; test whether the model can learn filters relating to object structure without supervised phase labels
- [ ] **Geometric Machine Learning for biphase** — study RNN, GNN, and Transformer architectures through the lens of geometric/equivariant ML and assess applicability to the biphase problem: [arXiv:2407.09468](https://arxiv.org/pdf/2407.09468) · [arXiv:2104.13478](https://arxiv.org/pdf/2104.13478)

---

## Code Quality & Testing

- [ ] **`setup_data()` dimension validation in `ModelTrainer`** — compare dataset input dimensions to model expectations at setup time and raise a clear error on mismatch; avoids confusing failures at inference
- [ ] **Update `test_cosPhi`** — replace the current average correlation with a more sophisticated one that accounts for high variance at specific pixels

---

## Literature Review

- [ ] **Bispectral signal/pulse recovery and multimode fibers** — review papers on ML-based bispectral inversion and fiber imaging:
  - [arXiv:2103.01551](https://arxiv.org/pdf/2103.01551)
  - [Sciencedirect: bispectrum signal recovery](https://www.sciencedirect.com/science/article/pii/S1296214701012793)
  - [Deep learning of ultrafast pulses with multimode fiber](https://pubs.aip.org/aip/app/article/5/9/096106/1024553/Deep-learning-of-ultrafast-pulses-with-a-multimode)
  - [Resolution limits of fiber imaging (Stanford)](https://ee.stanford.edu/~jmk/pubs/res.limits.fiber.imaging.OpEx.1-13.pdf)
- [ ] **Review paper** — [Science Advances (2023)](https://www.science.org/doi/10.1126/sciadv.adh5396)

---

## Background Concepts

- [ ] **Dynamic programming** — study Bellman-Ford algorithm and dynamic programming fundamentals as potential tools for phase-unwrapping or sign-ambiguity resolution
