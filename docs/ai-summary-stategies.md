# `scAICME`: Strategy Architecture & Design Philosophy

`scAICME` (Semi-Supervised Annotation for Single-Cell RNA) is built around a highly modular, decoupled **Strategy Pattern**. Instead of locking users into a rigid, monolithic pipeline, the library treats every algorithmic step—from weak seed generation to final ensemble propagation—as an independent, interchangeable object adhering to the `BaseLabelingStrategy` contract.

This document details the expected behavior, mathematical foundations, and design considerations behind each implemented strategy.

---

## 🏗️ Core Architectural Principles

1. **The Unix Philosophy:** Each strategy does one thing perfectly. We do not enforce preprocessing (like PCA or scaling) inside the strategies. We trust the computational biologist to prepare their `AnnData` object correctly using `scanpy` standard practices before passing it to our estimators.
2. **Rich Data Transfer Objects (DTO):** Strategies do not mutate `adata` directly during computation. They return a `LabelingResult` object containing the categorical labels, `.obs` metrics (confidence, margins), `.obsm` matrices (continuous scores, posterior probabilities), and `.uns` metadata. The dispatcher (`tl.label`) handles the safe, sequential writing of this DTO back to the main thread.
3. **Concurrency:** Computations are embarrassingly parallel where possible. Batching multiple strategies (or fitting independent per-cell-type distributions in DPMM) is offloaded to a `ThreadPoolExecutor` to minimize latency.

---

## Phase 1: Seed Generation (Weak Labeling)

The goal of Phase 1 is to convert continuous marker gene expression into sparse, ultra-high-confidence categorical labels ("seeds"). The priority here is **precision over recall**; it is better to leave a cell as `"unknown"` than to generate a false positive seed.

### 1. QCQ Adaptive Thresholding on Scored Markers (`QCQScoredAdaptiveSeeding`)

**Concept:** Quality-Checked Quantile (QCQ) acts as a robust statistical heuristic. It scores each marker gene set with `scanpy.tl.score_genes`, then gates cells using a population quantile plus a minimum score floor.

* **Expected Behavior:** Computes a continuous marker score for each cell type with `scanpy.tl.score_genes`. For each cell type, a cell must pass both gates: score above the configured `quantile` threshold for that score distribution and score above `min_score`.
* **Design Considerations:** * *Adaptive:* Quantile gating scales with dataset-specific score distributions.
* *The QC Floor:* `min_score` prevents low-signal tails from being labeled solely due to quantile rank.



### 1b. QCQ Per-Gene Adaptive Thresholding (`QCQAdaptiveSeeding`)

**Concept:** Quality-Checked Quantile (QCQ) acts as a robust statistical heuristic. It thresholds each marker gene independently using its positive-expression quantile, then scores cells by the fraction of active markers they express. Cells are selected only when that active-marker fraction clears a fixed confidence floor.

* **Expected Behavior:** Computes a per-gene threshold from positive expression values, converts each marker into a binary active/inactive feature, and uses the active marker fraction as the cell-type score. It then compares that score against `min_confidence` rather than applying a second quantile.
* **Design Considerations:** * *Adaptive:* Hardcoded thresholds fail across different datasets. Quantiles scale dynamically with the dataset's specific expression distribution.
* *The QC Floor:* The `min_confidence` parameter prevents the algorithm from labeling cells whose active marker fraction is too low, even if the distribution is shifted upward by noise.



### 2. Otsu Score-Based Thresholding (`OtsuScoredAdaptiveSeeding`)

**Concept:** A parameter-free variance optimization technique borrowed from computer vision. It scores marker sets with `scanpy.tl.score_genes` and automatically finds the optimal threshold to separate a bimodal distribution into "Background" and "Signal".

* **Expected Behavior:** Computes a continuous marker score for each cell type with `scanpy.tl.score_genes`, then builds a 1D histogram and searches for the threshold bin that maximizes inter-class variance :



\sigma_b^2 = \omega_1(t) \omega_2(t) \left[ \mu_1(t) - \mu_2(t) \right]^2



Where  represents class probabilities and  represents class means.
* **Design Considerations:** * *Parameter-Free Splitting:* Unlike QCQ, which requires guessing a quantile, Otsu dynamically adapts to the size of the cluster. If 40% of the dataset is T-cells, Otsu will naturally set a threshold that captures the whole 40%, whereas a 95th-percentile QCQ would drop millions of valid cells.
* *Speed:* Implemented in pure, vectorized NumPy. It calculates the optimal split in milliseconds.



### 2b. Otsu AMF Thresholding (`OtsuAdaptiveSeeding`)

**Concept:** This is the AMF-based companion to the score-based Otsu variant. It thresholds each marker gene independently using Otsu's method, then scores cells by the fraction of active markers they express.

* **Expected Behavior:** Computes a per-gene Otsu threshold from each gene's expression distribution, converts each marker into a binary active/inactive feature, and uses the active marker fraction as the cell-type score. It then compares that score against `min_confidence`.
* **Design Considerations:** * *Adaptive:* Hardcoded thresholds fail across different datasets. Otsu adapts to each gene's expression histogram and separates background from signal without a hand-tuned cutoff.
* *The QC Floor:* The `min_confidence` parameter prevents the algorithm from labeling cells whose active marker fraction is too low, even if the distribution is shifted upward by noise.



### 3. Graph Score Propagation (`GraphScorePropagation`)

**Concept:** Manifold diffusion. It leverages the global topology of the cell-cell -NN graph to smooth out local technical noise (dropout) in raw marker scores before gating the seeds.

* **Expected Behavior:** Extracts raw marker scores (), normalizes them to , and diffuses them over the symmetrically normalized adjacency matrix  via Personalized PageRank / Label Spreading:



Y_{t+1} = \alpha \hat{A} Y_t + (1-\alpha) Y_0



After  iterations, a "margin gate" is applied: a cell is only labeled if its highest diffused score beats its second-highest diffused score by a minimum `margin`.
* **Design Considerations:** * *Lightweight Math:* Explicitly avoids heavy deep learning frameworks (PyTorch/GCN) in favor of fast, deterministic `scipy.sparse` matrix multiplication.
* *The Margin Gate:* Crucial for resolving ambiguous cells positioned on the boundaries between closely related lineages in the latent space.



### 4. Dirichlet Process Mixture Models (`DirichletProcessLabeling`)

**Concept:** Automatic expression regime detection using nonparametric Bayesian methods. Instead of assuming a fixed bimodal structure (Signal vs. Noise), the algorithm discovers the optimal number of expression clusters inherent in the marker gene data via the Dirichlet Process prior.

* **Expected Behavior:** For each cell type, slices the log-normalized data into an $N \times M$ matrix (where $M$ is the number of markers). Fits a Bayesian Gaussian Mixture Model with $n_{components} = \max(2, N/10)$ and Dirichlet Process prior ($\gamma$). At convergence, the DP prior automatically collapses inactive components.

  The algorithm then:
  1. Ranks components by effective membership size (summed posterior probabilities)
  2. Designates the largest component as "background" (typically low/absent expression)
  3. Filters: keeps only components with mean marker expression **higher** than the background component
  4. Returns $\text{prob\_signal}$ = sum of posterior probabilities for all high-expression components
  5. Assigns labels via majority rule: if a cell is confident ($\text{prob\_signal} \geq \theta$) in multiple cell types, picks the highest

* **Design Considerations:**
  * *True Probabilities:* Yields rigorous mathematical confidence bounds ($\text{prob\_signal} \in [0,1]$) rather than arbitrary continuous scores, mapped directly to `.obsm['posterior_probabilities']`.
  * *Automatic Model Selection:* No external hyperparameter for the number of clusters. The DP prior ($\gamma$) controls collapse aggressiveness; lower values favor fewer active components.
  * *Threaded Execution:* DPMM fitting is computationally expensive. Independent per-cell-type model fits are mapped to a `ThreadPoolExecutor`, drastically reducing wall-clock time.
  * *Background-Relative:* By anchoring signal detection to the largest (typically noisiest) component, the method is robust to marker gene quality and avoids absolute expression thresholds.



---

## Phase 2: Label Propagation (Generalization via Independent Seeds)

**Architecture Key Change:** Rather than consensus-ing Phase 1 seeds into a single vector and propagating from that, we propagate from **each Phase 1 seed independently**. This allows:
- Each propagation method to learn from diverse seed sources
- Comparison of how different propagators respond to different seeds
- Preservation of seed diversity until the final consensus step

The goal of Phase 2 is **recall**. Taking the sparse labelings from Phase 1 (seeds), these strategies train classical Machine Learning algorithms to classify the remaining unlabeled manifold.

**Shared Design Considerations for Phase 2:**

* **The `obsm_key` Paradigm:** Phase 2 strategies do not train on raw genes ( dimensions). They train on compressed feature spaces to avoid the curse of dimensionality. By making `obsm_key` dynamic (defaulting to `"X_pca"`), the library natively supports standard PCA as well as modern deep-learning autoencoders (e.g., passing `"X_scVI"`).
* **`sklearn` Isolation:** `scikit-learn` is treated as a hard dependency, but is isolated entirely to the `strategies.ml` modules.

### 1. k-Nearest Neighbors (`KNNPropagation`)

* **Concept:** Local topological propagation.
* **Expected Behavior:** Fits a -NN classifier on the coordinates of the seed cells. It predicts the labels of unknown cells based on a distance-weighted majority vote of their  closest seed neighbors in the PCA/latent space. Highly effective for resolving non-linear, branching trajectories.

### 2. Random Forest (`RandomForestPropagation`)

* **Concept:** Non-linear decision boundaries.
* **Expected Behavior:** Fits an ensemble of decision trees on the seed cells.
* **Design Considerations:** Highly robust to noisy, unscaled latent spaces. Capable of outputting rigorous `.obsm['probabilities']` via the fraction of trees voting for a specific class, providing an excellent confidence metric for downstream filtering.

### 3. Nearest Centroid (`NearestCentroidPropagation`)

* **Concept:** Distance to the cluster mean.
* **Expected Behavior:** Calculates the exact geometric center (centroid) of the seed cells for each cell type. Assigns unknown cells to the class of the closest centroid.
* **Design Considerations:** The fastest and simplest algorithm. Acts as a baseline. It assumes clusters are convex and spherical in the latent space; while less accurate for complex topologies, its inclusion in a final Consensus Vote anchors the ensemble against wild extrapolations by the Random Forest.
