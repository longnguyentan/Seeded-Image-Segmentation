# Seeded Image Segmentation via s-t Min-Cut / Max-Flow

---

## Overview

This repository contains the paper and all companion code for:

**"Seeded Image Segmentation: Extending the s-t Min-Cut Formulation
with Seed Constraints and Edge-Weight Design"**

The paper extends the classic minimum s-t cut formulation for image
segmentation (Kleinberg & Tardos, §7.10) by:

1. **Seeded segmentation:** users pre-label pixels as definite
   foreground/background; enforced as hard constraints via a big-M argument.
2. **N-link penalty analysis:** comparing constant, gradient-based, and
   Gaussian intensity-difference edge weights.

All algorithms are implemented from scratch in Python using NetworkX's
Edmonds–Karp max-flow solver.

---

## Repository Structure

```
├── produce_panels.py          # Main experiment runner (generates all figures)
├── synth_panels.py            # Lightweight synthetic-only experiment script
├── images/                    # Input images (image_1.png ... image_5.png)
├── figures/panels/            # Output figures (auto-generated)
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install networkx numpy matplotlib scipy pillow scikit-image

# Generate all figure panels (synthetic + real images + penalties)
python produce_panels.py --images-dir images --out-dir figures/panels

# Or run the lightweight synthetic-only script
python synth_panels.py
```

---

## Experiments

### 1. Brute-Force Verification

Runs exhaustive search (all 2^n labelings) on 3×3 and 4×4 patches and
confirms graph-cut returns the same minimum energy. Verifies correctness.

```
Patch 3×3 (512 labelings):     BF energy == GC energy  PASS ✓
Patch 4×4 (65,536 labelings):  BF energy == GC energy  PASS ✓
```

### 2. Method Comparison (5 trials)

Compares three methods across 5 independent random trials on the
40×40 synthetic image:

- **Otsu Thresholding:** naïve baseline, O(n) time
- **Unseeded Graph-Cut:** basic min-cut, no user input
- **Seeded Graph-Cut:** extended model with seed constraints

### 3. Real Image Segmentation

Evaluates all three methods on five 70×70 natural images (mountain, dog,
human, flower, eagle) with manually placed seeds.

### 4. Seed Count Ablation

Shows how Dice coefficient changes with the number of seeds per class
(1 → 15), averaged over 4 trials with mean ± std bands.

### 5. Penalty Formula Comparison

Compares segmentation quality for three n-link formulas:

- Constant: `w_ij = λ`
- Gradient-based: `w_ij = λ / (|I_i − I_j| + 1)`
- Gaussian: `w_ij = λ · exp(−(I_i − I_j)² / 2σ²)`

---

## Algorithm Summary

```
SEEDED-SEGMENT(image I, seeds F, Bs, λ, σ):

  // Phase 1: Estimate likelihoods from seeds
  μ_F = mean intensity of F;  μ_B = mean intensity of Bs

  // Phase 2: Compute big-M
  C_finite = sum of all non-seed edge capacities
  M = C_finite + 1

  // Phase 3: Build flow network
  for each pixel i:
    if i in F:   cap(s, vi) = M,        cap(vi, t) = 0
    if i in Bs:  cap(s, vi) = 0,        cap(vi, t) = M
    else:        cap(s, vi) = D_i(BG),  cap(vi, t) = D_i(FG)
  for each neighbor pair (i,j):
    w_ij = λ · exp(−(I_i − I_j)² / 2σ²)
    add bidirected edges with capacity w_ij

  // Phase 4: Edmonds-Karp max-flow
  run max-flow on network

  // Phase 5: Extract labeling
  foreground = s-side of residual-graph BFS
```

**Complexity:** O(n³) time, O(n) space where n = H × W pixels.

---

## Key Theoretical Results

**Theorem (Optimality of Seeded Graph-Cut):** Algorithm SEEDED-SEGMENT
returns the labeling L\* minimizing the energy

```
E(L) = Σ_i D_i(L_i) + Σ_{(i,j)∈E} w_ij · 𝟙[L_i ≠ L_j]
```

subject to seed constraints L_i = FG for all i ∈ F, L_j = BG for all j ∈ Bs.

The proof relies on three lemmas:

1. **Bijection Lemma:** cuts and labelings are in bijection.
2. **Energy-Capacity Lemma:** cut capacity equals E(L) term-by-term.
3. **Big-M Lemma:** seed-violating cuts have cost > any feasible cut.

---

## References

- Kleinberg & Tardos (2005). _Algorithm Design._ Pearson. Chapter 7.10.
- Boykov & Jolly (2001). _Interactive graph cuts for optimal boundary and
  region segmentation._ ICCV.
- Rother, Kolmogorov & Blake (2004). _GrabCut._ SIGGRAPH.
- Kolmogorov & Zabih (2004). _What energy functions can be minimized via
  graph cuts?_ TPAMI.
- Edmonds & Karp (1972). _Theoretical improvements in algorithmic efficiency
  for network flow problems._ JACM.
- Ford & Fulkerson (1956). _Maximal flow through a network._ CJM.
