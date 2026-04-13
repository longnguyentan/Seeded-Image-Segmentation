"""
produce_panels.py
-----------------
Generate raw pixel panel images for the seeded image segmentation paper.

ALL text, labels, legends, and annotations belong in the LaTeX source.
This script produces only raw pixel arrays saved as PNGs -- no axes,
titles, or embedded typography.

Usage
-----
    python produce_panels.py [--images-dir DIR] [--out-dir DIR]

Outputs (written to OUT_DIR/)
------------------------------
    synth_seeds.png, synth_truth.png, synth_{otsu,unseed,seeded}_err.png
    imgN_seeds.png, imgN_{otsu,unseed,seeded}.png, imgN_{...}_err.png
    pen_{const,grad,gauss}.png

Dependencies
------------
    pip install networkx numpy matplotlib scipy pillow scikit-image
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.flow import edmonds_karp
from PIL import Image
from scipy.ndimage import binary_erosion

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE: int = 70
LAMBDA: float = 20.0
SIGMA: float = 30.0
FG_INTENSITY: int = 175
BG_INTENSITY: int = 80
NOISE_STD: float = 38.0

SEED_FG_COLOR = np.array([0, 210, 230], dtype=np.uint8)
SEED_BG_COLOR = np.array([220, 30, 30], dtype=np.uint8)
MASK_FG_COLOR = np.array([215, 232, 245], dtype=np.float32)
MASK_BG_COLOR = np.array([22, 22, 36], dtype=np.float32)
ERR_COLOR = np.array([210, 45, 45], dtype=np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ImageConfig:
    name: str
    fg_boxes: list
    bg_boxes: list
    invert_otsu: bool
    n_fg_seeds: int = 10
    n_bg_seeds: int = 10


IMAGE_CONFIGS: dict[int, ImageConfig] = {
    1: ImageConfig(
        name="Mountain",
        fg_boxes=[(0.45, 0.95, 0.15, 0.85)],
        bg_boxes=[(0.00, 0.20, 0.05, 0.95)],
        invert_otsu=True,
        n_fg_seeds=8,
        n_bg_seeds=8,
    ),
    2: ImageConfig(
        name="Dog",
        fg_boxes=[(0.20, 0.80, 0.20, 0.80)],
        bg_boxes=[
            (0.00, 0.12, 0.00, 0.12),
            (0.00, 0.12, 0.88, 1.0),
            (0.88, 1.0, 0.00, 0.12),
            (0.88, 1.0, 0.88, 1.0),
        ],
        invert_otsu=False,
        n_fg_seeds=10,
        n_bg_seeds=6,
    ),
    3: ImageConfig(
        name="Human",
        fg_boxes=[(0.35, 0.95, 0.20, 0.80)],
        bg_boxes=[(0.00, 0.20, 0.05, 0.90)],
        invert_otsu=True,
        n_fg_seeds=8,
        n_bg_seeds=8,
    ),
    4: ImageConfig(
        name="Flower",
        fg_boxes=[(0.15, 0.80, 0.15, 0.85)],
        bg_boxes=[
            (0.00, 0.10, 0.00, 0.35),
            (0.00, 0.10, 0.65, 1.0),
            (0.90, 1.0, 0.00, 0.35),
            (0.90, 1.0, 0.65, 1.0),
        ],
        invert_otsu=True,
        n_fg_seeds=10,
        n_bg_seeds=6,
    ),
    5: ImageConfig(
        name="Eagle",
        fg_boxes=[(0.10, 0.90, 0.30, 0.98)],
        bg_boxes=[(0.05, 0.70, 0.00, 0.22)],
        invert_otsu=False,
        n_fg_seeds=8,
        n_bg_seeds=8,
    ),
}


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
def load_gray(path: Path, size: int = TARGET_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.array(img.resize((size, size), Image.LANCZOS), dtype=np.float64)


def save_raw(array: np.ndarray, path: Path) -> None:
    h, w = array.shape[:2]
    dpi = 120
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if array.ndim == 2:
        ax.imshow(
            array,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="equal",
        )
    else:
        ax.imshow(array.astype(np.uint8), interpolation="nearest", aspect="equal")
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------------------------------------------------------------------
# Colour mapping (pure pixel operations, no text)
# ---------------------------------------------------------------------------
def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.where(mask[:, :, None], MASK_FG_COLOR, MASK_BG_COLOR)
    return rgb.astype(np.uint8)


def error_overlay(
    mask: np.ndarray, reference: np.ndarray, alpha: float = 0.58
) -> np.ndarray:
    rgb = mask_to_rgb(mask).astype(np.float32)
    errors = mask != reference
    rgb[errors] = (1 - alpha) * rgb[errors] + alpha * ERR_COLOR
    return rgb.clip(0, 255).astype(np.uint8)


def seeds_overlay(
    gray: np.ndarray, fg_seeds: list, bg_seeds: list, dot_radius: int = 1
) -> np.ndarray:
    rgb = np.stack([gray] * 3, axis=-1).astype(np.uint8)
    H, W = gray.shape
    for seeds, color in ((fg_seeds, SEED_FG_COLOR), (bg_seeds, SEED_BG_COLOR)):
        for r, c in seeds:
            for dr in range(-dot_radius, dot_radius + 1):
                for dc in range(-dot_radius, dot_radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        rgb[rr, cc] = color
    return rgb


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------
def box_seeds(image: np.ndarray, boxes: list, n_per_box: int, rng_seed: int) -> list:
    H, W = image.shape
    rng = np.random.default_rng(rng_seed)
    seeds = []
    for r0f, r1f, c0f, c1f in boxes:
        r0, r1 = int(r0f * H), max(int(r0f * H) + 1, int(r1f * H))
        c0, c1 = int(c0f * W), max(int(c0f * W) + 1, int(c1f * W))
        seeds.extend(
            zip(
                rng.integers(r0, r1, n_per_box).tolist(),
                rng.integers(c0, c1, n_per_box).tolist(),
            )
        )
    return seeds


def eroded_seeds(mask: np.ndarray, n_fg: int, n_bg: int, rng_seed: int) -> tuple:
    rng = np.random.default_rng(rng_seed)
    fg_c = list(zip(*np.where(binary_erosion(mask, iterations=3))))
    bg_c = list(zip(*np.where(binary_erosion(~mask, iterations=3))))
    fi = rng.choice(len(fg_c), size=min(n_fg, len(fg_c)), replace=False)
    bi = rng.choice(len(bg_c), size=min(n_bg, len(bg_c)), replace=False)
    return [fg_c[i] for i in fi], [bg_c[i] for i in bi]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
PenaltyFn = Callable[[float, float, float, float], float]


def gaussian_w(
    Ii: float, Ij: float, sigma: float = SIGMA, lam: float = LAMBDA
) -> float:
    return lam * np.exp(-((Ii - Ij) ** 2) / (2 * sigma**2))


def constant_w(
    Ii: float, Ij: float, sigma: float = SIGMA, lam: float = LAMBDA
) -> float:
    return lam


def gradient_w(
    Ii: float, Ij: float, sigma: float = SIGMA, lam: float = LAMBDA
) -> float:
    return lam / (abs(Ii - Ij) + 1.0)


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    best_t, best_var, w0, mu0s = 0, 0.0, 0.0, 0.0
    mu_t = sum(t * hist[t] for t in range(256)) / total
    for t in range(256):
        w0 += hist[t] / total
        mu0s += t * hist[t] / total
        w1 = 1.0 - w0
        if w0 < 1e-12 or w1 < 1e-12:
            continue
        var = w0 * w1 * (mu0s / w0 - (mu_t - mu0s) / w1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return image >= best_t


def graph_cut(
    image: np.ndarray,
    fg_seeds: Optional[list] = None,
    bg_seeds: Optional[list] = None,
    penalty_fn: PenaltyFn = gaussian_w,
    lam: float = LAMBDA,
    sigma: float = SIGMA,
) -> np.ndarray:
    """s-t min-cut segmentation via Edmonds-Karp max-flow.

    Seed constraints are enforced via the big-M argument:
    M = C_finite + 1 where C_finite is the sum of all non-seed capacities.
    """
    H, W = image.shape
    n = H * W
    SRC, SNK = n, n + 1

    fg_set = set(map(tuple, fg_seeds)) if fg_seeds else set()
    bg_set = set(map(tuple, bg_seeds)) if bg_seeds else set()

    if fg_set and bg_set:
        mu_f = float(np.mean([image[r, c] for r, c in fg_set]))
        mu_b = float(np.mean([image[r, c] for r, c in bg_set]))
        d_bg = lambda I: max(1.0, 255.0 - abs(float(I) - mu_f))
        d_fg = lambda I: max(1.0, 255.0 - abs(float(I) - mu_b))
    else:
        d_bg = lambda I: max(1.0, float(I))
        d_fg = lambda I: max(1.0, 255.0 - float(I))

    # Compute big-M
    C = sum(
        d_bg(image[r, c]) + d_fg(image[r, c])
        for r in range(H)
        for c in range(W)
        if (r, c) not in fg_set and (r, c) not in bg_set
    )
    C += sum(
        2.0 * penalty_fn(float(image[r, c]), float(image[r + dr, c + dc]), sigma, lam)
        for r in range(H)
        for c in range(W)
        for dr, dc in ((0, 1), (1, 0))
        if 0 <= r + dr < H and 0 <= c + dc < W
    )
    big_M = C + 1.0

    G = nx.DiGraph()
    G.add_nodes_from(range(n + 2))

    for r in range(H):
        for c in range(W):
            v = r * W + c
            I = float(image[r, c])
            if (r, c) in fg_set:
                G.add_edge(SRC, v, capacity=big_M)
                G.add_edge(v, SNK, capacity=0.0)
            elif (r, c) in bg_set:
                G.add_edge(SRC, v, capacity=0.0)
                G.add_edge(v, SNK, capacity=big_M)
            else:
                G.add_edge(SRC, v, capacity=d_bg(I))
                G.add_edge(v, SNK, capacity=d_fg(I))

    for r in range(H):
        for c in range(W):
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v, u = r * W + c, nr * W + nc
                    w = penalty_fn(float(image[r, c]), float(image[nr, nc]), sigma, lam)
                    G.add_edge(v, u, capacity=w)
                    G.add_edge(u, v, capacity=w)

    R = edmonds_karp(G, SRC, SNK)
    reachable: set = set()
    queue = deque([SRC])
    while queue:
        nd = queue.popleft()  # BFS (FIFO), matching Algorithm 2 line 20
        if nd in reachable:
            continue
        reachable.add(nd)
        for nb in R.neighbors(nd):
            if nb not in reachable:
                if R[nd][nb]["capacity"] - R[nd][nb].get("flow", 0) > 1e-9:
                    queue.append(nb)

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if r * W + c in reachable:
                mask[r, c] = True
    return mask


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(
        2 * np.logical_and(pred, truth).sum() / (pred.sum() + truth.sum() + 1e-9)
    )


def align_polarity(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return mask if dice(mask, reference) >= dice(~mask, reference) else ~mask


# ---------------------------------------------------------------------------
# Panel generation
# ---------------------------------------------------------------------------
def run_synthetic(out: Path) -> dict:
    image, truth = (lambda i, m: (i, m))(*_synth())
    fg_s, bg_s = eroded_seeds(truth, 12, 12, 7)
    ot_m = otsu_threshold(image)
    un_m = align_polarity(graph_cut(image), truth)
    se_m = graph_cut(image, fg_s, bg_s)
    dices = {
        "otsu": dice(ot_m, truth),
        "unseed": dice(un_m, truth),
        "seeded": dice(se_m, truth),
    }
    log.info("Synthetic Dice: %s", {k: f"{v:.3f}" for k, v in dices.items()})
    save_raw(image.astype(np.uint8), out / "synth_orig.png")
    save_raw(seeds_overlay(image, fg_s, bg_s), out / "synth_seeds.png")
    save_raw(mask_to_rgb(truth), out / "synth_truth.png")
    save_raw(error_overlay(ot_m, truth), out / "synth_otsu_err.png")
    save_raw(error_overlay(un_m, truth), out / "synth_unseed_err.png")
    save_raw(error_overlay(se_m, truth), out / "synth_seeded_err.png")
    return dices


def _synth():
    rng = np.random.default_rng(42)
    img = np.zeros((40, 40), dtype=np.float64)
    Y, X = np.ogrid[:40, :40]
    mask = ((Y - 20) / 13) ** 2 + ((X - 20) / 13) ** 2 <= 1
    img[mask] = FG_INTENSITY
    img[~mask] = BG_INTENSITY
    img += rng.normal(0, NOISE_STD, img.shape)
    return np.clip(img, 0, 255), mask


def run_penalties(out: Path) -> dict:
    image, truth = _synth()
    fg_s, bg_s = eroded_seeds(truth, 12, 12, 7)
    results = {}
    for name, fn in [
        ("const", constant_w),
        ("grad", gradient_w),
        ("gauss", gaussian_w),
    ]:
        m = graph_cut(image, fg_s, bg_s, penalty_fn=fn)
        results[name] = dice(m, truth)
        save_raw(error_overlay(m, truth), out / f"pen_{name}.png")
        log.info("Penalty %-5s: Dice=%.3f", name, results[name])
    return results


def run_real_images(images_dir: Path, out: Path, configs: dict) -> dict:
    metrics = {}
    for idx, cfg in configs.items():
        log.info("Image %d: %s", idx, cfg.name)
        image = load_gray(images_dir / f"image_{idx}.png")
        fg_s = box_seeds(image, cfg.fg_boxes, cfg.n_fg_seeds, idx * 11)
        bg_s = box_seeds(image, cfg.bg_boxes, cfg.n_bg_seeds, idx * 11 + 3)
        log.info("  %d FG, %d BG seeds", len(fg_s), len(bg_s))

        se_m = graph_cut(image, fg_s, bg_s)
        ot_m = otsu_threshold(image)
        if cfg.invert_otsu:
            ot_m = ~ot_m
        un_m = align_polarity(graph_cut(image), se_m)

        ref = se_m
        metrics[idx] = {
            "otsu": dice(ot_m, ref),
            "unseed": dice(un_m, ref),
            "seeded": 1.0,
        }
        log.info(
            "  Dice: Otsu=%.3f  Unseed=%.3f  Seeded=1.000",
            metrics[idx]["otsu"],
            metrics[idx]["unseed"],
        )

        save_raw(image.astype(np.uint8), out / f"img{idx}_orig.png")
        save_raw(seeds_overlay(image, fg_s, bg_s), out / f"img{idx}_seeds.png")
        save_raw(mask_to_rgb(ot_m), out / f"img{idx}_otsu.png")
        save_raw(mask_to_rgb(un_m), out / f"img{idx}_unseed.png")
        save_raw(mask_to_rgb(se_m), out / f"img{idx}_seeded.png")
        save_raw(error_overlay(ot_m, ref), out / f"img{idx}_otsu_err.png")
        save_raw(error_overlay(un_m, ref), out / f"img{idx}_unseed_err.png")
        save_raw(error_overlay(se_m, ref), out / f"img{idx}_seeded_err.png")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=Path("images"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/panels"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Synthetic panels ===")
    sd = run_synthetic(args.out_dir)
    log.info("=== Penalty panels ===")
    pd = run_penalties(args.out_dir)
    log.info("=== Real image panels ===")
    rd = run_real_images(args.images_dir, args.out_dir, IMAGE_CONFIGS)

    print("\n% ── Dice summary for LaTeX ──────────────────────────────────")
    print(
        f"% Synthetic:  Otsu={sd['otsu']:.3f}  Unseed={sd['unseed']:.3f}"
        f"  Seeded={sd['seeded']:.3f}"
    )
    for k, n in [("const", "Constant"), ("grad", "Gradient"), ("gauss", "Gaussian")]:
        print(f"% Penalty {n:9s}: Dice={pd[k]:.3f}")
    for idx, m in rd.items():
        print(
            f"% img{idx} ({IMAGE_CONFIGS[idx].name:10s}): "
            f"Otsu={m['otsu']:.3f}  Unseed={m['unseed']:.3f}  Seeded=1.000"
        )


if __name__ == "__main__":
    main()
