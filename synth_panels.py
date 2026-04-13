import os, warnings, numpy as np
from collections import deque
from scipy.ndimage import binary_erosion
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.flow import edmonds_karp

warnings.filterwarnings("ignore")
os.makedirs("figures/panels", exist_ok=True)

FG_C = np.array([0, 210, 230], dtype=np.uint8)
BG_C = np.array([220, 30, 30], dtype=np.uint8)
FG_M = np.array([215, 232, 245])
BG_M = np.array([22, 22, 36])


def synth(H=40, W=40, fg=175, bg=80, noise=38, seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float64)
    cy, cx, ry, rx = H // 2, W // 2, H // 3, W // 3
    Y, X = np.ogrid[:H, :W]
    mask = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1
    img[mask] = fg
    img[~mask] = bg
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255), mask


def mkseeds(mask, nf, nb, seed):
    rng = np.random.default_rng(seed)
    fe = binary_erosion(mask, iterations=3)
    be = binary_erosion(~mask, iterations=3)
    fc = list(zip(*np.where(fe)))
    bc = list(zip(*np.where(be)))
    fi = rng.choice(len(fc), size=min(nf, len(fc)), replace=False)
    bi = rng.choice(len(bc), size=min(nb, len(bc)), replace=False)
    return [fc[i] for i in fi], [bc[i] for i in bi]


def pid(r, c, W):
    return r * W + c


def gw(a, b, s=30.0, l=20.0):
    return l * np.exp(-((float(a) - float(b)) ** 2) / (2 * s**2))


def otsu(img):
    h, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    h = h.astype(float)
    tot = h.sum()
    bt, bv, w0, ms = 0, 0.0, 0.0, 0.0
    mt = sum(i * h[i] for i in range(256)) / tot
    for t in range(256):
        w0 += h[t] / tot
        ms += t * h[t] / tot
        w1 = 1.0 - w0
        if w0 < 1e-12 or w1 < 1e-12:
            continue
        v = w0 * w1 * (ms / w0 - (mt - ms) / w1) ** 2
        if v > bv:
            bv = v
            bt = t
    return img >= bt


def run_gc(img, fg=None, bg=None, lam=20.0, sig=30.0):
    H, W = img.shape
    n = H * W
    S = n
    T = n + 1
    fgs = set(map(tuple, fg)) if fg else set()
    bgs = set(map(tuple, bg)) if bg else set()
    if fgs and bgs:
        mf = np.mean([img[r, c] for r, c in fgs])
        mb = np.mean([img[r, c] for r, c in bgs])
        db = lambda I: max(1.0, 255.0 - abs(float(I) - mf))
        df = lambda I: max(1.0, 255.0 - abs(float(I) - mb))
    else:
        db = lambda I: max(1.0, float(I))
        df = lambda I: max(1.0, 255.0 - float(I))
    C = 0.0
    for r in range(H):
        for c in range(W):
            if (r, c) not in fgs and (r, c) not in bgs:
                C += db(img[r, c]) + df(img[r, c])
    for r in range(H):
        for c in range(W):
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    C += 2 * gw(img[r, c], img[nr, nc], sig, lam)
    M = C + 1.0
    G = nx.DiGraph()
    G.add_nodes_from(range(n + 2))
    for r in range(H):
        for c in range(W):
            v = pid(r, c, W)
            I = img[r, c]
            if (r, c) in fgs:
                G.add_edge(S, v, capacity=M)
                G.add_edge(v, T, capacity=0.0)
            elif (r, c) in bgs:
                G.add_edge(S, v, capacity=0.0)
                G.add_edge(v, T, capacity=M)
            else:
                G.add_edge(S, v, capacity=db(I))
                G.add_edge(v, T, capacity=df(I))
    for r in range(H):
        for c in range(W):
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v = pid(r, c, W)
                    u = pid(nr, nc, W)
                    w = gw(img[r, c], img[nr, nc], sig, lam)
                    G.add_edge(v, u, capacity=w)
                    G.add_edge(u, v, capacity=w)
    R = edmonds_karp(G, S, T)
    rch = set()
    q = deque([S])
    while q:
        nd = q.popleft()
        if nd in rch:
            continue
        rch.add(nd)
        for nb in R.neighbors(nd):
            if (
                nb not in rch
                and R[nd][nb]["capacity"] - R[nd][nb].get("flow", 0) > 1e-9
            ):
                q.append(nb)
    m = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if pid(r, c, W) in rch:
                m[r, c] = True
    return m


def dice(p, t):
    return 2 * np.logical_and(p, t).sum() / (p.sum() + t.sum() + 1e-9)


def save_raw(arr, path):
    h, w = arr.shape[:2]
    dpi = 120
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if arr.ndim == 2:
        ax.imshow(
            arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="equal"
        )
    else:
        ax.imshow(arr.astype(np.uint8), interpolation="nearest", aspect="equal")
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def mrgb(m):
    h, w = m.shape
    r = np.zeros((h, w, 3), dtype=np.float32)
    r[m] = FG_M
    r[~m] = BG_M
    return r.astype(np.uint8)


def ergb(m, ref):
    r = mrgb(m).astype(np.float32)
    e = m != ref
    r[e] = 0.42 * r[e] + 0.58 * np.array([210, 45, 45])
    return r.clip(0, 255).astype(np.uint8)


def srgb(gray, fgs, bgs, rd=1):
    r = np.stack([gray] * 3, -1).astype(np.uint8)
    H, W = gray.shape
    for row, col in fgs:
        for dr in range(-rd, rd + 1):
            for dc in range(-rd, rd + 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < H and 0 <= cc < W:
                    r[rr, cc] = FG_C
    for row, col in bgs:
        for dr in range(-rd, rd + 1):
            for dc in range(-rd, rd + 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < H and 0 <= cc < W:
                    r[rr, cc] = BG_C
    return r


# ── Run (5 trials) ─────────────────────────────────────────────────────────────
N_TRIALS = 5
TRIAL_SEEDS = [42, 123, 7, 99, 2024]

all_dice = {"otsu": [], "unseed": [], "seeded": []}
all_acc = {"otsu": [], "unseed": [], "seeded": []}
trial_data = []  # store (img, ref, fgs, bgs, ot_m, un_m, se_m, dices) per trial

for i, ts in enumerate(TRIAL_SEEDS):
    img_t, ref_t = synth(seed=ts)
    fgs_t, bgs_t = mkseeds(ref_t, 12, 12, ts + 1)
    ot_t = otsu(img_t)
    un_t = run_gc(img_t)
    se_t = run_gc(img_t, fgs_t, bgs_t)
    if dice(un_t, ref_t) < dice(~un_t, ref_t):
        un_t = ~un_t

    d = {
        "otsu": dice(ot_t, ref_t),
        "unseed": dice(un_t, ref_t),
        "seeded": dice(se_t, ref_t),
    }
    a = {
        "otsu": np.mean(ot_t == ref_t),
        "unseed": np.mean(un_t == ref_t),
        "seeded": np.mean(se_t == ref_t),
    }

    for k in d:
        all_dice[k].append(d[k])
        all_acc[k].append(a[k])
    trial_data.append((img_t, ref_t, fgs_t, bgs_t, ot_t, un_t, se_t, d))
    print(
        f"Trial {i+1} (seed={ts}): Otsu={d['otsu']:.3f}  "
        f"Unseed={d['unseed']:.3f}  Seeded={d['seeded']:.3f}"
    )

# ── Print Table 5 stats ───────────────────────────────────────────────────────
print(f"\n{'Method':<14s} {'Pixel Accuracy':<22s} {'Dice'}")
for k, label in [("otsu", "Otsu"), ("unseed", "Unseeded GC"), ("seeded", "Seeded GC")]:
    a_mu, a_std = np.mean(all_acc[k]), np.std(all_acc[k])
    d_mu, d_std = np.mean(all_dice[k]), np.std(all_dice[k])
    print(f"{label:<14s} {a_mu:.3f} +/- {a_std:.3f}       {d_mu:.3f} +/- {d_std:.3f}")

# ── Pick representative trial (closest to mean Dice) ──────────────────────────
mean_vec = np.array(
    [
        np.mean(all_dice["otsu"]),
        np.mean(all_dice["unseed"]),
        np.mean(all_dice["seeded"]),
    ]
)
best_idx = 0
best_dist = float("inf")
for i, (_, _, _, _, _, _, _, d) in enumerate(trial_data):
    v = np.array([d["otsu"], d["unseed"], d["seeded"]])
    dist = np.sum((v - mean_vec) ** 2)
    if dist < best_dist:
        best_dist = dist
        best_idx = i

img, ref, fgs, bgs, ot_m, un_m, se_m, rep_d = trial_data[best_idx]
print(f"\nRepresentative trial: #{best_idx+1} (seed={TRIAL_SEEDS[best_idx]})")
print(
    f"  Otsu={rep_d['otsu']:.3f}  Unseed={rep_d['unseed']:.3f}  "
    f"Seeded={rep_d['seeded']:.3f}"
)
print(f"\n% ── Use these in Figure 3 headers ──")
print(
    f"% Otsu ({rep_d['otsu']:.3f})   Unseeded GC ({rep_d['unseed']:.3f})   "
    f"Seeded GC ({rep_d['seeded']:.3f})"
)

# ── Save panels from representative trial ─────────────────────────────────────
b = "figures/panels/synth"
save_raw(img.astype(np.uint8), f"{b}_orig.png")
save_raw(srgb(img, fgs, bgs), f"{b}_seeds.png")
save_raw(mrgb(ref), f"{b}_truth.png")
save_raw(ergb(ot_m, ref), f"{b}_otsu_err.png")
save_raw(ergb(un_m, ref), f"{b}_unseed_err.png")
save_raw(ergb(se_m, ref), f"{b}_seeded_err.png")
print("Done.")
