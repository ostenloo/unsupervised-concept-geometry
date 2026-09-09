"""§18.24 — the (layer × position) selection grid, on Arditi's cheap metric.

Capture at token positions -1..-5 (all 32 layers), build the difference-in-means
direction at every (layer, position), ablate it network-wide, and score by
refusal-opener log-odds at the last prompt position. One forward per prompt per
cell; no decode, which is what makes the full 32x5 grid affordable -- the same
economy that let Arditi compute their Figure 11.

The opener token set is the project's already-verified one
(`capture_stageb.py:30-48`), so this metric is continuous with `refusal_prob`
in §14 rather than a new definition.

Storage note: the full (800, 5, 32, 4096) tensor would be 2.1 GB. Only the
per-class means are needed for `v_ref` at every cell, and those are 5 MB; full
activations are retained solely at the layers where `d` and PC1 must be rebuilt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import torch

import config
from src import ablation as AB
from src import model as M
from scripts.capture_stageb import refusal_opener_ids

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"
POSITIONS = [-1, -2, -3, -4, -5]
KEEP_LAYERS = [10, 11, 28]      # where d and PC1 get rebuilt
BATCH = 16
CACHE = NPY / "stageb_positions.npz"


@torch.no_grad()
def opener_logodds(L, chats, opener_ids, batch=BATCH):
    """Refusal-opener probability mass at the last prompt position, as log-odds.

    Arditi's selection metric: no generation, so a full grid is affordable. It is
    also continuous and non-saturating, unlike the substring rate.
    """
    tok, out = L.tokenizer, []
    for i in range(0, len(chats), batch):
        enc = tok(chats[i:i + batch], return_tensors="pt", padding=True,
                  padding_side="left").to("cuda")
        p = torch.softmax(L.model(**enc).logits[:, -1, :].float(), dim=-1)
        mass = p[:, opener_ids].sum(-1).clamp(1e-9, 1 - 1e-9)
        out.append(torch.log(mass / (1 - mass)).cpu().numpy())
    return np.concatenate(out)


def build_cache(L, df):
    """Capture at each position; keep per-class means everywhere, acts at KEEP_LAYERS."""
    rendered = [M.chat(L, p) for p in df.prompt]
    src = df.source.to_numpy()
    mh, ml = src == "harmful", src == "harmless"
    means_h = np.zeros((len(POSITIONS), config.N_LAYERS, config.HIDDEN_DIM), np.float32)
    means_l = np.zeros_like(means_h)
    kept = np.zeros((len(POSITIONS), len(KEEP_LAYERS), len(df), config.HIDDEN_DIM), np.float32)
    for pi, pos in enumerate(POSITIONS):
        acts = M.capture(L, rendered, positions=[pos] * len(rendered), batch_size=8).numpy()
        means_h[pi] = acts[mh].mean(0)
        means_l[pi] = acts[ml].mean(0)
        for li, layer in enumerate(KEEP_LAYERS):
            kept[pi, li] = acts[:, layer, :]
        print(f"  captured position {pos}: {acts.shape}", flush=True)
        del acts
    np.savez(CACHE, means_h=means_h, means_l=means_l, kept=kept,
             positions=np.array(POSITIONS), keep_layers=np.array(KEEP_LAYERS))
    return means_h, means_l, kept


def main():
    df = pd.read_csv(SB / "prompts.csv")
    ev = json.loads((SB / "eval_18_5.json").read_text())
    L = M.load()
    opener = refusal_opener_ids(L.tokenizer)
    opener_ids = sorted(set(opener.values()))
    print(f"refusal-opener tokens ({len(opener_ids)}): {opener_ids}")

    if CACHE.exists():
        z = np.load(CACHE)
        means_h, means_l, kept = z["means_h"], z["means_l"], z["kept"]
        print(f"loaded {CACHE.name}")
    else:
        print("capturing at positions", POSITIONS)
        means_h, means_l, kept = build_cache(L, df)

    hchats = [M.chat(L, p) for p in ev["harmful"]]
    base = opener_logodds(L, hchats, opener_ids)
    print(f"\nbaseline refusal-opener log-odds: {base.mean():+.3f} (n={len(base)})")

    grid = np.zeros((config.N_LAYERS, len(POSITIONS)), np.float32)
    for pi, pos in enumerate(POSITIONS):
        for layer in range(config.N_LAYERS):
            v = means_h[pi, layer] - means_l[pi, layer]
            v = v / np.linalg.norm(v)
            u = torch.tensor(v, dtype=torch.float32)
            with AB.ablate(L, u / u.norm(), alpha=1.0):
                s = opener_logodds(L, hchats, opener_ids)
            grid[layer, pi] = base.mean() - s.mean()      # bypass score
        print(f"  position {pos} done, best layer "
              f"{int(np.argmax(grid[:, pi]))} at {grid[:, pi].max():+.3f}", flush=True)

    print(f"\n=== bypass score (baseline - ablated log-odds), higher = more bypass ===")
    print("layer " + "".join(f"{p:>9}" for p in POSITIONS))
    for layer in range(config.N_LAYERS):
        star = "  <-" if layer in (10, 11) else ""
        print(f"{layer:>5} " + "".join(f"{grid[layer, pi]:9.3f}"
                                       for pi in range(len(POSITIONS))) + star)

    flat = [(int(l), POSITIONS[pi], float(grid[l, pi]))
            for l in range(config.N_LAYERS) for pi in range(len(POSITIONS))]
    flat.sort(key=lambda t: -t[2])
    print("\ntop 8 cells:")
    for l, p, s in flat[:8]:
        print(f"  layer {l:>2}, position {p}: {s:+.3f}")
    i10 = grid[10, POSITIONS.index(-1)]
    print(f"\n(10, -1) -- the inherited cell -- scores {i10:+.3f}, "
          f"rank {1 + sum(1 for _, _, s in flat if s > i10)} of {len(flat)}")

    json.dump(dict(positions=POSITIONS, baseline_logodds=float(base.mean()),
                   grid=grid.tolist(), top_cells=flat[:12],
                   inherited_cell_score=float(i10),
                   inherited_cell_rank=int(1 + sum(1 for _, _, s in flat if s > i10))),
              open(config.RESULTS / "stageb_18_24_grid.json", "w"), indent=2)
    print(f"\nwrote {config.RESULTS / 'stageb_18_24_grid.json'}")


if __name__ == "__main__":
    main()
