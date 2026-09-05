"""SPEC §3a: reproduce the Wurgaft weekday loop on this model.

    Q: What day is {k} days after {entity}?
    A:                                        7 entities x 7 increments = 49 prompts

Centroid by answer day, PCA to 3-D, look for the loop in PC1/PC2 at layer 28.
This is a substrate check, not a result: if the loop is absent, the model variant
or the hook point is wrong and everything downstream is measuring nothing.

    python scripts/smoke_weekday.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from src import model as M
from src import geometry as G
from src import evaluation as E

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = ["one", "two", "three", "four", "five", "six", "seven"]


def main():
    prompts, answers = [], []
    for entity_i, entity in enumerate(DAYS):
        for k_i, k_word in enumerate(WORDS, start=1):
            prompts.append(config.WEEKDAY_TEMPLATE.format(k=k_word, entity=entity))
            answers.append(DAYS[(entity_i + k_i) % 7])

    print(f"{len(prompts)} prompts. Example:\n---\n{prompts[0]}\n---\n")

    L = M.load()
    acts = M.capture(L, prompts).numpy()          # [49, 32, 4096] float32
    print(f"captured {acts.shape}, dtype {acts.dtype}")

    results = {}
    for layer in (config.WURGAFT_LAYER, 31, 24, 20, 16):
        X = np.ascontiguousarray(acts[:, layer, :])
        # Centroid by answer day -- 7 points, as in Wurgaft Table 1.
        cents = np.stack([X[[i for i, a in enumerate(answers) if a == d]].mean(0)
                          for d in DAYS]).astype(np.float32)
        Y, _ = G.pca_project(cents, n_components=3)

        # The loop is present if the 7 answer-day centroids sit in cyclic order
        # around PC1/PC2. Measured two ways so a single statistic cannot flatter it.
        ang = np.arctan2(Y[:, 1], Y[:, 0])
        order = np.argsort(ang)
        # Cyclic order means the sorted-by-angle sequence is a rotation of
        # 0..6 or of its reversal.
        seq = list(order)
        rots = [seq[i:] + seq[:i] for i in range(7)]
        forward = list(range(7))
        backward = forward[::-1]
        cyclic = any(r == forward for r in rots) or any(r == backward for r in rots)
        gap = E.cyclic_score(Y)

        results[layer] = {"cyclic_order": bool(cyclic), "max_angular_gap_ratio": round(gap, 3)}
        print(f"layer {layer:2d}: cyclic_order={cyclic}  max_gap_ratio={gap:.2f}  "
              f"angle_order={[DAYS[i][:3] for i in order]}")

    (config.RESULTS / "smoke_weekday.json").write_text(json.dumps(results, indent=2))

    ok = results[config.WURGAFT_LAYER]["cyclic_order"]
    print(f"\nSPEC §3a / acceptance test 4 at layer {config.WURGAFT_LAYER}: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Loop absent at layer 28 -- do not proceed to §3b capture until this is "
              "resolved (wrong model variant or wrong hook point).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
