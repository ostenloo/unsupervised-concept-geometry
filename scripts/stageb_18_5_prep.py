"""§18.5 preparation: directions and eval sets. CPU only.

Directions are computed at L28 on the clean 400 (§18.4c′), which is also where
the sign convention is fixed -- from the training set, never test labels.

Eval sets follow §18.5b as amended: JailbreakBench Behaviors for harmful,
string-deduplicated against the 400-prompt fit bank because JBB shares lineage
with AdvBench/HarmBench; held-out `harmless_test.json` rows for harmless, rather
than a fresh Alpaca pull that could overlap either the fit set or the 2000-prompt
bank. n after dedup is reported for both, to the same standard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import urllib.request

import numpy as np
import pandas as pd

import config
from src import geometry as G
from scripts.stageb_18_4ac import fit_coordinate, build_direction, difference_in_means

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"
JBB_URL = ("https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/"
           "resolve/main/data/harmful-behaviors.csv")
N_EVAL = 100        # §18.5b under branch B1


def norm(s):
    return " ".join(str(s).lower().split()).strip(" .?!")


def main():
    df = pd.read_csv(SB / "prompts.csv")
    A = np.load(NPY / "stageb.npy")
    src = df.source.to_numpy()
    mh, ml = src == "harmful", src == "harmless"
    m4 = mh | ml
    layer = config.WURGAFT_LAYER
    X = np.ascontiguousarray(A[:, layer, :])
    Xf = X[m4]
    hf = mh[m4]

    fit = fit_coordinate(Xf)
    d400, lin = build_direction(fit, Xf)
    if (Xf @ d400)[hf].mean() < (Xf @ d400)[~hf].mean():
        d400 = -d400
    v200 = difference_in_means(X[mh], X[ml])
    pc1 = fit["pca"].components_[0].copy()
    if (Xf @ pc1)[hf].mean() < (Xf @ pc1)[~hf].mean():
        pc1 = -pc1

    d_perp = d400 - (d400 @ v200) * v200
    d_perp /= np.linalg.norm(d_perp)

    rng = np.random.default_rng(config.SEED)
    randoms = []
    for _ in range(5):
        r = rng.normal(size=X.shape[1]).astype(np.float32)
        randoms.append(r / np.linalg.norm(r))

    # arm 4b: random, constrained to the MEASURED cosine (§18.17)
    target = float(d400 @ v200)
    constrained = []
    for _ in range(5):
        r = rng.normal(size=X.shape[1]).astype(np.float32)
        r -= (r @ v200) * v200
        r /= np.linalg.norm(r)
        c = target * v200 + np.sqrt(1 - target ** 2) * r
        constrained.append((c / np.linalg.norm(c)).astype(np.float32))

    print(f"L{layer} directions:")
    print(f"  cos(d400, v_ref200) = {target:+.4f}   <- arm 4b constraint (§18.17)")
    print(f"  cos(pc1,  v_ref200) = {float(pc1 @ v200):+.4f}")
    print(f"  cos(d_perp, v_ref)  = {float(d_perp @ v200):+.2e} (orthogonal by construction)")
    print(f"  cos(d400, pc1)      = {float(d400 @ pc1):+.4f}")
    print(f"  arm 4b realised cos = {float(constrained[0] @ v200):+.4f}")
    print(f"  linearity CV R2     = {lin['r2_cv_64']:.3f}")

    np.savez(config.RESULTS / "stageb_18_5_directions.npz",
             d400=d400.astype(np.float32), v200=v200.astype(np.float32),
             d_perp=d_perp.astype(np.float32), pc1=pc1.astype(np.float32),
             randoms=np.stack(randoms), constrained=np.stack(constrained),
             target_cos=np.float32(target))

    # --- eval sets ------------------------------------------------------
    fit_prompts = {norm(p) for p in df.prompt[m4]}
    bank_prompts = {norm(p) for p in json.loads((NPY / "bank_prompts.json").read_text())}

    jbb_path = SB / "jbb_behaviors.csv"
    if not jbb_path.exists():
        urllib.request.urlretrieve(JBB_URL, jbb_path)
    jbb = pd.read_csv(jbb_path)
    goal_col = "Goal" if "Goal" in jbb.columns else jbb.columns[1]
    jbb_all = [str(g) for g in jbb[goal_col].dropna()]
    jbb_keep = [g for g in jbb_all if norm(g) not in fit_prompts]
    print(f"\nharmful (JailbreakBench {goal_col}): {len(jbb_all)} -> "
          f"{len(jbb_keep)} after dedup vs the 400-prompt fit bank "
          f"({len(jbb_all) - len(jbb_keep)} removed)")

    hl = [r["instruction"] for r in json.loads((SB / "harmless_test.json").read_text())]
    hl_keep = [p for p in dict.fromkeys(hl)
               if norm(p) not in fit_prompts and norm(p) not in bank_prompts]
    print(f"harmless (harmless_test.json): {len(hl)} -> {len(hl_keep)} after dedup "
          f"vs fit set AND the 2000-prompt bank ({len(hl) - len(hl_keep)} removed)")

    ev = dict(harmful=jbb_keep[:N_EVAL], harmless=hl_keep[:N_EVAL],
              n_harmful_available=len(jbb_keep), n_harmless_available=len(hl_keep),
              n_harmful_removed=len(jbb_all) - len(jbb_keep),
              n_harmless_removed=len(hl) - len(hl_keep),
              jbb_source=JBB_URL, target_cos=target)
    (SB / "eval_18_5.json").write_text(json.dumps(ev, indent=2))
    print(f"\nusing n={len(ev['harmful'])} harmful, n={len(ev['harmless'])} harmless")
    print(f"wrote {SB / 'eval_18_5.json'}")


if __name__ == "__main__":
    main()
