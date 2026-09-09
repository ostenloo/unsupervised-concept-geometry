"""§18.24 branches 3 and 4 — how much do `d` and PC1 move with the read position?

CPU, on the cache the grid script wrote. For each (layer, position) it rebuilds
the whole pipeline and reports the quantities the two branches turn on:

  branch 4: cos(PC1, v_ref) across positions -- the headline recovery number
  branch 3: cos(d, v_ref) across positions   -- what arms 2/3/4b are conditional on

AUROC is recomputed too, since the XSTest evaluation is equally position-dependent
and §18.12's 0.9935 was measured at -1 like everything else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

import config
from scripts.stageb_18_4ac import fit_coordinate, build_direction

SB = config.DATA / "stageb"
NPY = config.ACTS / "npy"


def main():
    df = pd.read_csv(SB / "prompts.csv")
    z = np.load(NPY / "stageb_positions.npz")
    kept, positions, keep_layers = z["kept"], list(z["positions"]), list(z["keep_layers"])

    src = df.source.to_numpy()
    mh, ml = src == "harmful", src == "harmless"
    m4 = mh | ml
    mx = (src == "borderline") | (src == "xstest_contrast")
    lab = (src[mx] == "xstest_contrast").astype(int)

    print(f"{'layer':>5} {'pos':>4} {'cos(d,vref)':>12} {'cos(PC1,vref)':>14} "
          f"{'cos(d,PC1)':>11} {'PearsonR':>9} {'CV R2':>7} {'AUROC(d)':>9} {'AUROC(PC1)':>11}")
    out = {}
    for li, layer in enumerate(keep_layers):
        for pi, pos in enumerate(positions):
            X = np.ascontiguousarray(kept[pi, li])
            Xf, hf = X[m4], mh[m4]
            v = X[mh].mean(0) - X[ml].mean(0)
            v = v / np.linalg.norm(v)
            fit = fit_coordinate(Xf)
            d, lin = build_direction(fit, Xf)
            if (Xf @ d)[hf].mean() < (Xf @ d)[~hf].mean():
                d = -d
            pc1 = fit["pca"].components_[0].copy()
            if (Xf @ pc1)[hf].mean() < (Xf @ pc1)[~hf].mean():
                pc1 = -pc1
            proj = fit["Xp"] @ (fit["pca"].components_ @ v)
            row = dict(cos_d_vref=float(d @ v), cos_pc1_vref=float(pc1 @ v),
                       cos_d_pc1=float(d @ pc1),
                       pearson_r=float(abs(pearsonr(proj, fit["coord"]).statistic)),
                       cv_r2=float(lin["r2_cv_64"]),
                       auroc_d=float(roc_auc_score(lab, X[mx] @ d)),
                       auroc_pc1=float(roc_auc_score(lab, X[mx] @ pc1)))
            out[f"L{layer}_p{pos}"] = row
            mark = "  <- inherited" if (layer == 10 and pos == -1) else ""
            print(f"{layer:>5} {pos:>4} {row['cos_d_vref']:12.4f} {row['cos_pc1_vref']:14.4f} "
                  f"{row['cos_d_pc1']:11.4f} {row['pearson_r']:9.4f} {row['cv_r2']:7.3f} "
                  f"{row['auroc_d']:9.4f} {row['auroc_pc1']:11.4f}{mark}")

    for layer in keep_layers:
        cd = [out[f"L{layer}_p{p}"]["cos_d_vref"] for p in positions]
        cp = [out[f"L{layer}_p{p}"]["cos_pc1_vref"] for p in positions]
        ad = [out[f"L{layer}_p{p}"]["auroc_d"] for p in positions]
        print(f"\nL{layer} across positions: cos(d,vref) {min(cd):.3f}-{max(cd):.3f} "
              f"(spread {max(cd)-min(cd):.3f}) | cos(PC1,vref) {min(cp):.3f}-{max(cp):.3f} "
              f"(spread {max(cp)-min(cp):.3f}) | AUROC(d) {min(ad):.3f}-{max(ad):.3f}")
        if layer == 10:
            b4 = all(0.90 <= c <= 0.96 for c in cp)
            print(f"  branch 4: cos(PC1,vref) {'stays inside' if b4 else 'LEAVES'} ~0.90-0.96")
            print(f"  branch 3: cos(d,vref) spread {max(cd)-min(cd):.3f} "
                  f"-> {'material' if max(cd)-min(cd) > 0.10 else 'immaterial'} movement")

    json.dump(out, open(config.RESULTS / "stageb_18_24_dirs.json", "w"), indent=2)
    print(f"\nwrote {config.RESULTS / 'stageb_18_24_dirs.json'}")


if __name__ == "__main__":
    main()
