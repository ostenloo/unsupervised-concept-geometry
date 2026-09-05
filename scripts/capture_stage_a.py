"""SPEC §3d -- activation capture for Stage A.

Family 1 (structure recovery, Levels 1-2, §3f-§3h):
    "The chemical compound {name} is"
    captured at the LAST TOKEN OF THE NAME, not the last prompt token.

Family 2 (steering, Level 3):
    "The straight-chain alkane with {n} carbon atoms is called"
    captured at the last prompt token, plus the next-token distribution
    restricted to the ten concept names and an `other` bin.

All 32 layers in a single pass, float32 on save.

    python scripts/capture_stage_a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from src import model as M
from src.geometry import assert_geometry_dtype


def _verify_name_positions(L, prompts, names, positions, n_show=5):
    """The name is mid-prompt, so the read position is computed, not assumed.

    BPE is not guaranteed to tokenize a prefix consistently, so check that the
    token actually sitting at each computed index belongs to the name. A silent
    off-by-one here would make every Stage A number a measurement of the wrong
    token, and nothing downstream would look wrong.
    """
    tok = L.tokenizer
    bad = []
    for i, (p, nm, pos) in enumerate(zip(prompts, names, positions)):
        ids = tok.encode(p, add_special_tokens=False)
        piece = tok.decode([ids[pos]])
        if piece.strip() and piece.strip() not in nm and not nm.endswith(piece.strip()):
            bad.append((nm, pos, piece))
        if i < n_show:
            ctx = tok.decode(ids[max(0, len(ids) + pos - 2): len(ids) + pos + 1])
            print(f"    {nm:22s} pos={pos:3d} token={piece!r:14s} context={ctx!r}")
    if bad:
        for nm, pos, piece in bad[:10]:
            print(f"    MISMATCH {nm!r} pos={pos} token={piece!r}")
        raise SystemExit(f"read position does not land on the name for {len(bad)} compounds")
    print(f"    read position verified for all {len(prompts)} compounds")


def capture_family1(L, df):
    prompts, positions = [], []
    for name in df["name"]:
        p = config.FAMILY1_TEMPLATE.format(name=name)
        prompts.append(p)
        positions.append(M.last_token_index_of(L.tokenizer, p, name))

    print(f"\nFamily 1: {len(prompts)} prompts. Example: {prompts[0]!r}")
    print("  read-position check (last token of the NAME, not of the prompt):")
    _verify_name_positions(L, prompts, list(df["name"]), positions)

    acts = M.capture(L, prompts, positions=positions)
    assert_geometry_dtype(acts, "family1")
    print(f"  captured {tuple(acts.shape)} {acts.dtype}")
    torch.save({"acts": acts, "names": list(df["name"])}, config.ACTS / "family1.pt")
    return acts


SERIES_NAMES = {
    "alkane": ["methane", "ethane", "propane", "butane", "pentane",
               "hexane", "heptane", "octane", "nonane", "decane"],
    "alcohol": ["methanol", "ethanol", "propanol", "butanol", "pentanol",
                "hexanol", "heptanol", "octanol", "nonanol", "decanol"],
}


def _name_token_ids(tok, names):
    """Token ids opening each concept name, aggregating leading-space and case
    variants (Wurgaft App. A.2)."""
    out = {}
    for nm in names:
        ids = set()
        for var in (nm, " " + nm, nm.capitalize(), " " + nm.capitalize()):
            enc = tok.encode(var, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[nm] = sorted(ids)
    return out


@torch.no_grad()
def capture_family2(L, series):
    names = SERIES_NAMES[series]
    prompts = [config.FAMILY2_TEMPLATES[series].format(n=n) for n in range(1, 11)]
    print(f"\nFamily 2 [{series}]: example {prompts[0]!r}")

    gen = M.generate_greedy(L, prompts, max_new_tokens=6)
    correct = [names[i].lower() in g.lower() for i, g in enumerate(gen)]
    print(f"  greedy accuracy {sum(correct)}/10  " +
          ", ".join(f"{n}->{g.strip()[:12]!r}" for n, g in list(zip(names, gen))[:5]))
    if sum(correct) < 8:
        print(f"  SPEC §3d: accuracy < 8/10, series '{series}' is DROPPED from Level 3.")

    acts = M.capture(L, prompts)
    assert_geometry_dtype(acts, f"family2/{series}")

    tok = L.tokenizer
    name_ids = _name_token_ids(tok, names)
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    logits = L.model(**enc).logits[:, -1, :].to(torch.float32)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # [10 prompts, 11 bins]: ten concept names plus `other`.
    dist = np.zeros((len(prompts), len(names) + 1), dtype=np.float32)
    for j, nm in enumerate(names):
        dist[:, j] = probs[:, name_ids[nm]].sum(axis=1)
    dist[:, -1] = np.clip(1.0 - dist[:, :-1].sum(axis=1), 0.0, None)

    print(f"  mass on concept names (per n): "
          f"{np.round(1 - dist[:, -1], 2).tolist()}")
    torch.save({"acts": acts, "dist": dist, "names": names,
                "correct": correct, "series": series},
               config.ACTS / f"family2_{series}.pt")
    return acts, dist, sum(correct)


def main():
    df = pd.read_csv(config.DATA / "compounds.csv")
    print(f"{len(df)} gated compounds")
    L = M.load()

    capture_family1(L, df)
    summary = {}
    for series in SERIES_NAMES:
        _, _, n_ok = capture_family2(L, series)
        summary[series] = {"greedy_correct": int(n_ok), "kept_for_level3": bool(n_ok >= 8)}

    (config.RESULTS / "capture_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote acts/family1.pt and acts/family2_*.pt")


if __name__ == "__main__":
    main()
