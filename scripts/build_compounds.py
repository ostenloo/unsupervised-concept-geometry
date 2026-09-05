"""SPEC §3b gates 1 and 2, plus the series formula assertion.

Gate 3 (the knowledge gate) needs the model and lives in scripts/knowledge_gate.py.
Every gate logs what it removed -- the pass rates are a result in their own right
(SPEC §9 risk 1).

    python scripts/build_compounds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import config
from src import chemistry as C
from data.compound_source import candidate_pool, SERIES_FORMULA

# SPEC §3b gate 2, refined -- see SPEC §11c.
#
# The spec's rationale is explicit: "the last-token activation of a fragmented
# name measures the fragment". Family 1 reads the LAST token of the name, so the
# quantity that matters is how the name's final word tokenizes, not the whole
# string. For every two-word "X acid" name that final token is `Ġacid` -- a
# complete word, never a fragment -- so a flat <=3-token budget rejects
# butyric/caproic/enanthic/pelargonic acid for stem fragmentation that the read
# position never sees. That budget costs 4/10 of the acid series.
#
# So: gate the final word at <=2 tokens, with a total cap to exclude names that
# are shredded overall. Single-word names are unaffected (final word == name, so
# the <=3 budget still binds through MAX_TOKENS).
MAX_TOKENS = 3          # single-word names, unchanged
MAX_TOKENS_TOTAL = 5    # multi-word names: overall cap
MAX_TOKENS_FINAL = 2    # multi-word names: the read position itself


def main():
    pool = candidate_pool()
    log = {"pool_size": len(pool)}
    print(f"candidate pool: {len(pool)}")

    # --- gate 1: RDKit validity --------------------------------------------
    rows, dropped_parse = [], []
    for name, smiles, cls, series, sidx in pool:
        mol = C.parse(smiles)
        if mol is None:
            dropped_parse.append((name, smiles))
            continue
        d = C.descriptors(smiles)
        rows.append(dict(name=name, smiles=smiles, cls=cls, series=series,
                         series_index=sidx, n_carbons=d["n_carbons"],
                         molecular_weight=d["molecular_weight"], formula=d["formula"]))
    log["gate1_removed"] = len(dropped_parse)
    print(f"gate 1 (RDKit validity): removed {len(dropped_parse)} -> {len(rows)}")
    for n, s in dropped_parse:
        print(f"    DROP {n}: {s}")

    df = pd.DataFrame(rows)

    # --- series formula assertion ------------------------------------------
    # Not a gate: a correctness check on the hand-curated data. A name/structure
    # mismatch inside a homologous series must fail loudly, not quietly distort
    # the Level 2 ordinal recovery that the whole Stage A argument rests on.
    bad = []
    for _, r in df[df.series.notna()].iterrows():
        expect = SERIES_FORMULA[r.series](int(r.series_index))
        if r.formula != expect:
            bad.append((r["name"], r.series, int(r.series_index), r.formula, expect))
    if bad:
        for b in bad:
            print(f"    FORMULA MISMATCH {b}")
        raise SystemExit(f"series formula assertion failed on {len(bad)} rows")
    print(f"series formula assertion: OK ({int(df.series.notna().sum())} members)")

    # --- gate 2: tokenizer --------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    # Leading-space variant: this is how the name actually appears mid-prompt in
    # the Family-1 template, and it tokenizes differently from the bare string.
    n_tokens = [len(tok.encode(" " + n, add_special_tokens=False)) for n in df["name"]]
    df["n_tokens"] = n_tokens
    df["n_tokens_final"] = [len(tok.encode(" " + n.split()[-1], add_special_tokens=False))
                            for n in df["name"]]
    df["multiword"] = [len(n.split()) > 1 for n in df["name"]]
    keep = ((~df.multiword) & (df.n_tokens <= MAX_TOKENS)) | (
        df.multiword & (df.n_tokens <= MAX_TOKENS_TOTAL)
        & (df.n_tokens_final <= MAX_TOKENS_FINAL))
    log["gate2_removed"] = int((~keep).sum())
    print(f"\ngate 2 (single-word <= {MAX_TOKENS} tok; multi-word <= {MAX_TOKENS_TOTAL} tok "
          f"with final word <= {MAX_TOKENS_FINAL}): removed {int((~keep).sum())} -> {int(keep.sum())}")
    admitted = df[keep & df.multiword & (df.n_tokens > MAX_TOKENS)]
    if len(admitted):
        print(f"\n  multi-word names admitted by the refinement ({len(admitted)}) -- "
              f"read position is the final token, shown last:")
        for _, r in admitted.sort_values("name").iterrows():
            print(f"    {r['name']:22s} {r.n_tokens} tok  {tok.tokenize(' ' + r['name'])}")
        print()
    for _, r in df[~keep].sort_values("n_tokens", ascending=False).iterrows():
        pieces = tok.tokenize(" " + r["name"])
        print(f"    DROP {r['name']:24s} {r.n_tokens:2d} tok  {pieces}")

    df = df[keep].reset_index(drop=True)

    # --- dedup on canonical SMILES -----------------------------------------
    # Two names for one structure sit at Tanimoto distance 0 while being distinct
    # points in activation space, which is a direct contradiction in D_truth.
    # Run after gate 2 so the survivor is the name that actually tokenizes
    # (e.g. "methylene chloride" survives, "dichloromethane" does not).
    from rdkit import Chem
    df["canonical_smiles"] = [Chem.MolToSmiles(C.parse(s_)) for s_ in df.smiles]
    dup = df.duplicated(subset="canonical_smiles", keep=False)
    if dup.any():
        print(f"\ncanonical-SMILES collisions ({int(dup.sum())} rows):")
        for cs, grp in df[dup].groupby("canonical_smiles"):
            print(f"    {cs}: {list(grp['name'])}")
    # Prefer series members: dropping one truncates a Level 2/3 series.
    df["_series_priority"] = df.series.notna().astype(int)
    df = (df.sort_values(["_series_priority", "n_tokens"], ascending=[False, True])
            .drop_duplicates(subset="canonical_smiles", keep="first")
            .drop(columns="_series_priority")
            .sort_index()
            .reset_index(drop=True))
    log["dedup_removed"] = int(dup.sum()) and int(keep.sum()) - len(df)
    print(f"dedup: -> {len(df)} compounds")

    # Series members are the Level 2/3 targets; losing one truncates a series.
    print("\nseries survival after gate 2:")
    for s in sorted(SERIES_FORMULA):
        members = df[df.series == s]
        idxs = sorted(int(i) for i in members.series_index)
        print(f"    {s:8s} {len(members):2d}/10  n_carbons present: {idxs}")

    print("\nclass counts after gate 2:")
    for cls, n in df.cls.value_counts().items():
        print(f"    {cls:18s} {n}")

    out = config.DATA / "compounds_gated12.csv"
    df.to_csv(out, index=False)
    (config.DATA / "gate_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out} ({len(df)} compounds)")
    print("gate 3 (knowledge) still to run: scripts/knowledge_gate.py")


if __name__ == "__main__":
    main()
