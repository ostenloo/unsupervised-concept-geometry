"""SPEC §3b gate 3 -- the knowledge gate.

For each surviving compound, ask two closed questions greedily:
  (i)  molecular formula
  (ii) functional-group class (multiple choice)

Keep only compounds where both are correct, and report the pass rate. Per SPEC
§9 risk 1, the pass rate is a result in its own right: if it is below 70% the
answer is to expand the pool, not to lower the bar.

    python scripts/knowledge_gate.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src import chemistry as C
from src import model as M

# "inorganic" is required, not padding: water, ammonia and carbon dioxide carry
# no functional group from this list, so without it the question has no correct
# answer for them and a right response is graded wrong. "inorganic" rather than
# "none of these" because it is the answer a chemist would actually give, and
# the compound set has an inorganic family by design (SPEC §3b).
NONE_OPTION = "inorganic"
CHOICES = ["alkane", "alkene", "alkyne", "alcohol", "aldehyde", "ketone",
           "carboxylic acid", "ester", "ether", "amide", "amine", "aromatic",
           "halide", NONE_OPTION]
LETTERS = "ABCDEFGHIJKLMN"
_CANON = {c.replace(" ", "_"): c for c in CHOICES}


def formula_prompt(name: str) -> str:
    return (f"What is the molecular formula of {name}? "
            f"Reply with the formula only, no other words.")


def class_prompt(name: str) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(CHOICES))
    return (f"Which functional-group class does {name} belong to?\n{opts}\n"
            f"Reply with the single letter only.")


_FORMULA_CHARS = r"[A-Za-z0-9₀-₉()\[\]=#·\.\+\-]"


def extract_formula(text: str) -> str:
    """The formula in the reply, as the longest parseable run.

    Two traps, both of which silently score correct answers wrong:
      - anchoring on [A-Z] skips a leading parenthesis, so "(C2H5)3N" is read as
        the unbalanced "C2H5)3N" and fails to parse;
      - the run can pick up trailing punctuation ("C4H9OH)." ).
    So: take candidate runs that may begin with "(", then trim from the right
    until src.chemistry.parse_formula accepts, and keep the longest survivor.
    That function rejects prose, so this cannot turn a non-answer into a pass.
    """
    t = text.strip().splitlines()[0] if text.strip() else ""
    best = ""
    for m in re.finditer(rf"[A-Z(]{_FORMULA_CHARS}*", t):
        cand = m.group(0)
        while cand:
            if C.parse_formula(cand) is not None:
                break
            cand = cand[:-1]
        if len(cand) > len(best):
            best = cand
    return best


def extract_letter(text: str) -> str | None:
    """Answer letter. The character class is derived from LETTERS on purpose --
    hard-coding a range silently discards the last option whenever CHOICES grows."""
    m = re.search(rf"\b([{LETTERS[0]}-{LETTERS[-1]}])\b", text.strip().upper())
    return m.group(1) if m else None


def main():
    df = pd.read_csv(config.DATA / "compounds_gated12.csv")
    print(f"{len(df)} compounds entering gate 3")

    L = M.load()
    print("\nOne fully rendered prompt:\n---")
    print(M.chat(L, formula_prompt("ethanol")))
    print("---\n")

    f_prompts = [M.chat(L, formula_prompt(n)) for n in df["name"]]
    c_prompts = [M.chat(L, class_prompt(n)) for n in df["name"]]

    # 16 was too short: condensed answers like "C6H5CH=CH2" were being clipped
    # mid-formula and scored as wrong.
    f_out = M.generate_greedy(L, f_prompts, max_new_tokens=32)
    c_out = M.generate_greedy(L, c_prompts, max_new_tokens=8)

    rows = []
    for i, r in df.iterrows():
        got_f = extract_formula(f_out[i])
        f_ok = C.formula_matches(got_f, r.formula)

        letter = extract_letter(c_out[i])
        picked = CHOICES[LETTERS.index(letter)] if letter else None
        # Accept any class the molecule actually contains, not just the
        # hand-assigned label -- see src/chemistry.functional_classes.
        acceptable = {_CANON.get(c, c) for c in C.functional_classes(r.smiles)}
        if C.is_inorganic(r.smiles):
            acceptable.add(NONE_OPTION)
        if not acceptable:
            acceptable = {NONE_OPTION}
        c_ok = picked is not None and picked in acceptable

        rows.append(dict(name=r["name"], formula_truth=r.formula, formula_got=got_f,
                         formula_raw=f_out[i].strip().replace("\n", " ")[:60],
                         class_raw=c_out[i].strip().replace("\n", " ")[:30],
                         formula_ok=f_ok, class_picked=picked,
                         class_acceptable=",".join(sorted(acceptable)), class_ok=c_ok,
                         passed=bool(f_ok and c_ok)))

    res = pd.DataFrame(rows)

    # --- gate 3b: isomer disambiguation (SPEC §11c) -------------------------
    # Bare series names are ambiguous: "butanol" does not distinguish 1- from
    # 2-butanol, "butene" does not distinguish 1- from 2-butene. We assign the
    # straight-chain primary/terminal SMILES. Neither gate-3 question can catch a
    # misassignment -- isomers share a molecular formula AND a functional-group
    # class -- so ask a third question of the ambiguous names only. This does not
    # gate; it tests whether our SMILES assignment matches the model's reading.
    # Probe the DENOTATION, not the classification, and use more than one
    # phrasing. "Is butanol primary or secondary?" is worthless here: forced to
    # one word the model calls decanol tertiary, open-ended it says primary --
    # that measures format compliance. Multiple probes are required because a
    # single one misleads: asking for alkenes' "IUPAC name" returns but-2-ene
    # while "systematic name" returns but-1-ene, from the same model.
    amb_alc = df[(df.series == "alcohol") & (df.series_index >= 3)]
    amb_alk = df[(df.series == "alkene") & (df.series_index >= 4)]
    PROBES = {
        "alcohol": [("What is the IUPAC name of {n}? Reply with the name only.",
                     lambda t: "1-ol" in t.lower() or "1-" in t.lower()),
                    ("Give the systematic name of {n}. Name only.",
                     lambda t: "1-ol" in t.lower() or "1-" in t.lower()),
                    ("Is the hydroxyl group in {n} on a terminal carbon? Reply yes or no.",
                     lambda t: t.strip().lower().startswith("yes"))],
        "alkene":  [("What is the IUPAC name of {n}? Reply with the name only.",
                     lambda t: "1-ene" in t.lower()),
                    ("Give the systematic name of {n}. Name only.",
                     lambda t: "1-ene" in t.lower()),
                    ("Is the double bond in {n} at a terminal position? Reply yes or no.",
                     lambda t: t.strip().lower().startswith("yes"))],
    }
    dis_rows = []
    for series, members in (("alcohol", amb_alc), ("alkene", amb_alk)):
        if not len(members):
            continue
        names = list(members["name"])
        agree = {n: [] for n in names}
        for tmpl, test in PROBES[series]:
            outs = M.generate_greedy(L, [M.chat(L, tmpl.format(n=n)) for n in names],
                                     max_new_tokens=16)
            for n, o in zip(names, outs):
                agree[n].append(bool(test(o.strip().splitlines()[0] if o.strip() else "")))
        for n in names:
            v = agree[n]
            dis_rows.append(dict(name=n, series=series, n_probes=len(v),
                                 n_agree=sum(v), unanimous=all(v)))

    if dis_rows:
        dd = pd.DataFrame(dis_rows)
        dd.to_csv(config.DATA / "isomer_disambiguation.csv", index=False)
        print("\n--- gate 3b: isomer denotation across 3 probes (validates our SMILES) ---")
        for series, grp in dd.groupby("series"):
            frac = grp.n_agree.sum() / (grp.n_probes.sum())
            print(f"    {series:8s} {int(grp.unanimous.sum())}/{len(grp)} names unanimous; "
                  f"probe agreement {frac:.0%}")
            for _, r in grp.iterrows():
                print(f"        {r['name']:12s} {r.n_agree}/{r.n_probes}")
        print("    Our SMILES assigns the straight-chain primary/terminal isomer. A name "
              "well below 3/3 has no stable denotation for this model (SPEC §11c).")

    df2 = df.join(res.drop(columns="name"))

    n = len(res)
    print()
    print(f"formula correct : {res.formula_ok.sum():3d}/{n}  ({res.formula_ok.mean():.1%})")
    print(f"class correct   : {res.class_ok.sum():3d}/{n}  ({res.class_ok.mean():.1%})")
    print(f"both correct    : {res.passed.sum():3d}/{n}  ({res.passed.mean():.1%})")

    print("\nfailures (first 40):")
    for _, r in res[~res.passed].head(40).iterrows():
        why = []
        if not r.formula_ok:
            why.append(f"formula got {r.formula_got!r} want {r.formula_truth!r}")
        if not r.class_ok:
            why.append(f"class picked {r.class_picked!r} acceptable {{{r.class_acceptable}}}")
        print(f"    {r['name']:22s} {'; '.join(why)}")

    kept = df2[df2.passed].drop(columns=["passed"]).reset_index(drop=True)

    print("\nseries survival after gate 3:")
    for s in ["alkane", "alkene", "alcohol", "acid"]:
        m = kept[kept.series == s]
        print(f"    {s:8s} {len(m):2d}/10  n_carbons: {sorted(int(i) for i in m.series_index)}")

    print("\nclass counts after gate 3:")
    for cls, cnt in kept.cls.value_counts().items():
        print(f"    {cls:18s} {cnt}")

    out = config.DATA / "compounds.csv"
    kept.to_csv(out, index=False)
    df2.to_csv(config.DATA / "gate3_detail.csv", index=False)
    (config.DATA / "gate3_log.json").write_text(json.dumps({
        "n_in": int(n), "n_out": int(len(kept)),
        "formula_pass_rate": float(res.formula_ok.mean()),
        "class_pass_rate": float(res.class_ok.mean()),
        "both_pass_rate": float(res.passed.mean()),
    }, indent=2))
    print(f"\nwrote {out} ({len(kept)} compounds)")
    if res.passed.mean() < 0.70:
        print("PASS RATE BELOW 70% -- SPEC §3b says expand the pool, do not lower the bar.")


if __name__ == "__main__":
    main()
