"""Ground truth (SPEC §3c): Tanimoto distance on molecular fingerprints.

d_ij = 1 - Tanimoto(fp_i, fp_j) is the Jaccard distance, a proper metric on
binary fingerprints. Acceptance test 7 checks symmetry, zero diagonal and the
triangle inequality on random triples.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

import config


def parse(smiles: str):
    """SMILES -> Mol, or None. This is gate 1."""
    return Chem.MolFromSmiles(smiles)


def _morgan_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS
    )


def morgan_count_fp(mol):
    """Morgan COUNT fingerprint -- the default ground truth. See note below."""
    return _morgan_generator().GetCountFingerprint(mol)


def morgan_fp(mol):
    """Morgan BINARY fingerprint, exactly as SPEC §3c specifies.

    Kept as the comparison arm, not the default, because it is degenerate on the
    structure Stage A exists to measure. In a straight chain every interior
    carbon has the same radius-2 environment, so past ~7 carbons the *set* of
    environments stops growing and the bit vector stops changing: heptane,
    octane, nonane, decane, undecane and eicosane are all at Tanimoto distance
    exactly 0.000 from each other. The alkane homologous series is the spec's own
    continuous-manifold arm and the Level 2 / Level 3 target, so a ground truth
    that cannot resolve its last six members cannot validate anything. Across the
    whole compound set, 36% of pairs sit at exactly 1.0 (no shared bits), which
    also floods the Level 1 Spearman with ties.

    Counts fix it: they record how many times each environment occurs, so
    heptane and decane differ by their interior-CH2 count. The resulting
    generalised (MinMax/Soergel) Tanimoto is still a proper metric, and the
    alkane series becomes a strictly monotone distance gradient in carbon count.
    Both fingerprints are reported (SPEC §3j asks for fingerprint robustness);
    disagreement between them is a finding, not a nuisance.
    """
    return _morgan_generator().GetFingerprint(mol)


def maccs_fp(mol):
    """MACCS structural keys. Saturates on homologous series like binary Morgan
    (heptane through decane are again all at distance 0)."""
    return MACCSkeys.GenMACCSKeys(mol)


_FP = {"morgan_count": morgan_count_fp, "morgan": morgan_fp, "maccs": maccs_fp}
DEFAULT_FP = "morgan_count"


def fingerprints(smiles_list, kind=None):
    kind = DEFAULT_FP if kind is None else kind
    fn = _FP[kind]
    out = []
    for s in smiles_list:
        mol = parse(s)
        if mol is None:
            raise ValueError(f"unparsable SMILES reached fingerprinting: {s!r}")
        out.append(fn(mol))
    return out


def tanimoto_distance_matrix(smiles_list, kind=None) -> np.ndarray:
    """N x N float32 Jaccard distance matrix. SPEC §3c."""
    fps = fingerprints(smiles_list, kind=kind)
    n = len(fps)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # BulkTanimotoSimilarity is O(n) per row and much faster than pairwise calls.
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        D[i, :] = 1.0 - np.asarray(sims, dtype=np.float32)
    # Kill float asymmetry from the bulk path; the diagonal must be exactly zero.
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def descriptors(smiles: str) -> dict:
    """Per-compound ground-truth scalars (SPEC §3c)."""
    mol = parse(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    return {
        "n_carbons": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C"),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "formula": CalcMolFormula(mol),
    }


def check_metric_axioms(D: np.ndarray, n_triples: int = 1000, seed: int = 0):
    """Acceptance test 7. Returns (ok, message)."""
    rng = np.random.default_rng(seed)
    if not np.allclose(D, D.T, atol=1e-6):
        return False, "distance matrix not symmetric"
    if not np.allclose(np.diag(D), 0.0, atol=1e-7):
        return False, "diagonal not zero"
    if (D < -1e-7).any():
        return False, "negative distances"
    n = D.shape[0]
    idx = rng.integers(0, n, size=(n_triples, 3))
    # Jaccard is a metric, so d(i,k) <= d(i,j) + d(j,k) exactly; allow fp slack only.
    viol = D[idx[:, 0], idx[:, 2]] > D[idx[:, 0], idx[:, 1]] + D[idx[:, 1], idx[:, 2]] + 1e-5
    if viol.any():
        return False, f"triangle inequality violated on {int(viol.sum())}/{n_triples} triples"
    return True, f"symmetric, zero diagonal, triangle inequality holds on {n_triples} triples"


# --- functional-group membership (for SPEC §3b gate 3) ---------------------
# A compound legitimately belongs to several classes at once: aspirin is an
# ester, an aromatic and a carboxylic acid. Grading the gate-3 multiple choice
# against a single hand-assigned label would therefore fail correct answers and
# understate the model's knowledge. We derive the acceptable answer set by
# substructure match instead, and accept any class the molecule actually
# contains. This doubles as a check on the hand-assigned `cls` column.

_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester":           "[CX3](=[OX1])[OX2H0][#6]",
    # Recursive form so formaldehyde (CX3H2=O, no carbon neighbour) counts as an
    # aldehyde. The plain [CX3H1](=O)[#6] pattern excludes it, which made a
    # correct model answer ungradeable.
    "aldehyde":        "[$([CX3H1](=O)[#6]),$([CX3H2]=O)]",
    "ketone":          "[#6][CX3](=[OX1])[#6]",
    "alcohol":         "[#6;!$(C=O)][OX2H]",
    "amine":           "[NX3;H2,H1,H0;!$(N[CX3]=[OX1]);!$(N=*);!$([N+])]",
    "halide":          "[#6][F,Cl,Br,I]",
    "alkene":          "[CX3]=[CX3]",
    "alkyne":          "[CX2]#[CX2]",
    # Ether and amide are here so that "no listed functional group" cannot be
    # confused with "inorganic": diethyl ether, dioxane, ethylene oxide, urea
    # and formamide are organic, and without these patterns they fell through to
    # the inorganic bucket and made a correct model answer ungradeable.
    "ether":           "[OX2;!$(O[CX3]=[OX1]);!$(O[#1])]([#6])[#6]",
    "amide":           "[NX3][CX3](=[OX1])",
    "aromatic":        "[a]",
    # Basic aromatic nitrogen (pyridine, imidazole, purine). Standard texts do
    # class these as heterocyclic amines, so "amine" is an acceptable -- not the
    # primary -- answer for them, and grading it wrong understates the model.
    "amine_aromatic":  "[nX2]",
}
_COMPILED = None


def _compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}
        missing = [k for k, v in _COMPILED.items() if v is None]
        if missing:
            raise RuntimeError(f"bad SMARTS for {missing}")
    return _COMPILED


def functional_classes(smiles: str) -> set[str]:
    """Every functional-group class the molecule actually contains."""
    mol = parse(smiles)
    if mol is None:
        return set()
    found = {name for name, patt in _compiled().items() if mol.HasSubstructMatch(patt)}
    if mol.HasSubstructMatch(_compiled()["amine_aromatic"]):
        found.add("amine")
    found.discard("amine_aromatic")
    # A ketone pattern also matches a carboxylic acid/ester carbonyl; drop the
    # weaker claim when the stronger one is present.
    if found & {"carboxylic_acid", "ester", "aldehyde"}:
        found.discard("ketone")
    # Amide nitrogen is not a free amine.
    if "amide" in found:
        found.discard("amine")
    # Alkane means a saturated hydrocarbon: carbon and hydrogen only, and no
    # multiple bonds. Without the unsaturation check, acetylene falls through to
    # "alkane" because no other pattern claims it.
    if not found and all(a.GetSymbol() in ("C", "H") for a in mol.GetAtoms()):
        unsaturated = any(b.GetBondTypeAsDouble() > 1.0 for b in mol.GetBonds())
        found.add("alkane" if not unsaturated else "alkene")
    return found


_INORGANIC_CARBON = [
    "[C-]#[O+]",                       # carbon monoxide
    "O=C=O",                           # carbon dioxide
    "S=C=S",                           # carbon disulfide
    "[CX3](=[OX1])([OX1H0-,OX2H1])[OX1H0-,OX2H1]",   # carbonate / carbonic acid
    "[C-]#N",                          # cyanide
]
_INORG_COMPILED = None


def is_inorganic(smiles: str) -> bool:
    """True for compounds outside organic functional-group taxonomy.

    Carbon-free, or one of an explicit list of conventional inorganic carbon
    species (the oxides, carbonates, cyanides). Deliberately a list and not a
    heuristic: "carbon bearing no hydrogen" seems reasonable and classes urea as
    inorganic, which is wrong in the one direction chemistry students are taught
    to remember.

    Needed because an empty functional-class set means only "no group from our
    list", which is also true of ethers and amides -- calling those inorganic
    would grade a correct model answer wrong.
    """
    global _INORG_COMPILED
    mol = parse(smiles)
    if mol is None:
        return False
    if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        return True
    if _INORG_COMPILED is None:
        _INORG_COMPILED = [Chem.MolFromSmarts(p) for p in _INORGANIC_CARBON]
    n_carbon = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    for patt in _INORG_COMPILED:
        m = mol.GetSubstructMatches(patt)
        if m and sum(1 for idx in {i for t in m for i in t}
                     if mol.GetAtomWithIdx(idx).GetSymbol() == "C") == n_carbon:
            return True
    return False


# --- molecular formula comparison ------------------------------------------

def parse_formula(text: str) -> dict | None:
    """Parse a molecular OR condensed structural formula into element counts.

    'C2H6O', 'HCl', '(CH3)2NH', 'CH3CH2OH', 'C6H5CH=CH2', 'C6H4(OH)2' all parse.

    Condensed forms matter: asked for a formula, the model frequently answers
    '(C2H5)3N' for triethylamine, which is correct. A parser that stops at the
    first parenthesis reads that as 'C2H5' and scores a right answer wrong --
    which is what an earlier version of this function did, costing ~8 points of
    apparent pass rate. Comparison is on element multisets, so 'HCl' matches
    RDKit's canonical 'ClH'.
    """
    import re

    t = text.strip()
    t = t.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    t = t.translate(str.maketrans("[]", "()"))
    # Bond glyphs carry no stoichiometry in a condensed formula.
    for ch in "=#\u2013\u2014-\u00b7.":
        t = t.replace(ch, "")
        t = t.replace("+", "").replace("\u2212", "")
    t = t.replace(" ", "")
    if not t:
        return None

    tokens = re.findall(r"[A-Z][a-z]?|\d+|\(|\)|.", t)
    # Anything outside the formula grammar means this is prose, not a formula.
    for tk in tokens:
        if not (re.fullmatch(r"[A-Z][a-z]?", tk) or tk.isdigit() or tk in "()"):
            return None

    def merge(dst, src, mult=1):
        for k, v in src.items():
            dst[k] = dst.get(k, 0) + v * mult

    pos = 0

    def parse_group():
        nonlocal pos
        counts: dict[str, int] = {}
        while pos < len(tokens):
            tk = tokens[pos]
            if tk == "(":
                pos += 1
                inner = parse_group()
                if pos >= len(tokens) or tokens[pos] != ")":
                    raise ValueError("unbalanced")
                pos += 1
                mult = 1
                if pos < len(tokens) and tokens[pos].isdigit():
                    mult = int(tokens[pos]); pos += 1
                merge(counts, inner, mult)
            elif tk == ")":
                return counts
            elif tk.isdigit():
                # A bare leading number is a stoichiometric coefficient ("2H2O").
                raise ValueError("unexpected digit")
            else:
                pos += 1
                n = 1
                if pos < len(tokens) and tokens[pos].isdigit():
                    n = int(tokens[pos]); pos += 1
                merge(counts, {tk: n})
        return counts

    try:
        counts = parse_group()
    except ValueError:
        return None
    if pos != len(tokens) or not counts:
        return None
    return counts


def formula_matches(answer: str, truth: str) -> bool:
    a, b = parse_formula(answer), parse_formula(truth)
    if a is None or b is None:
        return False
    if a == b:
        return True
    # RDKit writes ionic species with an explicit charge (choline -> 'C5H14NO+');
    # the model answers the neutral skeleton. Compare heavy atoms and allow a
    # one-proton difference in that case only.
    if truth.strip().endswith(("+", "-")):
        ah = {k: v for k, v in a.items() if k != "H"}
        bh = {k: v for k, v in b.items() if k != "H"}
        return ah == bh and abs(a.get("H", 0) - b.get("H", 0)) <= 1
    return False
