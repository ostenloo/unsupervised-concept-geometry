"""Candidate compound pool (SPEC §3b), before gating.

Three families, per the spec:
  1. homologous series  -- the continuous-manifold arm (§3f Level 2/3 targets)
  2. functional-group classes -- the hierarchy arm (§3i)
  3. inorganics / everyday compounds -- breadth

SMILES are hand-curated, not LLM-generated (§3b gate 1 forbids that). Series
members are built programmatically and their molecular formulas are asserted
against the closed form for the series, so a name/structure mismatch fails loudly.

Every row: (name, smiles, cls, series, series_index)
`series` is None for non-series compounds; `series_index` is the carbon count
for series members and None otherwise. n_carbons is computed by RDKit later.
"""

# --- 1. homologous series --------------------------------------------------

_ALKANE_NAMES = ["methane", "ethane", "propane", "butane", "pentane",
                 "hexane", "heptane", "octane", "nonane", "decane"]

# NAMING DECISION (drives §3b gate 2, and it is load-bearing).
# The IUPAC "1-" locant prefix is catastrophic for the Llama-3.1 tokenizer:
# "1-propanol" -> ['1','-pro','pan','ol'] (5 tokens with the leading space), so
# the <=3-token gate deleted 8/10 alcohols and 7/9 alkenes -- i.e. it destroyed
# both of the §3f Level 3 steering series. The bare names cost at most 3 tokens
# each and recover 10/10 and 9/9.
#
# The price is isomer ambiguity: bare "butanol" does not distinguish 1- from
# 2-butanol, and "butene" does not distinguish 1- from 2-butene. We assign the
# straight-chain primary/terminal SMILES, which is the dominant reading of the
# bare name. Gate 3 cannot catch a wrong reading (both isomers share a molecular
# formula and a functional-group class), so this is a stated limitation, not a
# tested one. It does not threaten Level 2: every isomer of a given member has
# the same carbon count, so the ordinal recovery is unaffected; it perturbs only
# individual D_truth entries.
_ALKENE_NAMES = ["ethene", "propene", "butene", "pentene", "hexene",
                 "heptene", "octene", "nonene", "decene"]  # n = 2..10

_ALCOHOL_NAMES = ["methanol", "ethanol", "propanol", "butanol", "pentanol",
                  "hexanol", "heptanol", "octanol", "nonanol", "decanol"]

# Common names throughout: the model knows "butyric acid" far better than
# "butanoic acid", and the tokenizer gate treats them very differently.
# Neither naming survives gate 2 intact. Common names keep 6/10 (n = 1,2,3,5,8,10);
# the IUPAC "-anoic acid" forms keep 0/10, every one costing 4-5 tokens. So the
# acid series is reported at 6/10 with gaps at n = 4,6,7,9. Level 2's Spearman
# against n_carbons does not require consecutive members, so a gapped series is
# still a valid ordinal test -- just a weaker one. Acids are Level 2 only; the
# Level 3 steering series (alkane, alcohol) are both complete.
_ACID_NAMES = ["formic acid", "acetic acid", "propionic acid", "butyric acid",
               "valeric acid", "caproic acid", "enanthic acid", "caprylic acid",
               "pelargonic acid", "capric acid"]


def _series_rows():
    rows = []
    for i, name in enumerate(_ALKANE_NAMES, start=1):
        rows.append((name, "C" * i, "alkane", "alkane", i))
    for j, name in enumerate(_ALKENE_NAMES):
        n = j + 2
        rows.append((name, "C=C" + "C" * (n - 2), "alkene", "alkene", n))
    for i, name in enumerate(_ALCOHOL_NAMES, start=1):
        rows.append((name, "C" * i + "O", "alcohol", "alcohol", i))
    for i, name in enumerate(_ACID_NAMES, start=1):
        rows.append((name, "C" * (i - 1) + "C(=O)O", "carboxylic_acid", "acid", i))
    return rows


# Closed-form molecular formulas, asserted against RDKit in build_compounds.py.
SERIES_FORMULA = {
    "alkane":  lambda n: f"C{n}H{2 * n + 2}" if n > 1 else "CH4",
    "alkene":  lambda n: f"C{n}H{2 * n}",
    "alcohol": lambda n: (f"C{n}H{2 * n + 2}O" if n > 1 else "CH4O"),
    "acid":    lambda n: (f"C{n}H{2 * n}O2" if n > 1 else "CH2O2"),
}


# --- 2. functional-group classes -------------------------------------------
# (name, smiles) per class. Straight-chain series members live above and are not
# repeated here; these are the branched/cyclic/aromatic members that give each
# class breadth without collapsing onto the series.

_CLASSES = {
"alkane": [
    ("isobutane", "CC(C)C"), ("isopentane", "CCC(C)C"), ("neopentane", "CC(C)(C)C"),
    ("cyclopropane", "C1CC1"), ("cyclobutane", "C1CCC1"), ("cyclopentane", "C1CCCC1"),
    ("cyclohexane", "C1CCCCC1"), ("cycloheptane", "C1CCCCCC1"),
    ("cyclooctane", "C1CCCCCCC1"), ("methylcyclohexane", "CC1CCCCC1"),
    ("2-methylpentane", "CCCC(C)C"), ("3-methylpentane", "CCC(C)CC"),
    ("2,2-dimethylbutane", "CCC(C)(C)C"), ("2,3-dimethylbutane", "CC(C)C(C)C"),
    ("isooctane", "CC(C)CC(C)(C)C"), ("undecane", "CCCCCCCCCCC"),
    ("dodecane", "CCCCCCCCCCCC"), ("hexadecane", "CCCCCCCCCCCCCCCC"),
    ("octadecane", "CCCCCCCCCCCCCCCCCC"), ("eicosane", "CCCCCCCCCCCCCCCCCCCC"),
],
"alcohol": [
    ("isopropanol", "CC(C)O"), ("isobutanol", "CC(C)CO"), ("tert-butanol", "CC(C)(C)O"),
    ("sec-butanol", "CCC(C)O"), ("cyclohexanol", "OC1CCCCC1"),
    ("benzyl alcohol", "OCc1ccccc1"), ("ethylene glycol", "OCCO"),
    ("propylene glycol", "CC(O)CO"), ("glycerol", "OCC(O)CO"),
    ("menthol", "CC(C)C1CCC(C)CC1O"), ("geraniol", "CC(C)=CCCC(C)=CCO"),
    ("1-dodecanol", "CCCCCCCCCCCCO"), ("allyl alcohol", "C=CCO"),
    ("propargyl alcohol", "C#CCO"), ("2-pentanol", "CCCC(C)O"),
    ("3-pentanol", "CCC(O)CC"), ("1,4-butanediol", "OCCCCO"),
    ("xylitol", "OCC(O)C(O)C(O)CO"), ("sorbitol", "OCC(O)C(O)C(O)C(O)CO"),
    ("cholesterol", "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"),
],
"ketone": [
    ("acetone", "CC(C)=O"), ("butanone", "CCC(C)=O"), ("2-pentanone", "CCCC(C)=O"),
    ("3-pentanone", "CCC(=O)CC"), ("2-hexanone", "CCCCC(C)=O"),
    ("cyclohexanone", "O=C1CCCCC1"), ("cyclopentanone", "O=C1CCCC1"),
    ("acetophenone", "CC(=O)c1ccccc1"), ("benzophenone", "O=C(c1ccccc1)c1ccccc1"),
    ("camphor", "CC1(C)C2CCC1(C)C(=O)C2"), ("2-heptanone", "CCCCCC(C)=O"),
    ("2-octanone", "CCCCCCC(C)=O"), ("diacetyl", "CC(=O)C(C)=O"),
    ("mesityl oxide", "CC(C)=CC(C)=O"), ("4-heptanone", "CCCC(=O)CCC"),
    ("carvone", "CC(=C)C1CC=C(C)C(=O)C1"), ("menthone", "CC(C)C1CCC(C)CC1=O"),
    ("2-nonanone", "CCCCCCCC(C)=O"), ("2-decanone", "CCCCCCCCC(C)=O"),
],
"aldehyde": [
    ("formaldehyde", "C=O"), ("acetaldehyde", "CC=O"), ("propionaldehyde", "CCC=O"),
    ("butyraldehyde", "CCCC=O"), ("valeraldehyde", "CCCCC=O"), ("hexanal", "CCCCCC=O"),
    ("heptanal", "CCCCCCC=O"), ("octanal", "CCCCCCCC=O"), ("nonanal", "CCCCCCCCC=O"),
    ("decanal", "CCCCCCCCCC=O"), ("benzaldehyde", "O=Cc1ccccc1"),
    ("cinnamaldehyde", "O=C/C=C/c1ccccc1"), ("vanillin", "O=Cc1ccc(O)c(OC)c1"),
    ("citral", "CC(C)=CCCC(C)=CC=O"), ("glyoxal", "O=CC=O"), ("acrolein", "C=CC=O"),
    ("salicylaldehyde", "O=Cc1ccccc1O"), ("furfural", "O=Cc1ccco1"),
],
"ester": [
    ("methyl acetate", "COC(C)=O"), ("ethyl acetate", "CCOC(C)=O"),
    ("propyl acetate", "CCCOC(C)=O"), ("butyl acetate", "CCCCOC(C)=O"),
    ("pentyl acetate", "CCCCCOC(C)=O"), ("isoamyl acetate", "CC(C)CCOC(C)=O"),
    ("methyl formate", "COC=O"), ("ethyl formate", "CCOC=O"),
    ("methyl benzoate", "COC(=O)c1ccccc1"), ("ethyl benzoate", "CCOC(=O)c1ccccc1"),
    ("methyl salicylate", "COC(=O)c1ccccc1O"), ("benzyl acetate", "CC(=O)OCc1ccccc1"),
    ("ethyl butyrate", "CCCC(=O)OCC"), ("methyl butyrate", "CCCC(=O)OC"),
    ("vinyl acetate", "C=COC(C)=O"), ("ethyl propionate", "CCC(=O)OCC"),
    ("triacetin", "CC(=O)OCC(COC(C)=O)OC(C)=O"),
    ("dimethyl phthalate", "COC(=O)c1ccccc1C(=O)OC"),
],
"amine": [
    ("methylamine", "CN"), ("ethylamine", "CCN"), ("propylamine", "CCCN"),
    ("butylamine", "CCCCN"), ("pentylamine", "CCCCCN"), ("hexylamine", "CCCCCCN"),
    ("dimethylamine", "CNC"), ("trimethylamine", "CN(C)C"), ("diethylamine", "CCNCC"),
    ("triethylamine", "CCN(CC)CC"), ("aniline", "Nc1ccccc1"),
    ("benzylamine", "NCc1ccccc1"), ("ethylenediamine", "NCCN"),
    ("cyclohexylamine", "NC1CCCCC1"), ("piperidine", "C1CCNCC1"),
    ("pyrrolidine", "C1CCNC1"), ("morpholine", "C1COCCN1"),
    ("putrescine", "NCCCCN"), ("cadaverine", "NCCCCCN"),
    ("histamine", "NCCc1c[nH]cn1"), ("dopamine", "NCCc1ccc(O)c(O)c1"),
    ("serotonin", "NCCc1c[nH]c2ccc(O)cc12"), ("ethanolamine", "NCCO"),
],
"aromatic": [
    ("benzene", "c1ccccc1"), ("toluene", "Cc1ccccc1"), ("xylene", "Cc1ccccc1C"),
    ("ethylbenzene", "CCc1ccccc1"), ("styrene", "C=Cc1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"), ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("phenanthrene", "c1ccc2c(c1)ccc1ccccc12"), ("phenol", "Oc1ccccc1"),
    ("cresol", "Cc1ccc(O)cc1"), ("catechol", "Oc1ccccc1O"),
    ("resorcinol", "Oc1cccc(O)c1"), ("hydroquinone", "Oc1ccc(O)cc1"),
    ("benzoic acid", "OC(=O)c1ccccc1"), ("salicylic acid", "OC(=O)c1ccccc1O"),
    ("aspirin", "CC(=O)Oc1ccccc1C(O)=O"), ("nitrobenzene", "O=[N+]([O-])c1ccccc1"),
    ("mesitylene", "Cc1cc(C)cc(C)c1"), ("cumene", "CC(C)c1ccccc1"),
    ("biphenyl", "c1ccc(-c2ccccc2)cc1"), ("indole", "c1ccc2[nH]ccc2c1"),
    ("quinoline", "c1ccc2ncccc2c1"), ("pyridine", "c1ccncc1"), ("furan", "c1ccoc1"),
    ("thiophene", "c1ccsc1"), ("pyrrole", "c1cc[nH]c1"), ("imidazole", "c1c[nH]cn1"),
    ("phenylalanine", "NC(Cc1ccccc1)C(O)=O"),
],
"halide": [
    ("chloromethane", "CCl"), ("dichloromethane", "ClCCl"), ("chloroform", "ClC(Cl)Cl"),
    ("carbon tetrachloride", "ClC(Cl)(Cl)Cl"), ("bromomethane", "CBr"),
    ("iodomethane", "CI"), ("chloroethane", "CCCl"), ("bromoethane", "CCBr"),
    ("iodoethane", "CCI"), ("1-chloropropane", "CCCCl"), ("1-bromopropane", "CCCBr"),
    ("1-chlorobutane", "CCCCCl"), ("1-bromobutane", "CCCCBr"),
    ("vinyl chloride", "C=CCl"), ("trichloroethylene", "ClC=C(Cl)Cl"),
    ("tetrachloroethylene", "ClC(Cl)=C(Cl)Cl"), ("fluoromethane", "CF"),
    ("1,2-dichloroethane", "ClCCCl"), ("allyl chloride", "C=CCCl"),
    ("benzyl chloride", "ClCc1ccccc1"), ("chlorobenzene", "Clc1ccccc1"),
    ("bromobenzene", "Brc1ccccc1"),
],
"carboxylic_acid": [
    ("oxalic acid", "OC(=O)C(O)=O"), ("malonic acid", "OC(=O)CC(O)=O"),
    ("succinic acid", "OC(=O)CCC(O)=O"), ("glutaric acid", "OC(=O)CCCC(O)=O"),
    ("adipic acid", "OC(=O)CCCCC(O)=O"), ("lactic acid", "CC(O)C(O)=O"),
    ("citric acid", "OC(=O)CC(O)(CC(O)=O)C(O)=O"), ("malic acid", "OC(=O)CC(O)C(O)=O"),
    ("tartaric acid", "OC(=O)C(O)C(O)C(O)=O"), ("acrylic acid", "C=CC(O)=O"),
    ("oleic acid", "CCCCCCCCC=CCCCCCCCC(O)=O"),
    ("stearic acid", "CCCCCCCCCCCCCCCCCC(O)=O"),
    ("palmitic acid", "CCCCCCCCCCCCCCCC(O)=O"), ("lauric acid", "CCCCCCCCCCCC(O)=O"),
    ("myristic acid", "CCCCCCCCCCCCCC(O)=O"), ("glycolic acid", "OCC(O)=O"),
    ("pyruvic acid", "CC(=O)C(O)=O"), ("fumaric acid", "OC(=O)/C=C/C(O)=O"),
],
"inorganic": [
    ("water", "O"), ("ammonia", "N"), ("carbon dioxide", "O=C=O"),
    ("carbon monoxide", "[C-]#[O+]"), ("sodium chloride", "[Na+].[Cl-]"),
    ("sulfuric acid", "OS(=O)(=O)O"), ("nitric acid", "O[N+](=O)[O-]"),
    ("hydrochloric acid", "Cl"), ("sodium hydroxide", "[Na+].[OH-]"),
    ("potassium chloride", "[K+].[Cl-]"), ("hydrogen peroxide", "OO"),
    ("ozone", "[O-][O+]=O"), ("sulfur dioxide", "O=S=O"),
    ("nitrous oxide", "[N-]=[N+]=O"), ("hydrogen sulfide", "S"),
    ("carbon disulfide", "S=C=S"), ("phosphoric acid", "OP(O)(O)=O"),
    ("calcium carbonate", "[Ca+2].[O-]C([O-])=O"),
],
"other": [
    ("glucose", "OCC1OC(O)C(O)C(O)C1O"), ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("nicotine", "CN1CCCC1c1cccnc1"), ("urea", "NC(N)=O"), ("acetonitrile", "CC#N"),
    ("dimethyl sulfoxide", "CS(C)=O"), ("tetrahydrofuran", "C1CCOC1"),
    ("dioxane", "C1COCCO1"), ("diethyl ether", "CCOCC"),
    ("acetic anhydride", "CC(=O)OC(C)=O"), ("formamide", "NC=O"),
    ("glycine", "NCC(O)=O"), ("alanine", "CC(N)C(O)=O"),
    ("ascorbic acid", "OCC(O)C1OC(=O)C(O)=C1O"), ("ethylene oxide", "C1CO1"),
    ("propylene oxide", "CC1CO1"), ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(O)=O)cc1"),
    ("adrenaline", "CNCC(O)c1ccc(O)c(O)c1"),
    ("testosterone", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O"),
],
}


# --- 2b. additions filling classes that gate 2 thinned out -----------------
# Every name below is tokenizer-verified at <=3 tokens under the leading-space
# variant. Excluded on purpose: macromolecules (cellulose, starch, collagen,
# insulin) have no single well-defined SMILES, and stereoisomer-only pairs
# (galactose against glucose) would collide to Tanimoto distance 0 because our
# SMILES carry no stereochemistry.
_ADDITIONS = {}

_ADDITIONS["ketone"] = [
    ("pentanone", "CCCC(C)=O"),
    ("hexanone", "CCCCC(C)=O"),
    ("octanone", "CCCCCCC(C)=O"),
    ("nonanone", "CCCCCCCC(C)=O"),
    ("decanone", "CCCCCCCCC(C)=O"),
    ("quinone", "O=C1C=CC(=O)C=C1"),
    ("acetoin", "CC(O)C(C)=O"),
    ("ionone", "CC(=O)/C=C/C1=C(C)CCCC1(C)C"),
    ("progesterone", "CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"),
]

_ADDITIONS["halide"] = [
    ("methylene chloride", "ClCCl"),
    ("bromoform", "BrC(Br)Br"),
    ("iodoform", "IC(I)I"),
    ("halothane", "FC(F)(F)C(Cl)Br"),
]

_ADDITIONS["alkene"] = [
    ("isoprene", "CC(=C)C=C"),
    ("butadiene", "C=CC=C"),
    ("acetylene", "C#C"),
    ("propyne", "C#CC"),
    ("limonene", "CC(=C)C1CCC(C)=CC1"),
    ("pinene", "CC1=CCC2CC1C2(C)C"),
]

_ADDITIONS["amine"] = [
    ("toluidine", "Cc1ccc(N)cc1"),
    ("melamine", "Nc1nc(N)nc(N)n1"),
    ("adenine", "Nc1ncnc2[nH]cnc12"),
    ("guanine", "Nc1nc2[nH]cnc2c(=O)[nH]1"),
    ("cytosine", "Nc1cc[nH]c(=O)n1"),
    ("creatine", "CN(CC(O)=O)C(N)=N"),
    ("choline", "C[N+](C)(C)CCO"),
    ("taurine", "NCCS(O)(=O)=O"),
]

_ADDITIONS["aromatic"] = [
    ("pyrimidine", "c1cncnc1"),
    ("pyrazine", "c1cnccn1"),
    ("purine", "c1ncc2[nH]cnc2n1"),
    ("uracil", "O=c1cc[nH]c(=O)[nH]1"),
    ("thymine", "Cc1c[nH]c(=O)[nH]c1=O"),
    ("thymol", "Cc1ccc(C(C)C)cc1O"),
    ("eugenol", "C=CCc1ccc(O)c(OC)c1"),
    ("niacin", "OC(=O)c1cccnc1"),
    ("melatonin", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1"),
    ("tyrosine", "NC(Cc1ccc(O)cc1)C(O)=O"),
    ("histidine", "NC(Cc1c[nH]cn1)C(O)=O"),
]

_ADDITIONS["other"] = [
    ("valine", "CC(C)C(N)C(O)=O"),
    ("leucine", "CC(C)CC(N)C(O)=O"),
    ("proline", "OC(=O)C1CCCN1"),
    ("serine", "OCC(N)C(O)=O"),
    ("threonine", "CC(O)C(N)C(O)=O"),
    ("cysteine", "SCC(N)C(O)=O"),
    ("methionine", "CSCCC(N)C(O)=O"),
    ("lysine", "NCCCCC(N)C(O)=O"),
    ("arginine", "NC(=N)NCCCC(N)C(O)=O"),
    ("glutamic acid", "OC(=O)CCC(N)C(O)=O"),
    ("fructose", "OCC(O)C(O)C(O)C(=O)CO"),
    ("ribose", "OCC(O)C(O)C(O)C=O"),
    ("estradiol", "CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O"),
]

for _cls, _members in _ADDITIONS.items():
    _CLASSES.setdefault(_cls, []).extend(_members)


def candidate_pool():
    """All candidates as (name, smiles, cls, series, series_index), deduped by name."""
    rows = list(_series_rows())
    seen = {r[0] for r in rows}
    for cls, members in _CLASSES.items():
        for name, smiles in members:
            if name in seen:
                continue
            seen.add(name)
            rows.append((name, smiles, cls, None, None))
    return rows


CLASS_NAMES = sorted(_CLASSES.keys())
