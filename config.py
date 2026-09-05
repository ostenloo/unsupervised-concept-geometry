"""Fixed frame for the whole project (SPEC §2).

Every constant the spec pins lives here. Nothing below is a tuning knob;
changing one invalidates comparisons against runs already logged.
"""
from pathlib import Path

# --- model -----------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Pinned 2026-09-04 from the HF API; `gated: manual`, access confirmed for this token.
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

MODEL_DTYPE = "bfloat16"    # weights
GEOMETRY_DTYPE = "float32"  # every activation is cast on save; see SPEC §2 and
                            # acceptance test 6 -- bf16 ties break every neighbor method.

N_LAYERS = 32
HIDDEN_DIM = 4096

# --- capture ---------------------------------------------------------------
# Hook point: output of decoder block L, i.e. model.model.layers[L].
# transformers on this box is 5.12.1, where block outputs are no longer a bare
# tuple in all paths -- capture.py resolves the tensor explicitly and asserts shape
# rather than trusting either the v4 or v5 convention.
HOOK_MODULE_TEMPLATE = "model.layers.{layer}"

# --- geometry --------------------------------------------------------------
PCA_DIM = 64                 # matches Wurgaft
K_SWEEP = (8, 12, 16, 24)    # Isomap neighbour count; report stability
ID_MLE_K = (10, 20)          # Levina-Bickel

# Fingerprints (SPEC §3c)
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Steering (SPEC §3f Level 3)
N_WAYPOINTS = 50
N_BASE_PROMPTS = 8
MAX_STEER_PAIRS = 20

# Layers reported in Stage B regardless of what wins (SPEC §4d)
WURGAFT_LAYER = 28

# --- reproducibility -------------------------------------------------------
SEED = 0

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ACTS = ROOT / "acts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (DATA, ACTS, RESULTS, FIGURES):
    _p.mkdir(exist_ok=True)

# --- prompt templates ------------------------------------------------------
# SPEC §3d: exactly one Family-1 template in v1. Stated as an arbitrary choice
# in the writeup (SPEC §9 risk 10); a template ablation is future work.
FAMILY1_TEMPLATE = "The chemical compound {name} is"

FAMILY2_TEMPLATES = {
    "alkane":  "The straight-chain alkane with {n} carbon atoms is called",
    "alcohol": "The straight-chain primary alcohol with {n} carbon atoms is called",
}

WEEKDAY_TEMPLATE = "Q: What day is {k} days after {entity}?\nA:"
