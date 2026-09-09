"""§18.5a′ acceptance tests. These gate everything in §18.5.

"An unasserted ablation is not evidence." Three tests, all of which must pass
before any eval arm runs:

1. Null-op: alpha=0 through the same code path reproduces baseline generation
   token for token. Compared generation-to-generation, because §17a established
   that `capture`'s forward and `generate()`'s prompt pass are not numerically
   identical in bf16.
2. Completeness at sub-layer outputs AND at block outputs, over every position
   including generated ones. v2's block-boundary-only assertion would have passed
   while the within-block leak persisted (E6); this checks both.
3. KV-cache consistency: ablated generation with and without the cache must
   agree. Divergence means keys/values were computed pre-ablation, i.e. silent
   partial ablation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import torch

import config
from src import ablation as AB
from src import model as M

PROMPTS = [
    "Give three tips for staying healthy.",
    "Explain what a hash table is.",
    "Write a haiku about rain.",
    "What is the capital of Peru?",
    "Summarise the water cycle in two sentences.",
]
MAX_NEW = 24


def _gen(L, prompts, max_new=MAX_NEW, use_cache=True):
    tok = L.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    out = L.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           use_cache=use_cache, pad_token_id=tok.eos_token_id)
    return out[:, enc["input_ids"].shape[1]:].cpu()


def main():
    torch.manual_seed(config.SEED)
    L = M.load()
    chats = [M.chat(L, p) for p in PROMPTS]
    d = np.load(config.RESULTS / "stageb_18_5_directions.npz")["d400"]
    u = torch.tensor(d, dtype=torch.float32)
    u = u / u.norm()
    results = {}

    print("=== test 1: null-op (alpha=0) reproduces baseline token-for-token ===")
    base = _gen(L, chats)
    with AB.ablate(L, u, alpha=0.0) as st:
        null = _gen(L, chats)
    ok1 = bool(torch.equal(base, null))
    print(f"  writes hooked {st.writes_hooked} (expected {AB.EXPECTED_WRITES_PER_FORWARD}), "
          f"fired {st.fired} over {st.forward_calls} forwards")
    print(f"  identical: {ok1}")
    if not ok1:
        n = int((base != null).sum())
        print(f"  MISMATCH in {n} of {base.numel()} generated tokens")
    results["test1_nullop_identical"] = ok1
    results["writes_hooked"] = st.writes_hooked
    results["fired_per_forward"] = st.fired / max(st.forward_calls, 1)

    print("\n=== test 2: completeness at sub-layer AND block outputs, all positions ===")
    # §18.18: relative criteria. An absolute epsilon measures bf16's storage
    # rounding, not the code. The unablated pass supplies the scale.
    with AB.ablate(L, u, alpha=0.0, monitor=True) as st0:
        _ = _gen(L, chats)
    with AB.ablate(L, u, alpha=1.0, monitor=True) as st:
        _ = _gen(L, chats)
    dt_eps = float(torch.finfo(next(L.model.parameters()).dtype).eps)
    frac_writes = st.max_leak_at_writes / st0.max_leak_at_writes
    bound = float(np.sqrt(AB.EXPECTED_WRITES_PER_FORWARD)) * dt_eps * st0.max_leak_at_blocks
    ok2 = frac_writes <= 0.01 and st.max_leak_at_blocks <= bound
    print(f"  forwards {st.forward_calls} (1 prompt pass + {MAX_NEW - 1} decode steps)")
    print(f"  unablated max |u.x|: writes {st0.max_leak_at_writes:.3f} "
          f"blocks {st0.max_leak_at_blocks:.3f}")
    print(f"  ablated   max |u.x|: writes {st.max_leak_at_writes:.4f} "
          f"blocks {st.max_leak_at_blocks:.4f}")
    print(f"  (1) removed at writes {1 - frac_writes:.4%} (need >= 99%): "
          f"{frac_writes <= 0.01}")
    print(f"  (2) block leak {st.max_leak_at_blocks:.4f} <= sqrt(65)*eps*scale "
          f"= {bound:.4f}: {st.max_leak_at_blocks <= bound}")
    results.update(test2_leak_writes=st.max_leak_at_writes,
                   test2_leak_blocks=st.max_leak_at_blocks,
                   test2_unablated_writes=st0.max_leak_at_writes,
                   test2_unablated_blocks=st0.max_leak_at_blocks,
                   test2_fraction_removed_writes=float(1 - frac_writes),
                   test2_rounding_bound=bound, test2_pass=bool(ok2))

    print("\n=== test 3: KV-cache consistency under ablation ===")
    with AB.ablate(L, u, alpha=1.0):
        with_cache = _gen(L, chats, use_cache=True)
    with AB.ablate(L, u, alpha=1.0):
        without_cache = _gen(L, chats, use_cache=False)
    ok3 = bool(torch.equal(with_cache, without_cache))
    print(f"  identical: {ok3}")
    if not ok3:
        n = int((with_cache != without_cache).sum())
        print(f"  {n} of {with_cache.numel()} tokens differ -> disable cache for the eval")
    results["test3_kv_cache_consistent"] = ok3

    print("\n=== hook removal ===")
    after = _gen(L, chats)
    ok4 = bool(torch.equal(base, after))
    print(f"  baseline reproduced after all hook contexts exited: {ok4}")
    results["test4_hooks_removed"] = ok4

    allpass = ok1 and ok2 and ok3 and ok4
    results["all_pass"] = bool(allpass)
    (config.RESULTS / "stageb_18_5_tests.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'ALL TESTS PASS' if allpass else 'FAILURE — §18.5 does not run'}")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
