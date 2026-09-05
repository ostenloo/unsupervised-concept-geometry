"""SPEC §6 acceptance tests 4, 8, 9 — the ones needing the model.

Test 9 is the one that matters most and had never been run: it checks that the
capture/replace path is faithful. If replacing an activation with the value that
was captured there does NOT reproduce the unhooked generation token-for-token,
then the hook machinery is corrupting the forward pass and every Level 3 (§13)
and §4f number computed through it is suspect.

    python -m pytest tests/test_acceptance_gpu.py -q -s
"""
import json

import numpy as np
import pytest
import torch

import config
from src import model as M


@pytest.fixture(scope="module")
def L():
    return M.load()


# --- test 4: weekday loop (already run by scripts/smoke_weekday.py) --------

def test_4_weekday_loop_recorded():
    p = config.RESULTS / "smoke_weekday.json"
    assert p.exists(), "run scripts/smoke_weekday.py"
    r = json.loads(p.read_text())
    assert r[str(config.WURGAFT_LAYER)]["cyclic_order"] is True


# --- test 8: gate removal counts logged -----------------------------------

def test_8_gate_counts_logged():
    g3 = json.loads((config.DATA / "gate3_log.json").read_text())
    assert {"n_in", "n_out", "both_pass_rate"} <= set(g3)
    assert 0 < g3["n_out"] <= g3["n_in"]


# --- test 9: the replace path is faithful, and hooks do not leak -----------

def test_9_replacement_reproduces_base_and_hooks_are_clean(L):
    """Faithfulness of the capture/replace path, plus hook hygiene.

    Two constraints this test had to be written around, both measured:

    1. The hook must be GATED to the prompt forward pass. Under generate(), an
       ungated `t[:, -1, :] = rep` overwrites every newly generated position too
       and pins the model to one token (" propane propane propane ...").
       Production steering runs a single forward and asserts the hook fires
       exactly once, so it is not exposed; any reuse under generation must gate.
    2. The replacement value must come from the SAME forward path it is injected
       into. `M.capture` runs a plain forward while generate() runs a
       cache-enabled prompt pass, and in bf16 the two differ by up to 0.125
       (mean 0.0046, ~3x bf16 epsilon at this magnitude). Injecting a value
       captured through the other path perturbs generation enough to flip a token
       ~9 steps in. This does not affect Level 3 or §4f, which do one forward and
       read logits immediately with no autoregressive compounding -- but it means
       "alpha = 0 reproduces base" is only exactly true within one path.
    """
    prompts = [
        "The straight-chain alkane with 3 carbon atoms is called",
        "The chemical compound ethanol is",
        M.chat(L, "Describe the water cycle briefly."),
    ]
    layer = config.WURGAFT_LAYER
    tok = L.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    prompt_len = enc["input_ids"].shape[1]

    def run(hook=None, n_new=16):
        h = L.blocks[layer].register_forward_hook(hook) if hook else None
        try:
            with torch.no_grad():
                g = L.model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.eos_token_id)
        finally:
            if h is not None:
                h.remove()
        return [tok.decode(g[b, prompt_len:], skip_special_tokens=True)
                for b in range(len(prompts))]

    base = run()
    fired = {"n": 0}
    store = []

    def read_hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if t.shape[1] == prompt_len:
            store.append(t[:, -1, :].detach().float().cpu().numpy())
        return out

    run(read_hook, n_new=1)
    assert store, "read hook never fired at the prompt position"

    # --- alpha = 0: write back the captured value, through float32 -----------
    rep = torch.as_tensor(store[0], dtype=torch.bfloat16, device="cuda")

    def replace_hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if t.shape[1] == prompt_len:
            t[:, -1, :] = rep
            fired["n"] += 1
        return (t,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else t

    steered = run(replace_hook)
    assert fired["n"] == 1, f"hook fired {fired['n']} times, expected 1"
    assert len(L.blocks[layer]._forward_hooks) == 0, "hook leaked after removal"
    for p, b, s_ in zip(prompts, base, steered):
        assert b == s_, (f"alpha=0 replacement did not reproduce base\n"
                         f"  prompt : {p[:60]!r}\n  base   : {b!r}\n  steered: {s_!r}")

    # --- control: a DIFFERENT activation must change the output --------------
    rep2 = torch.as_tensor(np.roll(store[0], 1, axis=0),
                           dtype=torch.bfloat16, device="cuda")

    def replace_hook2(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if t.shape[1] == prompt_len:
            t[:, -1, :] = rep2
        return (t,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else t

    rolled = run(replace_hook2)
    assert rolled != base, "hook has no effect even with a different activation"
    assert len(L.blocks[layer]._forward_hooks) == 0


def test_9b_ungated_hook_is_the_known_hazard(L):
    """Documents the failure mode test 9 was written around, so it stays known."""
    tok = L.tokenizer
    layer = config.WURGAFT_LAYER
    prompts = ["The straight-chain alkane with 3 carbon atoms is called"]
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    pl = enc["input_ids"].shape[1]
    store = []

    def read_hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if t.shape[1] == pl:
            store.append(t[:, -1, :].detach().clone())
        return out

    h = L.blocks[layer].register_forward_hook(read_hook)
    try:
        with torch.no_grad():
            L.model(**enc)
    finally:
        h.remove()

    def ungated(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        t[:, -1, :] = store[0]          # NO length gate -- the hazard
        return (t,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else t

    h = L.blocks[layer].register_forward_hook(ungated)
    try:
        with torch.no_grad():
            g = L.model.generate(**enc, max_new_tokens=12, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    out = tok.decode(g[0, pl:], skip_special_tokens=True)
    words = out.split()
    # The hazard's signature: the same token repeated.
    assert len(set(words)) <= 2, f"expected degenerate repetition, got {out!r}"
