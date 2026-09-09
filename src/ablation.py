"""Directional ablation at every residual-stream write (SPEC §18.5a′, N6/N8).

`config.py:25` pins the capture hook at `model.layers.{i}` -- block *outputs*.
Ablating only there leaves a within-block leak: attention writes a û-component
that the MLP reads before the next ablation fires. §18.5a′ records why that is
not cosmetic -- the leak biases toward a false null on exactly the arms whose
nulls are pre-registered as supporting the completeness conclusion.

So ablation is applied at the three places the residual stream is actually
written: the embedding output, and every attention and MLP output. Because the
stream is the sum of those writes, making each write û-free makes the stream
û-free -- and `monitor=True` verifies that end-to-end at the block outputs rather
than trusting the argument.

Sub-layer return conventions vary by attention implementation, and `config.py:23`
is stale about the version in use (§18.7: the runs are on transformers 5.16.1,
not 5.12.1), so outputs are resolved explicitly rather than by convention.
"""
from __future__ import annotations

import contextlib

import torch

import config
from src.model import Loaded, _resolve_block_output

# embedding + one attention and one MLP write per layer
EXPECTED_WRITES_PER_FORWARD = 1 + 2 * config.N_LAYERS


def _rebuild(out, new: torch.Tensor):
    if isinstance(out, tuple):
        return (new,) + tuple(out[1:])
    if isinstance(out, list):
        return [new] + list(out[1:])
    return new


def _resolve_any(out) -> torch.Tensor:
    """Like `_resolve_block_output` but tolerant of the embedding's bare tensor."""
    t = out[0] if isinstance(out, (tuple, list)) else out
    if not isinstance(t, torch.Tensor) or t.ndim != 3:
        raise TypeError(f"unresolvable residual-stream write: {type(out)}")
    return t


def _writes(L: Loaded):
    base = L.model.model
    yield base.embed_tokens
    for blk in base.layers:
        yield blk.self_attn
        yield blk.mlp


class AblationState:
    def __init__(self):
        self.writes_hooked = 0
        self.forward_calls = 0     # counted at the embedding, once per forward
        self.fired = 0
        self.max_leak_at_writes = 0.0
        self.max_leak_at_blocks = 0.0   # the end-to-end check (§18.5a′ test 2)

    def __repr__(self):
        return (f"<Ablation fired={self.fired} forwards={self.forward_calls} "
                f"leak_writes={self.max_leak_at_writes:.2e} "
                f"leak_blocks={self.max_leak_at_blocks:.2e}>")


@contextlib.contextmanager
def ablate(L: Loaded, u: torch.Tensor, alpha: float = 1.0, monitor: bool = False):
    """`x <- x - alpha * (x·û) û` at every residual-stream write, every position.

    alpha=0.0 is the null-op used by §18.5a′ test 1: the arithmetic still runs, so
    a pass through this code path with alpha=0 must reproduce baseline generation
    token for token.

    Hooks are removed on exit even if the body raises; a leaked hook silently
    contaminates every later forward in the process (§17a).
    """
    if abs(float(u.norm()) - 1.0) > 1e-4:
        raise ValueError(f"direction must be unit norm, got {float(u.norm()):.6f}")

    st = AblationState()
    handles = []
    cache: dict[tuple, torch.Tensor] = {}

    def _u_like(t):
        key = (t.device, t.dtype)
        if key not in cache:
            cache[key] = u.to(device=t.device, dtype=t.dtype)
        return cache[key]

    def write_hook(is_embed):
        def hook(_m, _i, out):
            t = _resolve_any(out)
            uu = _u_like(t)
            st.fired += 1
            if is_embed:
                st.forward_calls += 1
            new = t - alpha * (t @ uu).unsqueeze(-1) * uu
            if monitor:
                st.max_leak_at_writes = max(st.max_leak_at_writes,
                                            float((new @ uu).abs().max()))
            return _rebuild(out, new)
        return hook

    def block_monitor(_m, _i, out):
        t = _resolve_block_output(out)
        st.max_leak_at_blocks = max(st.max_leak_at_blocks,
                                    float((t @ _u_like(t)).abs().max()))

    try:
        for i, mod in enumerate(_writes(L)):
            handles.append(mod.register_forward_hook(write_hook(i == 0)))
            st.writes_hooked += 1
        if st.writes_hooked != EXPECTED_WRITES_PER_FORWARD:
            raise RuntimeError(f"hooked {st.writes_hooked} writes, "
                               f"expected {EXPECTED_WRITES_PER_FORWARD}")
        if monitor:
            for blk in L.blocks:
                handles.append(blk.register_forward_hook(block_monitor))
        yield st
    finally:
        for h in handles:
            h.remove()
        handles.clear()


@contextlib.contextmanager
def add_direction(L: Loaded, u: torch.Tensor, alpha: float, layers):
    """`x <- x + alpha*û` at the block output of each layer in `layers`.

    §18.22: `layers` is REQUIRED and has no default. The first version of this
    applied the addition at all 65 residual-stream writes, by analogy with
    `ablate`. That analogy is wrong: removing a component at every write is what
    makes the stream û-free, but *adding* at every write accumulates, so a
    nominal dose of alpha lands as ~65*alpha and destroyed the model at every
    dose tested. Steering along a direction means adding at a site.
    """
    if isinstance(layers, int):
        layers = [layers]
    st = AblationState()
    handles = []
    cache: dict[tuple, torch.Tensor] = {}

    def hook(_m, _i, out):
        t = _resolve_block_output(out)
        key = (t.device, t.dtype)
        if key not in cache:
            cache[key] = u.to(device=t.device, dtype=t.dtype)
        st.fired += 1
        return _rebuild(out, t + alpha * cache[key])

    try:
        for i in layers:
            handles.append(L.blocks[i].register_forward_hook(hook))
            st.writes_hooked += 1
        yield st
    finally:
        for h in handles:
            h.remove()
        handles.clear()
