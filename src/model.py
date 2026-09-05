"""Model loading, hooked activation capture, and greedy generation.

transformers on this box is 5.16.1. The decoder-block output convention differs
between the v4 and v5 lines, so nothing here trusts either: the hook resolves
the tensor explicitly and asserts its shape. Every captured activation is cast
to float32 on the way out (SPEC §2; acceptance test 6).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

import config


@dataclass
class Loaded:
    model: object
    tokenizer: object

    @property
    def blocks(self):
        """The 32 decoder blocks. Hook point per SPEC §2."""
        return self.model.model.layers


def load(device: str = "cuda", dtype=torch.bfloat16) -> Loaded:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, revision=config.MODEL_REVISION)
    if tok.pad_token is None:
        # Llama-3.1 ships no pad token. eos is the conventional stand-in; with
        # left padding and the attention mask, no pad position is ever read.
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION,
        dtype=dtype, device_map=device,
    )
    model.eval()
    n = len(model.model.layers)
    if n != config.N_LAYERS:
        raise RuntimeError(f"expected {config.N_LAYERS} decoder blocks, found {n}")
    return Loaded(model=model, tokenizer=tok)


def _resolve_block_output(out) -> torch.Tensor:
    """Decoder blocks return a Tensor in some versions and a tuple in others."""
    t = out[0] if isinstance(out, (tuple, list)) else out
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"could not resolve block output to a tensor: {type(out)}")
    if t.ndim != 3:
        raise ValueError(f"expected [batch, seq, hidden] from a decoder block, got {tuple(t.shape)}")
    return t


class HookCounter:
    """Acceptance test 9 requires that hooks fire and are removed after every run."""

    def __init__(self):
        self.fired = 0
        self.registered = 0


@contextlib.contextmanager
def capture_all_layers(L: Loaded, counter: HookCounter | None = None):
    """Context manager yielding a dict {layer: Tensor[batch, seq, hidden]}.

    Hooks are removed on exit even if the body raises -- a leaked hook silently
    contaminates every later forward pass in the process.
    """
    counter = counter or HookCounter()
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(i):
        def hook(_module, _inp, out):
            store[i] = _resolve_block_output(out).detach()
            counter.fired += 1
        return hook

    try:
        for i, block in enumerate(L.blocks):
            handles.append(block.register_forward_hook(make(i)))
            counter.registered += 1
        yield store, counter
    finally:
        for h in handles:
            h.remove()
        handles.clear()


@torch.no_grad()
def capture(L: Loaded, prompts: list[str], positions: list[int] | None = None,
            batch_size: int = 16, device: str = "cuda") -> torch.Tensor:
    """Residual-stream activations at one position per prompt, at every layer.

    positions: index into each prompt's token sequence. None means the last
    prompt token. Negative indices count from the end.

    Returns float32 [n_prompts, n_layers, hidden]. The float32 cast happens here,
    once, so nothing downstream ever sees bf16 (SPEC §2).
    """
    from src.geometry import assert_geometry_dtype

    tok = L.tokenizer
    out = torch.empty(len(prompts), config.N_LAYERS, config.HIDDEN_DIM, dtype=torch.float32)

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        # Left padding puts the last real token at index -1 for every row, so a
        # right-padded batch cannot silently hand us a pad-token activation.
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to(device)
        with capture_all_layers(L) as (store, counter):
            L.model(**enc)
            if counter.fired != config.N_LAYERS:
                raise RuntimeError(f"expected {config.N_LAYERS} hook firings, got {counter.fired}")
            for i in range(config.N_LAYERS):
                h = store[i]
                if positions is None:
                    sel = h[:, -1, :]
                else:
                    idx = positions[start:start + batch_size]
                    sel = torch.stack([h[b, idx[b], :] for b in range(h.shape[0])])
                out[start:start + len(chunk), i, :] = sel.to(torch.float32).cpu()
        del store

    assert_geometry_dtype(out, "capture")
    return out


def last_token_index_of(tok, prompt: str, substring: str) -> int:
    """Index of the final token of `substring` inside `prompt`, after left padding.

    Used by Family 1, which captures at the last token of the compound *name*,
    not at the last token of the prompt (SPEC §3d).

    Returned as a negative index (counting from the end of the sequence) so it
    stays valid under left padding, where absolute positions shift per row.
    """
    pos = prompt.rindex(substring)
    end_char = pos + len(substring)
    n_prefix = len(tok.encode(prompt[:end_char], add_special_tokens=False))
    n_total = len(tok.encode(prompt, add_special_tokens=False))
    return n_prefix - n_total - 1 if n_prefix < n_total else -1


@torch.no_grad()
def generate_greedy(L: Loaded, prompts: list[str], max_new_tokens: int = 24,
                    batch_size: int = 16, device: str = "cuda") -> list[str]:
    tok = L.tokenizer
    outs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to(device)
        gen = L.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.eos_token_id)
        for b in range(len(chunk)):
            new = gen[b, enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def chat(L: Loaded, user_msg: str) -> str:
    """Render one user turn through the chat template. Print one and read it (SPEC §4b)."""
    return L.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
    )
