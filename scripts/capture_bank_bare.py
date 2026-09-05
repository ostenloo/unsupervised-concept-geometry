"""Bare-format activation bank for §3h Stage A whitening.

The Stage B bank is chat-formatted, but Family 1 prompts are bare completions
("The chemical compound X is"). Whitening Stage A with a chat-formatted bank
would fold prompt FORMAT into the covariance being removed, which is a different
transform from the one §3h specifies.
"""
import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json, numpy as np, pandas as pd, torch
import config
from src import model as M
SB=config.DATA/"stageb"
rows=[r["instruction"] for r in json.loads((SB/"harmless_train.json").read_text())]
rows=list(dict.fromkeys(rows))
rng=np.random.default_rng(config.SEED)
bank=[rows[i] for i in rng.choice(len(rows),size=2000,replace=False)]
L=M.load()
acts=M.capture(L,bank,batch_size=16)   # NO chat template -- bare, like Family 1
print("bare bank captured",tuple(acts.shape),acts.dtype)
torch.save({"acts":acts},config.ACTS/"bank_bare.pt")
