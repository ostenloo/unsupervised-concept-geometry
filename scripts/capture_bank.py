"""General-instruction activation bank (SPEC §3h whitening, §4d ID reference).

2,000 prompts from the harmless_train split, disjoint from the Stage B primary
set, same layer and same position as everything else.
"""
import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json, numpy as np, pandas as pd, torch
import config
from src import model as M
SB=config.DATA/"stageb"
used=set(pd.read_csv(SB/"prompts.csv").prompt)
rows=[r["instruction"] for r in json.loads((SB/"harmless_train.json").read_text())]
rows=[r for r in dict.fromkeys(rows) if r not in used]
rng=np.random.default_rng(config.SEED)
bank=[rows[i] for i in rng.choice(len(rows),size=2000,replace=False)]
print(f"bank: {len(bank)} prompts, disjoint from the Stage B set")
L=M.load()
acts=M.capture(L,[M.chat(L,p) for p in bank],batch_size=8)
print("captured",tuple(acts.shape),acts.dtype)
torch.save({"acts":acts,"prompts":bank},config.ACTS/"bank.pt")
