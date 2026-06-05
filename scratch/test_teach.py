# scratch/test_teach.py
import sys
import os
import torch

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from gfaml.config import GFAMLConfig
from experiments.exp_12_scaled_benchmark import DecoupledGFAML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

d_model = hf_model.config.hidden_size
vocab_size = hf_model.config.vocab_size
config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0)
model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
model.eval()

# Teach
prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer
ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)

print("Recording to GFAML memory...")
model._record(ids)

# Train cortex (LoRA) for 30 steps
model.train()
plen = len(tokenizer.encode(prompt, add_special_tokens=False))
for step in range(30):
    model.online_opt.zero_grad()
    logits, _ = model._forward_clean(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
        ids[:, plen:].reshape(-1)
    )
    loss.backward()
    model.online_opt.step()
model.eval()
print("LoRA trained.")

# Query
query_ids = tokenizer.encode(prompt, add_special_tokens=False)
ids = torch.tensor([query_ids], dtype=torch.long, device=device)

# Clean predictions
with torch.no_grad():
    clean_logits, _ = model._forward_clean(ids)
clean_top_vals, clean_top_idxs = torch.topk(clean_logits[0, -1, :], 5)
print("\nClean Predictions (GFAML Off):")
for idx, val in zip(clean_top_idxs.tolist(), clean_top_vals.tolist()):
    print(f"Token: {idx} ({repr(tokenizer.decode([idx]))}) | Logit: {val:.4f}")

# Augmented predictions
with torch.no_grad():
    aug_logits = model._forward_augmented(ids)
aug_top_vals, aug_top_idxs = torch.topk(aug_logits[0, -1, :], 5)
print("\nAugmented Predictions (GFAML Active):")
for idx, val in zip(aug_top_idxs.tolist(), aug_top_vals.tolist()):
    print(f"Token: {idx} ({repr(tokenizer.decode([idx]))}) | Logit: {val:.4f}")
