# scratch/test_qwen.py
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

# Ask a question before teach
user_input = "Onur'un en sevdiği programlama dili hangisidir?"
prompt_ids = tokenizer.encode(user_input, add_special_tokens=False)
ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

print("Prompt IDs:", prompt_ids)
print("Decoded prompt tokens:", [tokenizer.decode([i]) for i in prompt_ids])

with torch.no_grad():
    logits = model._forward_augmented(ids)
    
pred_ids = torch.topk(logits[0, -1, :], 5)
print("Top 5 predictions before teach:")
for idx, val in zip(pred_ids.indices.tolist(), pred_ids.values.tolist()):
    print(f"Token: {idx} ({repr(tokenizer.decode([idx]))}), Logit: {val}")

# Try normal generation without GFAML wrapper (clean)
with torch.no_grad():
    clean_logits, _ = model._forward_clean(ids)
clean_pred_ids = torch.topk(clean_logits[0, -1, :], 5)
print("\nTop 5 clean predictions (GFAML off):")
for idx, val in zip(clean_pred_ids.indices.tolist(), clean_pred_ids.values.tolist()):
    print(f"Token: {idx} ({repr(tokenizer.decode([idx]))}), Logit: {val}")
