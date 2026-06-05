# scratch/test_hidden_states_change.py
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

prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer
ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)

query_ids = tokenizer.encode(prompt, add_special_tokens=False)
query_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

# 1. Get hidden states for alpha = 0.0
config0 = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=0.0)
model0 = DecoupledGFAML(hf_model, tokenizer, config0, device=device)
model0.eval()
model0._record(ids)
model0.gfaml_mode = "augment"
with torch.no_grad():
    out0 = model0.hf_model(query_tensor, output_hidden_states=True)
model0.gfaml_mode = "off"
h0 = out0.hidden_states[-1]

# 2. Get hidden states for alpha = 0.01
config1 = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=0.01)
model1 = DecoupledGFAML(hf_model, tokenizer, config1, device=device)
model1.eval()
model1._record(ids)
model1.gfaml_mode = "augment"
with torch.no_grad():
    out1 = model1.hf_model(query_tensor, output_hidden_states=True)
model1.gfaml_mode = "off"
h1 = out1.hidden_states[-1]

# Compare
diff = h1 - h0
diff_norm = torch.norm(diff).item()
h0_norm = torch.norm(h0).item()
h1_norm = torch.norm(h1).item()

print(f"h0 norm: {h0_norm:.4f}")
print(f"h1 norm: {h1_norm:.4f}")
print(f"diff norm: {diff_norm:.4f}")
print(f"Relative difference: {diff_norm / h0_norm:.6f}")
