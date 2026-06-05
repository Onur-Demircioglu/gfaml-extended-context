# scratch/test_hidden_equality.py
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

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
full_ids = tokenizer.encode(full, add_special_tokens=False)

ids_16 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
ids_17 = torch.tensor([full_ids], dtype=torch.long, device=device)

# We will run forward passes through hf_model
hf_model.eval()
with torch.no_grad():
    out_16 = hf_model(ids_16, output_hidden_states=True)
    out_17 = hf_model(ids_17, output_hidden_states=True)

h_16 = out_16.hidden_states[-1][0, 15, :]
h_17 = out_17.hidden_states[-1][0, 15, :]

diff = h_17 - h_16
print("h_16 norm:", torch.norm(h_16).item())
print("h_17 norm:", torch.norm(h_17).item())
print("diff norm:", torch.norm(diff).item())
print("Are they identical?", torch.allclose(h_16, h_17, atol=1e-5))
