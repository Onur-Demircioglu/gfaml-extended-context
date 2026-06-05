# scratch/test_grad.py
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

config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=0.0)
model = DecoupledGFAML(hf_model, tokenizer, config, device=device)

prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer
ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
plen = len(tokenizer.encode(prompt, add_special_tokens=False))

model.train()
model.online_opt.zero_grad()
logits, _ = model._forward_clean(ids)

# Check if logits requires grad
print("Logits requires_grad:", logits.requires_grad)

loss = torch.nn.functional.cross_entropy(
    logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
    ids[:, plen:].reshape(-1)
)
print("Loss requires_grad:", loss.requires_grad)

loss.backward()

print("Cortex A grad:", model.cortex.A.grad)
if model.cortex.A.grad is not None:
    print("Cortex A grad sum:", model.cortex.A.grad.sum().item())
