# scratch/test_cortex_logits.py
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
plen = len(tokenizer.encode(prompt, add_special_tokens=False))
target_token_id = ids[0, plen].item()

query_ids = tokenizer.encode(prompt, add_special_tokens=False)
query_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

for alpha_val in [0.0, 0.01]:
    config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=alpha_val)
    model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
    model.eval()
    
    # Record
    model._record(ids)
    
    # Train cortex (LoRA) for 20 steps
    model.train()
    for _ in range(20):
        model.online_opt.zero_grad()
        logits, _ = model._forward_clean(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
            ids[:, plen:].reshape(-1)
        )
        loss.backward()
        model.online_opt.step()
    model.eval()
    
    # Custom augmented pass to print base and cortex logits
    model.gfaml_mode = "augment"
    with torch.no_grad():
        out = model.hf_model(query_tensor, output_hidden_states=True)
    model.gfaml_mode = "off"
    
    base_logits = out.logits
    cortex_logits = model.cortex(out.hidden_states[-1])
    
    base_target = base_logits[0, -1, target_token_id].item()
    cortex_target = cortex_logits[0, -1, target_token_id].item()
    total_target = base_target + cortex_target
    
    print(f"\nAlpha: {alpha_val:.2f}")
    print(f"  Base Logits (' Python'): {base_target:.4f}")
    print(f"  Cortex Logits (' Python'): {cortex_target:.4f}")
    print(f"  Total Logits (' Python'): {total_target:.4f}")
    
    # Check max logits of cortex
    cortex_top_val, cortex_top_idx = torch.max(cortex_logits[0, -1, :], dim=-1)
    print(f"  Cortex Top predicted: {repr(tokenizer.decode([cortex_top_idx.item()]))} (logit: {cortex_top_val.item():.4f})")
