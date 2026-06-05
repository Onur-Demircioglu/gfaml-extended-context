# scratch/test_lora_learning.py
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

# Let's inspect the target token
prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer

ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
plen = len(tokenizer.encode(prompt, add_special_tokens=False))
target_ids = ids[:, plen:]

print("Prompt token IDs:", ids[0, :plen].tolist())
print("Target token IDs:", target_ids[0].tolist())

# Record (Hebbian write)
model.eval()
model._record(ids)

# Train cortex (LoRA)
model.train()
print("\nTraining LoRA cortex...")
for step in range(50): # try 50 steps instead of 10
    model.online_opt.zero_grad()
    logits, _ = model._forward_clean(ids)
    
    # Calculate loss
    pred_logits = logits[:, plen-1:-1, :]
    loss = torch.nn.functional.cross_entropy(
        pred_logits.reshape(-1, logits.size(-1)),
        ids[:, plen:].reshape(-1)
    )
    loss.backward()
    model.online_opt.step()
    
    if step % 5 == 0 or step == 49:
        # Check target logit value vs max logit value
        with torch.no_grad():
            clean_logits, _ = model._forward_clean(ids[:, :plen])
            top_vals, top_idxs = torch.topk(clean_logits[0, -1, :], 3)
            target_logit = clean_logits[0, -1, target_ids[0, 0]].item()
            print(f"Step {step:02d} | Loss: {loss.item():.4f} | Target Logit: {target_logit:.4f} | Top: {[tokenizer.decode([i]) for i in top_idxs.tolist()]} (values: {top_vals.tolist()})")
