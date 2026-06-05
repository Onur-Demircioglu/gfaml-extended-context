# scratch/test_chat_complete_flow.py
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

config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=0.1)
model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
model.eval()

# 1. Ask before teach
print("--- Ask before teach ---")
prompt = "Onur'un en sevdiği programlama dili hangisidir?"
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
with torch.no_grad():
    logits = model._forward_augmented(ids)
pred_id = torch.argmax(logits[0, -1, :]).item()
print("Prediction before teach:", repr(tokenizer.decode([pred_id])))

# 2. Teach
print("\n--- Teaching: Onur'un en sevdiği programlama dili hangisidir? = Python ---")
answer = " Python"
full = prompt + answer
ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)

# Record
model._record(ids)

# Train cortex (LoRA)
model.train()
plen = len(tokenizer.encode(prompt, add_special_tokens=False))
for step in range(15):
    model.online_opt.zero_grad()
    logits, _ = model._forward_clean(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
        ids[:, plen:].reshape(-1)
    )
    loss.backward()
    model.online_opt.step()
model.eval()
print("Taught successfully.")

# 3. Ask after teach
print("\n--- Ask after teach ---")
ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
with torch.no_grad():
    logits = model._forward_augmented(ids)

pred_id = torch.argmax(logits[0, -1, :]).item()
generated = tokenizer.decode([pred_id])
curr_ids = torch.cat([ids, torch.tensor([[pred_id]], device=device)], dim=1)

for _ in range(5):
    with torch.no_grad():
        logits = model._forward_augmented(curr_ids)
    next_id = torch.argmax(logits[0, -1, :]).item()
    if next_id == tokenizer.eos_token_id:
        break
    generated += tokenizer.decode([next_id])
    curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]], device=device)], dim=1)

print("Generated response:", repr(generated.strip()))
