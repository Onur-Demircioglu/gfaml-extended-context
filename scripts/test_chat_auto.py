# scripts/test_chat_auto.py
"""Non-interactive test: teach facts with replay, then query them."""
import sys, os, torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from gfaml.config import GFAMLConfig
from experiments.exp_12_scaled_benchmark import DecoupledGFAML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_model.eval()

d_model = hf_model.config.hidden_size
vocab_size = hf_model.config.vocab_size
config = GFAMLConfig(
    vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0,
    alpha=0.1,  # Conservative memory contribution
)

model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
model.eval()
print("Model loaded.\n")

# === TEACH with REPLAY ===
facts = [
    ("Onur's favorite programming language is", " Python"),
    ("The capital of Turkey is", " Ankara"),
    ("The color of the sky is", " blue"),
]

taught = []
for prompt, answer in facts:
    full = prompt + answer
    ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
    model._record(ids)
    
    model.train()
    plen = len(tokenizer.encode(prompt, add_special_tokens=False))
    for epoch in range(10):
        # Train new fact
        model.online_opt.zero_grad()
        logits, _ = model._forward_clean(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
            ids[:, plen:].reshape(-1)
        )
        loss.backward()
        model.online_opt.step()
        
        # Replay past facts
        for past_p, past_a in taught:
            past_ids = tokenizer.encode(past_p + past_a, return_tensors="pt", add_special_tokens=False).to(device)
            past_plen = len(tokenizer.encode(past_p, add_special_tokens=False))
            if past_plen >= past_ids.size(1):
                continue
            model.online_opt.zero_grad()
            past_logits, _ = model._forward_clean(past_ids)
            past_loss = torch.nn.functional.cross_entropy(
                past_logits[:, past_plen-1:-1, :].reshape(-1, past_logits.size(-1)),
                past_ids[:, past_plen:].reshape(-1)
            )
            past_loss.backward()
            model.online_opt.step()
    
    taught.append((prompt, answer))
    model.eval()
    print(f"Taught: '{prompt} ={answer}'")

print("\n=== EVALUATION ===\n")

# Query each fact
for prompt, expected_answer in facts:
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_clean, _ = model._forward_clean(ids)
        logits_aug = model._forward_augmented(ids)
    
    clean_pred = torch.argmax(logits_clean[0, -1, :]).item()
    aug_pred = torch.argmax(logits_aug[0, -1, :]).item()
    expected_id = tokenizer.encode(expected_answer, add_special_tokens=False)[0]
    
    clean_tok = tokenizer.decode([clean_pred]).encode('ascii', errors='replace').decode()
    aug_tok = tokenizer.decode([aug_pred]).encode('ascii', errors='replace').decode()
    exp_tok = tokenizer.decode([expected_id]).encode('ascii', errors='replace').decode()
    
    clean_match = "OK" if clean_pred == expected_id else "MISS"
    aug_match = "OK" if aug_pred == expected_id else "MISS"
    
    print(f"Query: '{prompt}'")
    print(f"  Expected: '{exp_tok}' (id={expected_id})")
    print(f"  Clean:    '{clean_tok}' (id={clean_pred}) [{clean_match}]")
    print(f"  Augment:  '{aug_tok}' (id={aug_pred}) [{aug_match}]")
    print()

# Also test base model knowledge (not taught)
print("=== BASE MODEL KNOWLEDGE (not taught) ===\n")
base_queries = [
    "The capital of France is",
    "Water boils at 100 degrees",
]
for q in base_queries:
    ids = torch.tensor([tokenizer.encode(q, add_special_tokens=False)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_aug = model._forward_augmented(ids)
    pred = torch.argmax(logits_aug[0, -1, :]).item()
    tok = tokenizer.decode([pred]).encode('ascii', errors='replace').decode()
    print(f"Q: '{q}' -> '{tok}' (id={pred})")
