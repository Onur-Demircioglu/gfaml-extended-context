# scripts/test_instant_learn.py
"""End-to-end test: Gradient-free instant learning via Hebbian boundary write."""
import sys, os, torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from gfaml.config import GFAMLConfig
from experiments.exp_12_scaled_benchmark import DecoupledGFAML
from scripts.chat_gfaml import generate_augmented, wrap_prompt

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
config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=0.3)

model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
model.eval()
print("Model loaded.\n")

# === INSTANT TEACH (no gradient!) ===
facts = [
    ("Onur's favorite programming language is", " Python"),
    ("The capital of Turkey is", " Ankara"),
    ("The best pizza topping is", " pepperoni"),
    ("GFAML stands for Gradient-Free Associative", " Memory"),
]

print("=== TEACHING (Gradient-Free, Instant) ===\n")
import time
for prompt, answer in facts:
    full = prompt + answer
    ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
    plen = len(tokenizer.encode(prompt, add_special_tokens=False))
    
    t0 = time.time()
    model._record(ids, prompt_len=plen)
    dt = (time.time() - t0) * 1000
    print(f"  Taught: '{prompt} ={answer}' ({dt:.0f}ms)")

# === QUERY ===
print("\n=== RECALL TEST ===\n")
for prompt, answer in facts:
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model._forward_augmented(ids)
    
    pred_id = torch.argmax(logits[0, -1, :]).item()
    expected_id = tokenizer.encode(answer, add_special_tokens=False)[0]
    match = pred_id == expected_id
    
    pred_tok = tokenizer.decode([pred_id]).encode('ascii', errors='replace').decode()
    exp_tok = tokenizer.decode([expected_id]).encode('ascii', errors='replace').decode()
    print(f"  Q: '{prompt}'")
    print(f"     Expected: '{exp_tok}' | Got: '{pred_tok}' [{'OK' if match else 'MISS'}]")

# === GENERATION TEST ===
print("\n=== GENERATION TEST (Full Sentences) ===\n")
gen_queries = [
    "Onur's favorite programming language is",
    "The capital of Turkey is",
    "The capital of France is",  # base knowledge
    "What is 2+2?",             # base knowledge
]

for q in gen_queries:
    formatted = wrap_prompt(q)
    raw = generate_augmented(model, tokenizer, formatted, device, max_new_tokens=40)
    response = raw.split('\n')[0].strip() if raw else '(empty)'
    for prefix in ['User:', 'Assistant:', 'Q:', 'A:']:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    print(f"  Q: {q}")
    print(f"  A: {response}\n")
