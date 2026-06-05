# scripts/test_pure_memory.py
"""Test: Hetero-associative memory with SELECTIVE WRITE (only boundary tokens)."""
import sys, os, torch, copy

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

facts = [
    ("Onur's favorite programming language is", " Python"),
    ("The capital of Turkey is", " Ankara"),
    ("The color of the sky is", " blue"),
]

base_knowledge = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
]

alpha_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

for alpha in alpha_values:
    print(f"\n{'='*60}")
    print(f"  ALPHA = {alpha} (Selective Boundary Write)")
    print(f"{'='*60}")
    
    hf_copy = copy.deepcopy(hf_model)
    config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=alpha)
    model = DecoupledGFAML(hf_copy, tokenizer, config, device=device)
    model.eval()
    
    # Record ONLY at prompt-answer boundary
    for prompt, answer in facts:
        full = prompt + answer
        ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
        plen = len(tokenizer.encode(prompt, add_special_tokens=False))
        model._record(ids, prompt_len=plen)  # <-- selective write!
    
    print(f"\n  Taught facts (Hebbian only, boundary write, no LoRA):")
    taught_correct = 0
    for prompt, answer in facts:
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_aug = model._forward_augmented(ids)
        
        pred_id = torch.argmax(logits_aug[0, -1, :]).item()
        expected_id = tokenizer.encode(answer, add_special_tokens=False)[0]
        match = pred_id == expected_id
        if match:
            taught_correct += 1
        
        pred_tok = tokenizer.decode([pred_id]).encode('ascii', errors='replace').decode()
        exp_tok = tokenizer.decode([expected_id]).encode('ascii', errors='replace').decode()
        print(f"    Q: '{prompt}' -> '{pred_tok}' (expected '{exp_tok}') [{'OK' if match else 'MISS'}]")
    
    print(f"\n  Base model knowledge (should be PRESERVED):")
    base_correct = 0
    for prompt, answer in base_knowledge:
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_aug = model._forward_augmented(ids)
        
        pred_id = torch.argmax(logits_aug[0, -1, :]).item()
        expected_id = tokenizer.encode(answer, add_special_tokens=False)[0]
        match = pred_id == expected_id
        if match:
            base_correct += 1
        
        pred_tok = tokenizer.decode([pred_id]).encode('ascii', errors='replace').decode()
        exp_tok = tokenizer.decode([expected_id]).encode('ascii', errors='replace').decode()
        print(f"    Q: '{prompt}' -> '{pred_tok}' (expected '{exp_tok}') [{'OK' if match else 'MISS'}]")
    
    print(f"\n  SCORE: Taught={taught_correct}/{len(facts)} | Base={base_correct}/{len(base_knowledge)}")
    del model, hf_copy
