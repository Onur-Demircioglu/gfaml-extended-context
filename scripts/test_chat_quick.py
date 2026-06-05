# scripts/test_chat_quick.py
"""Quick non-interactive test of the chat prompt template."""
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
config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=0.1)

model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
model.eval()
print("Model loaded.\n")

# Test conversations
test_inputs = [
    "merhaba",
    "naber",
    "What is the capital of France?",
    "Tell me a joke",
    "2+2 equals?",
]

for user_text in test_inputs:
    formatted = wrap_prompt(user_text)
    print(f"--- Prompt template ---")
    print(formatted)
    print(f"--- End template ---\n")
    
    raw = generate_augmented(model, tokenizer, formatted, device, max_new_tokens=60)
    
    # Clean up
    response = raw.split('\n')[0].strip() if raw else '(empty)'
    for prefix in ['User:', 'Assistant:', 'Q:', 'A:']:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    print(f"User: {user_text}")
    print(f"GFAML: {response}")
    print()
