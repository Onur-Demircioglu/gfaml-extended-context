# scratch/test_hook_values.py
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
model.eval()

# Teach
prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer
ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)

model._record(ids)

# Let's inspect norms during query
query_ids = tokenizer.encode(prompt, add_special_tokens=False)
ids = torch.tensor([query_ids], dtype=torch.long, device=device)

# We will modify the hook temporarily to print norms
for i, block in enumerate(model.layers):
    attn_module = getattr(block, model.attn_name)
    
    # Remove old hook
    model.hook_handles[i].remove()
    
    def make_debug_hook(layer_idx):
        def debug_hook_fn(module, args, kwargs, hook_output):
            if model.gfaml_mode != "augment":
                return hook_output
            
            x_norm = args[0] if len(args) > 0 else kwargs["hidden_states"]
            state = model.state_list[layer_idx]
            
            gfaml_out, _, _ = model.gfaml_layers[layer_idx](
                x_norm, state=state, write=False
            )
            
            # Print norms of x_norm, hook_output, and gfaml_out
            x_norm_norm = torch.norm(x_norm).item()
            hook_out_norm = torch.norm(hook_output[0] if isinstance(hook_output, tuple) else hook_output).item()
            gfaml_out_norm = torch.norm(gfaml_out).item()
            
            print(f"Layer {layer_idx:02d} | x_norm norm: {x_norm_norm:.4f} | hook_output norm: {hook_out_norm:.4f} | gfaml_out norm: {gfaml_out_norm:.4f}")
            
            if isinstance(hook_output, tuple):
                return (hook_output[0] + gfaml_out,) + hook_output[1:]
            else:
                return hook_output + gfaml_out
        return debug_hook_fn
        
    model.hook_handles[i] = attn_module.register_forward_hook(make_debug_hook(i), with_kwargs=True)

print("Running augmented forward pass and printing norms...")
with torch.no_grad():
    model._forward_augmented(ids)
