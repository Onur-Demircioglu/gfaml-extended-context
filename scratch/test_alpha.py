# scratch/test_alpha.py
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

# Teach
prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer
ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
plen = len(tokenizer.encode(prompt, add_special_tokens=False))
target_token_id = ids[0, plen].item()

query_ids = tokenizer.encode(prompt, add_special_tokens=False)
query_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

for alpha_val in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
    config = GFAMLConfig(vocab_size=vocab_size, d_model=d_model, rank=32, eta=1.0, alpha=alpha_val)
    model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
    
    # We register the corrected hook
    for i, block in enumerate(model.layers):
        attn_module = getattr(block, model.attn_name)
        model.hook_handles[i].remove()
        
        def make_corrected_hook(layer_idx):
            def hook_fn(module, args, kwargs, hook_output):
                if model.gfaml_mode == "off":
                    return hook_output
                
                x_norm = args[0] if len(args) > 0 else kwargs["hidden_states"]
                state = model.state_list[layer_idx] if model.state_list is not None else None
                
                if model.gfaml_mode == "record":
                    _, next_state, _ = model.gfaml_layers[layer_idx](
                        x_norm, state=state, write=True
                    )
                    if model.state_list is None:
                        model.state_list = [None] * len(model.layers)
                    model.state_list[layer_idx] = next_state
                    return hook_output
                
                elif model.gfaml_mode == "augment":
                    gfaml_out, _, _ = model.gfaml_layers[layer_idx](
                        x_norm, state=state, write=False
                    )
                    # Corrected addition: only add the memory contribution (gfaml_out - x_norm)
                    memory_contrib = gfaml_out - x_norm
                    if isinstance(hook_output, tuple):
                        attn_out = hook_output[0] + memory_contrib
                        return (attn_out,) + hook_output[1:]
                    else:
                        return hook_output + memory_contrib
                
                return hook_output
            return hook_fn
        
        model.hook_handles[i] = attn_module.register_forward_hook(make_corrected_hook(i), with_kwargs=True)
        
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
    
    # Query with augmented
    with torch.no_grad():
        aug_logits = model._forward_augmented(query_tensor)
        
    top_val, top_idx = torch.max(aug_logits[0, -1, :], dim=-1)
    target_logit = aug_logits[0, -1, target_token_id].item()
    
    print(f"Alpha: {alpha_val:.2f} | Target Logit (' Python'): {target_logit:.4f} | Top predicted: {repr(tokenizer.decode([top_idx.item()]))} (logit: {top_val.item():.4f})")
