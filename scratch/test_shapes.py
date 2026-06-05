# scratch/test_shapes.py
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

# Inspect shapes in hook
query_ids = tokenizer.encode(prompt, add_special_tokens=False)
query_ids_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

block = model.layers[0]
attn_module = getattr(block, model.attn_name)
model.hook_handles[0].remove()

def shape_hook(module, args, kwargs, hook_output):
    if model.gfaml_mode != "augment":
        return hook_output
        
    print("args length:", len(args))
    for i, a in enumerate(args):
        if torch.is_tensor(a):
            print(f"arg {i} shape: {a.shape}, dtype: {a.dtype}, device: {a.device}")
        else:
            print(f"arg {i}: {type(a)}")
            
    print("kwargs keys:", list(kwargs.keys()))
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            print(f"kwarg {k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
        else:
            print(f"kwarg {k}: {type(v)}")
            
    if isinstance(hook_output, tuple):
        print(f"hook_output is tuple of length {len(hook_output)}")
        for i, o in enumerate(hook_output):
            if torch.is_tensor(o):
                print(f"  tuple element {i} shape: {o.shape}, dtype: {o.dtype}")
            else:
                print(f"  tuple element {i}: {type(o)}")
    elif torch.is_tensor(hook_output):
        print(f"hook_output shape: {hook_output.shape}")
        
    # Run GFAML Layer
    x_norm = args[0] if len(args) > 0 else kwargs["hidden_states"]
    state = model.state_list[0]
    gfaml_out, _, _ = model.gfaml_layers[0](x_norm, state=state, write=False)
    print(f"gfaml_out shape: {gfaml_out.shape}, dtype: {gfaml_out.dtype}")
    
    return hook_output

model.hook_handles[0] = attn_module.register_forward_hook(shape_hook, with_kwargs=True)

with torch.no_grad():
    model._forward_augmented(query_ids_tensor)
