# scratch/decode_ids.py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
ids = [47385, 83637, 279, 93, 22238, 102606, 92955, 33501, 109726, 30]
print("Decoded whole list:", repr(tokenizer.decode(ids)))
print("Decoded token by token:")
for i in ids:
    print(f"{i} -> {repr(tokenizer.decode([i]))}")
