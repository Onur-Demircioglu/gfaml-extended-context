# scratch/check_tokenizer_class.py
import sys
import os
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Tokenizer class:", tokenizer.__class__.__name__)
print("Tokenizer vocab size:", len(tokenizer))
print("Tokenizer vocab_size attr:", tokenizer.vocab_size)

test_str = "Onur'un en sevdiği programlama dili hangisidir?"
ids = tokenizer.encode(test_str, add_special_tokens=False)
print("Encoded IDs:", ids)
print("Decoded token by token:")
for i in ids:
    try:
        print(f"{i} -> {repr(tokenizer.decode([i]))}")
    except Exception as e:
        print(f"{i} -> ERROR: {e}")
