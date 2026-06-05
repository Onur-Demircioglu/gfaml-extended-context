# scratch/test_qwen_encoding.py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
full_ids = tokenizer.encode(full, add_special_tokens=False)

print("Prompt IDs:", prompt_ids)
print("Prompt Decoded tokens:")
for i in prompt_ids:
    print(f"  {i} -> {repr(tokenizer.decode([i]))}")

print("\nFull IDs:", full_ids)
print("Full Decoded tokens:")
for i in full_ids:
    print(f"  {i} -> {repr(tokenizer.decode([i]))}")
