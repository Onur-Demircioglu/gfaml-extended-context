# scratch/test_tokens.py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
full_ids = tokenizer.encode(full, add_special_tokens=False)

print("Prompt tokens:")
for i in prompt_ids:
    print(f"{i}: {repr(tokenizer.decode([i]))}")

print("\nFull tokens:")
for i in full_ids:
    print(f"{i}: {repr(tokenizer.decode([i]))}")
