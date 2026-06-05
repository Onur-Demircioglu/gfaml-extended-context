# scratch/print_sizes.py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

prompt = "Onur'un en sevdiği programlama dili hangisidir?"
answer = " Python"
full = prompt + answer

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
full_ids = tokenizer.encode(full, add_special_tokens=False)

print("prompt_ids length:", len(prompt_ids))
print("full_ids length:", len(full_ids))
print("plen:", len(prompt_ids))
print("diff:", len(full_ids) - len(prompt_ids))
