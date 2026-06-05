# scratch/test_single_word.py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

for word in ["hangisidir", "hangisidir?", "Onur", "en", "sevdiği", "programlama", "dili"]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    print(f"Word: {repr(word)} -> IDs: {ids} -> Decoded: {repr(tokenizer.decode(ids))}")
