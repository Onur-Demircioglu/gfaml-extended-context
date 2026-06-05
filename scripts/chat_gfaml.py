# scripts/chat_gfaml.py
import sys
import os
import torch

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from gfaml.config import GFAMLConfig
from experiments.exp_12_scaled_benchmark import DecoupledGFAML


PROMPT_TEMPLATE = """The following is a conversation between a user and a helpful assistant.

User: {input}
Assistant:"""


def wrap_prompt(user_text):
    """Wrap raw user text in a QA template so the base model generates natural language."""
    return PROMPT_TEMPLATE.format(input=user_text)


def generate_augmented(model, tokenizer, prompt_text, device, max_new_tokens=50, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2, debug=False):
    """
    Autoregressive generation with GFAML memory augmentation.
    Uses sampling (temperature + top-k + top-p) instead of raw greedy argmax.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    generated_ids = []
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model._forward_augmented(curr_ids)
        
        next_logits = logits[0, -1, :].float()
        
        # Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            for token_id in set(generated_ids):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty
        
        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
            next_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        # Stop on EOS
        if next_id == tokenizer.eos_token_id:
            break
        
        # Stop on newline after at least a few tokens
        decoded_token = tokenizer.decode([next_id])
        if '\n' in decoded_token and len(generated_ids) > 3:
            pre_newline = decoded_token.split('\n')[0]
            if pre_newline.strip():
                generated_ids.append(next_id)
            break
        
        generated_ids.append(next_id)
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]], device=device)], dim=1)
    
    if debug:
        safe_tokens = []
        for tid in generated_ids:
            try:
                safe_tokens.append(tokenizer.decode([tid]).encode('ascii', errors='replace').decode())
            except:
                safe_tokens.append(f"<id={tid}>")
        print(f"  [DEBUG] Token IDs: {generated_ids}")
        print(f"  [DEBUG] Tokens: {safe_tokens}")
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def chat_with_memory():
    print("=" * 75)
    print("  DECOUPLED GFAML (Cift Sistemli Akilli Bellek) Sohbet Arayuzu")
    print("  Kullanim:")
    print("  - Bir soru yazip modelin cevabini gorebilirsin.")
    print("  - `/teach Soru = Cevap` formatiyla modele anlik bilgi ogretebilirsin.")
    print("    Ornek: `/teach The capital of Turkey is = Ankara`")
    print("  - `/debug` ile token-seviye hata ayiklama modunu ac/kapat.")
    print("  - Cikmak icin 'exit' yazabilirsin.")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modeller yukleniyor ({device} aktif)...")

    debug_mode = False
    taught_facts = []  # Replay buffer for catastrophic forgetting prevention

    try:
        model_name = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_model.eval()

        d_model = hf_model.config.hidden_size if hasattr(hf_model.config, "hidden_size") else hf_model.config.n_embd
        vocab_size = hf_model.config.vocab_size if hasattr(hf_model.config, "vocab_size") else tokenizer.vocab_size
        config = GFAMLConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            rank=32, eta=1.0,
            alpha=0.3,  # Optimal: strong enough for recall, preserves base knowledge
        )
        
        model = DecoupledGFAML(hf_model, tokenizer, config, device=device)
        model.eval()
        print(f"\nGFAML Mimarisi ({model_name}) basariyla kuruldu!\n")
    except Exception as e:
        print(f"Hata: {e}")
        import traceback; traceback.print_exc()
        return

    while True:
        try:
            user_input = input("\nSen: ")
            if not user_input.strip():
                continue
            
            user_input = user_input.strip()
            if user_input.startswith("Sen:"):
                user_input = user_input[4:].strip()
                
            if user_input.lower() == 'exit':
                print("Gorusmek uzere!")
                break
            
            if user_input.lower() == '/debug':
                debug_mode = not debug_mode
                print(f">> Debug modu: {'ACIK' if debug_mode else 'KAPALI'}")
                continue
            
            # 1. Bilgi Ogretme Modu — Gradient-Free Instant Learning
            if user_input.startswith("/teach"):
                try:
                    parts = user_input[7:].split("=")
                    if len(parts) != 2:
                        print("Hata: Lutfen `/teach Soru = Cevap` formatinda yazin.")
                        continue
                    prompt = parts[0].strip()
                    answer = " " + parts[1].strip()
                    
                    full = prompt + answer
                    ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False).to(device)
                    plen = len(tokenizer.encode(prompt, add_special_tokens=False))
                    
                    # Hebbian Boundary Write: key=h(prompt_last) → value=h(answer_first)
                    # No LoRA, no gradient, no training loop — instant one-shot learning
                    model._record(ids, prompt_len=plen)
                    
                    taught_facts.append((prompt, answer))
                    
                    if debug_mode:
                        test_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
                        with torch.no_grad():
                            test_logits = model._forward_augmented(test_ids)
                        pred_id = torch.argmax(test_logits[0, -1, :]).item()
                        expected_id = tokenizer.encode(answer, add_special_tokens=False)[0]
                        pred_tok = tokenizer.decode([pred_id]).encode('ascii', errors='replace').decode()
                        exp_tok = tokenizer.decode([expected_id]).encode('ascii', errors='replace').decode()
                        print(f"  [DEBUG] Predicted: '{pred_tok}' (id={pred_id})")
                        print(f"  [DEBUG] Expected:  '{exp_tok}' (id={expected_id})")
                        print(f"  [DEBUG] Match: {'OK' if pred_id == expected_id else 'MISS'}")
                    
                    print(f">> GFAML: Aninda ogrenildi! (Gradient-free, {len(taught_facts)} fact bellekte)")
                except Exception as e:
                    print(f"Ogretme hatasi: {e}")
                    import traceback; traceback.print_exc()
                continue
                
            # 2. Soru/Sorgulama Modu (Augmented Inference)
            saved_state = getattr(model, 'state_list', None)
            
            # Wrap in QA template so base model generates natural language, not code
            formatted_prompt = wrap_prompt(user_input)
            
            generated = generate_augmented(
                model, tokenizer, formatted_prompt, device,
                max_new_tokens=80,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                debug=debug_mode,
            )
            
            if saved_state is not None:
                model.state_list = saved_state
            
            # Clean up: take only first line of response (avoid rambling)
            response = generated.split('\n')[0].strip() if generated else '(bos cikti)'
            # Remove any echoed prompt fragments
            for prefix in ['User:', 'Assistant:', 'Q:', 'A:']:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            print(f"GFAML: {response}")
            
        except KeyboardInterrupt:
            print("\nSohbet kapatildi.")
            break
        except Exception as e:
            print(f"\nBir hata olustu: {e}")
            import traceback; traceback.print_exc()
            break

if __name__ == "__main__":
    chat_with_memory()
