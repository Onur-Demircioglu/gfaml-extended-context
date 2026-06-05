# scripts/chat_gpt2.py
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat():
    print("=" * 60)
    print("  GPT-2 (124M) İnteraktif Sohbet Modülü")
    print("  (Çıkmak için 'exit' yazabilirsin)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model yükleniyor ({device} aktif)...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        model.eval()
        print("\nModel başarıyla yüklendi! Sohbet başlayabilir.\n")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    # Base modellerde sohbeti yönlendirmek için bir başlangıç bağlamı (context) veriyoruz
    chat_history = (
        "The following is a conversation between a helpful Human and an intelligent AI assistant.\n\n"
        "Human: Hello, who are you?\n"
        "AI: Hello! I am GPT-2, a language model trained by OpenAI. How can I help you today?\n"
    )

    while True:
        try:
            user_input = input("\nSen: ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() == 'exit':
                print("Görüşmek üzere!")
                break
            
            # Format inputs
            chat_history += f"Human: {user_input}\nAI:"
            
            # Tokenize & Generate
            inputs = tokenizer.encode(chat_history, return_tensors="pt").to(device)
            
            # GPT-2'nin saçmalamasını önlemek için sıcaklık ve top_p ayarları
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=40,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.90,
                    temperature=0.8,
                    repetition_penalty=1.1
                )
            
            # Çıktıyı decode et
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = full_text[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            
            # Bir sonraki satıra geçmesini veya kendi kendine konuşmasını (Human: diyerek) engelle
            response = new_text.split("Human:")[0].split("\n")[0].strip()
            
            print(f"GPT-2: {response}")
            chat_history += f" {response}\n"
            
            # Bellek şişmesini önlemek için son 4 konuşmayı sakla
            lines = chat_history.split("\n")
            if len(lines) > 15:
                # Başlangıç promptunu koruyarak geçmişi kırp
                chat_history = "\n".join(lines[:3]) + "\n" + "\n".join(lines[-10:])
                
        except KeyboardInterrupt:
            print("\nSohbet sonlandırıldı.")
            break
        except Exception as e:
            print(f"\nBir hata oluştu: {e}")
            break

if __name__ == "__main__":
    chat()
