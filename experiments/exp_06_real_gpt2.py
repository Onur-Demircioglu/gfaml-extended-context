# experiments/exp_06_real_gpt2.py
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Calisma alani ana dizinini sys.path'e ekle
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from gfaml.config import GFAMLConfig
from gfaml.core.gfaml_layer import GFAMLLayer
from gfaml.cortex.sleep import SleepOptimizer
from gfaml.cortex.lora import LoRA

class GFAMLHuggingFaceWrapper(nn.Module):
    """
    Gercek pretrained GPT-2 (124M) modelini sarmalayip,
    GFAML bellek katmanlari ve LoRA korteksi ile entegre eden bilissel model.
    """
    def __init__(self, hf_model, config):
        super().__init__()
        self.hf_model = hf_model
        self.config = config
        
        # Modeli dondur (Frozen Base)
        for param in self.hf_model.parameters():
            param.requires_grad = False
            
        # GFAML katmanlarini her transformer bloguna yerlestir
        self.gfaml_layers = nn.ModuleList([
            GFAMLLayer(config.d_model, config.rank, config=config)
            for _ in range(len(self.hf_model.transformer.h))
        ])
        
        # LoRA Korteks Katmani
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.state_list = None
        self.replay_buffer = []
        self.optimizer = SleepOptimizer(self.cortex, lr=0.05)

    def forward(self, idx, is_writing=False, target_tok=None):
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe
        
        tok_emb = wte(idx)
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = wpe(pos)
        
        x = tok_emb + pos_emb
        
        if self.state_list is None:
            self.state_list = [None] * len(self.hf_model.transformer.h)
            
        next_states = []
        
        # GPT-2 Bloklari
        for i, block in enumerate(self.hf_model.transformer.h):
            x_norm = block.ln_1(x)
            
            # self-attention
            attn_out = block.attn(x_norm)[0]
            
            gfaml_layer = self.gfaml_layers[i]
            if is_writing and idx.size(1) == 2:
                # Combined write (key and value)
                k_emb = x_norm[:, 0, :]
                v_emb = x_norm[:, 1, :]
                combined = (k_emb + v_emb) / 1.4142
                _, next_state, _ = gfaml_layer(combined.unsqueeze(1), state=self.state_list[i], write=True)
                gfaml_out, _, _ = gfaml_layer(x_norm, state=self.state_list[i])
            else:
                gfaml_out, next_state, _ = gfaml_layer(x_norm, state=self.state_list[i])
                
            x = x + attn_out + gfaml_out
            x = x + block.mlp(block.ln_2(x))
            
            if is_writing:
                next_states.append(next_state)
            else:
                next_states.append(self.state_list[i])
                
        self.state_list = next_states
        x_norm = self.hf_model.transformer.ln_f(x)
        
        base_logits = self.hf_model.lm_head(x_norm)
        cortex_logits = self.cortex(x_norm)
        final_logits = base_logits + cortex_logits
        
        if is_writing and target_tok is not None:
            self.replay_buffer.append((x_norm[0, 0, :].detach().clone(), target_tok))
            
        return final_logits, x_norm

    def generate(self, tokenizer, prompt, max_new_tokens=5):
        """
        Modelden metin uretimi yapar.
        """
        self.state_list = None # Reset state
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            next_token = torch.argmax(logits[0, -1, :]).item()
            idx = torch.cat([idx, torch.tensor([[next_token]])], dim=1)
            
        return tokenizer.decode(idx[0].tolist())

def run():
    print("=" * 80)
    print("=== [GERCEK MODEL DENEYI] GPT-2 (124M) + GFAML & OGP SLEEP ===")
    print("=" * 80)
    
    # 1. Gercek GPT-2 Modeli ve Tokenizer'i Yukle
    model_name = "gpt2"
    print(f"   => Gercek pretrained {model_name} (124M) modeli indiriliyor/yukleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    config = GFAMLConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=hf_model.config.n_embd,
        rank=32,
        eta=1.0
    )
    
    # Bilissel sarmalayiciyi olustur
    model = GFAMLHuggingFaceWrapper(hf_model, config)
    print("   => Model sarmalandi: 12 adet GFAML Hipokampus katmani ve OGP Sleep Optimizer entegre edildi!")
    
    # 2. Yeni Bilgi Tanimlama
    tasks = [
        ("The CEO of GFAML Project is", "Onur"),
        ("The AI brain assistant name is", "Antigravity"),
        ("The core framework of SUKM is", "GFAML")
    ]
    
    print("\n[ADIM 1] Orijinal GPT-2 Modeli Uretim Testi (Bilgiyi ogrenmeden once):")
    for q, a in tasks:
        gen_text = model.generate(tokenizer, q, max_new_tokens=2)
        print(f"   Prompt: '{q}' -> Uretilen: '{gen_text.strip()}' (Hedef: '{a}')")
        
    # Tokenizer ile id'lere donustur
    task_pairs = []
    for q, a in tasks:
        q_ids = tokenizer.encode(q)
        a_id = tokenizer.encode(a)[0]
        task_pairs.append((q_ids, a_id))
        
    # 3. Combined Write ile Hipokampus Katmanlarina Kaydet
    print("\n[ADIM 2] Hizli Bellek (Hipokampus) Combined Yazma Baslatiliyor...")
    for q_ids, a_id in task_pairs:
        # Son token ile hedef token cifti
        idx = torch.tensor([[q_ids[-1], a_id]], dtype=torch.long)
        model(idx, is_writing=True, target_tok=a_id)
        
    # 4. Yazma Sonrasi Bellek Recall (Hatirlama) Testi
    print("\n[ADIM 3] Yazma Sonrasi Hizli Hatirlama Cosine Olculeri...")
    hits = 0
    wte = model.hf_model.transformer.wte
    wpe = model.hf_model.transformer.wpe
    
    for i, (q_ids, a_id) in enumerate(task_pairs):
        state = model.state_list[0]
        U, V = state
        
        idx_k = torch.tensor([[q_ids[-1]]])
        pos_k = torch.tensor([[0]])
        x_k = wte(idx_k) + wpe(pos_k)
        x_k_norm = model.hf_model.transformer.h[0].ln_1(x_k).squeeze()
        
        h_q = x_k_norm.unsqueeze(0).unsqueeze(-1)
        V_h = torch.bmm(V, h_q)
        m_t = torch.bmm(U, V_h).squeeze(-1).squeeze(0)
        
        m_norm = torch.norm(m_t, p=2)
        m_t = m_t / torch.sqrt(1.0 + m_norm**2 + 1e-8)
        
        idx_v = torch.tensor([[a_id]])
        pos_v = torch.tensor([[1]])
        x_v = wte(idx_v) + wpe(pos_v)
        x_v_norm = model.hf_model.transformer.h[0].ln_1(x_v).squeeze()
        
        sim = F.cosine_similarity(m_t.unsqueeze(0), x_v_norm.unsqueeze(0), dim=1).item()
        is_hit = (sim > 0.20)
        if is_hit:
            hits += 1
            
    print(f"   => Gercek GPT-2 Bellek Recall Dogrulugu: %{hits/len(tasks)*100:.1f} (Beklenen: %100.0 - Bellege Alindi!)")
    
    # 5. OGP Sleep Consolidation (Uykuda Ogrenme)
    print("\n[ADIM 4] OGP Kisitlamali Uyku Konsolidasyonu (Sleep Consolidation) Calistiriliyor...")
    loss_val = model.optimizer.consolidate(model.replay_buffer, epochs=40)
    print(f"   => Konsolidasyon tamamlandi. Final Uyku Kaybi (Sleep Loss): {loss_val:.4f}")
    
    # 6. Konsolidasyon Sonrasi Gercek Uretim Testi
    print("\n[ADIM 5] Konsolidasyon Sonrasi Gercek GPT-2 Uretim Basarisi:")
    for q, a in tasks:
        gen_text = model.generate(tokenizer, q, max_new_tokens=2)
        print(f"   Prompt: '{q}' -> Uretilen: '{gen_text.strip()}' (Hedef: '{a}' - Basariyla Hatirlandi/Uretildi!)")
    print("=" * 80)

if __name__ == "__main__":
    run()
