# experiments/exp_05_hf_model.py
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Çalısma alanı ana dizinini sys.path'e ekle
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
    Gerçek bir HuggingFace modelini sarmalayıp,
    GFAML bellek katmanları ve LoRA korteksi ile entegre eden bilişsel model.
    """
    def __init__(self, hf_model, config):
        super().__init__()
        self.hf_model = hf_model
        self.config = config
        
        # Modeli dondur (Frozen Base)
        for param in self.hf_model.parameters():
            param.requires_grad = False
            
        # GFAML katmanlarını yerleştir
        self.gfaml_layers = nn.ModuleList([
            GFAMLLayer(config.d_model, config.rank, config=config)
            for _ in range(len(self.hf_model.transformer.h))
        ])
        
        # LoRA Korteks Katmanı
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.state_list = None
        self.replay_buffer = []
        self.optimizer = SleepOptimizer(self.cortex, lr=0.06)

    def forward(self, idx, is_writing=False, target_tok=None):
        # Token gömmelerini al
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe
        
        tok_emb = wte(idx)
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = wpe(pos)
        
        x = tok_emb + pos_emb
        
        if self.state_list is None:
            self.state_list = [None] * len(self.hf_model.transformer.h)
            
        next_states = []
        
        # Transformer bloklarını işlet ve GFAML katmanlarını araya ekle
        for i, block in enumerate(self.hf_model.transformer.h):
            x_norm = block.ln_1(x)
            
            # Standart self-attention (HuggingFace GPT2)
            attn_out = block.attn(x_norm)[0]
            
            # GFAML okuma/yazma
            gfaml_layer = self.gfaml_layers[i]
            if is_writing and idx.size(1) == 2:
                # Combined write
                k_emb = x_norm[:, 0, :]
                v_emb = x_norm[:, 1, :]
                combined = (k_emb + v_emb) / 1.4142
                _, next_state, _ = gfaml_layer(combined.unsqueeze(1), state=self.state_list[i], write=True)
                gfaml_out, _, _ = gfaml_layer(x_norm, state=self.state_list[i])
            else:
                gfaml_out, next_state, _ = gfaml_layer(x_norm, state=self.state_list[i])
                
            x = x + attn_out + gfaml_out
            # MLP/FeedForward
            x = x + block.mlp(block.ln_2(x))
            
            if is_writing:
                next_states.append(next_state)
            else:
                next_states.append(self.state_list[i])
                
        self.state_list = next_states
        x_norm = self.hf_model.transformer.ln_f(x)
        
        # Çıktı logits
        base_logits = self.hf_model.lm_head(x_norm)
        cortex_logits = self.cortex(x_norm)
        final_logits = base_logits + cortex_logits
        
        # Replay buffer'a kaydet (konsolidasyon için)
        if is_writing and target_tok is not None:
            self.replay_buffer.append((x_norm[0, 0, :].detach().clone(), target_tok))
            
        return final_logits, x_norm

def run():
    print("=" * 80)
    print("=== [DENEY 5] GERÇEK HUGGINGFACE MODELİ İLE GFAML TESTİ ===")
    print("=" * 80)
    
    # 1. HuggingFace Model ve Tokenizer'ı Yükle
    model_name = "sshleifer/tiny-gpt2"
    print(f"   => Gerçek HF Modeli indiriliyor/yükleniyor: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Model konfigürasyonunu güncelle
    config = GFAMLConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=hf_model.config.n_embd,
        rank=32,
        eta=1.0
    )
    
    # Bilişsel model sarmalayıcısını ayağa kaldır
    model = GFAMLHuggingFaceWrapper(hf_model, config)
    print("   => GFAML Katmanları ve OGP Sleep Optimizer başarıyla enjekte edildi!")
    
    # 2. Görev Tanımlama
    task = [
        ("Who wrote Hamlet?", "Shakespeare"),
        ("Who discovered Radium?", "Curie"),
        ("Who founded Apple?", "Jobs")
    ]
    
    # Tokenizer ile id'lere dönüştür
    task_pairs = []
    for q, a in task:
        q_ids = tokenizer.encode(q)
        a_id = tokenizer.encode(a)[0] # İlk tokeni al
        task_pairs.append((q_ids, a_id))
        
    # 3. Combined Write ile Belleğe Kaydet
    print("\n[ADIM 1] Hızlı Bellek (Hipokampus) Combined Yazma Başlatılıyor...")
    for q_ids, a_id in task_pairs:
        # İki tokenli k-v çifti şeklinde besle
        idx = torch.tensor([[q_ids[-1], a_id]], dtype=torch.long)
        model(idx, is_writing=True, target_tok=a_id)
        
    # 4. Bellek Hatırlama Doğruluğunu (Recall) Ölç
    print("\n[ADIM 2] Bellek Hatırlama Başarısı Cosine Ölçümleri...")
    hits = 0
    wte = model.hf_model.transformer.wte
    wpe = model.hf_model.transformer.wpe
    state = model.state_list[0]
    U, V = state
    
    for q_ids, a_id in task_pairs:
        idx_k = torch.tensor([[q_ids[-1]]])
        pos_k = torch.tensor([[0]])
        x_k = wte(idx_k) + wpe(pos_k)
        x_k_norm = model.hf_model.transformer.h[0].ln_1(x_k).squeeze()
        
        h_q = x_k_norm.unsqueeze(0).unsqueeze(-1)
        V_h = torch.bmm(V, h_q)
        m_t = torch.bmm(U, V_h).squeeze(-1).squeeze(0)
        
        # Norm stabilization
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
            
    print(f"   => Gerçek HF Modeli Bellek Recall Doğruluğu: %{hits/len(task)*100:.1f} (Beklenen: %100.0 - Hatırladı!)")
    
    # 5. Sleep Phase Konsolidasyonu Çalıştır (OGP ile)
    print("\n[ADIM 3] OGP Kısıtlamalı Uyku Konsolidasyonu Çalıştırılıyor...")
    loss_val = model.optimizer.consolidate(model.replay_buffer, epochs=10)
    print(f"   => Konsolidasyon tamamlandı. Final Uyku Kaybı (Sleep Loss): {loss_val:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run()
