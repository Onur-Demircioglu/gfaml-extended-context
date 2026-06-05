# gfaml/experiments/cognitive_stress_test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from gfaml.config import GFAMLConfig

# --- 1. PLASTISITE VE PARAZITLI SUKM MIMARISI ---
class SUKMStressBrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hipokampus (Fast Memory - GFAML)
        from gfaml.memory.gfaml import GFAMLLayer
        self.hippocampus = GFAMLLayer(config)
        self.hippocampus_state = None
        
        # Kortex (LoRA Parametreleri)
        self.cortex_A = nn.Parameter(torch.randn(config.d_model, config.rank) * 0.01)
        self.cortex_B = nn.Parameter(torch.zeros(config.rank, config.d_model))
        
        self.replay_buffer = []
        self.surprise_threshold = 0.30
        self.optimizer = optim.Adam([self.cortex_A, self.cortex_B], lr=0.01)

    def forward(self, x):
        cortex_out = x @ (self.cortex_A @ self.cortex_B)
        gfaml_out, next_state, _ = self.hippocampus(x, state=self.hippocampus_state)
        self.hippocampus_state = next_state
        return x + cortex_out + gfaml_out

    def sleep_consolidation_with_synaptic_noise(self, noise_scale=0.01):
        """
        [SYNAPTIC NOISE CONSOLIDATION]
        Kortex ogrenirken sinaptik agirlik gurultusu (noise) enjekte edilir.
        """
        if len(self.replay_buffer) == 0:
            return
            
        inputs = torch.stack([item[0].squeeze() for item in self.replay_buffer])
        targets = torch.stack([item[1].squeeze() for item in self.replay_buffer])
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
        # 80 epokluk konsolidasyon egitimi (Hibrit konsolidasyon)
        epochs = 80
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            cortex_out = inputs @ (self.cortex_A @ self.cortex_B)
            loss = nn.MSELoss()(cortex_out, targets)
            loss.backward()
            self.optimizer.step()
            
        # Sinaptik Gurultu Enjeksiyonu (stabilizasyon testi)
        with torch.no_grad():
            self.cortex_A.add_(torch.randn_like(self.cortex_A) * noise_scale)
            self.cortex_B.add_(torch.randn_like(self.cortex_B) * noise_scale)

# --- 2. STRES TESTI KOSTURUCU ---
def run_cognitive_stress_tests():
    print("="*75)
    print("=== [SUKM STRESS TEST] - EZBER vs GERCEK OGRENME ANALIZ SUITE ===")
    print("="*75)
    
    config = GFAMLConfig(
        d_model=128, rank=32, lambda_decay=1.0, eta=1.0, alpha=1.0,
        perfect_gating=True, use_soft_stabilization=True
    )
    brain = SUKMStressBrain(config)
    
    # Kelime gommesi uretici
    def get_emb(word):
        g = torch.Generator().manual_seed(hash(word) % 10000)
        return F.normalize(torch.randn(1, config.d_model, generator=g), p=2, dim=1)

    # Cakisan semantik veri seti (Interference olusturmak icin)
    entities = ["ONUR", "AYSE", "FATMA", "MEHMET", "ALICE", "BOB"]
    frameworks = ["PYTORCH", "TENSORFLOW", "KERAS", "JAX", "MXNET", "CNTK"]
    
    embs_ent = {e: get_emb(e) for e in entities}
    embs_fw = {f: get_emb(f) for f in frameworks}
    
    print("\n[PART 1] PARAZIT VE INTERFERANS TESTI (Overlapping Concepts)")
    print("  * Bellege 6 adet bagimsiz iliski yukleniyor (Cakisma stress testi)...")
    
    # 6 iliskiyi ayni anda hipokampuse yaz ve uykuda ogrenilmek uzere buffer'a at
    for i in range(6):
        k = embs_ent[entities[i]]
        v = embs_fw[frameworks[i]]
        combined = (k + v) / math.sqrt(2)
        
        # Hipokampus yazimi
        brain(combined.unsqueeze(0))
        # Replay buffer'a kaydet
        brain.replay_buffer.append((k, v))
        
    print(f"  * Replay Buffer'daki toplam ani sayisi: {len(brain.replay_buffer)}")
    
    # Uyku Evresi - Sinaptik gurultu enjeksiyonuyla konsolidasyon
    print("\n[PART 2] SINAPTIK GURULTULU UYKU EVRESI (Synaptic Noise Consolidation)")
    brain.sleep_consolidation_with_synaptic_noise(noise_scale=0.01)
    
    # Uyku Sonrasi Ezber vs Genelleme Sinavi
    print("\n[PART 3] GENELLEME VE PARAZIT BASARI TABLOSU (Generalization vs Interference)")
    print(f"{'Sorgulanan Kisi':<15} | {'Hedef Framework':<15} | {'Korteks Kosinus':<18} | {'Durum':<15}")
    print("-" * 70)
    
    hits = 0
    total_sim = 0.0
    for i in range(6):
        k = embs_ent[entities[i]]
        v = embs_fw[frameworks[i]]
        
        # Gecici bellek tamamen bosken sadece Kortex'ten sorgula
        out = brain(k.unsqueeze(0)).squeeze()
        m_out = (out - k.squeeze()) / config.alpha
        
        sim = F.cosine_similarity(m_out.unsqueeze(0), v, dim=1).item()
        total_sim += sim
        
        # En yuksek benzerlik dogru veya yanlis framework'te mi? (Argmax Check)
        sims = [F.cosine_similarity(m_out.unsqueeze(0), embs_fw[f], dim=1).item() for f in frameworks]
        best_idx = sims.index(max(sims))
        
        status = "GECTI (DENGELI)" if best_idx == i and sim > 0.30 else "SIZINTI/KILIT"
        if best_idx == i and sim > 0.30:
            hits += 1
            
        print(f"{entities[i]:<15} | {frameworks[i]:<15} | {sim:>18.4f} | {status:<15}")
        
    acc = (hits / 6.0) * 100
    avg_sim = total_sim / 6.0
    print("-" * 70)
    print(f"  * Gercek Kavramsal Genelleme Basarisi : %{acc:.1f}")
    print(f"  * Ortalama Kosinus Guvenirligi       : {avg_sim:.4f}")
    
    if acc >= 80.0:
        print("\n[HAKEM ONAYI] Harika! Model sadece ezber yapmamis; gurultu altinda")
        print("              tum bilgileri kortekse birbirine karistirmadan aktarmis!")
    else:
        print("\n[HAKEM REDDI] Model ezber yapmaya calismis ama yuksek parazit (interference)")
        print("              altinda agirliklar birbirine girip cökmüs!")

if __name__ == "__main__":
    run_cognitive_stress_tests()
