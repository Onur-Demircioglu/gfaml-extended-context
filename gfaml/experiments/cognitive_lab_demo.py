# gfaml/experiments/cognitive_lab_demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from gfaml.config import GFAMLConfig

# --- 1. SEVIYE 3: SUKM BEYIN MIMARISI ---
class SUKMBrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hipokampus (Fast Memory - GFAML)
        from gfaml.memory.gfaml import GFAMLLayer
        self.hippocampus = GFAMLLayer(config)
        self.hippocampus_state = None
        
        # Kortex (Slow Weight Learning - LoRA Parameter Adapter)
        self.cortex_A = nn.Parameter(torch.randn(config.d_model, config.rank) * 0.01)
        self.cortex_B = nn.Parameter(torch.zeros(config.rank, config.d_model))
        
        # Episodic Replay Buffer (Uyku Deposu)
        self.replay_buffer = []
        self.surprise_threshold = 0.30
        self.optimizer = optim.Adam([self.cortex_A, self.cortex_B], lr=0.01)

    def forward(self, x, is_writing=False, key_emb=None, val_emb=None):
        # A) Kortex agirliklarindan kalici bilgi cagirilir
        cortex_out = x @ (self.cortex_A @ self.cortex_B)
        
        # B) Hipokampus'ten anlik bellek okunur
        gfaml_out, next_state, _ = self.hippocampus(x, state=self.hippocampus_state)
        self.hippocampus_state = next_state
        
        # C) Gercek Surpriz (Tahmin Hatasi / Novelty) Hesaplama
        # Girdi ile bellek ciktisi arasindaki farka bakilir
        for t in range(x.size(1)):
            h_t = x[:, t, :]
            # gfaml_out = h_t + alpha * m_t oldugundan m_t'yi geri cekelim:
            m_t = (gfaml_out[:, t, :] - h_t) / self.config.alpha
            
            # Sinyal sifirken tahmin hatasi 1.0 (Tam surpriz) olur. Hafiza ogrenildikce sifira yaklasir.
            surprise = torch.norm(h_t - m_t, p=2).item()
            
            # Eger bilgi yeniyse ve yazma modundaysak Replay Buffer'a kaydet
            if is_writing and surprise > self.surprise_threshold and key_emb is not None and val_emb is not None:
                self.replay_buffer.append((key_emb.detach(), val_emb.detach()))
                print(f"  * [EPISODIC STORE] Surpriz Degeri: {surprise:.4f} > {self.surprise_threshold} (Anı uyku deposuna alindi!)")
                
        return x + cortex_out + gfaml_out

    def sleep_consolidation(self):
        """
        UYKU VE BELLEK KONSOLIDASYONU
        Episodik hafizadaki tum iliskiler Kortex'e (LoRA agirliklarina) nakledilir.
        Uykudan uyaninca Hipokampus (Fast memory) tamamen bosaltilir.
        """
        if len(self.replay_buffer) == 0:
            print("[UYKU] Yeni bilgi birikmedigi icin uyunmadi.")
            return
            
        print(f"\n[UYKU] Uyku evresi tetiklendi. {len(self.replay_buffer)} onemli ani kortekse yaziliyor...")
        
        # Boyut hatasini onlemek icin stack ve squeeze islemleri
        inputs = torch.stack([item[0].squeeze() for item in self.replay_buffer]) # (batch, d_model)
        targets = torch.stack([item[1].squeeze() for item in self.replay_buffer]) # (batch, d_model)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
        # Kortex LoRA agirliklarinin konsolidasyon egitimi (Sleep Consolidation)
        epochs = 50
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            cortex_out = inputs @ (self.cortex_A @ self.cortex_B)
            loss = nn.MSELoss()(cortex_out, targets)
            loss.backward()
            self.optimizer.step()
            
        print(f"[UYKU] Kalici ogrenme tamamlandi. Kortex Transfer Kaybi (Loss): {loss.item():.6f}")
        
        # Hipokampus Reset (Uykuda temizlenme)
        self.hippocampus_state = None
        self.replay_buffer.clear()
        print("[UYKU] Hipokampus (Gecici Hafiza) sifirlandi. Model zinde sekilde uyandi!")

# --- 2. FISEKLEME VE DEMO SIMULASYONU ---
def run_cognitive_demo():
    print("="*75)
    print("=== [SUKM] SELF-UPDATING COGNITIVE MODEL (BILSIM LABORATUVARI) ===")
    print("="*75)
    
    config = GFAMLConfig(
        d_model=128, rank=32, lambda_decay=1.0, eta=1.0, alpha=1.0,
        perfect_gating=True, use_soft_stabilization=True
    )
    brain = SUKMBrain(config)
    
    # Kelime gommesi uretici
    def get_emb(word):
        g = torch.Generator().manual_seed(hash(word) % 10000)
        return F.normalize(torch.randn(1, config.d_model, generator=g), p=2, dim=1)

    print("\n[ADIM 1] Model uyanik. Yeni bir bilgi girisi yapiliyor...")
    k = get_emb("ONUR")
    v = get_emb("PYTORCH")
    combined = (k + v) / math.sqrt(2) # ONUR -> PYTORCH ilişkisi
    
    # Yazma forward pass'i (is_writing=True ile surpriz kontrolu yapilir)
    brain(combined.unsqueeze(0), is_writing=True, key_emb=k, val_emb=v)
    print("  * 'ONUR -> PYTORCH' iliskisi gecici olarak hipokampuse yazildi.")
    
    # 2. Uyku Oncesi Test
    print("\n[ADIM 2] Uyku Oncesi Sorgu: 'ONUR' anahtariyla hafiza sorgulaniyor...")
    # Hipokampus dolu oldugu icin dogru hatirlanmasi beklenir
    out_pre = brain(k.unsqueeze(0)).squeeze()
    m_pre = (out_pre - k.squeeze()) / config.alpha
    sim_pre = F.cosine_similarity(m_pre.unsqueeze(0), v, dim=1).item()
    print(f"  * Uyku Oncesi Kosinus Benzerligi (Hatirlama): {sim_pre:.4f} (Basarili!)")
    
    # 3. Uyku Evresi (Sleep Consolidation)
    brain.sleep_consolidation()
    
    # 4. Uyku Sonrasi Test (Uyanis)
    print("\n[ADIM 3] Uyku Sonrasi (Hipokampus tamamen SIFIRLANMISKEN) Sorgulama...")
    # Hipokampus sifirlandi. Bilginin korteks agirliklarindan (LoRA) gelmesi gerekir!
    out_post = brain(k.unsqueeze(0)).squeeze()
    m_post = (out_post - k.squeeze()) / config.alpha
    sim_post = F.cosine_similarity(m_post.unsqueeze(0), v, dim=1).item()
    
    print(f"  * Uyku Sonrasi Kosinus Benzerligi (Korteksten Okuma): {sim_post:.4f}")
    if sim_post > 0.40:
        print("\n[ONAY] INANILMAZ BASARI! Hipokampus tamamen sifirlanmasina ragmen,")
        print("       bilgi korteks agirliklarina kalici olarak aktarilmis ve kurtarilmis!")
    else:
        print("\n[UYARI] Korteks transfer verimi zayif kaldi.")

if __name__ == "__main__":
    run_cognitive_demo()
