# gfaml/config.py
from dataclasses import dataclass

@dataclass
class GFAMLConfig:
    # Model Boyutları
    d_model: int = 128
    rank: int = 8
    
    # Bellek Dinamikleri
    lambda_decay: float = 0.995  # Sönümleme oranı (lambda)
    eta: float = 0.01            # Gradyansız öğrenme/yazma hızı (eta)
    alpha: float = 1.0           # Bellekten okunan değerin çarpanı
    
    # Doyum (Saturation) Kontrolü
    saturation_threshold: float = 5.0
    use_hard_clamp: bool = False  # True ise torch.clamp, False ise dinamik norm bölmesi yapar
    
    # Gating (Kapılama) Ayarları
    use_gating: bool = True
    gate_bias: float = -1.0       # Seyreklik sağlamak için gate başlangıç bias'ı
    
    # Sentetik Veri Ayarları
    vocab_size: int = 5000
    embedding_dim: int = 128
    
    # Simülasyon Ayarları
    perfect_gating: bool = False
    
    # GFAML v2 & v2.1 Enerji Stabilizasyon Ayarları
    use_unit_normalization: bool = False  # Sert küre yüzeyi projeksiyonunu kapat (v2)
    use_soft_stabilization: bool = True   # Yumuşak enerji stabilizasyonunu aktif et (v2.1)
    normalization_threshold: float = 1e-6
