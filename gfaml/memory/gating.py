# gfaml/memory/gating.py
import torch
import torch.nn as nn
from gfaml.config import GFAMLConfig

class MemoryGate(nn.Module):
    """
    Hangi tokenlerin bellek matrisine yazılacağını kontrol eden kapılama katmanı.
    """
    def __init__(self, config: GFAMLConfig):
        super().__init__()
        self.config = config
        
    def forward(self, h_t: torch.Tensor, W_g: nn.Linear, target_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        h_t: (batch_size, d_model)
        W_g: Eğitilebilir lineer projeksiyon katmanı
        target_embeddings: (num_targets, d_model) simülasyon modu için hedef vektörler
        
        Dönen değer:
        - g_t: (batch_size, 1) - Kapı katsayıları [0, 1]
        """
        if not self.config.use_gating:
            # Gating kapalıysa her şeyi 1.0 ile çarpıp doğrudan yazıyoruz
            return torch.ones(h_t.size(0), 1, device=h_t.device)
            
        if self.config.perfect_gating and target_embeddings is not None:
            # Kusursuz Gating Modu (Simülasyon): Gürültüleri sıfırlar, sadece anahtar kelimeleri geçirir
            # h_t: (batch_size, d_model), target_embeddings: (num_targets, d_model)
            import torch.nn.functional as F
            # Kosinüs benzerlik matrisi hesapla: (batch_size, num_targets)
            cos_sims = F.cosine_similarity(h_t.unsqueeze(1), target_embeddings.unsqueeze(0), dim=2)
            max_sim, _ = torch.max(cos_sims, dim=1, keepdim=True)
            g_t = (max_sim > 0.95).float()
            return g_t
            
        # Standart Sigmoid(W_g * h_t) kuralı
        g_t = torch.sigmoid(W_g(h_t)) # (batch_size, 1)
        return g_t
