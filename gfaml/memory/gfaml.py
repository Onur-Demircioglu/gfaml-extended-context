# gfaml/memory/gfaml.py
import torch
import torch.nn as nn
from gfaml.config import GFAMLConfig
from gfaml.memory.gating import MemoryGate
from gfaml.memory.update_rules import apply_memory_update, apply_norm_constraint

class GFAMLLayer(nn.Module):
    """
    Gradient-Free Associative Memory Layer (GFAML).
    Geriye yayılım (backpropagation) olmadan dinamik olarak güncellenen
    düşük ranklı (low-rank) bellek matrisi katmanı.
    """
    def __init__(self, config: GFAMLConfig):
        super().__init__()
        self.config = config
        
        # Gating (Kapılama) Mekanizması
        self.gate = MemoryGate(config)
        
        # Öğrenilebilir parametreler (gating matrisi vb.)
        # Bellek matrisleri U ve V gradyansız (gradient-free) güncellenirken,
        # gate parametreleri gradyan tabanlı eğitilebilir.
        self.W_g = nn.Linear(config.d_model, 1, bias=True)
        with torch.no_grad():
            self.W_g.bias.fill_(config.gate_bias)
            
    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Başlangıç U ve V durumlarını sıfır olarak üretir.
        U: (batch_size, d_model, rank)
        V: (batch_size, rank, d_model)
        """
        U = torch.randn(batch_size, self.config.d_model, self.config.rank, device=device) * 0.01
        V = torch.randn(batch_size, self.config.rank, self.config.d_model, device=device) * 0.01
        return U, V
        
    def forward(self, h: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] = None, target_embeddings: torch.Tensor = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        h: (batch_size, seq_len, d_model) veya (batch_size, d_model)
        state: (U, V) bellek durumu tuple'ı
        target_embeddings: simülasyon kapısı için hedef anahtar kelime gömmeleri
        
        Dönen değerler:
        - h_out: (batch_size, seq_len, d_model) veya (batch_size, d_model) - Bellekten okuma eklenmiş çıktı
        - next_state: Güncellenmiş (U, V) bellek durumu
        - gates: (batch_size, seq_len) - Gating katsayıları
        """
        # Girdiyi 3D formata uyarla (batch_size, seq_len, d_model)
        is_2d = len(h.shape) == 2
        if is_2d:
            h = h.unsqueeze(1)
            
        batch_size, seq_len, d_model = h.shape
        device = h.device
        
        if state is None:
            state = self.initial_state(batch_size, device)
            
        U, V = state
        h_out_list = []
        gate_list = []
        
        # Token-by-token (adım adım) bellek güncellemesi ve okuma
        for t in range(seq_len):
            h_t = h[:, t, :] # (batch_size, d_model)
            
            # 1. Okuma (Readout): m_t = U * (V * h_t)
            # h_t'yi 3D yapalım: (batch_size, d_model, 1)
            h_t_uns = h_t.unsqueeze(-1)
            
            # V * h_t -> (batch_size, rank, 1)
            V_h = torch.bmm(V, h_t_uns)
            
            # U * (V * h_t) -> (batch_size, d_model, 1)
            m_t = torch.bmm(U, V_h).squeeze(-1) # (batch_size, d_model)
            
            # GFAML v2: Enerji ve Sinyal Genliği Stabilizasyonu (Unit Normalization Head)
            if self.config.use_unit_normalization:
                m_norm = torch.norm(m_t, p=2, dim=-1, keepdim=True)
                scale = torch.where(m_norm > self.config.normalization_threshold, 
                                    1.0 / (m_norm + 1e-8), 
                                    torch.zeros_like(m_norm))
                m_t = m_t * scale
            
            # GFAML v2.1: Yumuşak Enerji Saturasyonu (Soft Energy Stabilization)
            elif self.config.use_soft_stabilization:
                m_norm = torch.norm(m_t, p=2, dim=-1, keepdim=True)
                scale = 1.0 / torch.sqrt(1.0 + m_norm**2 + 1e-8)
                m_t = m_t * scale
            
            # Çıktı: h'_t = h_t + alpha * m_t
            h_prime_t = h_t + self.config.alpha * m_t
            h_out_list.append(h_prime_t)
            
            # 2. Kapılama (Gating)
            g_t = self.gate(h_t, self.W_g, target_embeddings=target_embeddings) # (batch_size, 1)
            gate_list.append(g_t)
            
            # 3. Gradyansız Güncelleme (Gradient-Free Update)
            # z_t = U^T * h_t -> (batch_size, rank, 1)
            z_t = torch.bmm(U.transpose(1, 2), h_t_uns)
            
            # U ve V güncellemesini uygulayalım
            U, V = apply_memory_update(U, V, h_t_uns, z_t, g_t, self.config)
            
            # 4. Bellek Taşması / Doyum Kontrolü (Norm clip)
            U = apply_norm_constraint(U, self.config)
            V = apply_norm_constraint(V, self.config)
            
        h_out = torch.stack(h_out_list, dim=1)
        gates = torch.cat(gate_list, dim=1)
        
        if is_2d:
            h_out = h_out.squeeze(1)
            
        return h_out, (U, V), gates
