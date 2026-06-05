# gfaml/core/gfaml_layer.py
import torch
import torch.nn as nn
import math
from gfaml.config import GFAMLConfig
from gfaml.core.memory_state import apply_memory_update, apply_norm_constraint
from gfaml.memory.gating import MemoryGate

class GFAMLLayer(nn.Module):
    """
    Hetero-Associative Episodic Memory Engine.
    
    Write: key=h_t → value=h_{t+1} eşlemesi kurar (prompt → answer)
    Read:  query=h_t → m_t ≈ h_{t+1} retrieve eder (bellekten cevabı getirir)
    """
    def __init__(self, d_model: int, rank: int = 32, config: GFAMLConfig = None):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.config = config if config is not None else GFAMLConfig(d_model=d_model, rank=rank)
        
        # Gating (Kapılama) Mekanizması
        self.gate = MemoryGate(self.config)
        
        # Öğrenilebilir kapılama projeksiyon katmanı
        self.W_g = nn.Linear(d_model, 1, bias=True)
        with torch.no_grad():
            self.W_g.bias.fill_(self.config.gate_bias)
            
    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Başlangıç U ve V durumlarını küçük rastgele değerlerle üretir.
        U: (batch_size, d_model, rank) — value reconstruction
        V: (batch_size, rank, d_model) — key projection
        """
        U = torch.randn(batch_size, self.d_model, self.rank, device=device, dtype=dtype) * 0.01
        V = torch.randn(batch_size, self.rank, self.d_model, device=device, dtype=dtype) * 0.01
        return U, V
        
    def forward(self, h, state=None, write=False, write_start=0, target_embeddings=None):
        """
        h: (batch_size, seq_len, d_model) veya (batch_size, d_model)
        state: (U, V) bellek durumu tuple'ı
        write_start: sadece bu pozisyondan itibaren belleğe yaz (öncesini atla)
        
        Dönen değerler:
        - h_out: bellek ile zenginleştirilmiş hidden states
        - next_state: güncellenmiş (U, V) bellek durumu
        - gates: gating katsayıları
        """
        is_2d = len(h.shape) == 2
        if is_2d:
            h = h.unsqueeze(1)
            
        batch_size, seq_len, d_model = h.shape
        device = h.device
        
        if state is None:
            state = self.initial_state(batch_size, device, dtype=h.dtype)
            
        U, V = state
        h_out_list = []
        gate_list = []
        
        for t in range(seq_len):
            h_t = h[:, t, :]
            h_t_uns = h_t.unsqueeze(-1)
            
            # 1. Read: query bellekten retrieve et
            #    m_t = U @ V @ h_t  →  eğer (key, value) yazılmışsa, key'e benzeyen
            #    query için value döndürür
            V_h = torch.bmm(V, h_t_uns)
            m_t = torch.bmm(U, V_h).squeeze(-1)
            
            # soft energy stabilization
            m_norm = torch.norm(m_t, p=2, dim=-1, keepdim=True)
            scale = 1.0 / torch.sqrt(1.0 + m_norm**2 + 1e-8)
            m_t = m_t * scale
            
            h_prime_t = h_t + self.config.alpha * m_t
            h_out_list.append(h_prime_t)
            
            # 2. Gating
            g_t = self.gate(h_t, self.W_g, target_embeddings=target_embeddings)
            gate_list.append(g_t)
            
            # 3. Hetero-Associative Write: key=h_t → value=h_{t+1}
            #    Son token'da yazma yapma (h_{t+1} yok)
            should_write = (write and t >= write_start) or (self.config.use_gating and (g_t > 0.5).any())
            has_next = (t < seq_len - 1)
            
            if should_write and has_next:
                h_next = h[:, t + 1, :]           # value: sonraki token
                h_next_uns = h_next.unsqueeze(-1)
                
                z_t = torch.bmm(U.transpose(1, 2), h_t_uns)  # key projection
                U, V = apply_memory_update(
                    U, V, 
                    key_h=h_t_uns,      # key: mevcut token
                    value_h=h_next_uns, # value: sonraki token
                    z_t=z_t, g_t=g_t, config=self.config
                )
                U = apply_norm_constraint(U, self.config)
                V = apply_norm_constraint(V, self.config)
                
        h_out = torch.stack(h_out_list, dim=1)
        gates = torch.cat(gate_list, dim=1)
        
        if is_2d:
            h_out = h_out.squeeze(1)
            
        return h_out, (U, V), gates
