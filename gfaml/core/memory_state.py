# gfaml/core/memory_state.py
import torch
from gfaml.config import GFAMLConfig

def apply_memory_update(U: torch.Tensor, V: torch.Tensor, 
                        key_h: torch.Tensor, value_h: torch.Tensor,
                        z_t: torch.Tensor, 
                        g_t: torch.Tensor, config: GFAMLConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hetero-associative Hebbian update: key → value eşlemesi kurar.
    
    U: (batch_size, d_model, rank)  — value reconstruction matrix
    V: (batch_size, rank, d_model)  — key projection matrix
    key_h: (batch_size, d_model, 1)   — anahtar vektör (mevcut token h_t)
    value_h: (batch_size, d_model, 1) — değer vektör (sonraki token h_{t+1})
    z_t: (batch_size, rank, 1)        — key'in rank-space projeksiyonu
    g_t: (batch_size, 1)              — gating katsayısı
    
    Read sırasında: m = U @ V @ query  →  query'ye en yakın key'in value'sunu döndürür.
    """
    g_t_uns = g_t.unsqueeze(-1) # (batch_size, 1, 1)
    
    # U value'yu kodlar: U += η * g * (value_h ⊗ z_t^T)
    # Read'de U @ z → value_h'ye yakın bir vektör üretir
    delta_U = torch.bmm(value_h, z_t.transpose(1, 2))
    U_next = config.lambda_decay * U + config.eta * g_t_uns * delta_U
    
    # V key'i kodlar: V += η * g * (z_t ⊗ key_h^T)
    # Read'de V @ query → key ile eşleşme skoru üretir
    delta_V = torch.bmm(z_t, key_h.transpose(1, 2))
    V_next = config.lambda_decay * V + config.eta * g_t_uns * delta_V
    
    return U_next, V_next

def apply_norm_constraint(W: torch.Tensor, config: GFAMLConfig) -> torch.Tensor:
    """
    Doyumu (saturation) ve sayısal patlamayı önlemek için Frobenius norm sınırlaması uygular.
    """
    if config.use_hard_clamp:
        return torch.clamp(W, min=-config.saturation_threshold, max=config.saturation_threshold)
    else:
        norm = torch.norm(W, p='fro', dim=(1, 2), keepdim=True)
        scale = torch.clamp(norm / config.saturation_threshold, min=1.0)
        return W / scale
