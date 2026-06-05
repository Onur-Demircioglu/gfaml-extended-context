# gfaml/memory/update_rules.py
import torch
from gfaml.config import GFAMLConfig

def apply_memory_update(U: torch.Tensor, V: torch.Tensor, 
                        h_t: torch.Tensor, z_t: torch.Tensor, 
                        g_t: torch.Tensor, config: GFAMLConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    U ve V matrislerini gradyansız (Hebbian dış çarpım) kuralına göre günceller.
    
    U: (batch_size, d_model, rank)
    V: (batch_size, rank, d_model)
    h_t: (batch_size, d_model, 1)
    z_t: (batch_size, rank, 1)
    g_t: (batch_size, 1) - Kapı değeri (Gating coefficient)
    """
    # Gating katsayısını çarpım için genişletelim: (batch_size, 1, 1)
    g_t_uns = g_t.unsqueeze(-1)
    
    # U güncellemesi: lambda * U_t + eta * g_t * (h_t * z_t^T)
    # h_t * z_t^T -> (batch_size, d_model, rank)
    delta_U = torch.bmm(h_t, z_t.transpose(1, 2))
    U_next = config.lambda_decay * U + config.eta * g_t_uns * delta_U
    
    # V güncellemesi: lambda * V_t + eta * g_t * (z_t * h_t^T)
    # z_t * h_t^T -> (batch_size, rank, d_model)
    delta_V = torch.bmm(z_t, h_t.transpose(1, 2))
    V_next = config.lambda_decay * V + config.eta * g_t_uns * delta_V
    
    return U_next, V_next

def apply_norm_constraint(W: torch.Tensor, config: GFAMLConfig) -> torch.Tensor:
    """
    Bellek taşmasını (Saturation) önlemek için norm sınırlama kuralını uygular.
    """
    if config.use_hard_clamp:
        # Eleman seviyesinde sert sınırlama (clamp)
        return torch.clamp(W, min=-config.saturation_threshold, max=config.saturation_threshold)
    else:
        # Frobenius Norm tabanlı yumuşak ölçekleme
        # Her batch elemanı için matris normunu hesapla: (batch_size, 1, 1)
        norm = torch.norm(W, p='fro', dim=(1, 2), keepdim=True)
        scale = torch.clamp(norm / config.saturation_threshold, min=1.0)
        return W / scale
