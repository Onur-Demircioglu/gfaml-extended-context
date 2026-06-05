# scratch/test_norm_math.py
import torch

m_t = torch.randn(1, 896, dtype=torch.bfloat16, device="cuda") * 1000.0
m_norm = torch.norm(m_t, p=2, dim=-1, keepdim=True)
scale = 1.0 / torch.sqrt(1.0 + m_norm**2 + 1e-8)
m_t_scaled = m_t * scale

print("Original norm:", m_norm.item())
print("Scaled norm:", torch.norm(m_t_scaled).item())
