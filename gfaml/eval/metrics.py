# gfaml/eval/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_representation_drift(x_before: torch.Tensor, x_after: torch.Tensor) -> float:
    """
    Görev kaymaları sonrasında anahtar kelimenin temsil vektöründeki sapmayı/ drifti hesaplar.
    Drift = 1 - cosine_similarity(before, after)
    """
    return 1.0 - F.cosine_similarity(x_before.unsqueeze(0), x_after.unsqueeze(0), dim=1).item()

def calculate_temporal_consistency(p_0: torch.Tensor, p_2: torch.Tensor) -> float:
    """
    Görev kaymaları boyunca tahmin olasılık dağılımlarının zamansal tutarlılığını (TCS) ölçer.
    TCS = cosine_similarity(p_0, p_2)
    """
    return F.cosine_similarity(p_0.unsqueeze(0), p_2.unsqueeze(0), dim=1).item()

def simulate_gradient_interference(model, idx_1, target_1, idx_3, target_3) -> float:
    """
    Task 1 ve Task 3 güncellemeleri arasındaki gradyan çakışmasını (Gradient Interference) hesaplar.
    """
    # Task 1 gradyanlarını hesapla
    model.optimizer.zero_grad()
    logits_1, _ = model(idx_1, is_writing=True, target_tok=target_1)
    loss_1 = nn.CrossEntropyLoss()(logits_1[0, -1, :].unsqueeze(0), torch.tensor([target_1], device=idx_1.device))
    loss_1.backward()
    grad_1 = torch.cat([model.cortex.lora_A.grad.flatten(), model.cortex.lora_B.grad.flatten()])
    
    # Task 3 gradyanlarını hesapla
    model.optimizer.zero_grad()
    logits_3, _ = model(idx_3, is_writing=True, target_tok=target_3)
    loss_3 = nn.CrossEntropyLoss()(logits_3[0, -1, :].unsqueeze(0), torch.tensor([target_3], device=idx_3.device))
    loss_3.backward()
    grad_3 = torch.cat([model.cortex.lora_A.grad.flatten(), model.cortex.lora_B.grad.flatten()])
    
    # Cosine similarity ve parazit skoru (1 - benzerlik)
    grad_sim = F.cosine_similarity(grad_1.unsqueeze(0), grad_3.unsqueeze(0), dim=1).item()
    return 1.0 - grad_sim
