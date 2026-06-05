# gfaml/evaluation/drift.py
import torch
import torch.nn.functional as F

def calculate_representation_drift(x_before: torch.Tensor, x_after: torch.Tensor) -> float:
    """
    Görev kaymaları sonrasında embedding temsillerinin kayma (drift) oranını hesaplar.
    Drift = 1 - cosine_similarity(before, after)
    """
    return 1.0 - F.cosine_similarity(x_before.unsqueeze(0), x_after.unsqueeze(0), dim=1).item()
