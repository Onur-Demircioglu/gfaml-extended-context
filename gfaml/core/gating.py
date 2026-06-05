# gfaml/core/gating.py
import torch

def surprise_score(pred, target):
    """
    Tahmin ile hedef arasındaki sürpriz ( prediction error) skorunu hesaplar.
    """
    return torch.mean((pred - target) ** 2)
