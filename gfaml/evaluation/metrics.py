# gfaml/evaluation/metrics.py
import torch

def accuracy(pred, target):
    """
    Doğruluk oranını hesaplar.
    """
    return (pred == target).float().mean().item()

def forgetting(before, after):
    """
    Unutma oranını (FR) hesaplar.
    """
    return before - after
