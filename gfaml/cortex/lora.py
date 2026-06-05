# gfaml/cortex/lora.py
import torch
import torch.nn as nn

class LoRA(nn.Module):
    """
    Parametrik yavaş korteks LoRA katmanı.
    """
    def __init__(self, d_model: int, rank: int, vocab_size: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, vocab_size) * 0.01)

    @property
    def lora_A(self):
        return self.A

    @property
    def lora_B(self):
        return self.B

    def forward(self, x):
        return (x @ self.A) @ self.B

