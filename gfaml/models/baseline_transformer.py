# gfaml/models/baseline_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gfaml.config import GFAMLConfig

class CausalSelfAttention(nn.Module):
    """
    GPT tarzı nedensel (causal) çoklu kafalı öz-dikkat (Multi-Head Self-Attention) bloğu.
    """
    def __init__(self, config: GFAMLConfig, n_head: int = 4):
        super().__init__()
        assert config.d_model % n_head == 0
        self.d_model = config.d_model
        self.n_head = n_head
        self.head_dim = config.d_model // n_head
        
        # Q, K, V projeksiyonları
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        
        # Nedensel Maske (Causal Mask)
        self.register_buffer("bias", torch.tril(torch.ones(4096, 4096)).view(1, 1, 4096, 4096))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Q, K, V hesaplama
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Kafalara bölme: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Nedensel Maskeleme uygula
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Kafaları geri birleştir
        
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config: GFAMLConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config: GFAMLConfig, n_head: int = 4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config, n_head)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class BaselineTransformer(nn.Module):
    """
    Karşılaştırma için kullanılacak standart Transformer Decoder mimarisi.
    """
    def __init__(self, config: GFAMLConfig, n_layer: int = 2, n_head: int = 4):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(4096, config.d_model),
            h = nn.ModuleList([TransformerBlock(config, n_head) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(config.d_model)
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Embedding ve Output weights bağlama (weight sharing)
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        
        # Token ve pozisyon gömmelerini topla
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
