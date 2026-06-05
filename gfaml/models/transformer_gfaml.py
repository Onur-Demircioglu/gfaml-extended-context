# gfaml/models/transformer_gfaml.py
import torch
import torch.nn as nn
from gfaml.core.gfaml_layer import GFAMLLayer
from gfaml.models.baseline_transformer import CausalSelfAttention, MLP
from gfaml.config import GFAMLConfig

class TransformerBlockWithGFAML(nn.Module):
    """
    Episodik bellek (GFAML) entegrasyonuna sahip Transformer Decoder bloğu.
    """
    def __init__(self, config: GFAMLConfig, n_head: int = 4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config, n_head)
        
        # GFAML katmanı
        rank = getattr(config, 'rank', 32)
        if hasattr(config, 'gfaml') and isinstance(config.gfaml, dict):
            rank = config.gfaml.get('rank', rank)
        self.gfaml = GFAMLLayer(config.d_model, rank, config=config)
        
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln_1(x)
        attn_out = self.attn(x_norm)
        gfaml_out, _ = self.gfaml(x_norm)
        x = x + attn_out + gfaml_out
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerWithGFAML(nn.Module):
    """
    Episodik bellek (GFAML) katmanlı tam Causal Transformer Modeli.
    """
    def __init__(self, config, n_layer: int = 1, n_head: int = 4):
        super().__init__()
        self.config = config
        
        # Robust config parameters
        if hasattr(config, 'model') and isinstance(config.model, dict):
            vocab_size = config.model.get('vocab_size', 150)
            d_model = config.model.get('d_model', 128)
            n_layer = config.model.get('n_layer', n_layer)
            n_head = config.model.get('n_head', n_head)
        else:
            vocab_size = getattr(config, 'vocab_size', 150)
            d_model = getattr(config, 'd_model', 128)
            n_layer = getattr(config, 'n_layer', n_layer)
            n_head = getattr(config, 'n_head', n_head)
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            wpe = nn.Embedding(4096, d_model),
            h = nn.ModuleList([TransformerBlockWithGFAML(config, n_head) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model)
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
