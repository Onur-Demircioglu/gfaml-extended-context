# experiments/exp_02_gfaml.py
import sys
import os

# Çalısma alanı ana dizinini sys.path'e ekle
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from gfaml.config import GFAMLConfig
from gfaml.eval.runner import SUKMCognitiveBrain
from gfaml.datasets import ContinualLearningBenchmark
import torch
import torch.nn.functional as F
import math

def main():
    print("[DENEY] exp_02: GFAML Bilişsel Bellek Modeli Test Ediliyor...")
    bench = ContinualLearningBenchmark()
    task_1, _, _ = bench.get_task_pairs()
    
    config = GFAMLConfig(vocab_size=150, d_model=128, rank=32, eta=1.0)
    # Model D: use_gfaml=True, use_sleep=True
    model = SUKMCognitiveBrain(config, use_gfaml=True, use_sleep=True)
    
    # Task 1 eğit (combined write schema)
    wte = model.base_model.transformer.wte
    for k, v in task_1:
        idx = torch.tensor([[k, v]], dtype=torch.long)
        model(idx, is_writing=True, target_tok=v)
        
    # Değerlendir (Cosine similarity metrics)
    state = model.state_list[0]
    U, V = state
    hits = 0
    wpe = model.base_model.transformer.wpe
    for k, v in task_1:
        # Key representation at position 0
        idx_k = torch.tensor([[k]])
        pos_k = torch.tensor([[0]], dtype=torch.long)
        x_k = wte(idx_k) + wpe(pos_k)
        x_k_norm = model.base_model.transformer.h[0].ln_1(x_k).squeeze()
        
        h_q = x_k_norm.unsqueeze(0).unsqueeze(-1)
        V_h = torch.bmm(V, h_q)
        m_t = torch.bmm(U, V_h).squeeze(-1).squeeze(0)
        
        # Norm stabilization
        m_norm = torch.norm(m_t, p=2)
        m_t = m_t / torch.sqrt(1.0 + m_norm**2 + 1e-8)
        
        # Value representation at position 1
        idx_v = torch.tensor([[v]])
        pos_v = torch.tensor([[1]], dtype=torch.long)
        x_v = wte(idx_v) + wpe(pos_v)
        x_v_norm = model.base_model.transformer.h[0].ln_1(x_v).squeeze()
        
        sim = F.cosine_similarity(m_t.unsqueeze(0), x_v_norm.unsqueeze(0), dim=1).item()
        is_hit = (sim > 0.25)
        if is_hit:
            hits += 1
            
    print(f"   => Task 1 Memory Recall Accuracy: %{hits/len(task_1)*100:.1f} (Beklenen: %100.0 - Hatırladı!)")


if __name__ == "__main__":
    main()
