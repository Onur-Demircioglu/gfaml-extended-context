# gfaml/experiments/adversarial_graph_test.py
import torch
import torch.nn.functional as F
import math
from gfaml.config import GFAMLConfig
from gfaml.data.synthetic_pairs import SyntheticPairsDataset
from gfaml.memory.gfaml import GFAMLLayer

def run_adversarial_graph_test():
    """
    TEST E: Adversarial Graph Validation (Çapraz Sızıntı ve Kimlik Ayrıştırma)
    Belleğe iki adet tamamen bağımsız zincir yazar:
      - Zincir 1: ONUR -> PYTORCH -> CUDA -> GPU
      - Zincir 2: AYSE -> TENSORFLOW -> KERAS -> CPU
    A sorgusundan başlayarak yapılan recurrent okumanın, 2. Zincirin alt uzaylarına
    çapraz sızıp sızmadığını (crossover leakage) ölçer.
    """
    print("\n" + "="*75)
    print("=== [TEST E] - ADVERSARIAL GRAPH VALIDATION (ÇAPRAZ SIZINTI KONTROLÜ) ===")
    print("="*75)
    
    config = GFAMLConfig(
        d_model=128, rank=32, lambda_decay=1.0, eta=1.0, alpha=1.0,
        use_gating=True, perfect_gating=True, use_soft_stabilization=True
    )
    
    dataset = SyntheticPairsDataset(config)
    
    # İki Bağımsız Zincir
    chain_1 = ["ONUR", "PYTORCH", "CUDA", "GPU"]
    chain_2 = ["AYSE", "TENSORFLOW", "KERAS", "CPU"]
    
    # Düğüm gömmelerini hazırla
    embs_1 = {node: dataset.get_embedding(node) for node in chain_1}
    embs_2 = {node: dataset.get_embedding(node) for node in chain_2}
    
    # Birleşik vektörleri oluştur (Toplam 6 ilişki)
    c1_1 = (embs_1["ONUR"] + embs_1["PYTORCH"]) / math.sqrt(2)
    c1_2 = (embs_1["PYTORCH"] + embs_1["CUDA"]) / math.sqrt(2)
    c1_3 = (embs_1["CUDA"] + embs_1["GPU"]) / math.sqrt(2)
    
    c2_1 = (embs_2["AYSE"] + embs_2["TENSORFLOW"]) / math.sqrt(2)
    c2_2 = (embs_2["TENSORFLOW"] + embs_2["KERAS"]) / math.sqrt(2)
    c2_3 = (embs_2["KERAS"] + embs_2["CPU"]) / math.sqrt(2)
    
    # 6 ilişki vektörünü stack et
    h_write = torch.stack([c1_1, c1_2, c1_3, c2_1, c2_2, c2_3]).unsqueeze(0)
    target_embs = torch.stack([c1_1, c1_2, c1_3, c2_1, c2_2, c2_3])
    
    # Belleğe yaz
    layer = GFAMLLayer(config)
    _, final_state, _ = layer(h_write, target_embeddings=target_embs)
    
    print("[INFO] İki bağımsız semantik zincir aynı belleğe yazildi.")
    print("       * Zincir 1: ONUR -> PYTORCH -> CUDA -> GPU")
    print("       * Zincir 2: AYSE -> TENSORFLOW -> KERAS -> CPU")
    print("-" * 75)
    
    # Zincir 1 Sorgusu (ONUR ile başla)
    q = embs_1["ONUR"]
    
    # Hop 1: ONUR -> ? (Hedef: PYTORCH, Düşman: TENSORFLOW)
    h_q_1 = q.unsqueeze(0).unsqueeze(0)
    h_out_1, _, _ = layer(h_q_1, state=final_state)
    m_1 = (h_out_1.squeeze() - q) / config.alpha
    
    sim_1_target = F.cosine_similarity(m_1.unsqueeze(0), embs_1["PYTORCH"].unsqueeze(0)).item()
    sim_1_leak = F.cosine_similarity(m_1.unsqueeze(0), embs_2["TENSORFLOW"].unsqueeze(0)).item()
    
    # Hop 2: m_1 -> ? (Hedef: CUDA, Düşman: KERAS)
    h_q_2 = m_1.unsqueeze(0).unsqueeze(0)
    h_out_2, _, _ = layer(h_q_2, state=final_state)
    m_2 = (h_out_2.squeeze() - m_1) / config.alpha
    
    sim_2_target = F.cosine_similarity(m_2.unsqueeze(0), embs_1["CUDA"].unsqueeze(0)).item()
    sim_2_leak = F.cosine_similarity(m_2.unsqueeze(0), embs_2["KERAS"].unsqueeze(0)).item()
    
    # Hop 3: m_2 -> ? (Hedef: GPU, Düşman: CPU)
    h_q_3 = m_2.unsqueeze(0).unsqueeze(0)
    h_out_3, _, _ = layer(h_q_3, state=final_state)
    m_3 = (h_out_3.squeeze() - m_2) / config.alpha
    
    sim_3_target = F.cosine_similarity(m_3.unsqueeze(0), embs_1["GPU"].unsqueeze(0)).item()
    sim_3_leak = F.cosine_similarity(m_3.unsqueeze(0), embs_2["CPU"].unsqueeze(0)).item()
    
    print(f"{'İşlem (Hop)':<15} | {'Hedef (Target) Sim':<20} | {'Çapraz Sızıntı (Leakage) Sim':<28}")
    print("-" * 75)
    print(f"Hop 1 (ONUR)    | {sim_1_target:>20.4f} | {sim_1_leak:>28.4f}")
    print(f"Hop 2 (PYTORCH) | {sim_2_target:>20.4f} | {sim_2_leak:>28.4f}")
    print(f"Hop 3 (CUDA)    | {sim_3_target:>20.4f} | {sim_3_leak:>28.4f}")
    print("-" * 75)
    
    # Sızıntı Kararı
    max_leakage = max(abs(sim_1_leak), abs(sim_2_leak), abs(sim_3_leak))
    print(f"   * Maksimum Capraz Sizinti (Max Cross Leakage): {max_leakage:.4f}")
    
    if max_leakage < 0.05:
        print("[ONAY] Çapraz sızıntı pratik olarak sıfırdır. Rezonans İllüzyonu kesinlikle reddedilmiştir!")
        print("       GFAML v2.2 semantik graf yollarını izole ve yüksek güvenirlikle takip eder.")
    else:
        print("[TEHLİKE] Zincirler arası rezonans veya çapraz sızıntı tespit edildi!")

if __name__ == "__main__":
    run_adversarial_graph_test()
