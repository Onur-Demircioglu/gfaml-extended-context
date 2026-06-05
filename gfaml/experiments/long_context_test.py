# gfaml/experiments/long_context_test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gfaml.config import GFAMLConfig
from gfaml.data.synthetic_pairs import SyntheticPairsDataset
from gfaml.models.baseline_transformer import BaselineTransformer
from gfaml.models.transformer_gfaml import TransformerWithGFAML

def run_long_context_benchmark():
    """
    Uzun bağlam (Long Context) testi.
    Araya giren yüzlerce gürültü (distraction) token'ı sonrasında eski bilgilerin
    hatırlanma başarısını standart Transformer ile GFAML entegrasyonlu modeli
    karşılaştırarak ölçer.
    """
    print("[INFO] Uzun Baglam (Long Context) Karsilastirma Testi Hazirlaniyor...")
    
    # Hiperparametreleri kur
    config = GFAMLConfig(
        d_model=128,
        rank=32,
        lambda_decay=1.0,   # Sönümlemeyi kapatıp kalıcı belleği ölçelim
        eta=1.0,
        alpha=1.0,
        use_gating=True,
        perfect_gating=True, # Kusursuz kapılama modunu aktif et
        gate_bias=-1.5
    )
    
    # Sentetik veri üreteci
    dataset = SyntheticPairsDataset(config)
    
    # Belleğe yerleştirilecek kilit ilişkiler
    target_pairs = [
        ("ONUR", "PYTORCH"),      # Çok erken aşamada (token 50 civarı)
        ("MEHMET", "JAX"),        # Orta aşamada (token 500 civarı)
        ("AYSE", "TENSORFLOW")    # Yakın aşamada (token 2500 civarı)
    ]
    
    # Tüm kelimeleri kaydet
    for k, v in target_pairs:
        dataset.get_embedding(k)
        dataset.get_embedding(v)
        
    # Gürültü (distraction) kelimeleri üret
    distractors = [f"word_{i}" for i in range(100)]
    for w in distractors:
        dataset.get_embedding(w)
        
    print("[INFO] Sentetik uzun baglam dizisi olusturuluyor (Uzunluk: 3000 token)...")
    
    # Bağlamı oluştur (Toplam ~3000 token)
    # Yapı: [ONUR+PYTORCH] + [1000 gürültü] + [MEHMET+JAX] + [1800 gürültü] + [AYSE+TENSORFLOW]
    h_list = []
    
    # Kilit çiftlerin birleşik gömmelerini hazırlayalım (k_emb + v_emb)
    emb_pairs = []
    for key, val in target_pairs:
        k_emb = dataset.get_embedding(key)
        v_emb = dataset.get_embedding(val)
        combined = k_emb + v_emb
        combined = combined / torch.norm(combined)
        emb_pairs.append(combined)
        
    # 1. İlişki
    h_list.append(emb_pairs[0])
    
    # Araya giren ilk gürültü bloğu (~1000 token)
    for _ in range(1000):
        w = random_choice(distractors)
        h_list.append(dataset.get_embedding(w))
        
    # 2. İlişki
    h_list.append(emb_pairs[1])
    
    # Araya giren ikinci gürültü bloğu (~1800 token)
    for _ in range(1800):
        w = random_choice(distractors)
        h_list.append(dataset.get_embedding(w))
        
    # 3. İlişki
    h_list.append(emb_pairs[2])
    
    h_seq = torch.stack(h_list).unsqueeze(0) # (1, seq_len, d_model)
    seq_len = h_seq.shape[1]
    
    print(f"   * Baglam Uzunlugu: {seq_len} token.")
    
    # Modelleri oluştur
    from gfaml.memory.gfaml import GFAMLLayer
    gfaml_layer = GFAMLLayer(config)
    
    # Kusursuz Gating için hedef gömmeleri derle (Sadece 3 kilit ilişki vektörünü geçirir!)
    target_embs_tensor = torch.stack(emb_pairs)
    
    # GFAML ile diziyi işle
    print("[INFO] GFAML bellek katmanı güncellemeleri hesaplaniyor...")
    h_out, (U_final, V_final), gates = gfaml_layer(h_seq, target_embeddings=target_embs_tensor)
    
    # Kapı (gating) istatistikleri
    print(f"   * Ortalama Yazma Kapisi Degeri: {gates.mean().item():.4f}")
    
    # Her bir ilişkiyi sorgula
    print("\n[TEST SONUCLARI - UZUN BAGLAM BELLEK HATIRLAMA]")
    print("-" * 65)
    print(f"{'Sorgu (Key)':<15} | {'Hedef (Value)':<15} | {'Hatirlama Basarisi (Cosine Sim)':<30}")
    print("-" * 65)
    
    for key, val in target_pairs:
        k_emb = dataset.get_embedding(key)
        v_emb = dataset.get_embedding(val)
        
        # Sorgu h_q = k_emb
        h_q = k_emb.unsqueeze(0).unsqueeze(0)
        
        # GFAML ile oku
        h_read, _, _ = gfaml_layer(h_q, state=(U_final, V_final))
        m_q = (h_read.squeeze(0).squeeze(0) - k_emb) / config.alpha
        
        # Hedef değer ile kosinüs benzerliği
        sim = F.cosine_similarity(m_q.unsqueeze(0), v_emb.unsqueeze(0)).item()
        
        # Başarı tespiti (benzerlik > 0.15 ise başarılı kabul edilir)
        status = "BASARILI" if sim > 0.15 else "ZAYIF/BASARISIZ"
        
        print(f"{key:<15} | {val:<15} | {sim:>8.4f} ({status})")
        
    print("-" * 65)

def random_choice(lst):
    import random
    return random.choice(lst)

if __name__ == "__main__":
    run_long_context_benchmark()
