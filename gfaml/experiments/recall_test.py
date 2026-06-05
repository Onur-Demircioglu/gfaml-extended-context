# gfaml/experiments/recall_test.py
import torch
import torch.nn.functional as F
from gfaml.config import GFAMLConfig
from gfaml.data.synthetic_pairs import SyntheticPairsDataset
from gfaml.memory.gfaml import GFAMLLayer

def run_recall_benchmark(num_pairs: int, config: GFAMLConfig):
    """
    Belirli sayıda ilişki çifti için GFAML katmanının hatırlama doğruluğunu ölçer.
    """
    # Veri setini oluştur
    dataset = SyntheticPairsDataset(config)
    pairs = dataset.generate_pairs(num_pairs)
    
    # Katmanı oluştur
    layer = GFAMLLayer(config)
    
    # Yazma aşaması (Birleşik şema kullanarak)
    X_write, key_embs, val_embs = dataset.create_batch_combined(pairs)
    
    # Belleğe yaz
    _, final_state, _ = layer(X_write)
    
    # Sorgu ve Hatırlama Aşaması
    correct_recalls = 0
    total_cosine_sim = 0.0
    
    # Tüm değer gömmelerini bir matris haline getirelim: (num_pairs, d_model)
    val_matrix = torch.stack(val_embs)
    
    for i, (key, val) in enumerate(pairs):
        k_emb = key_embs[i] # (d_model,)
        v_emb = val_embs[i] # (d_model,)
        
        # Sadece bu anahtar ile bellekten okuma yapalım (state koruyarak)
        # h_q = k_emb -> (1, 1, d_model)
        h_q = k_emb.unsqueeze(0).unsqueeze(0)
        
        # forward çağrısında sadece okuma yapar, yeni durum güncellense de biz final_state'i bozmayız
        h_out, _, _ = layer(h_q, state=final_state)
        
        # Okunan bellek vektörü (readout): m_q = (h_out - h_q) / alpha
        m_q = (h_out.squeeze(0).squeeze(0) - k_emb) / config.alpha
        
        # Tüm değerlerle olan kosinüs benzerliğini hesaplayalım
        # m_q: (d_model,), val_matrix: (num_pairs, d_model)
        cos_sims = F.cosine_similarity(m_q.unsqueeze(0), val_matrix, dim=1)
        
        # En yakın değerin indeksini bul
        pred_idx = torch.argmax(cos_sims).item()
        
        # Doğruluk kontrolü
        if pred_idx == i:
            correct_recalls += 1
            
        total_cosine_sim += cos_sims[i].item()
        
    accuracy = (correct_recalls / num_pairs) * 100
    avg_cosine = total_cosine_sim / num_pairs
    
    print(f"[TEST - {num_pairs} Iliski]")
    print(f"   * Recall Accuracy: %{accuracy:.2f}")
    print(f"   * Avg Target Cosine Similarity: {avg_cosine:.4f}")
    print("-" * 50)
    
    return accuracy, avg_cosine

if __name__ == "__main__":
    print("[INFO] GFAML Ilk Bellek Hatirlama (Recall) Benchmark Testi Basliyor...\n")
    
    # Farklı boyutlardaki deneyleri çalıştıralım
    test_sizes = [10, 100, 1000]
    
    # Geniş bir model boyutu ve rank tanımlayalım
    # (Büyük kapasiteleri ölçmek için config değerlerini güncelleyelim)
    config = GFAMLConfig(
        d_model=128,
        rank=32,          # En uygun rank (Grid Search ile belirlendi)
        lambda_decay=1.0, # Bellek sönümlemesini kapatıp saf kapasiteyi ölçelim
        eta=1.0,          # En uygun yazma hızı (Grid Search ile belirlendi)
        alpha=1.0,
        use_gating=False  # Saf kapasite testi için gating kapalı
    )
    
    for size in test_sizes:
        run_recall_benchmark(size, config)
