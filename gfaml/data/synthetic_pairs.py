# gfaml/data/synthetic_pairs.py
import torch
import random
from gfaml.config import GFAMLConfig

class SyntheticPairsDataset:
    """
    İlişkisel bellek testleri için sentetik veri üreteci.
    Kelimeleri yüksek boyutlu, birbirine dik (orthogonal) birim vektörlere eşler.
    """
    def __init__(self, config: GFAMLConfig):
        self.config = config
        self.vocab = {}
        self.embeddings = {}
        
        # Özel token'ları ekleyelim
        self._get_or_create_word("->")
        self._get_or_create_word("<SEP>")
        self._get_or_create_word("<PAD>")

    def _get_or_create_word(self, word: str) -> torch.Tensor:
        """
        Kelimeye ait benzersiz bir yüksek boyutlu birim vektör üretir.
        """
        if word not in self.vocab:
            # Rastgele normal dağılımdan vektör üret
            vec = torch.randn(self.config.d_model)
            # Birim vektör haline getir (diklik olasılığını artırmak için)
            vec = vec / torch.norm(vec)
            
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.embeddings[word] = vec
            
        return self.embeddings[word]

    def get_embedding(self, word: str) -> torch.Tensor:
        return self._get_or_create_word(word)

    def generate_pairs(self, num_pairs: int) -> list[tuple[str, str]]:
        """
        Rastgele anahtar-değer çiftleri üretir.
        Örn: [("ONUR", "PYTORCH"), ("AYSE", "TENSORFLOW")]
        """
        keys = [f"KEY_{i}" for i in range(num_pairs)]
        values = [f"VAL_{i}" for i in range(num_pairs)]
        
        # Benzersiz vektörleri oluştur
        for k in keys:
            self._get_or_create_word(k)
        for v in values:
            self._get_or_create_word(v)
            
        return list(zip(keys, values))

    def create_batch_combined(self, pairs: list[tuple[str, str]]) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Birleşik Vektör Şeması (Summed Representation):
        Her bir çift için h_t = key_emb + val_emb beslenir.
        
        Dönen değerler:
        - X: (1, num_pairs, d_model) -> Belleğe yazılacak birleşik vektörler dizisi
        - key_embs: Anahtar gömmeleri listesi
        - val_embs: Değer gömmeleri listesi
        """
        h_list = []
        key_embs = []
        val_embs = []
        
        for k, v in pairs:
            k_emb = self.get_embedding(k)
            v_emb = self.get_embedding(v)
            
            # h_t = k_emb + v_emb
            h_t = k_emb + v_emb
            h_t = h_t / torch.norm(h_t) # Birim vektör yap
            
            h_list.append(h_t)
            key_embs.append(k_emb)
            val_embs.append(v_emb)
            
        X = torch.stack(h_list).unsqueeze(0) # (1, num_pairs, d_model)
        return X, key_embs, val_embs

    def create_batch_sequential(self, pairs: list[tuple[str, str]], query_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ardışık Şema (Sequential Schema):
        Tokens: KEY_0, ->, VAL_0, <SEP>, KEY_1, ->, VAL_1, <SEP> ...
        Ardından sorgu için: QUERY_KEY, -> beslenir.
        
        Dönen değerler:
        - X_write: (1, write_seq_len, d_model) -> Belleğe yazma akışı
        - X_query: (1, 2, d_model) -> Sorgu akışı ("KEY", "->")
        - y_target: (d_model,) -> Beklenen değer vektörü
        """
        write_tokens = []
        for k, v in pairs:
            write_tokens.extend([k, "->", v, "<SEP>"])
            
        # Gömmeleri al
        write_embs = [self.get_embedding(tok) for tok in write_tokens]
        X_write = torch.stack(write_embs).unsqueeze(0) # (1, write_seq_len, d_model)
        
        # Sorgu
        query_embs = [self.get_embedding(query_key), self.get_embedding("->")]
        X_query = torch.stack(query_embs).unsqueeze(0) # (1, 2, d_model)
        
        # Hedef (Target Value)
        # pairs içinden query_key eşini bul
        target_val = None
        for k, v in pairs:
            if k == query_key:
                target_val = v
                break
        
        y_target = self.get_embedding(target_val)
        
        return X_write, X_query, y_target
