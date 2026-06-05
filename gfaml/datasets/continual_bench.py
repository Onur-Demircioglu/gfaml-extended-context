# gfaml/datasets/continual_bench.py
import torch

class RealContinualDatasetPipeline:
    """
    Sürekli Öğrenme (Continual Learning) araştırmaları için gerçekçi veri seti boru hattı.
    1. SQuAD-lite QA (Soru -> Cevap)
    2. Wiki Factual Triples (Entity -> Relation -> Entity)
    3. GSM8K Reasoning Chains (Basit Matematiksel Akıl Yürütme)
    4. Sentetik Rastgele İlişkiler (Synthetic Relational Pairs)
    """
    def __init__(self, vocab_size=150):
        # 1. Ham Veri Grupları
        self.qa_data = [
            ("Who wrote Hamlet?", "Shakespeare"),
            ("Who discovered Radium?", "Curie"),
            ("Who founded Apple?", "Jobs"),
            ("What is the capital of Spain?", "Madrid"),
            ("Who painted Mona Lisa?", "DaVinci")
        ]
        
        self.facts_data = [
            ("Paris is the capital of", "France"),
            ("Tokyo is the capital of", "Japan"),
            ("Berlin is the capital of", "Germany"),
            ("Rome is the capital of", "Italy"),
            ("London is the capital of", "UK")
        ]
        
        self.reasoning_data = [
            ("Double eight plus four makes", "Twenty"),
            ("Five times six minus ten is", "Twenty"),
            ("Ten divided by two plus three is", "Eight"),
            ("Three times four plus two is", "Fourteen"),
            ("Square of five minus five makes", "Twenty")
        ]

        self.synthetic_data = [
            ("KEY_ALPHA", "VAL_ALPHA"),
            ("KEY_BETA", "VAL_BETA"),
            ("KEY_GAMMA", "VAL_GAMMA"),
            ("KEY_DELTA", "VAL_DELTA"),
            ("KEY_EPSILON", "VAL_EPSILON")
        ]
        
        self.paraphrased_qa_data = [
            ("Hamlet was written by whom", "Shakespeare"),
            ("Radium was discovered by which scientist", "Curie"),
            ("Which company was founded by Steve Jobs", "Jobs"),
            ("What is Spains capital city", "Madrid"),
            ("Who is the creator of Mona Lisa", "DaVinci")
        ]
        
        # 2. Ortak Kelime Dağarcığı (Vocabulary Builder)
        all_words = set()
        for dataset in [self.qa_data, self.facts_data, self.reasoning_data, self.synthetic_data, self.paraphrased_qa_data]:
            for q, a in dataset:
                for word in q.replace("?", "").replace(".", "").split():
                    all_words.add(word)
                all_words.add(a)
                
        # Özel belirteçleri ekle
        self.special_tokens = ["<PAD>", "<UNK>", "->"]
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        
        # Kelimeleri vocab'e ekle
        for idx, word in enumerate(sorted(list(all_words))):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                
        # Gerekirse vocab_size'a kadar pad'le
        self.vocab_size = max(vocab_size, len(self.vocab))

    def _tokenize(self, sentence: str) -> list[int]:
        """
        Cümleyi tokenize edip id dizisine çevirir.
        """
        words = sentence.replace("?", "").replace(".", "").split()
        return [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]

    def get_task_batches(self, task_type="qa") -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Seçilen göreve ait token çiftlerini (x, y) PyTorch tensor batch'leri olarak döndürür.
        x: (batch_size, seq_len) -> Soru/Girdi token dizisi
        y: (batch_size,) -> Hedef/Cevap token ID'si
        """
        if task_type == "qa":
            data = self.qa_data
        elif task_type == "paraphrase_qa":
            data = self.paraphrased_qa_data
        elif task_type == "facts":
            data = self.facts_data
        elif task_type == "reasoning":
            data = self.reasoning_data
        else:
            data = self.synthetic_data
            
        batches = []
        for q, a in data:
            q_ids = self._tokenize(q)
            a_id = self.vocab.get(a, self.vocab["<UNK>"])
            
            # x = soru + "->" ayıracı (seq_len, )
            x_ids = q_ids + [self.vocab["->"]]
            
            x_tensor = torch.tensor(x_ids, dtype=torch.long).unsqueeze(0) # (1, seq_len)
            y_tensor = torch.tensor([a_id], dtype=torch.long) # (1,)
            
            batches.append((x_tensor, y_tensor))
            
        return batches


# Geriye dönük uyumluluk ve basit iskelet fonksiyonları
def get_task_1():
    return [(10, 11), (12, 13)]

def get_task_2():
    return [(20, 21), (22, 23)]

def get_task_3():
    return [(30, 31), (32, 33)]
