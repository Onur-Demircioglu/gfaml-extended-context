# gfaml/benchmarks/continual_bench.py
import torch

class ContinualLearningBenchmark:
    """
    Continual Learning değerlendirmesi için gerçekçi Mini Continual Bench veri yükleyici.
    Üç farklı bilişsel alanı (QA, Wiki Facts ve GSM8K Matematiksel Akıl Yürütme) barındırır.
    """
    def __init__(self):
        # 1. GERÇEKÇİ BENCHMARK VERİ YAPISI
        self.qa_data = {
            "Who_wrote_Hamlet": "Shakespeare",
            "Who_discovered_Radium": "Marie_Curie",
            "Who_founded_Apple": "Steve_Jobs"
        }
        
        self.facts_data = {
            "Paris": "capital_of_France",
            "Tokyo": "capital_of_Japan",
            "Berlin": "capital_of_Germany"
        }
        
        self.reasoning_data = {
            "Ten_toys_per_hour_makes_in_three_hours": "Thirty",
            "Double_of_eight_plus_four_equals": "Twenty",
            "Five_times_six_minus_ten_is": "Twenty"
        }
        
        # Benzersiz kelimeleri topla ve kelime havuzu oluştur
        all_words = set()
        for d in [self.qa_data, self.facts_data, self.reasoning_data]:
            for k, v in d.items():
                all_words.add(k)
                all_words.add(v)
                
        self.unique_words = sorted(list(all_words))
        self.vocab = {word: idx + 10 for idx, word in enumerate(self.unique_words)}
        
    def get_task_pairs(self) -> tuple[list, list, list]:
        """
        Üç göreve ait token çiftlerini (key_id, val_id) döndürür.
        """
        task_a = [(self.vocab[k], self.vocab[v]) for k, v in self.qa_data.items()]
        task_b = [(self.vocab[k], self.vocab[v]) for k, v in self.facts_data.items()]
        task_c = [(self.vocab[k], self.vocab[v]) for k, v in self.reasoning_data.items()]
        return task_a, task_b, task_c
