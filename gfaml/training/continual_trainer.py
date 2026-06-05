# gfaml/training/continual_trainer.py
import random

class ContinualTrainer:
    """
    Ömür boyu sürekli öğrenme görevlerinin karıştırılmasını (shuffling) 
    ve düşmansal besleme sıralamasını yöneten eğitici sınıfı.
    """
    def __init__(self, task_list: list):
        self.task_list = task_list
        
    def get_adversarial_order(self) -> list:
        """
        Geri beslemeli düşmansal görev sırasını döndürür.
        """
        return self.task_list
