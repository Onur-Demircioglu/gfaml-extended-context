# gfaml/evaluation/forgetting.py

def calculate_forgetting_rate(acc_before: float, acc_after: float) -> float:
    """
    Önceki doğruluğu ve sonraki doğruluğu kıyaslayarak unutma oranını (FR) hesaplar.
    FR = Acc(Before) - Acc(After)
    """
    return max(0.0, acc_before - acc_after)
