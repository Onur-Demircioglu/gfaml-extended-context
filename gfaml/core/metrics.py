# gfaml/core/metrics.py

def forgetting_rate(acc_before, acc_after):
    """
    Hesaplanan unutma oranı (Forgetting Rate).
    """
    return acc_before - acc_after

def memory_contribution_gain(baseline_scores, gfaml_scores):
    """
    GFAML modelinin baseline modele göre bellek katkı kazancı (Memory Contribution Gain).
    """
    return sum(gfaml_scores) - sum(baseline_scores)

def compute_metrics(history):
    """
    Eğitim geçmişinden metrikleri hesaplar.
    """
    return {
        "FR": history.get("forgetting", []),
        "MCG": history.get("mcg", []),
    }
