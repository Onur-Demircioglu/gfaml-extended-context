# experiments/exp_03_ablation.py
import sys
import os

# Çalısma alanı ana dizinini sys.path'e ekle
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from benchmarks.continual_learning.run_full_benchmark import main

if __name__ == "__main__":
    print("[DENEY] Ablasyon Matrisi Deneyi (A, B, C, D) Başlatılıyor...")
    main()
