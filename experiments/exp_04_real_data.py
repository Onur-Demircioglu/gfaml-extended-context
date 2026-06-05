# experiments/exp_04_real_data.py
import sys
import os

# Çalısma alanı ana dizinini sys.path'e ekle
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from benchmarks.continual_learning.run_full_benchmark import main

if __name__ == "__main__":
    print("[DENEY] exp_04: Gerçek WikiFacts QA ve GSM8K Verileriyle Sürekli Öğrenme Sınavı...")
    main()
