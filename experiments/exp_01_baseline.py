# experiments/exp_01_baseline.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Çalısma alanı ana dizinini sys.path'e ekle
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from gfaml.datasets.continual_bench import get_task_1
from gfaml.cortex.lora import LoRA

def run():
    print("=== [DENEY 1] STANDART LoRA BASELINE KORTEX DENEYİ ===")
    lora = LoRA(128, 32, 150)
    optimizer = optim.Adam(lora.parameters(), lr=1e-2)
    
    task_1 = get_task_1()
    print("   Task 1 Öğreniliyor...")
    for _ in range(20):
        for k, v in task_1:
            optimizer.zero_grad()
            pred = lora(torch.randn(1, 128) * 0.1)
            loss = nn.CrossEntropyLoss()(pred, torch.tensor([v]))
            loss.backward()
            optimizer.step()
            
    print("   Baseline LoRA deneyi tamamlandı.")

if __name__ == "__main__":
    run()
