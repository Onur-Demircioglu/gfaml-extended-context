# gfaml/cortex/sleep_optimizer.py
import torch
import torch.nn as nn

def sleep_train(model, replay_buffer, epochs=10):
    """
    Episodik replay buffer anılarını LoRA katmanına konsolide eder.
    """
    opt = torch.optim.Adam([model.A, model.B], lr=1e-2)

    for _ in range(epochs):
        for x, y in replay_buffer:
            opt.zero_grad()
            pred = model(x)
            loss = nn.CrossEntropyLoss()(pred.unsqueeze(0), torch.tensor([y], device=pred.device))
            loss.backward()
            opt.step()
