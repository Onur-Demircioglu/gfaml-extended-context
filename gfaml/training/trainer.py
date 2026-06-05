# gfaml/training/trainer.py
import torch

def train_step(model, batch, optimizer):
    """
    Modeli bir eğitim adımında günceller.
    """
    x, y = batch
    logits = model(x)

    loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
