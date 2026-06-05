# gfaml/core/trainer.py
import torch
import torch.nn as nn
import inspect
from gfaml.core.metrics import compute_metrics

class GFAMLTrainer:
    """
    GFAML araştırma omurgası için modüler eğitim ve değerlendirme yöneticisi.
    """
    def __init__(self, model, optimizer, config, logger):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger

        self.history = {
            "loss": [],
            "accuracy": [],
            "forgetting": [],
            "mcg": []
        }

    def _model_forward(self, x, is_writing=False, target_tok=None):
        """
        Hem SUKM çift-sistem bilişsel modellerini hem de standart PyTorch modellerini
        hatasız çalıştırmak için geliştirilmiş esnek ileri besleme sarmalayıcısı.
        """
        try:
            sig = inspect.signature(self.model.forward)
            has_writing = 'is_writing' in sig.parameters
        except Exception:
            has_writing = False

        if has_writing:
            out = self.model(x, is_writing=is_writing, target_tok=target_tok)
        else:
            out = self.model(x)

        if isinstance(out, tuple):
            return out[0]
        else:
            return out

    def train_step(self, batch):
        self.model.train()

        x, y = batch
        logits = self._model_forward(x, is_writing=True, target_tok=y)

        loss = nn.CrossEntropyLoss()(logits[:, -1, :], y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, model, dataset, name="task"):
        model.eval()

        hits = 0
        total = len(dataset)

        for x, y in dataset:
            logits = self._model_forward(x, is_writing=False)
            pred = torch.argmax(logits[:, -1, :]).item()
            hits += (pred == y)

        acc = hits / total
        self.logger.log_metric(f"{name}_acc", acc)

        return acc

    def run_task(self, train_data, eval_data, task_name):
        print(f"\n[TRAIN] {task_name}")

        epochs = getattr(self.config, 'epochs', 10)
        # Handle dict configs
        if isinstance(self.config, dict):
            epochs = self.config.get('epochs', epochs)
        elif hasattr(self.config, 'training') and isinstance(self.config.training, dict):
            epochs = self.config.training.get('epochs', epochs)

        for epoch in range(epochs):
            for batch in train_data:
                loss = self.train_step(batch)

            self.logger.log_metric(f"{task_name}_loss", loss)

        acc = self.evaluate(self.model, eval_data, task_name)
        return acc
