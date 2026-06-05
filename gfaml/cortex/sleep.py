# gfaml/cortex/sleep.py
import torch
import torch.nn as nn
import torch.optim as optim


class SleepOptimizer:
    """
    Sleep-phase consolidation manager with tunable OGP.

    Key insight from rigorous eval:
    - Full OGP (projection_strength=1.0, threshold=0.0) eliminates forgetting
      but also kills learning (accuracy stays near 0).
    - No OGP gives full learning but catastrophic forgetting.
    - The sweet spot is partial projection with a conflict threshold.

    Parameters:
        projection_strength: float in [0, 1].
            0.0 = no projection (standard SGD, maximum plasticity)
            1.0 = full orthogonal projection (maximum stability)
            0.3-0.5 = recommended sweet spot
        conflict_threshold: float, typically negative.
            Only project when dot(g, g_ref) < threshold.
            0.0 = project on any conflict (aggressive)
            -0.1 = allow mild conflicts (relaxed)
        warmup_epochs: int.
            Train freely for this many epochs before activating OGP.
            Allows initial learning before stability kicks in.
    """
    def __init__(
        self,
        lora_layer: nn.Module,
        lr: float = 0.06,
        projection_strength: float = 1.0,
        conflict_threshold: float = 0.0,
        warmup_epochs: int = 0,
    ):
        self.lora_layer = lora_layer
        self.optimizer = optim.Adam(self.lora_layer.parameters(), lr=lr)
        self.history = []
        self.projection_strength = projection_strength
        self.conflict_threshold = conflict_threshold
        self.warmup_epochs = warmup_epochs

    def consolidate(self, replay_buffer: list, epochs: int = 200) -> float:
        """
        Consolidate replay buffer into LoRA cortex with tunable OGP.
        """
        if len(replay_buffer) == 0:
            return 0.0

        inputs = torch.stack([item[0] for item in replay_buffer])
        device = inputs.device
        targets = torch.tensor([item[1] for item in replay_buffer], dtype=torch.long, device=device)

        # 1. Compute reference gradients from past tasks (for OGP)
        ref_grads = {}
        if len(self.history) > 0:
            past_inputs = torch.cat([item[0] for item in self.history], dim=0)
            past_targets = torch.cat([item[1] for item in self.history], dim=0).to(device)

            self.optimizer.zero_grad()
            ref_out = self.lora_layer(past_inputs)
            ref_loss = nn.CrossEntropyLoss()(ref_out, past_targets)
            ref_loss.backward()

            for name, param in self.lora_layer.named_parameters():
                if param.grad is not None:
                    ref_grads[name] = param.grad.clone()

        # 2. Training loop with tunable OGP
        loss_val = 0.0
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            cortex_out = self.lora_layer(inputs)
            loss = nn.CrossEntropyLoss()(cortex_out, targets)
            loss.backward()

            # Apply OGP only after warmup and only if we have history
            if epoch >= self.warmup_epochs and len(ref_grads) > 0:
                for name, param in self.lora_layer.named_parameters():
                    if param.grad is not None and name in ref_grads:
                        g = param.grad
                        g_ref = ref_grads[name]
                        dot = torch.sum(g * g_ref)

                        if dot < self.conflict_threshold:
                            norm_sq = torch.sum(g_ref * g_ref) + 1e-8
                            projection = (dot / norm_sq) * g_ref
                            # Partial projection: blend between original and projected
                            param.grad.copy_(g - self.projection_strength * projection)

            self.optimizer.step()
            loss_val = loss.item()

        # Store current memories for future OGP reference
        self.history.append((inputs.detach().clone(), targets.detach().clone()))
        if len(self.history) > 5:
            self.history.pop(0)

        replay_buffer.clear()
        return loss_val

    def zero_grad(self):
        """Standard PyTorch optimizer delegation."""
        self.optimizer.zero_grad()

    def step(self):
        """Standard PyTorch optimizer delegation."""
        self.optimizer.step()
