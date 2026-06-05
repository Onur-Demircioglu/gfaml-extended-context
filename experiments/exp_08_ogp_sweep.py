# experiments/exp_08_ogp_sweep.py
"""
OGP Stability-Plasticity Sweep on Real GPT-2.

Goal: Find the sweet spot where forgetting stays near zero
but accuracy actually climbs. This is the real research question.

Sweeps over:
- projection_strength: [0.0, 0.3, 0.5, 0.7, 1.0]
- warmup_epochs: [0, 5, 10]
- lr: [0.01, 0.03, 0.05]
"""
import sys
import os
import copy
import math
import itertools

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from gfaml.config import GFAMLConfig
from gfaml.core.gfaml_layer import GFAMLLayer
from gfaml.cortex.sleep import SleepOptimizer
from gfaml.cortex.lora import LoRA
from gfaml.eval.rigorous_metrics import RigorousEvaluator


# Tasks (same as exp_07)
TASKS = [
    [
        ("The speed of light is approximately", " 300"),
        ("Water boils at", " 100"),
        ("The chemical symbol for gold is", " Au"),
    ],
    [
        ("The capital of France is", " Paris"),
        ("The longest river in Africa is", " Nile"),
        ("Mount Everest is located in", " Nepal"),
    ],
    [
        ("World War II ended in", " 1945"),
        ("The first president of the United States was", " Washington"),
        ("The Berlin Wall fell in", " 1989"),
    ],
]


class GFAMLSweepModel(nn.Module):
    """GFAML model with configurable OGP parameters."""
    def __init__(self, hf_model, tokenizer, config,
                 lr=0.05, projection_strength=1.0,
                 conflict_threshold=0.0, warmup_epochs=0):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.config = config

        for param in self.hf_model.parameters():
            param.requires_grad = False

        self.gfaml_layers = nn.ModuleList([
            GFAMLLayer(config.d_model, config.rank, config=config)
            for _ in range(len(self.hf_model.transformer.h))
        ])

        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.state_list = None
        self.replay_buffer = []
        self.sleep_optimizer = SleepOptimizer(
            self.cortex,
            lr=lr,
            projection_strength=projection_strength,
            conflict_threshold=conflict_threshold,
            warmup_epochs=warmup_epochs,
        )

    def forward(self, input_ids, is_writing=False):
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe

        tok_emb = wte(input_ids)
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long,
                           device=input_ids.device).unsqueeze(0)
        pos_emb = wpe(pos)
        x = tok_emb + pos_emb

        if self.state_list is None:
            self.state_list = [None] * len(self.hf_model.transformer.h)

        next_states = []
        for i, block in enumerate(self.hf_model.transformer.h):
            x_norm = block.ln_1(x)
            attn_out = block.attn(x_norm)[0]
            gfaml_out, next_state, _ = self.gfaml_layers[i](
                x_norm, state=self.state_list[i], write=is_writing
            )
            x = x + attn_out + gfaml_out
            x = x + block.mlp(block.ln_2(x))
            next_states.append(next_state if is_writing else self.state_list[i])

        self.state_list = next_states
        x_norm = self.hf_model.transformer.ln_f(x)

        base_logits = self.hf_model.lm_head(x_norm)
        cortex_logits = self.cortex(x_norm)
        return base_logits + cortex_logits

    def train_on_task(self, task_data, epochs=50):
        self.train()
        for prompt, answer in task_data:
            full_text = prompt + answer
            ids = self.tokenizer.encode(full_text, return_tensors="pt",
                                        add_special_tokens=False)
            if ids is None or ids.size(1) < 2:
                continue

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            # Write to episodic memory
            with torch.no_grad():
                self.forward(ids, is_writing=True)

            # Store in replay buffer
            with torch.no_grad():
                wte = self.hf_model.transformer.wte
                wpe = self.hf_model.transformer.wpe
                tok_emb = wte(ids)
                pos = torch.arange(0, ids.size(1), dtype=torch.long).unsqueeze(0)
                pos_emb = wpe(pos)
                x = tok_emb + pos_emb
                x_norm = self.hf_model.transformer.h[0].ln_1(x)

                if prompt_len < ids.size(1):
                    repr_vec = x_norm[0, prompt_len - 1, :].detach().clone()
                    target_tok = ids[0, prompt_len].item()
                    self.replay_buffer.append((repr_vec, target_tok))

        if len(self.replay_buffer) > 0:
            self.sleep_optimizer.consolidate(self.replay_buffer, epochs=epochs)


class BaselineLoRA(nn.Module):
    """Standard LoRA fine-tuning baseline."""
    def __init__(self, hf_model, tokenizer, config, lr=0.01):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.config = config
        for param in self.hf_model.parameters():
            param.requires_grad = False
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.optimizer = torch.optim.Adam(self.cortex.parameters(), lr=lr)

    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.hf_model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        return outputs.logits + self.cortex(hidden)

    def train_on_task(self, task_data, epochs=50):
        self.train()
        for epoch in range(epochs):
            for prompt, answer in task_data:
                full_text = prompt + answer
                ids = self.tokenizer.encode(full_text, return_tensors="pt",
                                            add_special_tokens=False)
                if ids is None or ids.size(1) < 2:
                    continue
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_len = len(prompt_ids)
                self.optimizer.zero_grad()
                logits = self.forward(ids)
                if prompt_len < ids.size(1):
                    shift_logits = logits[:, prompt_len - 1:-1, :]
                    shift_labels = ids[:, prompt_len:]
                    loss = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                    )
                    loss.backward()
                    self.optimizer.step()


def evaluate_accuracy(model, task_data, tokenizer):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for prompt, answer in task_data:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(answer_ids) == 0:
                continue

            input_ids = torch.tensor([prompt_ids], dtype=torch.long)

            if hasattr(model, 'state_list'):
                saved = model.state_list
            logits = model.forward(input_ids, is_writing=False) if hasattr(model, 'state_list') \
                else model.forward(input_ids)
            if hasattr(model, 'state_list'):
                model.state_list = saved

            pred = torch.argmax(logits[0, -1, :]).item()
            if pred == answer_ids[0]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def evaluate_loss(model, task_data, tokenizer):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for prompt, answer in task_data:
            full_text = prompt + answer
            ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            if prompt_len >= len(ids):
                continue
            input_ids = torch.tensor([ids], dtype=torch.long)

            if hasattr(model, 'state_list'):
                saved = model.state_list
            logits = model.forward(input_ids, is_writing=False) if hasattr(model, 'state_list') \
                else model.forward(input_ids)
            if hasattr(model, 'state_list'):
                model.state_list = saved

            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = input_ids[:, prompt_len:]
            n_tokens = shift_labels.size(1)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def run_config(hf_model, tokenizer, config, tasks, model_factory, train_epochs=50):
    """Run one config, return ContinualResult."""
    model = model_factory(hf_model, tokenizer, config)
    n_tasks = len(tasks)
    evaluator = RigorousEvaluator(n_tasks)

    for task_id in range(n_tasks):
        model.train_on_task(tasks[task_id], epochs=train_epochs)
        evaluator.evaluate_after_task(
            model, task_id, tasks,
            accuracy_fn=lambda m, td: evaluate_accuracy(m, td, tokenizer),
            loss_fn=lambda m, td: evaluate_loss(m, td, tokenizer),
        )
    return evaluator.get_result()


def run():
    print("=" * 80)
    print("  OGP STABILITY-PLASTICITY SWEEP")
    print("  GPT-2 (124M) | Finding the sweet spot")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model.eval()

    config = GFAMLConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=hf_model.config.n_embd,
        rank=32, eta=1.0,
    )

    # Sweep grid
    sweep_configs = [
        # (name, projection_strength, warmup_epochs, lr, train_epochs)
        ("No-OGP (ps=0.0)",        0.0, 0,  0.03, 50),
        ("Soft-OGP (ps=0.3)",      0.3, 0,  0.03, 50),
        ("Medium-OGP (ps=0.5)",    0.5, 0,  0.03, 50),
        ("Strong-OGP (ps=0.7)",    0.7, 0,  0.03, 50),
        ("Full-OGP (ps=1.0)",      1.0, 0,  0.03, 50),
        ("Warmup-5 (ps=0.5)",      0.5, 5,  0.03, 50),
        ("Warmup-10 (ps=0.5)",     0.5, 10, 0.03, 50),
        ("LR-0.01 (ps=0.5)",       0.5, 5,  0.01, 50),
        ("LR-0.05 (ps=0.5)",       0.5, 5,  0.05, 50),
        ("Epochs-80 (ps=0.5,w=5)", 0.5, 5,  0.03, 80),
    ]

    results = {}

    # First: baseline
    print("\n[Baseline LoRA]")
    torch.manual_seed(42)
    np.random.seed(42)
    hf_copy = copy.deepcopy(hf_model)
    baseline_result = run_config(
        hf_copy, tokenizer, config, TASKS,
        lambda hf, tok, cfg: BaselineLoRA(hf, tok, cfg, lr=0.01),
        train_epochs=50,
    )
    results["Baseline LoRA"] = baseline_result
    print(f"  AA={baseline_result.average_accuracy():.4f} "
          f"F={baseline_result.forgetting_rate():.4f} "
          f"BWT={baseline_result.backward_transfer():+.4f}")

    # Sweep
    for name, ps, warmup, lr, epochs in sweep_configs:
        print(f"\n[{name}]")
        torch.manual_seed(42)
        np.random.seed(42)
        hf_copy = copy.deepcopy(hf_model)

        def factory(hf, tok, cfg, _ps=ps, _warmup=warmup, _lr=lr):
            return GFAMLSweepModel(hf, tok, cfg,
                                   lr=_lr,
                                   projection_strength=_ps,
                                   warmup_epochs=_warmup)

        result = run_config(hf_copy, tokenizer, config, TASKS, factory,
                            train_epochs=epochs)
        results[name] = result

        print(f"  AA={result.average_accuracy():.4f} "
              f"F={result.forgetting_rate():.4f} "
              f"BWT={result.backward_transfer():+.4f}")
        print(f"  R matrix: {result.accuracy_matrix.tolist()}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<30} | {'AvgAcc':>8} | {'Forget':>8} | {'BWT':>8} | {'Score':>8}")
    print("-" * 90)

    best_score = -999
    best_name = ""

    for name, result in results.items():
        aa = result.average_accuracy()
        fr = result.forgetting_rate()
        bwt = result.backward_transfer()
        # Composite score: accuracy matters, but forgetting penalty is harsh
        score = aa - 2.0 * max(fr, 0) + 0.5 * bwt
        if score > best_score:
            best_score = score
            best_name = name
        print(f"{name:<30} | {aa:>8.4f} | {fr:>8.4f} | {bwt:>+8.4f} | {score:>8.4f}")

    print("=" * 90)
    print(f"\n  >> BEST CONFIG: {best_name} (score={best_score:.4f})")
    print("=" * 90)

    # Save report
    reports_dir = os.path.join(WORKSPACE_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    report_lines = []
    report_lines.append("# OGP Stability-Plasticity Sweep Report")
    report_lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report_lines.append("## Sweep Results\n")
    report_lines.append(f"| Config | Avg Accuracy | Forgetting | BWT | Composite Score |")
    report_lines.append(f"| :--- | :---: | :---: | :---: | :---: |")

    for name, result in results.items():
        aa = result.average_accuracy()
        fr = result.forgetting_rate()
        bwt = result.backward_transfer()
        score = aa - 2.0 * max(fr, 0) + 0.5 * bwt
        marker = " **BEST**" if name == best_name else ""
        report_lines.append(
            f"| {name}{marker} | {aa:.4f} | {fr:.4f} | {bwt:+.4f} | {score:.4f} |"
        )

    report_lines.append(f"\n## Best Configuration\n")
    report_lines.append(f"**{best_name}** with composite score {best_score:.4f}")
    report_lines.append(f"\nComposite Score = Avg_Accuracy - 2 * max(Forgetting, 0) + 0.5 * BWT")

    report_lines.append(f"\n## Accuracy Matrices\n")
    for name, result in results.items():
        report_lines.append(f"### {name}")
        n = result.n_tasks
        header_cols = " | ".join([f"Task {j}" for j in range(n)])
        report_lines.append(f"| Trained | {header_cols} |")
        report_lines.append("| :--- | " + " | ".join([":---:"] * n) + " |")
        for i in range(n):
            cols = " | ".join([f"{result.accuracy_matrix[i, j]:.4f}" for j in range(n)])
            report_lines.append(f"| After Task {i} | {cols} |")
        report_lines.append("")

    report_path = os.path.join(reports_dir, "OGP_SWEEP_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run()
