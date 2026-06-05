# experiments/exp_11_decoupled_gfaml.py
"""
DECOUPLED GFAML: The architecturally correct approach.

Key insight from exp_10:
- GFAML layers injected during training ADD NOISE to LoRA gradients
- Simple experience replay (AA=0.98) beats GFAML Full (AA=0.78)
- Root cause: gfaml_out disrupts the optimization landscape

Fix: DECOUPLE training and inference paths.

Biological justification:
- WAKE (training): Cortex learns via gradients (LoRA + replay).
  Hippocampus RECORDS in parallel but does NOT interfere.
- SLEEP (consolidation): Hippocampal memories replay into cortex with OGP.
- RETRIEVAL (inference): Both systems contribute — cortex provides
  learned patterns, hippocampus provides episodic augmentation.

Architecture:
- Training: LoRA fine-tuning + experience replay (clean, no GFAML noise)
- Recording: GFAML writes facts to episodic memory (parallel, non-disruptive)
- Inference: base_logits + cortex_logits + gfaml_augmentation
"""
import sys
import os
import copy
import math
import random

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
from gfaml.eval.rigorous_metrics import (
    RigorousEvaluator, MultiSeedEvaluator,
    format_results_table, format_multi_seed_summary,
)


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


# ===========================================================================
# MODEL A: Baseline LoRA (no protection)
# ===========================================================================
class BaselineLoRA(nn.Module):
    def __init__(self, hf_model, tokenizer, config):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        for param in self.hf_model.parameters():
            param.requires_grad = False
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.optimizer = torch.optim.Adam(self.cortex.parameters(), lr=0.01)

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        return out.logits + self.cortex(out.hidden_states[-1])

    def train_on_task(self, task_data, epochs=50):
        self.train()
        for epoch in range(epochs):
            for prompt, answer in task_data:
                full = prompt + answer
                ids = self.tokenizer.encode(full, return_tensors="pt",
                                            add_special_tokens=False)
                plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if plen >= ids.size(1):
                    continue
                self.optimizer.zero_grad()
                logits = self.forward(ids)
                loss = F.cross_entropy(
                    logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
                    ids[:, plen:].reshape(-1))
                loss.backward()
                self.optimizer.step()


# ===========================================================================
# MODEL B: LoRA + Experience Replay (standard CL baseline)
# ===========================================================================
class LoRAReplay(nn.Module):
    def __init__(self, hf_model, tokenizer, config):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        for param in self.hf_model.parameters():
            param.requires_grad = False
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.optimizer = torch.optim.Adam(self.cortex.parameters(), lr=0.01)
        self.past_tasks = []

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        return out.logits + self.cortex(out.hidden_states[-1])

    def _train_sample(self, prompt, answer):
        full = prompt + answer
        ids = self.tokenizer.encode(full, return_tensors="pt",
                                    add_special_tokens=False)
        plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if plen >= ids.size(1):
            return
        self.optimizer.zero_grad()
        logits = self.forward(ids)
        loss = F.cross_entropy(
            logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
            ids[:, plen:].reshape(-1))
        loss.backward()
        self.optimizer.step()

    def train_on_task(self, task_data, epochs=50):
        self.train()
        for epoch in range(epochs):
            for prompt, answer in task_data:
                self._train_sample(prompt, answer)
            for past_task in self.past_tasks:
                if past_task:
                    p, a = random.choice(past_task)
                    self._train_sample(p, a)
        self.past_tasks.append(list(task_data))


# ===========================================================================
# MODEL C: DECOUPLED GFAML
# Training: clean LoRA + replay (no GFAML noise)
# Recording: parallel episodic write (non-disruptive)
# Inference: base + cortex + gfaml retrieval
# ===========================================================================
class DecoupledGFAML(nn.Module):
    """
    Decoupled dual-system architecture:

    Training path (wake):
        input -> GPT-2 (frozen) -> hidden -> LoRA -> logits
        + experience replay from past tasks

    Recording path (parallel, no gradients):
        input -> GPT-2 block norms -> GFAML write (Hebbian)

    Inference path (retrieval):
        input -> GPT-2 (frozen) -> hidden + GFAML read -> LoRA -> logits

    Sleep path (offline):
        replay buffer -> LoRA update with OGP protection
    """
    def __init__(self, hf_model, tokenizer, config,
                 projection_strength=0.5, warmup_epochs=5):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.config = config

        for param in self.hf_model.parameters():
            param.requires_grad = False

        # Episodic memory (hippocampus) - one per transformer block
        self.gfaml_layers = nn.ModuleList([
            GFAMLLayer(config.d_model, config.rank, config=config)
            for _ in range(len(self.hf_model.transformer.h))
        ])
        self.state_list = None

        # Parametric cortex (LoRA)
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size)
        self.online_optimizer = torch.optim.Adam(self.cortex.parameters(), lr=0.01)

        # Sleep consolidation
        self.sleep_optimizer = SleepOptimizer(
            self.cortex, lr=0.02,
            projection_strength=projection_strength,
            warmup_epochs=warmup_epochs,
        )

        # Experience replay buffer (raw text)
        self.past_tasks = []

    def _forward_clean(self, input_ids):
        """Training path: clean GPT-2 + LoRA, no GFAML interference."""
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        return out.logits + self.cortex(hidden), hidden

    def _forward_augmented(self, input_ids):
        """Inference path: GPT-2 + GFAML augmentation + LoRA."""
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe

        tok_emb = wte(input_ids)
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long,
                           device=input_ids.device).unsqueeze(0)
        x = tok_emb + wpe(pos)

        if self.state_list is None:
            self.state_list = [None] * len(self.hf_model.transformer.h)

        for i, block in enumerate(self.hf_model.transformer.h):
            x_norm = block.ln_1(x)
            attn_out = block.attn(x_norm)[0]

            # GFAML READ (retrieval augmentation)
            gfaml_out, _, _ = self.gfaml_layers[i](
                x_norm, state=self.state_list[i], write=False
            )

            x = x + attn_out + gfaml_out
            x = x + block.mlp(block.ln_2(x))

        x_final = self.hf_model.transformer.ln_f(x)
        base_logits = self.hf_model.lm_head(x_final)
        cortex_logits = self.cortex(x_final)
        return base_logits + cortex_logits

    def _record_to_memory(self, input_ids):
        """Recording path: write to episodic memory without disrupting anything."""
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe

        with torch.no_grad():
            tok_emb = wte(input_ids)
            pos = torch.arange(0, input_ids.size(1), dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            x = tok_emb + wpe(pos)

            if self.state_list is None:
                self.state_list = [None] * len(self.hf_model.transformer.h)

            next_states = []
            for i, block in enumerate(self.hf_model.transformer.h):
                x_norm = block.ln_1(x)
                attn_out = block.attn(x_norm)[0]

                # GFAML WRITE (Hebbian recording)
                _, next_state, _ = self.gfaml_layers[i](
                    x_norm, state=self.state_list[i], write=True
                )
                next_states.append(next_state)

                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

            self.state_list = next_states

    def forward(self, input_ids, mode="infer"):
        if mode == "train":
            logits, _ = self._forward_clean(input_ids)
            return logits
        elif mode == "infer":
            return self._forward_augmented(input_ids)
        elif mode == "record":
            self._record_to_memory(input_ids)
            return None

    def _train_sample(self, prompt, answer):
        full = prompt + answer
        ids = self.tokenizer.encode(full, return_tensors="pt",
                                    add_special_tokens=False)
        plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if plen >= ids.size(1):
            return
        self.online_optimizer.zero_grad()
        logits = self.forward(ids, mode="train")
        loss = F.cross_entropy(
            logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
            ids[:, plen:].reshape(-1))
        loss.backward()
        self.online_optimizer.step()

    def train_on_task(self, task_data, epochs=50):
        self.train()

        # Phase 1: Record facts to episodic memory (parallel, non-disruptive)
        for prompt, answer in task_data:
            full = prompt + answer
            ids = self.tokenizer.encode(full, return_tensors="pt",
                                        add_special_tokens=False)
            self.forward(ids, mode="record")

        # Phase 2: Online fine-tuning with experience replay (clean path)
        for epoch in range(epochs):
            for prompt, answer in task_data:
                self._train_sample(prompt, answer)
            for past_task in self.past_tasks:
                if past_task:
                    p, a = random.choice(past_task)
                    self._train_sample(p, a)

        # Phase 3: Sleep consolidation with OGP
        # Collect fresh representations for sleep replay
        sleep_buffer = []
        with torch.no_grad():
            for prompt, answer in task_data:
                full = prompt + answer
                ids = self.tokenizer.encode(full, return_tensors="pt",
                                            add_special_tokens=False)
                plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if plen >= ids.size(1):
                    continue
                _, hidden = self._forward_clean(ids)
                repr_vec = hidden[0, plen - 1, :].detach().clone()
                target_tok = ids[0, plen].item()
                sleep_buffer.append((repr_vec, target_tok))

        if sleep_buffer:
            self.sleep_optimizer.consolidate(sleep_buffer, epochs=30)

        self.past_tasks.append(list(task_data))


# ===========================================================================
# EVALUATION
# ===========================================================================
def evaluate_accuracy(model, task_data, tokenizer):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for prompt, answer in task_data:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not answer_ids:
                continue
            ids = torch.tensor([prompt_ids], dtype=torch.long)

            saved_state = getattr(model, 'state_list', None)

            if isinstance(model, DecoupledGFAML):
                logits = model.forward(ids, mode="infer")
            else:
                out = model.forward(ids)
                logits = out[0] if isinstance(out, tuple) else out

            if saved_state is not None:
                model.state_list = saved_state

            if torch.argmax(logits[0, -1, :]).item() == answer_ids[0]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def evaluate_loss(model, task_data, tokenizer):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for prompt, answer in task_data:
            full = prompt + answer
            ids_list = tokenizer.encode(full, add_special_tokens=False)
            plen = len(tokenizer.encode(prompt, add_special_tokens=False))
            if plen >= len(ids_list):
                continue
            ids = torch.tensor([ids_list], dtype=torch.long)

            saved_state = getattr(model, 'state_list', None)

            if isinstance(model, DecoupledGFAML):
                logits = model.forward(ids, mode="infer")
            else:
                out = model.forward(ids)
                logits = out[0] if isinstance(out, tuple) else out

            if saved_state is not None:
                model.state_list = saved_state

            shift_logits = logits[:, plen-1:-1, :]
            shift_labels = ids[:, plen:]
            n = shift_labels.size(1)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += n
    return total_loss / max(total_tokens, 1)


# ===========================================================================
# RUNNER
# ===========================================================================
def run_variant(name, model_factory, hf_model, tokenizer, config, tasks, seeds):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    n_tasks = len(tasks)
    multi = MultiSeedEvaluator(n_tasks=n_tasks, seeds=seeds)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        hf_copy = copy.deepcopy(hf_model)
        model = model_factory(hf_copy, tokenizer, config)

        evaluator = RigorousEvaluator(n_tasks)
        for tid in range(n_tasks):
            model.train_on_task(tasks[tid], epochs=50)
            evaluator.evaluate_after_task(
                model, tid, tasks,
                accuracy_fn=lambda m, td: evaluate_accuracy(m, td, tokenizer),
                loss_fn=lambda m, td: evaluate_loss(m, td, tokenizer),
            )

        result = evaluator.get_result()
        multi.submit(seed, result)

        print(f"  Seed {seed}: AA={result.average_accuracy():.4f} "
              f"F={result.forgetting_rate():.4f} "
              f"BWT={result.backward_transfer():+.4f}")
        for i in range(n_tasks):
            row = [f"{result.accuracy_matrix[i,j]:.2f}" for j in range(n_tasks)]
            print(f"    R[{i}] = [{', '.join(row)}]")

    from gfaml.eval.rigorous_metrics import ContinualResult
    all_results = list(multi.results.values())
    avg_matrix = np.mean([r.accuracy_matrix for r in all_results], axis=0)
    avg_ppl = np.mean([r.perplexity_matrix for r in all_results], axis=0)
    avg_loss = np.mean([r.loss_matrix for r in all_results], axis=0)
    avg_result = ContinualResult(avg_matrix, avg_ppl, avg_loss)

    return avg_result, multi.summary()


def run():
    print("=" * 70)
    print("  DECOUPLED GFAML EVALUATION")
    print("  GPT-2 (124M) | Train clean, retrieve with memory")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model.eval()

    config = GFAMLConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=hf_model.config.n_embd,
        rank=32, eta=1.0,
    )

    seeds = [42, 123, 456, 789, 1337]
    all_results = {}
    all_summaries = {}

    variants = [
        ("Model A: Baseline LoRA",
         lambda hf, tok, cfg: BaselineLoRA(hf, tok, cfg)),
        ("Model B: LoRA + Replay",
         lambda hf, tok, cfg: LoRAReplay(hf, tok, cfg)),
        ("Model C: Decoupled GFAML",
         lambda hf, tok, cfg: DecoupledGFAML(hf, tok, cfg,
                                              projection_strength=0.5,
                                              warmup_epochs=5)),
    ]

    for name, factory in variants:
        r, s = run_variant(name, factory, hf_model, tokenizer, config, TASKS, seeds)
        short_name = name.split(": ")[1]
        all_results[short_name] = r
        all_summaries[short_name] = s

    # ===== FINAL =====
    print("\n" + "=" * 85)
    print("  FINAL COMPARISON (AVERAGED OVER 5 SEEDS)")
    print("=" * 85)
    print(f"{'Model':<25} | {'AvgAcc':>8} | {'Forget':>8} | {'BWT':>8}")
    print("-" * 60)
    for name, result in all_results.items():
        print(f"{name:<25} | {result.average_accuracy():>8.4f} | "
              f"{result.forgetting_rate():>8.4f} | "
              f"{result.backward_transfer():>+8.4f}")
    print("=" * 85)

    print("\n  STATISTICAL SIGNIFICANCE (5 seeds)")
    print("-" * 70)
    for name, summary in all_summaries.items():
        aa = summary["average_accuracy"]
        fr = summary["forgetting_rate"]
        print(f"  {name}:")
        print(f"    AA = {aa['mean']:.4f} +/- {aa['std']:.4f} "
              f"  CI95: [{aa['ci_95_lower']:.4f}, {aa['ci_95_upper']:.4f}]")
        print(f"    F  = {fr['mean']:.4f} +/- {fr['std']:.4f} "
              f"  CI95: [{fr['ci_95_lower']:.4f}, {fr['ci_95_upper']:.4f}]")

    # Save report
    reports_dir = os.path.join(WORKSPACE_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    lines = []
    lines.append("# Decoupled GFAML Evaluation Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("## Architecture\n")
    lines.append("The decoupled approach separates GFAML into three non-interfering paths:\n")
    lines.append("1. **Training (wake):** Clean LoRA fine-tuning + experience replay")
    lines.append("   - No GFAML noise in gradient computation")
    lines.append("   - Experience replay prevents catastrophic forgetting")
    lines.append("2. **Recording (parallel):** GFAML writes to episodic memory via Hebbian updates")
    lines.append("   - Non-disruptive: no gradients, no interference with cortex")
    lines.append("3. **Inference (retrieval):** GFAML augments GPT-2 hidden states")
    lines.append("   - Episodic memory provides additional context for recall")
    lines.append("4. **Sleep (offline):** OGP-protected consolidation of replay buffer\n")

    lines.append(format_results_table(all_results))
    lines.append("")
    for name, summary in all_summaries.items():
        lines.append(f"### {name}")
        lines.append(format_multi_seed_summary(summary))
        lines.append("")

    # Accuracy matrices
    lines.append("## Accuracy Matrices (Seed-Averaged)\n")
    for name, result in all_results.items():
        n = result.n_tasks
        lines.append(f"### {name}")
        header = " | ".join([f"Task {j}" for j in range(n)])
        lines.append(f"| Trained | {header} |")
        lines.append("| :--- | " + " | ".join([":---:"] * n) + " |")
        for i in range(n):
            cols = " | ".join([f"{result.accuracy_matrix[i, j]:.4f}" for j in range(n)])
            lines.append(f"| After Task {i} | {cols} |")
        lines.append("")

    report_path = os.path.join(reports_dir, "DECOUPLED_GFAML_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run()
