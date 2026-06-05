# experiments/exp_09_fixed_eval.py
"""
FIXED Rigorous GPT-2 Continual Learning Evaluation.

Critical fix from exp_08 sweep analysis:
- Replay buffer was storing Layer 0 representations but LoRA cortex
  receives Layer 12 representations at inference. Complete representation
  space mismatch. No wonder accuracy was stuck at 0.11.

Fix: Store the SAME representation the cortex sees during inference
(final layer norm output from full forward pass).

Also added: Combined training (online fine-tune + sleep consolidation)
to properly compare against baseline.
"""
import sys
import os
import copy
import math

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
    RigorousEvaluator, ContinualResult, format_results_table,
    MultiSeedEvaluator, format_multi_seed_summary,
)

# ===========================================================================
# TASKS
# ===========================================================================
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
# MODEL A: Baseline LoRA (no memory, no consolidation)
# ===========================================================================
class BaselineLoRA(nn.Module):
    """Standard LoRA fine-tuning. Maximum plasticity, zero stability."""
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
                    ids[:, plen:].reshape(-1),
                )
                loss.backward()
                self.optimizer.step()


# ===========================================================================
# MODEL B: GFAML + Sleep (FIXED replay buffer)
# ===========================================================================
class GFAMLFixed(nn.Module):
    """
    GFAML with correctly aligned replay buffer.
    Key fix: replay stores final-layer representations, not Layer 0.
    """
    def __init__(self, hf_model, tokenizer, config,
                 projection_strength=0.5, warmup_epochs=5):
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
            self.cortex, lr=0.03,
            projection_strength=projection_strength,
            warmup_epochs=warmup_epochs,
        )

    def forward(self, input_ids, is_writing=False):
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe

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
            gfaml_out, ns, _ = self.gfaml_layers[i](
                x_norm, state=self.state_list[i], write=is_writing
            )
            x = x + attn_out + gfaml_out
            x = x + block.mlp(block.ln_2(x))
            next_states.append(ns if is_writing else self.state_list[i])

        self.state_list = next_states
        x_final = self.hf_model.transformer.ln_f(x)  # Final layer norm

        base_logits = self.hf_model.lm_head(x_final)
        cortex_logits = self.cortex(x_final)
        return base_logits + cortex_logits, x_final

    def train_on_task(self, task_data, epochs=50):
        self.train()
        for prompt, answer in task_data:
            full = prompt + answer
            ids = self.tokenizer.encode(full, return_tensors="pt",
                                        add_special_tokens=False)
            plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            if plen >= ids.size(1):
                continue

            # Phase 1: Write to episodic memory AND get final-layer repr
            with torch.no_grad():
                _, x_final = self.forward(ids, is_writing=True)

                # FIXED: Store FINAL layer representation (what cortex sees)
                repr_vec = x_final[0, plen - 1, :].detach().clone()
                target_tok = ids[0, plen].item()
                self.replay_buffer.append((repr_vec, target_tok))

        # Phase 2: Sleep consolidation with OGP
        if len(self.replay_buffer) > 0:
            self.sleep_optimizer.consolidate(self.replay_buffer, epochs=epochs)


# ===========================================================================
# MODEL C: GFAML + Online Fine-tuning + Sleep (Combined approach)
# ===========================================================================
class GFAMLCombined(nn.Module):
    """
    GFAML with both online fine-tuning and sleep consolidation.
    This is the dual-system approach:
    - Fast: episodic memory write (non-parametric)
    - Medium: online cortex fine-tuning (parametric, current task)
    - Slow: sleep consolidation with OGP (parametric, replay with protection)
    """
    def __init__(self, hf_model, tokenizer, config,
                 projection_strength=0.5, warmup_epochs=5):
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
        self.online_optimizer = torch.optim.Adam(self.cortex.parameters(), lr=0.01)
        self.sleep_optimizer = SleepOptimizer(
            self.cortex, lr=0.03,
            projection_strength=projection_strength,
            warmup_epochs=warmup_epochs,
        )

    def forward(self, input_ids, is_writing=False):
        wte = self.hf_model.transformer.wte
        wpe = self.hf_model.transformer.wpe

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
            gfaml_out, ns, _ = self.gfaml_layers[i](
                x_norm, state=self.state_list[i], write=is_writing
            )
            x = x + attn_out + gfaml_out
            x = x + block.mlp(block.ln_2(x))
            next_states.append(ns if is_writing else self.state_list[i])

        self.state_list = next_states
        x_final = self.hf_model.transformer.ln_f(x)

        base_logits = self.hf_model.lm_head(x_final)
        cortex_logits = self.cortex(x_final)
        return base_logits + cortex_logits, x_final

    def train_on_task(self, task_data, epochs=50):
        self.train()

        # Phase 1: Write to episodic memory
        for prompt, answer in task_data:
            full = prompt + answer
            ids = self.tokenizer.encode(full, return_tensors="pt",
                                        add_special_tokens=False)
            plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            if plen >= ids.size(1):
                continue

            with torch.no_grad():
                _, x_final = self.forward(ids, is_writing=True)
                repr_vec = x_final[0, plen - 1, :].detach().clone()
                target_tok = ids[0, plen].item()
                self.replay_buffer.append((repr_vec, target_tok))

        # Phase 2: Online fine-tuning (learn current task fast)
        for epoch in range(epochs):
            for prompt, answer in task_data:
                full = prompt + answer
                ids = self.tokenizer.encode(full, return_tensors="pt",
                                            add_special_tokens=False)
                plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                if plen >= ids.size(1):
                    continue
                self.online_optimizer.zero_grad()
                logits, _ = self.forward(ids, is_writing=False)
                loss = F.cross_entropy(
                    logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
                    ids[:, plen:].reshape(-1),
                )
                loss.backward()
                self.online_optimizer.step()

        # Phase 3: Sleep consolidation with OGP (protect past tasks)
        if len(self.replay_buffer) > 0:
            self.sleep_optimizer.consolidate(self.replay_buffer, epochs=epochs)


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

            saved = getattr(model, 'state_list', None)
            out = model.forward(ids)
            logits = out[0] if isinstance(out, tuple) else out
            if saved is not None:
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
            full = prompt + answer
            ids_list = tokenizer.encode(full, add_special_tokens=False)
            plen = len(tokenizer.encode(prompt, add_special_tokens=False))
            if plen >= len(ids_list):
                continue
            ids = torch.tensor([ids_list], dtype=torch.long)

            saved = getattr(model, 'state_list', None)
            out = model.forward(ids)
            logits = out[0] if isinstance(out, tuple) else out
            if saved is not None:
                model.state_list = saved

            shift_logits = logits[:, plen-1:-1, :]
            shift_labels = ids[:, plen:]
            n = shift_labels.size(1)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), reduction='sum',
            )
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
    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
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
        all_results.append(result)

        print(f"  Seed {seed}: AA={result.average_accuracy():.4f} "
              f"F={result.forgetting_rate():.4f} "
              f"BWT={result.backward_transfer():+.4f}")
        # Print accuracy matrix
        for i in range(n_tasks):
            row = [f"{result.accuracy_matrix[i,j]:.2f}" for j in range(n_tasks)]
            print(f"    R[{i}] = [{', '.join(row)}]")

    summary = multi.summary()
    return all_results[0], summary


def run():
    print("=" * 70)
    print("  FIXED RIGOROUS EVALUATION")
    print("  GPT-2 (124M) | Correct Replay Buffer Alignment")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model.eval()

    config = GFAMLConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=hf_model.config.n_embd,
        rank=32, eta=1.0,
    )

    seeds = [42, 123, 456]
    all_results = {}
    all_summaries = {}

    # Model A: Baseline LoRA
    r, s = run_variant(
        "Model A: Baseline LoRA (no memory)",
        lambda hf, tok, cfg: BaselineLoRA(hf, tok, cfg),
        hf_model, tokenizer, config, TASKS, seeds,
    )
    all_results["Baseline LoRA"] = r
    all_summaries["Baseline LoRA"] = s

    # Model B: GFAML + Sleep (fixed replay)
    r, s = run_variant(
        "Model B: GFAML + Sleep (fixed replay buffer)",
        lambda hf, tok, cfg: GFAMLFixed(hf, tok, cfg,
                                         projection_strength=0.5, warmup_epochs=5),
        hf_model, tokenizer, config, TASKS, seeds,
    )
    all_results["GFAML + Sleep"] = r
    all_summaries["GFAML + Sleep"] = s

    # Model C: GFAML + Online + Sleep (combined)
    r, s = run_variant(
        "Model C: GFAML + Online + Sleep (combined dual-system)",
        lambda hf, tok, cfg: GFAMLCombined(hf, tok, cfg,
                                            projection_strength=0.5, warmup_epochs=5),
        hf_model, tokenizer, config, TASKS, seeds,
    )
    all_results["GFAML Combined"] = r
    all_summaries["GFAML Combined"] = s

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<35} | {'AvgAcc':>8} | {'Forget':>8} | {'BWT':>8} | {'PPL Shift':>10}")
    print("-" * 80)

    for name, result in all_results.items():
        aa = result.average_accuracy()
        fr = result.forgetting_rate()
        bwt = result.backward_transfer()
        ppl = result.perplexity_shift()["mean"]
        print(f"{name:<35} | {aa:>8.4f} | {fr:>8.4f} | {bwt:>+8.4f} | {ppl:>+10.2f}")

    print("=" * 80)

    # Save report
    reports_dir = os.path.join(WORKSPACE_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    lines = []
    lines.append("# GFAML Fixed Rigorous Evaluation Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("## Critical Fix Applied")
    lines.append("Previous experiments had a **representation space mismatch**:")
    lines.append("- Replay buffer stored Layer 0 input representations")
    lines.append("- LoRA cortex receives final layer (Layer 12) output at inference")
    lines.append("- This mismatch prevented the cortex from learning anything useful")
    lines.append("- **Fix:** Store final-layer representations in replay buffer\n")

    lines.append("## Models Compared\n")
    lines.append("| Model | Description |")
    lines.append("| :--- | :--- |")
    lines.append("| **Baseline LoRA** | Standard LoRA fine-tuning, no memory, no consolidation |")
    lines.append("| **GFAML + Sleep** | Episodic memory + fixed replay + OGP sleep consolidation |")
    lines.append("| **GFAML Combined** | Episodic memory + online fine-tuning + OGP sleep consolidation |")
    lines.append("")

    lines.append(format_results_table(all_results))
    lines.append("")

    for name, summary in all_summaries.items():
        lines.append(f"### {name} - Statistical Summary ({len(seeds)} seeds)")
        lines.append(format_multi_seed_summary(summary))
        lines.append("")

    lines.append("## Perplexity Analysis\n")
    for name, result in all_results.items():
        shifts = result.perplexity_shift()
        lines.append(f"### {name}")
        for k, v in shifts.items():
            lines.append(f"- {k}: {v:+.4f}")
        lines.append("")

    report_path = os.path.join(reports_dir, "FIXED_EVAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run()
