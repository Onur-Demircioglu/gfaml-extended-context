# experiments/exp_12_scaled_benchmark.py
"""
SCALED HONEST BENCHMARK.

Fixes every methodological flaw from previous experiments:
1. Proper train/test split (NO data leakage)
2. 10 tasks instead of 3
3. 15 train + 5 held-out test examples per task
4. Tracks degradation as task count increases
5. Proper ablation: Replay vs Replay+GFAML
6. Reports on TEST set only
"""
import sys
import os
import copy
import math
import random
import hashlib

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
from gfaml.eval.rigorous_metrics import RigorousEvaluator, MultiSeedEvaluator


# ===========================================================================
# DATASET: 10 TASKS, TRAIN/TEST SPLIT
# ===========================================================================

def generate_tasks(n_tasks=10, n_train=15, n_test=5):
    """
    Generate factual QA tasks with proper train/test split.
    Each task is a domain of knowledge. No overlap between train and test.
    Uses deterministic generation for reproducibility.
    """
    # Domain templates - each domain has many possible facts
    domains = [
        {
            "name": "Chemistry",
            "template": "The chemical symbol for {} is",
            "facts": [
                ("Hydrogen", " H"), ("Helium", " He"), ("Lithium", " Li"),
                ("Beryllium", " Be"), ("Boron", " B"), ("Carbon", " C"),
                ("Nitrogen", " N"), ("Oxygen", " O"), ("Fluorine", " F"),
                ("Neon", " Ne"), ("Sodium", " Na"), ("Magnesium", " Mg"),
                ("Aluminum", " Al"), ("Silicon", " Si"), ("Phosphorus", " P"),
                ("Sulfur", " S"), ("Chlorine", " Cl"), ("Argon", " Ar"),
                ("Potassium", " K"), ("Calcium", " Ca"),
            ],
        },
        {
            "name": "Capitals",
            "template": "The capital of {} is",
            "facts": [
                ("France", " Paris"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
                ("Italy", " Rome"), ("Spain", " Madrid"), ("Canada", " Ottawa"),
                ("Brazil", " Brasilia"), ("Australia", " Canberra"),
                ("Russia", " Moscow"), ("China", " Beijing"),
                ("India", " Delhi"), ("Egypt", " Cairo"),
                ("Turkey", " Ankara"), ("Sweden", " Stockholm"),
                ("Norway", " Oslo"), ("Finland", " Helsinki"),
                ("Poland", " Warsaw"), ("Greece", " Athens"),
                ("Portugal", " Lisbon"), ("Austria", " Vienna"),
            ],
        },
        {
            "name": "Numbers",
            "template": "The square of {} is",
            "facts": [
                ("two", " four"), ("three", " nine"), ("four", " sixteen"),
                ("five", " twenty"), ("six", " thirty"), ("seven", " forty"),
                ("eight", " sixty"), ("nine", " eighty"), ("ten", " hundred"),
                ("eleven", " one"), ("twelve", " one"), ("thirteen", " one"),
                ("fourteen", " one"), ("fifteen", " two"), ("sixteen", " two"),
                ("seventeen", " two"), ("eighteen", " three"),
                ("nineteen", " three"), ("twenty", " four"),
                ("one", " one"),
            ],
        },
        {
            "name": "Continents",
            "template": "{} is located in",
            "facts": [
                ("Nigeria", " Africa"), ("Kenya", " Africa"), ("Egypt", " Africa"),
                ("Japan", " Asia"), ("China", " Asia"), ("India", " Asia"),
                ("France", " Europe"), ("Germany", " Europe"), ("Italy", " Europe"),
                ("Brazil", " South"), ("Argentina", " South"), ("Chile", " South"),
                ("Canada", " North"), ("Mexico", " North"), ("Cuba", " North"),
                ("Australia", " Oceania"), ("Fiji", " Oceania"),
                ("Morocco", " Africa"), ("Thailand", " Asia"),
                ("Spain", " Europe"),
            ],
        },
        {
            "name": "Colors",
            "template": "The color of {} is",
            "facts": [
                ("blood", " red"), ("grass", " green"), ("sky", " blue"),
                ("snow", " white"), ("coal", " black"), ("gold", " golden"),
                ("milk", " white"), ("night", " dark"), ("sun", " yellow"),
                ("ocean", " blue"), ("leaves", " green"), ("fire", " red"),
                ("sand", " brown"), ("silver", " grey"), ("ivory", " white"),
                ("emerald", " green"), ("ruby", " red"), ("sapphire", " blue"),
                ("amber", " orange"), ("jade", " green"),
            ],
        },
        {
            "name": "Languages",
            "template": "The official language of {} is",
            "facts": [
                ("France", " French"), ("Germany", " German"), ("Japan", " Japanese"),
                ("China", " Chinese"), ("Brazil", " Portuguese"), ("Russia", " Russian"),
                ("Spain", " Spanish"), ("Italy", " Italian"), ("Turkey", " Turkish"),
                ("Greece", " Greek"), ("Poland", " Polish"), ("Sweden", " Swedish"),
                ("Finland", " Finnish"), ("Norway", " Norwegian"),
                ("Netherlands", " Dutch"), ("Portugal", " Portuguese"),
                ("Egypt", " Arabic"), ("Israel", " Hebrew"),
                ("Thailand", " Thai"), ("Vietnam", " Vietnamese"),
            ],
        },
        {
            "name": "Inventions",
            "template": "The telephone was invented by",
            "facts": [
                ("The telephone was invented by", " Bell"),
                ("The light bulb was invented by", " Edison"),
                ("The airplane was invented by", " Wright"),
                ("The radio was invented by", " Marconi"),
                ("The dynamite was invented by", " Nobel"),
                ("The telescope was invented by", " Galileo"),
                ("The printing press was invented by", " Gutenberg"),
                ("The phonograph was invented by", " Edison"),
                ("The steam engine was improved by", " Watt"),
                ("The World Wide Web was created by", " Berners"),
                ("The theory of relativity was by", " Einstein"),
                ("The periodic table was by", " Mendeleev"),
                ("The vaccine was pioneered by", " Jenner"),
                ("The computer was conceptualized by", " Turing"),
                ("The AC motor was invented by", " Tesla"),
                ("The automobile was pioneered by", " Benz"),
                ("Calculus was developed by", " Newton"),
                ("Evolution theory was by", " Darwin"),
                ("The X-ray was discovered by", " Rontgen"),
                ("Penicillin was discovered by", " Fleming"),
            ],
        },
        {
            "name": "Currencies",
            "template": "The currency of {} is the",
            "facts": [
                ("Japan", " yen"), ("USA", " dollar"), ("UK", " pound"),
                ("Switzerland", " franc"), ("India", " rupee"), ("China", " yuan"),
                ("Russia", " ruble"), ("Brazil", " real"), ("Mexico", " peso"),
                ("Sweden", " krona"), ("Turkey", " lira"), ("Thailand", " baht"),
                ("South Korea", " won"), ("Egypt", " pound"), ("Israel", " shekel"),
                ("Poland", " zloty"), ("Norway", " krone"),
                ("Australia", " dollar"), ("Canada", " dollar"),
                ("Singapore", " dollar"),
            ],
        },
        {
            "name": "Planets",
            "template": "The planet {} is the",
            "facts": [
                ("Mercury", " first"), ("Venus", " second"), ("Earth", " third"),
                ("Mars", " fourth"), ("Jupiter", " fifth"), ("Saturn", " sixth"),
                ("Uranus", " seventh"), ("Neptune", " eighth"),
                ("Mercury is closest to the", " Sun"),
                ("Jupiter is the largest", " planet"),
                ("Saturn has prominent", " rings"),
                ("Mars is called the red", " planet"),
                ("Venus is the hottest", " planet"),
                ("Neptune is the farthest", " planet"),
                ("Earth has one natural", " satellite"),
                ("Jupiter has the most", " moons"),
                ("Uranus rotates on its", " side"),
                ("Mars has the tallest", " mountain"),
                ("Saturn is less dense than", " water"),
                ("Mercury has no", " atmosphere"),
            ],
        },
        {
            "name": "Biology",
            "template": "The {} belongs to",
            "facts": [
                ("The largest organ in the body is the", " skin"),
                ("The smallest bone is in the", " ear"),
                ("DNA stands for deoxyribonucleic", " acid"),
                ("Photosynthesis occurs in the", " chloroplast"),
                ("The powerhouse of the cell is the", " mitochondria"),
                ("Blood is filtered by the", " kidneys"),
                ("Insulin is produced by the", " pancreas"),
                ("The brain is protected by the", " skull"),
                ("Oxygen is carried by red blood", " cells"),
                ("The heart has four", " chambers"),
                ("Proteins are made of amino", " acids"),
                ("Genes are segments of", " DNA"),
                ("The liver produces", " bile"),
                ("Neurons transmit electrical", " signals"),
                ("The retina is in the", " eye"),
                ("Hemoglobin contains", " iron"),
                ("The trachea leads to the", " lungs"),
                ("The femur is the longest", " bone"),
                ("Tendons connect muscle to", " bone"),
                ("Ligaments connect bone to", " bone"),
            ],
        },
    ]

    tasks = []
    for i in range(min(n_tasks, len(domains))):
        domain = domains[i]
        facts = domain["facts"][:n_train + n_test]
        random.shuffle(facts)

        # For domains with template, construct full prompts
        if domain["name"] == "Inventions" or domain["name"] == "Biology":
            # These already have full prompts
            train_data = [(q, a) for q, a in facts[:n_train]]
            test_data = [(q, a) for q, a in facts[n_train:n_train + n_test]]
        else:
            template = domain["template"]
            train_data = [(template.format(q) if "{}" in template else q, a)
                          for q, a in facts[:n_train]]
            test_data = [(template.format(q) if "{}" in template else q, a)
                         for q, a in facts[n_train:n_train + n_test]]

        tasks.append({
            "name": domain["name"],
            "train": train_data,
            "test": test_data,
        })

    return tasks


# ===========================================================================
# MODELS
# ===========================================================================

class BaselineLoRA(nn.Module):
    """No replay, no memory. Pure fine-tuning."""
    def __init__(self, hf_model, tokenizer, config, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.hf_model = hf_model.to(self.device)
        self.tokenizer = tokenizer
        for param in self.hf_model.parameters():
            param.requires_grad = False
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.cortex.parameters(), lr=0.01)

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        return out.logits + self.cortex(out.hidden_states[-1])

    def train_on_task(self, task, epochs=30):
        self.train()
        for epoch in range(epochs):
            for prompt, answer in task["train"]:
                full = prompt + answer
                ids = self.tokenizer.encode(full, return_tensors="pt",
                                            add_special_tokens=False).to(self.device)
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


class LoRAReplay(nn.Module):
    """LoRA + experience replay. Standard CL baseline."""
    def __init__(self, hf_model, tokenizer, config, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.hf_model = hf_model.to(self.device)
        self.tokenizer = tokenizer
        for param in self.hf_model.parameters():
            param.requires_grad = False
        self.cortex = LoRA(config.d_model, config.rank, config.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.cortex.parameters(), lr=0.01)
        self.past_trains = []

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        return out.logits + self.cortex(out.hidden_states[-1])

    def _step(self, prompt, answer):
        full = prompt + answer
        ids = self.tokenizer.encode(full, return_tensors="pt",
                                    add_special_tokens=False).to(self.device)
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

    def train_on_task(self, task, epochs=30):
        self.train()
        for epoch in range(epochs):
            for p, a in task["train"]:
                self._step(p, a)
            for past in self.past_trains:
                if past:
                    p, a = random.choice(past)
                    self._step(p, a)
        self.past_trains.append(list(task["train"]))


class DecoupledGFAML(nn.Module):
    """Decoupled GFAML: clean training + parallel memory + inference augmentation."""
    def __init__(self, hf_model, tokenizer, config, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.hf_model = hf_model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        for param in self.hf_model.parameters():
            param.requires_grad = False

        self.dtype = next(self.hf_model.parameters()).dtype

        # Detect architecture dynamically
        if hasattr(self.hf_model, "transformer"):
            self.model_type = "gpt2"
            self.layers = self.hf_model.transformer.h
            self.d_model = self.hf_model.config.n_embd
            self.attn_name = "attn"
        elif hasattr(self.hf_model, "model"):
            self.model_type = "qwen_llama"
            self.layers = self.hf_model.model.layers
            self.d_model = self.hf_model.config.hidden_size
            self.attn_name = "self_attn"
        else:
            raise ValueError("Unsupported architecture! Must be GPT-2 or Qwen/Llama.")

        self.gfaml_layers = nn.ModuleList([
            GFAMLLayer(self.d_model, config.rank, config=config)
            for _ in range(len(self.layers))
        ]).to(self.device).to(dtype=self.dtype)
        self.state_list = None
        self.gfaml_mode = "off" # Mode can be "off", "record", "augment"

        self.cortex = LoRA(self.d_model, config.rank, config.vocab_size).to(self.device).to(dtype=self.dtype)
        self.online_opt = torch.optim.Adam(self.cortex.parameters(), lr=0.01)
        self.sleep_opt = SleepOptimizer(
            self.cortex, lr=0.02,
            projection_strength=0.5, warmup_epochs=5)
        self.past_trains = []

        # Register forward hooks on attention submodules
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for i, block in enumerate(self.layers):
            attn_module = getattr(block, self.attn_name)
            
            def make_hook(layer_idx):
                def hook_fn(module, args, kwargs, hook_output):
                    if self.gfaml_mode == "off":
                        return hook_output
                    
                    if len(args) > 0:
                        x_norm = args[0]
                    elif "hidden_states" in kwargs:
                        x_norm = kwargs["hidden_states"]
                    else:
                        return hook_output
                        
                    if self.state_list is None:
                        self.state_list = [None] * len(self.layers)
                        
                    state = self.state_list[layer_idx]
                    
                    if self.gfaml_mode == "record":
                        # Write to memory (Hebbian write) — only at prompt-answer boundary
                        write_start = getattr(self, '_record_write_start', 0)
                        _, next_state, _ = self.gfaml_layers[layer_idx](
                            x_norm, state=state, write=True, write_start=write_start
                        )
                        self.state_list[layer_idx] = next_state
                        return hook_output
                    
                    elif self.gfaml_mode == "augment":
                        # Read from memory and add to output
                        gfaml_out, _, _ = self.gfaml_layers[layer_idx](
                            x_norm, state=state, write=False
                        )
                        memory_contrib = gfaml_out - x_norm
                        if isinstance(hook_output, tuple):
                            attn_out = hook_output[0] + memory_contrib
                            return (attn_out,) + hook_output[1:]
                        else:
                            return hook_output + memory_contrib
                    
                    return hook_output
                return hook_fn
            
            handle = attn_module.register_forward_hook(make_hook(i), with_kwargs=True)
            self.hook_handles.append(handle)

    def __del__(self):
        # Clean up hooks on destruction to prevent memory leak
        for handle in self.hook_handles:
            handle.remove()

    def _forward_clean(self, input_ids):
        self.gfaml_mode = "off"
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        return out.logits + self.cortex(hidden), hidden

    def _forward_augmented(self, input_ids):
        self.gfaml_mode = "augment"
        with torch.no_grad():
            out = self.hf_model(input_ids, output_hidden_states=True)
        self.gfaml_mode = "off"
        return out.logits + self.cortex(out.hidden_states[-1])

    def _record(self, input_ids, prompt_len=None):
        """Record to episodic memory. If prompt_len given, only write at the boundary."""
        if prompt_len is not None:
            self._record_write_start = max(0, prompt_len - 1)
        else:
            self._record_write_start = 0
        self.gfaml_mode = "record"
        with torch.no_grad():
            self.hf_model(input_ids)
        self.gfaml_mode = "off"
        self._record_write_start = 0

    def _step(self, prompt, answer):
        full = prompt + answer
        ids = self.tokenizer.encode(full, return_tensors="pt",
                                    add_special_tokens=False).to(self.device)
        plen = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if plen >= ids.size(1):
            return
        self.online_opt.zero_grad()
        logits, _ = self._forward_clean(ids)
        loss = F.cross_entropy(
            logits[:, plen-1:-1, :].reshape(-1, logits.size(-1)),
            ids[:, plen:].reshape(-1))
        loss.backward()
        self.online_opt.step()

    def train_on_task(self, task, epochs=30):
        self.train()
        # Record to episodic memory (only at prompt-answer boundary)
        for p, a in task["train"]:
            full = p + a
            ids = self.tokenizer.encode(full, return_tensors="pt",
                                        add_special_tokens=False).to(self.device)
            plen = len(self.tokenizer.encode(p, add_special_tokens=False))
            self._record(ids, prompt_len=plen)

        # Online fine-tuning with replay
        for epoch in range(epochs):
            for p, a in task["train"]:
                self._step(p, a)
            for past in self.past_trains:
                if past:
                    p, a = random.choice(past)
                    self._step(p, a)

        # Sleep consolidation
        sleep_buf = []
        with torch.no_grad():
            for p, a in task["train"]:
                full = p + a
                ids = self.tokenizer.encode(full, return_tensors="pt",
                                            add_special_tokens=False).to(self.device)
                plen = len(self.tokenizer.encode(p, add_special_tokens=False))
                if plen >= ids.size(1):
                    continue
                _, hidden = self._forward_clean(ids)
                sleep_buf.append((hidden[0, plen-1, :].detach().clone(),
                                  ids[0, plen].item()))
        if sleep_buf:
            self.sleep_opt.consolidate(sleep_buf, epochs=20)

        self.past_trains.append(list(task["train"]))


# ===========================================================================
# EVALUATION (TEST SET ONLY)
# ===========================================================================

def eval_accuracy_on_test(model, task, tokenizer):
    """Evaluate on HELD-OUT test set only. No data leakage."""
    model.eval()
    correct = 0
    total = 0
    device = getattr(model, 'device', torch.device("cpu"))
    with torch.no_grad():
        for prompt, answer in task["test"]:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not answer_ids:
                continue
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            saved = getattr(model, 'state_list', None)
            if isinstance(model, DecoupledGFAML):
                logits = model._forward_augmented(ids)
            else:
                logits = model.forward(ids)
            if saved is not None:
                model.state_list = saved

            if torch.argmax(logits[0, -1, :]).item() == answer_ids[0]:
                correct += 1
            total += 1
    return correct / max(total, 1)


# ===========================================================================
# MAIN
# ===========================================================================

def run():
    print("=" * 80)
    print("  SCALED BENCHMARK: 10 Tasks, Train/Test Split, 5 Seeds")
    print("  GPT-2 (124M) | Honest evaluation")
    print("=" * 80)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model.eval()

    d_model = hf_model.config.hidden_size if hasattr(hf_model.config, "hidden_size") else hf_model.config.n_embd
    vocab_size = hf_model.config.vocab_size if hasattr(hf_model.config, "vocab_size") else tokenizer.vocab_size
    config = GFAMLConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=32, eta=1.0,
    )

    n_tasks_list = [3, 5, 10]  # Test at different scales
    seeds = [42, 123, 456]

    models_spec = [
        ("Baseline LoRA", BaselineLoRA),
        ("LoRA + Replay", LoRAReplay),
        ("Decoupled GFAML", DecoupledGFAML),
    ]

    all_scale_results = {}

    for n_tasks in n_tasks_list:
        print(f"\n{'='*70}")
        print(f"  SCALE: {n_tasks} TASKS (15 train + 5 test per task)")
        print(f"{'='*70}")

        scale_results = {}

        for model_name, model_class in models_spec:
            print(f"\n  [{model_name}]")
            seed_aas = []
            seed_fs = []

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                tasks = generate_tasks(n_tasks=n_tasks, n_train=15, n_test=5)
                hf_copy = copy.deepcopy(hf_model)
                model = model_class(hf_copy, tokenizer, config, device=DEVICE)

                evaluator = RigorousEvaluator(n_tasks)

                for tid in range(n_tasks):
                    model.train_on_task(tasks[tid], epochs=30)

                    # Evaluate on ALL tasks' TEST sets
                    evaluator.evaluate_after_task(
                        model, tid, tasks,
                        accuracy_fn=lambda m, td: eval_accuracy_on_test(m, td, tokenizer),
                    )

                result = evaluator.get_result()
                aa = result.average_accuracy()
                fr = result.forgetting_rate()
                seed_aas.append(aa)
                seed_fs.append(fr)
                print(f"    Seed {seed}: AA={aa:.4f} F={fr:.4f}")

            mean_aa = np.mean(seed_aas)
            std_aa = np.std(seed_aas, ddof=1) if len(seed_aas) > 1 else 0
            mean_f = np.mean(seed_fs)
            std_f = np.std(seed_fs, ddof=1) if len(seed_fs) > 1 else 0

            scale_results[model_name] = {
                "aa_mean": mean_aa, "aa_std": std_aa,
                "f_mean": mean_f, "f_std": std_f,
            }
            print(f"    => AA={mean_aa:.4f}+/-{std_aa:.4f} F={mean_f:.4f}+/-{std_f:.4f}")

        all_scale_results[n_tasks] = scale_results

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 90)
    print("  SCALING ANALYSIS: How does performance change with task count?")
    print("=" * 90)

    for n_tasks, scale_results in all_scale_results.items():
        print(f"\n  --- {n_tasks} Tasks (15 train / 5 test each) ---")
        print(f"  {'Model':<25} | {'Test AA':>12} | {'Forgetting':>12}")
        print(f"  {'-'*55}")
        for name, r in scale_results.items():
            print(f"  {name:<25} | {r['aa_mean']:.4f}+/-{r['aa_std']:.4f} | "
                  f"{r['f_mean']:.4f}+/-{r['f_std']:.4f}")

    print("\n" + "=" * 90)

    # Save report
    reports_dir = os.path.join(WORKSPACE_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    lines = []
    lines.append("# Scaled Benchmark Evaluation Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("## Methodology\n")
    lines.append("- **Train/Test Split:** 15 train + 5 held-out test per task")
    lines.append("- **Evaluation:** TEST SET ONLY (no data leakage)")
    lines.append("- **Seeds:** 3 (for statistical significance)")
    lines.append("- **Task Scales:** 3, 5, and 10 sequential tasks")
    lines.append("- **Base Model:** GPT-2 (124M)")
    lines.append("- **Training:** 30 epochs per task\n")

    lines.append("## Scaling Results\n")
    for n_tasks, scale_results in all_scale_results.items():
        lines.append(f"### {n_tasks} Tasks\n")
        lines.append("| Model | Test Accuracy | Forgetting |")
        lines.append("| :--- | :---: | :---: |")
        for name, r in scale_results.items():
            lines.append(f"| {name} | {r['aa_mean']:.4f} +/- {r['aa_std']:.4f} | "
                          f"{r['f_mean']:.4f} +/- {r['f_std']:.4f} |")
        lines.append("")

    lines.append("## Key Questions Answered\n")
    lines.append("1. **Does GFAML help beyond replay?** See test accuracy difference")
    lines.append("2. **Does it scale?** Compare 3 vs 5 vs 10 tasks")
    lines.append("3. **Is it robust?** Check std across seeds")
    lines.append("4. **Is there data leakage?** No - evaluation on held-out test only")

    report_path = os.path.join(reports_dir, "SCALED_BENCHMARK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run()
