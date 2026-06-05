# GFAML: Gradient-Free Associative Memory Layers for Extended Context

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Gradient-Free Associative Memory Layers (GFAML)** is an experimental dual-system continual learning architecture that implements complementary learning pathways directly inside pre-trained Transformer models without parameter co-adaptation.

Inspired by the mammalian brain's **Complementary Learning Systems (CLS)** theory, GFAML decouples fast episodic acquisition from slow semantic integration to achieve zero-gradient instant learning without catastrophic forgetting.

---

## 📋 Table of Contents

- [Core Architecture](#-core-architecture)
- [Key Achievements](#-key-achievements)
- [Failure Modes & Scientific Integrity](#-failure-modes--scientific-integrity)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Experiments & Benchmarks](#-experiments--benchmarks)
- [Research Documents](#-research-documents)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🧠 Core Architecture

GFAML splits the model's memory pathways into two parallel structures inspired by biological learning systems:

### 1. **Episodic Memory (Hippocampus)**
- Non-parametric, low-rank stateful associative layers ($U, V$ matrices)
- Embedded in the residual stream of Transformer blocks
- Updates are completely **gradient-free** using Hebbian learning rules
- Enables instant fact acquisition during inference (<200ms on consumer hardware)

### 2. **Parametric Cortex (Cortex)**
- Low-rank adapters (LoRA) optimized during offline consolidation phases
- Uses **Orthogonal Gradient Projection (OGP)** to prevent catastrophic forgetting
- Consolidates episodic memories into static parameters with minimal interference

### Wake-Sleep Cycle

**Wake Phase (Inference):**
- User provides new facts during natural conversation
- Hebbian Boundary Write applies to episodic memory matrices without backpropagation
- Model retrieves and integrates memories into output

**Sleep Phase (Training):**
- Collected episodic memories consolidated into LoRA parameters
- OGP prevents catastrophic forgetting on existing knowledge
- No gradients flow through backbone

### Architecture Diagram

```
                ┌──────────────────────────────────────┐
                │    Transformer Backbone (Frozen)     │
                └──────────┬────────────────┬──────────┘
                           │ (Residual)     │
                           ▼                ▼
                  ┌────────────────┐   ┌───────────────┐
                  │  GFAML Layer   │   │  LoRA Cortex  │
                  │ (Hippocampus)  │   │   (Cortex)    │
                  └────────┬───────┘   └────────┬──────┘
                           │                    │
         [Wake Phase]      │ Hebbian            │ Gradient [Sleep Phase]
         Instant Write <───┤ Update (U,V)       │ Descent (OGP)
                           │                    │
                           ▼                    ▼
                    Memory Retrieval ──► Combined Logits
```

---

## 🚀 Key Achievements

- **Zero-Gradient Instant Learning:** Using Hebbian updates, new facts are learned **in under 200ms on a consumer laptop** without backward passes.

- **Zero Catastrophic Forgetting:** Decoupling episodic memory from model parameters completely eliminates forgetting on general knowledge.

- **Negative Forgetting (Backward Transfer):** 10-task sequential benchmark yields **-0.0167 forgetting rate** (model improves by recalling past knowledge).

- **Hardware Efficient:** Works with consumer-grade GPUs and laptops. No expensive fine-tuning.

---

## 🛑 Failure Modes & Scientific Integrity

True AI engineering involves documenting the physical boundaries of our experiments:

### 1. Output Saturation & Decoder Bypass

The episodic memory vector injected into the residual stream becomes a strong local attractor. Without careful calibration, attention heads lock onto this vector instead of performing semantic reasoning.

**Mitigation:** Soft Energy Saturation enforces normalized magnitude constraints:
$$m_t^{\text{stabilized}} = \frac{m_t}{\sqrt{1.0 + \|m_t\|_2^2 + \epsilon}}$$

### 2. Exact String Lock (Lack of Semantic Grounding)

Hebbian updates link exact sequence activations, making learned memories sensitive to prompt formatting. Learning "Onur's favorite language is Python" does not automatically transfer to "What programming language does Onur prefer?"

**Root Cause:** Episodic memory stores **syntactic patterns**, not semantic concepts.

**Future Work:** Use continuous nearest-neighbor retrieval or semantic embeddings.

### 3. GPU Memory Bandwidth Bottleneck

Modern GPUs optimize for static weights (high arithmetic intensity). Dynamic weight updates trigger sequential DRAM load/write cycles, creating a bottleneck.

**Workaround:** Batch updates or use experience replay during sleep phases.

---

## 📦 Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy ≥1.22.0
- PyYAML ≥6.0
- (Optional) Transformers ≥4.30.0 for Hugging Face models
- (Optional) tqdm, matplotlib for visualization

### Setup

```bash
# Clone the repository
git clone https://github.com/Onur-Demircioglu/gfaml-extended-context.git
cd gfaml-extended-context

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Hugging Face Transformers
pip install transformers>=4.30.0
```

---

## 🚀 Quick Start

### 1. Interactive Chat with Instant Learning

Teach the model facts on-the-fly with Qwen-2.5-0.5B:

```bash
python scripts/chat_gfaml.py
```

**Commands:**
- `/teach <fact>` - Teach a fact (e.g., `/teach The capital of Turkey is = Ankara`)
- `/debug` - Toggle token-level visualization
- `/exit` - Quit

**Example:**
```
You: /teach Onur likes machine learning = True
[Hebbian update applied, <200ms]

You: Does Onur like machine learning?
Bot: Yes, based on what I learned, Onur likes machine learning.
```

### 2. Run Quick Tests

Test instant learning end-to-end:
```bash
python scripts/test_instant_learn.py
```

Quick sanity check:
```bash
python scripts/test_chat_quick.py
```

Test pure Hebbian memory component:
```bash
python scripts/test_pure_memory.py
```

### 3. Run 10-Task Sequential Benchmark

Evaluate catastrophic forgetting across 10 disjoint tasks:

```bash
python experiments/exp_12_scaled_benchmark.py
```

**Output:** Results saved to `outputs/` with CSV metrics and comparison report.

**Expected Results:**
```
Baseline GPT-2:          Forgetting ≈ 0.87 (high)
GFAML (Episodic):        Forgetting ≈ -0.0167 (negative = improvement!)
GFAML (+ OGP):           Forgetting ≈ 0.0 (zero forgetting)
```

### 4. Generate Visualizations

Create plots from experiment results:
```bash
python scripts/generate_plots.py
```

### 5. Run All Experiments

Sequential progression through all 12 experiments:
```bash
bash scripts/run_all.sh
```

---

## 📂 Project Structure

```
gfaml-extended-context/
│
├── gfaml/                          # Core library
│   ├── config.py                   # Hyperparameters (rank, eta, decay, etc.)
│   ├── core/
│   │   ├── gfaml_layer.py         # Main GFAML episodic memory layer
│   │   ├── gating.py              # Memory write gating mechanisms
│   │   ├── engine.py              # Experiment orchestrator
│   │   ├── trainer.py             # Sleep phase consolidation
│   │   ├── memory_state.py        # Memory state management
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── logger.py              # Logging utilities
│   ├── models/
│   │   ├── baseline_transformer.py # Baseline model wrapper
│   │   └── transformer_gfaml.py   # GFAML-augmented wrapper
│   └── run.py                      # Main entry point
│
├── experiments/                    # 12-phase R&D progression
│   ├── exp_01_baseline.py          # Baseline GPT-2
│   ├── exp_02_gfaml.py             # Basic GFAML integration
│   ├── exp_03_ablation.py          # Component ablations
│   ├── exp_04_real_data.py         # Real dataset tests
│   ├── exp_05_hf_model.py          # Hugging Face integration
│   ├── exp_06_real_gpt2.py         # Full GPT-2 GFAML
│   ├── exp_07_rigorous_eval.py     # Comprehensive metrics
│   ├── exp_08_ogp_sweep.py         # Hyperparameter tuning
│   ├── exp_09_fixed_eval.py        # Standardized protocol
│   ├── exp_10_experience_replay.py # Replay buffer tests
│   ├── exp_11_decoupled_gfaml.py   # Decoupled architecture
│   └── exp_12_scaled_benchmark.py  # Final 10-task benchmark ⭐
│
├── scripts/                        # Utilities and demos
│   ├── chat_gfaml.py              # Interactive GFAML chat
│   ├── chat_gpt2.py               # Baseline GPT-2 chat
│   ├── test_instant_learn.py      # Instant learning validation
│   ├── test_pure_memory.py        # Memory component test
│   ├── test_chat_auto.py          # Automated chat testing
│   ├── test_chat_quick.py         # Quick validation
│   ├── generate_plots.py          # Plot generation
│   └── run_all.sh                 # Batch runner
│
├── configs/                        # YAML configuration files
├── benchmarks/                     # Benchmark results
├── papers/                         # Reference materials
├── scratch/                        # Temporary files
│
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Project metadata
├── LICENSE                         # MIT License
├── README.md                       # This file
│
├── GFAML_FINAL_RESEARCH_REPORT.md          # 📊 Complete findings
├── GFAML_DEVELOPER_GUIDE.md               # 🛠️ Technical deep-dive
├── GFAML_MATHEMATICAL_HARDWARE_LIMITS.md  # 📐 Math formulations
├── COGNITIVE_PHYSICS_RAPORU.md            # 🧠 Biological grounding
├── HAFIZA_DENETIM_RAPORU.md               # 📈 Memory profiling
└── SUKM_COMPARE_RAPORU.md                 # 🔄 Comparative analysis
```

---

## 🔬 Experiments & Benchmarks

### 12-Phase R&D Progression

| Exp | Purpose | Model | Phase |
|-----|---------|-------|-------|
| 01 | Baseline GPT-2 | GPT-2 345M | Proof of Concept |
| 02 | GFAML integration | GPT-2 + Episodic | Initial Implementation |
| 03 | Component ablation | Lite GFAML | Testing |
| 04 | Real data validation | GPT-2 | Real-world data |
| 05 | Hugging Face support | Multiple HF models | Framework support |
| 06 | Full GPT-2 GFAML | GPT-2 + GFAML | Production-ready |
| 07 | Rigorous evaluation | GPT-2 + GFAML | Comprehensive metrics |
| 08 | Hyperparameter sweep | GPT-2 + OGP | Optimization |
| 09 | Standardized protocol | GPT-2 + GFAML | Protocol validation |
| 10 | Experience replay | GPT-2 + Replay | Replay buffer testing |
| 11 | Decoupled GFAML | GPT-2 Decoupled | Architecture finalization |
| 12 | **10-Task Benchmark** | **GPT-2 + Qwen** | **Final Validation** ⭐ |

### Running Experiments

**Individual experiment:**
```bash
python experiments/exp_12_scaled_benchmark.py --num_tasks 10 --seed 42
```

**All experiments (takes ~2-3 hours):**
```bash
bash scripts/run_all.sh
```

### Configuration

Edit `gfaml/config.py` to adjust hyperparameters:

```python
@dataclass
class GFAMLConfig:
    # Memory dimensions
    d_model: int = 128          # Hidden dimension
    rank: int = 8               # Low-rank approximation rank
    
    # Learning dynamics
    lambda_decay: float = 0.995 # Memory decay rate
    eta: float = 0.01           # Hebbian update rate
    alpha: float = 1.0          # Memory contribution weight
    
    # Stability
    use_soft_stabilization: bool = True  # Energy saturation
    saturation_threshold: float = 5.0
    
    # Gating
    use_gating: bool = True
    gate_bias: float = -1.0     # For sparsity
```

---

## 🔗 Hook Integration (Technical Details)

GFAML uses PyTorch `forward_hook` to intercept transformer residual streams:

```python
# From exp_12_scaled_benchmark.py
def _register_hooks(self):
    for i, block in enumerate(self.layers):
        attn_module = getattr(block, self.attn_name)
        
        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs, hook_output):
                # 'record' mode: Apply Hebbian update to U, V
                # 'augment' mode: Read memory and add to residual
                ...
                return modified_output
            return hook_fn
        
        handle = attn_module.register_forward_hook(
            make_hook(i), with_kwargs=True
        )
```

This allows GFAML to inject/retrieve memory without modifying the model architecture.

---

## 📚 Research Documents

Deep-dive into specific topics:

| Document | Contents |
|----------|----------|
| **[Final Research Report](GFAML_FINAL_RESEARCH_REPORT.md)** | Complete R&D summary, 4-phase evolution, ablation studies, final takeaways |
| **[Developer Guide](GFAML_DEVELOPER_GUIDE.md)** | Hook architecture, code walkthrough, porting to new models (Gemma, Llama, etc.) |
| **[Mathematical & Hardware Analysis](GFAML_MATHEMATICAL_HARDWARE_LIMITS.md)** | Exact Hebbian formulas, energy stabilization, decoder bypass analysis |
| **[Cognitive Physics Report](COGNITIVE_PHYSICS_RAPORU.md)** | Biological grounding in CLS theory and neuroscience |
| **[Memory Audit Report](HAFIZA_DENETIM_RAPORU.md)** | Detailed memory profiling and resource usage analysis |
| **[Comparative Analysis](SUKM_COMPARE_RAPORU.md)** | GFAML vs. alternative memory architectures |

---

## 🛠️ Troubleshooting

### Installation Issues

**PyTorch not found:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Transformers import error:**
```bash
pip install transformers>=4.30.0
```

### Runtime Issues

**CUDA out of memory:**
- Reduce batch size in configs
- Use CPU: set `device = 'cpu'` in config

**Memory not being learned (0% accuracy):**
- Increase Hebbian learning rate: `eta = 0.05`
- Check gate bias: `gate_bias = -0.5`

**Slow inference:**
```python
import torch
print(torch.cuda.is_available())  # Check GPU
# If False, reinstall PyTorch with CUDA support
```

**Model not loading:**
```bash
# Verify Hugging Face model availability
huggingface-cli repo info gpt2
huggingface-cli repo info Qwen/Qwen2.5-0.5B
```

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Profile memory:**
   ```bash
   python -m memory_profiler scripts/test_instant_learn.py
   ```

3. **Inspect model structure:**
   ```bash
   python scripts/test_chat_quick.py
   ```

4. **Validate data loading:**
   ```python
   from gfaml.datasets import load_benchmark
   load_benchmark('10_tasks')
   ```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📖 How to Use This Repository

1. **For Quick Demo:** Start with `scripts/chat_gfaml.py`
2. **For Understanding:** Read `GFAML_DEVELOPER_GUIDE.md` first
3. **For Deep Dive:** See `GFAML_FINAL_RESEARCH_REPORT.md`
4. **For Implementation:** Reference `gfaml/core/gfaml_layer.py`
5. **For Validation:** Run `experiments/exp_12_scaled_benchmark.py`

---

## 🎯 Key Takeaways

✅ **What Works:**
- Zero-gradient instant learning via Hebbian updates
- Episodic memory prevents catastrophic forgetting
- Works with frozen, pre-trained models
- Minimal computational overhead

❌ **What Doesn't:**
- Semantic generalization (exact string lock)
- Real-time performance on some GPUs (bandwidth bottleneck)
- Standalone without pre-trained backbone

🔮 **Future Directions:**
- Semantic memory grounding
- Neuromorphic hardware implementations
- Integration with more model families
- Continuous context expansion

---

**Last Updated:** June 5, 2024  
**Status:** Active Research  
**Maintainer:** Onur Demircioglu

For questions or issues, refer to the [Troubleshooting](#-troubleshooting) section or check the research documents above.
