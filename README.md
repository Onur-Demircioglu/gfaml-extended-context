# GFAML: Gradient-Free Associative Memory Layers for Extended Context

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An experimental **Dual-System Continual Learning Architecture** that implements complementary learning pathways directly inside pre-trained Transformer models (GPT-2, Qwen-2.5) without parameter corruption.

Inspired by the mammalian brain's **Complementary Learning Systems (CLS)** theory, GFAML decouples fast episodic acquisition from slow semantic integration to achieve zero-gradient instant learning.

---

## 🧠 Core Architecture

GFAML splits the model's memory pathways into two parallel structures:
1. **Episodic Memory (Hippocampus):** Non-parametric, low-rank stateful associative layers ($U, V$ matrices) embedded in the residual stream. Updates are completely **gradient-free** and utilize localized Hebbian outer-product updates.
2. **Parametric Cortex (Cortex):** Low-rank adapters (LoRA) optimized during offline sleep phases using **Orthogonal Gradient Projection (OGP)** to consolidate memories into static parameters without representation drift.

```text
               ┌──────────────────────────────────────┐
               │         Transformer Backbone         │
               └──────────┬────────────────┬──────────┘
                          │ (Residual)     │
                          ▼                ▼
                 ┌────────────────┐   ┌───────────────┐
                 │  GFAML Layers  │   │  LoRA Cortex  │
                 │ (Hippocampus)  │   │   (Cortex)    │
                 └────────┬───────┘   └────────┬──────┘
                          │                    │
        [Wake Phase]      │ Hebbian            │ Gradient [Sleep Phase]
        Instant Write <───┤ Write              │ Descent (OGP Consolidation)
                          │                    │
                          ▼                    ▼
                   Episodic Retrieval ──► Combined Logits
```

---

## 🚀 Key Achievements

*   **Zero-Gradient Instant Learning:** Using Hebbian Boundary Write, the model learns new facts during natural conversation in **under 200ms on a consumer laptop** without backward passes.
*   **Zero Catastrophic Forgetting:** Decoupling the episodic memory from model parameters completely eliminates forgetting on general knowledge baselines.
*   **Negative Forgetting (Backward Transfer):** In our 5-task sequential benchmark, episodic retrieval during inference assisted in recalling past tasks, yielding a forgetting rate of **-0.0167**.

---

## 🔬 Core Research Documents

For deep-dive reviews of the mathematical formulations, experimental metrics, and hardware constraints, refer to our project reports:

*   📄 **[Final Research Report](GFAML_FINAL_RESEARCH_REPORT.md):** The complete R&D summary of our 4-phase evolution, ablasion studies, and final takeaways.
*   📐 **[Mathematical & Hardware Limits Analysis](GFAML_MATHEMATICAL_HARDWARE_LIMITS.md):** Exact formulas for Hebbian updates, energy stabilization techniques (Soft Energy Saturation), and a detailed breakdown of the memory bandwidth bottleneck in GPU architectures.
*   🛠️ **[Developer & Integration Guide](GFAML_DEVELOPER_GUIDE.md):** PyTorch forward hook architecture, directories map, execution flows, and instructions for porting GFAML to other Hugging Face models (e.g., Llama-3, Gemma).

---

## 🛑 Failure Modes & Scientific Integrity

True AI engineering involves documenting the physical boundaries of our experiments. We highlight three critical limits discovered during testing:

### 1. Output Saturation & Decoder Bypass
The Hebbian memory vector ($m_t$) injected into the residual stream acts as an extremely strong local attractor state. If uncalibrated, attention query heads lock onto this vector, bypassing the decoder's semantic processing and leading to repetitive token loops (e.g., repeating `Python Python Python...`).

### 2. Exact String Lock (Lack of Semantic Grounding)
Because Hebbian updates link exact sequence activations, the learned memory is sensitive to prompt formatting. Learning "Onur's favorite language is Python" does not automatically translate to answering paraphrased queries like "Which programming language does Onur prefer?".

### 3. GPU Memory Bandwidth Bottleneck
Modern GPUs are optimized for high arithmetic intensity with static weights. Step-by-step dynamic weight updates (Hebbian writes) trigger sequential DRAM load/write cycles, causing a bottleneck under Von Neumann architectures. True lifelong learning requires **In-Memory Computing** or **Neuromorphic Hardware**.

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/yourusername/gfaml-context.git
cd gfaml-context
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Instant Learning Chat (Qwen-2.5-0.5B)
Interact with the decoupled model and teach it facts on-the-fly:
```bash
python scripts/chat_gfaml.py
```
*   Teach facts: `/teach The capital of Turkey is = Ankara`
*   Toggle token-level visualization: `/debug`

### Run 10-Task Sequential Benchmark (GPT-2)
Evaluate catastrophic forgetting across a sequence of 10 disjoint tasks:
```bash
python experiments/exp_12_scaled_benchmark.py
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
