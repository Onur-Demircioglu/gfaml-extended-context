# scripts/generate_plots.py
import os
import matplotlib.pyplot as plt
import numpy as np

# Premium styling (Minimalist, elegant)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
})

# Data from scaled GPU benchmark
scales = np.array([3, 5, 10])

# Average Accuracy (AA) results
lora_aa = np.array([0.0444, 0.0800, 0.0133]) * 100
replay_aa = np.array([0.0444, 0.1067, 0.0800]) * 100
gfaml_aa = np.array([0.0000, 0.1067, 0.0533]) * 100

# Forgetting Rate (F) results
lora_f = np.array([0.0000, 0.0667, 0.1111]) * 100
replay_f = np.array([0.0000, 0.0667, 0.0667]) * 100
gfaml_f = np.array([0.0000, -0.0167, 0.0667]) * 100

os.makedirs("papers/figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1: Forgetting Rate Curve (Lower is Better)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

ax.plot(scales, lora_f, marker='o', linewidth=2.5, color='#e056fd', label='Baseline LoRA')
ax.plot(scales, replay_f, marker='s', linewidth=2.5, color='#30336b', label='LoRA + Replay')
ax.plot(scales, gfaml_f, marker='^', linewidth=2.5, color='#22a6b3', label='Decoupled GFAML (Ours)')

# Annotate negative forgetting
ax.annotate('Negative Forgetting\n(Backward Transfer)', 
            xy=(5, -1.67), xytext=(6, -5),
            arrowprops=dict(facecolor='#22a6b3', shrink=0.08, width=1.5, headwidth=6),
            fontsize=9, color='#22a6b3', fontweight='semibold')

ax.set_title("Forgetting Rate vs. Task Horizon Scale", pad=15, fontweight='bold', color='#2c3e50')
ax.set_xlabel("Number of Sequential Tasks", labelpad=10)
ax.set_ylabel("Forgetting Rate (%)", labelpad=10)
ax.set_xticks(scales)
ax.set_ylim(-10, 20)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0')

plt.tight_layout()
plot_path = "papers/figures/forgetting_curve.png"
plt.savefig(plot_path)
plt.close()
print(f"Saved forgetting curve plot to: {plot_path}")

# ---------------------------------------------------------------------------
# Plot 2: Average Accuracy (Test Set)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

x = np.arange(len(scales))
width = 0.25

rects1 = ax.bar(x - width, lora_aa, width, label='Baseline LoRA', color='#dff9fb', edgecolor='#c7ecee')
rects2 = ax.bar(x, replay_aa, width, label='LoRA + Replay', color='#c7ecee', edgecolor='#95afc0')
rects3 = ax.bar(x + width, gfaml_aa, width, label='Decoupled GFAML (Ours)', color='#22a6b3', edgecolor='#138f9e')

ax.set_title("Average Test Accuracy across Scales", pad=15, fontweight='bold', color='#2c3e50')
ax.set_xlabel("Number of Sequential Tasks", labelpad=10)
ax.set_ylabel("Test Set Accuracy (%)", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(scales)
ax.set_ylim(0, 20)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0')

plt.tight_layout()
plot_path = "papers/figures/accuracy_comparison.png"
plt.savefig(plot_path)
plt.close()
print(f"Saved accuracy comparison plot to: {plot_path}")
