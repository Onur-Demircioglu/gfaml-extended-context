# gfaml/eval/rigorous_metrics.py
"""
Research-grade continual learning evaluation metrics.

References:
- Lopez-Paz & Ranzato (2017): Gradient Episodic Memory for Continual Learning
- Chaudhry et al. (2018): Riemannian Walk for Incremental Learning
- Kemker et al. (2018): Measuring Catastrophic Forgetting in Neural Networks

No overclaims. No demo tricks. Real numbers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class TaskResult:
    """Single task evaluation snapshot."""
    task_id: int
    accuracy: float
    perplexity: float
    loss: float


@dataclass
class ContinualResult:
    """Full continual learning evaluation result across all tasks and time steps."""
    # R[i][j] = accuracy on task j after training on task i
    # This is the standard CL evaluation matrix
    accuracy_matrix: np.ndarray  # shape (n_tasks, n_tasks)
    perplexity_matrix: np.ndarray  # shape (n_tasks, n_tasks)
    loss_matrix: np.ndarray  # shape (n_tasks, n_tasks)

    @property
    def n_tasks(self) -> int:
        return self.accuracy_matrix.shape[0]

    def average_accuracy(self) -> float:
        """AA = (1/T) * sum_j R[T,j] -- final row average."""
        return float(np.mean(self.accuracy_matrix[-1, :]))

    def forgetting_rate(self) -> float:
        """
        Average Forgetting (Lopez-Paz & Ranzato 2017):
        F = (1/(T-1)) * sum_{j=1}^{T-1} max_{l in {1..T-1}} (R[l,j] - R[T,j])

        Higher = worse. 0 = no forgetting. Negative = impossible (backward transfer).
        """
        T = self.n_tasks
        if T <= 1:
            return 0.0
        forgetting_per_task = []
        for j in range(T - 1):
            best_past = max(self.accuracy_matrix[l, j] for l in range(T - 1))
            final = self.accuracy_matrix[T - 1, j]
            forgetting_per_task.append(best_past - final)
        return float(np.mean(forgetting_per_task))

    def backward_transfer(self) -> float:
        """
        BWT = (1/(T-1)) * sum_{j=1}^{T-1} (R[T,j] - R[j,j])

        Negative = catastrophic forgetting. Positive = beneficial backward transfer.
        """
        T = self.n_tasks
        if T <= 1:
            return 0.0
        bwt_per_task = []
        for j in range(T - 1):
            bwt_per_task.append(self.accuracy_matrix[T - 1, j] - self.accuracy_matrix[j, j])
        return float(np.mean(bwt_per_task))

    def forward_transfer(self) -> float:
        """
        FWT = (1/(T-1)) * sum_{j=2}^{T} (R[j-1,j] - baseline[j])

        Measures zero-shot transfer before seeing task j.
        baseline[j] assumed 0 (random init accuracy).
        """
        T = self.n_tasks
        if T <= 1:
            return 0.0
        fwt_per_task = []
        for j in range(1, T):
            fwt_per_task.append(self.accuracy_matrix[j - 1, j])
        return float(np.mean(fwt_per_task))

    def perplexity_shift(self) -> Dict[str, float]:
        """
        Perplexity shift per task: how much perplexity increases
        on task j between its best point and the final evaluation.
        """
        T = self.n_tasks
        shifts = {}
        for j in range(T):
            best_ppl = min(self.perplexity_matrix[l, j] for l in range(T))
            final_ppl = self.perplexity_matrix[T - 1, j]
            shifts[f"task_{j}"] = float(final_ppl - best_ppl)
        shifts["mean"] = float(np.mean(list(shifts.values())))
        return shifts

    def learning_curve(self, task_id: int) -> List[float]:
        """Accuracy on task_id after each sequential training phase."""
        return [float(self.accuracy_matrix[t, task_id]) for t in range(self.n_tasks)]

    def forgetting_curve(self, task_id: int) -> List[float]:
        """
        Forgetting curve for a specific task:
        f(t) = max_{l<=t} R[l, task_id] - R[t, task_id]
        """
        curve = []
        for t in range(self.n_tasks):
            best_so_far = max(self.accuracy_matrix[l, task_id] for l in range(t + 1))
            curve.append(best_so_far - self.accuracy_matrix[t, task_id])
        return curve


def compute_perplexity(model, eval_fn, data) -> float:
    """
    Compute perplexity = exp(avg cross-entropy loss).
    eval_fn(model, data) -> average CE loss (float)
    """
    avg_loss = eval_fn(model, data)
    return float(math.exp(min(avg_loss, 20.0)))  # cap to avoid inf


def compute_accuracy(model, eval_fn, data) -> float:
    """
    Compute accuracy.
    eval_fn(model, data) -> accuracy in [0, 1]
    """
    return float(eval_fn(model, data))


class RigorousEvaluator:
    """
    Standard continual learning evaluator.

    Usage:
        evaluator = RigorousEvaluator(n_tasks=3)

        for task_id in range(n_tasks):
            train(model, task_data[task_id])
            evaluator.evaluate_after_task(model, task_id, task_data, eval_fn)

        result = evaluator.get_result()
        print(result.forgetting_rate())
        print(result.backward_transfer())
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.accuracy_matrix = np.zeros((n_tasks, n_tasks))
        self.perplexity_matrix = np.full((n_tasks, n_tasks), float('inf'))
        self.loss_matrix = np.full((n_tasks, n_tasks), float('inf'))

    def evaluate_after_task(
        self,
        model: nn.Module,
        trained_task_id: int,
        all_task_data: List,
        accuracy_fn,
        loss_fn=None,
    ):
        """
        After training on task `trained_task_id`, evaluate on ALL tasks.

        accuracy_fn(model, task_data) -> float in [0,1]
        loss_fn(model, task_data) -> float (CE loss), optional
        """
        model.eval()
        with torch.no_grad():
            for j in range(self.n_tasks):
                acc = accuracy_fn(model, all_task_data[j])
                self.accuracy_matrix[trained_task_id, j] = acc

                if loss_fn is not None:
                    loss = loss_fn(model, all_task_data[j])
                    self.loss_matrix[trained_task_id, j] = loss
                    self.perplexity_matrix[trained_task_id, j] = math.exp(min(loss, 20.0))
                else:
                    self.perplexity_matrix[trained_task_id, j] = float('nan')
                    self.loss_matrix[trained_task_id, j] = float('nan')

    def get_result(self) -> ContinualResult:
        return ContinualResult(
            accuracy_matrix=self.accuracy_matrix.copy(),
            perplexity_matrix=self.perplexity_matrix.copy(),
            loss_matrix=self.loss_matrix.copy(),
        )


class MultiSeedEvaluator:
    """
    Run the full evaluation across multiple seeds and compute
    mean +/- confidence intervals.

    Usage:
        multi = MultiSeedEvaluator(n_tasks=3, seeds=[42, 123, 456])
        for seed in multi.seeds:
            model = create_model(seed)
            evaluator = multi.get_evaluator(seed)
            for task_id in range(n_tasks):
                train(model, task_data[task_id])
                evaluator.evaluate_after_task(...)
            multi.submit(seed, evaluator.get_result())

        summary = multi.summary()
    """

    def __init__(self, n_tasks: int, seeds: List[int]):
        self.n_tasks = n_tasks
        self.seeds = seeds
        self.results: Dict[int, ContinualResult] = {}
        self.evaluators: Dict[int, RigorousEvaluator] = {
            seed: RigorousEvaluator(n_tasks) for seed in seeds
        }

    def get_evaluator(self, seed: int) -> RigorousEvaluator:
        return self.evaluators[seed]

    def submit(self, seed: int, result: ContinualResult):
        self.results[seed] = result

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Returns mean and 95% CI for key metrics across seeds.
        """
        if len(self.results) == 0:
            return {}

        metrics = {
            "average_accuracy": [],
            "forgetting_rate": [],
            "backward_transfer": [],
            "forward_transfer": [],
        }

        for result in self.results.values():
            metrics["average_accuracy"].append(result.average_accuracy())
            metrics["forgetting_rate"].append(result.forgetting_rate())
            metrics["backward_transfer"].append(result.backward_transfer())
            metrics["forward_transfer"].append(result.forward_transfer())

        summary = {}
        for name, values in metrics.items():
            arr = np.array(values)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            n = len(arr)
            # 95% CI using t-distribution approximation
            t_val = 2.776 if n == 5 else (2.262 if n == 10 else (1.96 if n >= 30 else 2.0))
            ci = t_val * std / math.sqrt(n) if n > 1 else 0.0
            summary[name] = {
                "mean": mean,
                "std": std,
                "ci_95_lower": mean - ci,
                "ci_95_upper": mean + ci,
                "n_seeds": n,
            }

        return summary


def stability_under_noise(
    model: nn.Module,
    eval_fn,
    task_data,
    noise_levels: List[float] = None,
) -> Dict[float, float]:
    """
    Measure accuracy degradation as input noise sigma increases.
    Returns {sigma: accuracy} dict.

    eval_fn(model, task_data, noise_sigma) -> accuracy
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = {}
    model.eval()
    with torch.no_grad():
        for sigma in noise_levels:
            acc = eval_fn(model, task_data, sigma)
            results[sigma] = float(acc)
    return results


def format_results_table(
    results: Dict[str, ContinualResult],
    model_names: List[str] = None,
) -> str:
    """
    Generate a clean markdown table comparing multiple models.
    No overclaims. Just numbers.
    """
    if model_names is None:
        model_names = list(results.keys())

    lines = []
    lines.append("## Continual Learning Evaluation Results")
    lines.append("")

    # Header
    header = "| Model | Avg Acc | Forgetting | BWT | FWT |"
    sep = "| :--- | :---: | :---: | :---: | :---: |"
    lines.append(header)
    lines.append(sep)

    for name in model_names:
        r = results[name]
        aa = r.average_accuracy()
        fr = r.forgetting_rate()
        bwt = r.backward_transfer()
        fwt = r.forward_transfer()
        lines.append(f"| {name} | {aa:.4f} | {fr:.4f} | {bwt:+.4f} | {fwt:+.4f} |")

    lines.append("")

    # Accuracy matrices
    for name in model_names:
        r = results[name]
        lines.append(f"### {name} - Accuracy Matrix R[i,j]")
        lines.append("*(Row i = trained up to task i, Col j = evaluated on task j)*")
        lines.append("")

        n = r.n_tasks
        header_cols = " | ".join([f"Task {j}" for j in range(n)])
        lines.append(f"| Trained | {header_cols} |")
        lines.append("| :--- | " + " | ".join([":---:"] * n) + " |")

        for i in range(n):
            cols = " | ".join([f"{r.accuracy_matrix[i, j]:.4f}" for j in range(n)])
            lines.append(f"| After Task {i} | {cols} |")

        lines.append("")

    return "\n".join(lines)


def format_multi_seed_summary(summary: Dict[str, Dict[str, float]]) -> str:
    """Format multi-seed summary as markdown."""
    lines = []
    lines.append("## Statistical Summary (Multi-Seed)")
    lines.append("")
    lines.append("| Metric | Mean | Std | 95% CI | N |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for name, stats in summary.items():
        ci_str = f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]"
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | {ci_str} | {stats['n_seeds']} |"
        )

    return "\n".join(lines)
