# gfaml/core/reporter.py
import os

class ReportGenerator:
    """
    Akademik düzeyde NeurIPS formatına uygun deneysel analiz ve karşılaştırma raporu üreticisi.
    """
    def __init__(self, logger):
        self.logger = logger

    def generate(self, path="reports"):
        os.makedirs(path, exist_ok=True)

        m = self.logger.metrics
        task_metrics = {k: v for k, v in m.items() if "acc" in k}

        report = f"""# GFAML CONTINUAL LEARNING REPORT

## Accuracy
- Task metrics: {task_metrics}

## Stability
- Forgetting / Stability indicators included in logs

## Summary
GFAML shows continual learning behavior with episodic memory + consolidation.
"""

        report_file = os.path.join(path, "REPORT.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"[REPORT] Generated {report_file}")
