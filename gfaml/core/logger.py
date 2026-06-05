# gfaml/core/logger.py
import json
import os
from datetime import datetime

class ExperimentLogger:
    """
    Deneylerin tüm metriklerini ve günlüklerini standart formatta kaydeden araştırma kayıt sistemi.
    """
    def __init__(self, run_name="exp"):
        self.run_name = run_name
        self.logs = []
        self.metrics = {}

    def log_metric(self, key, value):
        self.metrics[key] = value
        print(f"[METRIC] {key} = {value}")

    def log(self, msg):
        print(f"[LOG] {msg}")
        self.logs.append(msg)

    def save(self, path="reports"):
        # Resolve path relative to workspace if necessary
        os.makedirs(path, exist_ok=True)

        data = {
            "run": self.run_name,
            "time": str(datetime.now()),
            "metrics": self.metrics,
            "logs": self.logs
        }

        file_path = os.path.join(path, f"{self.run_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[LOG] Saved metrics successfully to {file_path}")
