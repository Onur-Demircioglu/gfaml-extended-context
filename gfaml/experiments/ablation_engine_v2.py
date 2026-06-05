# gfaml/experiments/ablation_engine_v2.py
import itertools
import copy

class AblationEngineV2:
    """
    Tüm bellek, uyku ve gürültü parametre kombinasyonlarını test eden
    akademik düzeyde 2. Nesil Kontrollü Ablasyon Matrisi Motoru.
    """
    def __init__(self, base_model, tasks, trainer, logger):
        self.base_model = base_model
        self.tasks = tasks
        self.trainer = trainer
        self.logger = logger

    def run_variant(self, config_flags):
        """
        Derin kopyalanmış model üzerinde ablation bayraklarını set eder ve eğitip test eder.
        """
        model = copy.deepcopy(self.base_model)

        # --- apply ablations ---
        model.use_gfaml = config_flags["gfaml"]
        if hasattr(model, 'use_sleep'):
            model.use_sleep = config_flags["sleep"]
            
        model.use_lora = config_flags["lora"]
        model.noise = config_flags["noise"]
        
        # Modeli trainer'a dinamik olarak bağlayarak hatasız çalışmasını garanti ederiz
        self.trainer.model = model

        results = []

        for train, test, name in self.tasks:
            acc = self.trainer.run_task(train, test, name)
            results.append(acc)

        return sum(results) / len(results)

    def run_full_matrix(self):
        """
        2x2x2x3 hiperparametre uzayını tarayan ablasyon matrisini koşturur.
        """
        grid = {
            "gfaml": [True, False],
            "lora": [True, False],
            "sleep": [True, False],
            "noise": [0.0, 0.01, 0.05]
        }

        keys = list(grid.keys())
        combos = list(itertools.product(*grid.values()))

        results_table = []

        print("\n[ABLATION START] FULL PARAMETER SPACE TESTING...\n")

        for combo in combos:
            config = dict(zip(keys, combo))

            score = self.run_variant(config)

            results_table.append((config, score))

            self.logger.log_metric(str(config), score)

        return results_table
