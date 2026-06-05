# gfaml/core/engine.py
from gfaml.core.trainer import GFAMLTrainer
from gfaml.core.logger import ExperimentLogger
from gfaml.core.reporter import ReportGenerator
import torch.optim as optim

class GFAMLEngine:
    """
    Tüm eğitim, günlükleme, değerlendirme ve raporlama akışını tek merkezden yöneten araştırma omurgası motoru.
    """
    def __init__(self, model, config):
        self.logger = ExperimentLogger("gfaml_run")

        lr = getattr(config, 'lr', 3e-4)
        if isinstance(config, dict):
            lr = config.get('lr', lr)
        elif hasattr(config, 'training') and isinstance(config.training, dict):
            lr = config.training.get('lr', lr)

        self.trainer = GFAMLTrainer(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=lr),
            config=config,
            logger=self.logger
        )

        self.reporter = ReportGenerator(self.logger)

    def run(self, tasks):
        results = []

        for i, task in enumerate(tasks):
            train, test, name = task

            acc = self.trainer.run_task(train, test, name)
            results.append(acc)

        self.logger.log_metric("final_score", sum(results))

        self.logger.save()
        self.reporter.generate()

        return results
