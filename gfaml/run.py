# gfaml/run.py
from gfaml.core.engine import GFAMLEngine
from gfaml.config import GFAMLConfig

def run_experiment(model, tasks):
    """
    GFAML araştırma omurgası motorunu çalıştırır ve sonuçları ekrana basar.
    """
    config = GFAMLConfig()
    engine = GFAMLEngine(model, config)

    results = engine.run(tasks)

    print("\n=== FINAL RESULTS ===")
    print(results)
