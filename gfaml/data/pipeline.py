# gfaml/data/pipeline.py
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

# -----------------------------
# 1. DATA SCHEMA
# -----------------------------
@dataclass
class Sample:
    input_text: str
    target_text: str
    task_id: str
    meta: Dict = None


# -----------------------------
# 2. SYNTHETIC + REAL MIX PIPELINE
# -----------------------------
class GFAMLDatasetPipeline:
    """
    GFAML araştırmaları için geliştirilmiş yüksek kaliteli SQuAD-style QA,
    Wiki factual triples ve multi-hop akıl yürütme veri seti işleme boru hattı.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

        self.datasets = {
            "qa": [],
            "wiki": [],
            "reasoning": []
        }

    # -----------------------------
    # QA STYLE DATA
    # -----------------------------
    def load_qa(self, raw_pairs: List[Tuple[str, str]]):
        """
        SQuAD / TriviaQA style:
        (question, answer)
        """
        for q, a in raw_pairs:
            self.datasets["qa"].append(
                Sample(q, a, task_id="qa")
            )

    # -----------------------------
    # WIKI FACT DATA
    # -----------------------------
    def load_wiki_facts(self, triples: List[Tuple[str, str, str]]):
        """
        (entity, relation, value)
        ONUR -> works_on -> PYTORCH
        """
        for e, r, v in triples:
            text = f"{e} {r} nedir?"
            self.datasets["wiki"].append(
                Sample(text, v, task_id="wiki")
            )

    # -----------------------------
    # REASONING CHAINS
    # -----------------------------
    def load_reasoning(self, chains: List[Dict]):
        """
        multi-hop reasoning:
        {
          "premise": "...",
          "steps": [...],
          "answer": "..."
        }
        """
        for c in chains:
            self.datasets["reasoning"].append(
                Sample(
                    input_text=c["premise"],
                    target_text=c["answer"],
                    task_id="reasoning",
                    meta={"steps": c.get("steps", [])}
                )
            )

    # -----------------------------
    # CONTINUAL LEARNING SPLIT
    # -----------------------------
    def build_tasks(self):
        """
        Task split for continual learning:
        T1: QA
        T2: Wiki Facts
        T3: Reasoning Chains
        """

        tasks = [
            ("T1_QA", self.datasets["qa"]),
            ("T2_WIKI", self.datasets["wiki"]),
            ("T3_REASONING", self.datasets["reasoning"]),
        ]

        return tasks

    # -----------------------------
    # SHUFFLED BATCH GENERATOR
    # -----------------------------
    def get_stream(self, task_data: List[Sample]):
        data = task_data.copy()
        random.shuffle(data)

        for sample in data:
            yield sample


# -----------------------------
# 3. DEMO USAGE
# -----------------------------
if __name__ == "__main__":
    pipeline = GFAMLDatasetPipeline()

    # QA
    pipeline.load_qa([
        ("Türkiye'nin başkenti neresidir?", "Ankara"),
        ("Python'u kim geliştirdi?", "Guido van Rossum"),
    ])

    # Wiki facts
    pipeline.load_wiki_facts([
        ("ONUR", "çalıştığı_framework", "PYTORCH"),
        ("PARIS", "ülkesi", "FRANCE"),
    ])

    # Reasoning
    pipeline.load_reasoning([
        {
            "premise": "Ali 3 elma aldı, 2'sini yedi",
            "steps": ["3-2"],
            "answer": "1"
        }
    ])

    tasks = pipeline.build_tasks()

    for name, data in tasks:
        print(f"\n[{name}] -> {len(data)} samples")

        for s in pipeline.get_stream(data):
            print(s)
