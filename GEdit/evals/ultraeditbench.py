"""UltraEditBench-specific evaluation."""

from evals.fever import eval_fever


def eval_ultraeditbench(cfg, model, tok, record: dict) -> dict:
    return eval_fever(cfg, model, tok, record)
