"""WikiBigEdit-specific evaluation."""

from evals.exact_match import score_scope
from evals.fever import eval_fever


def eval_wikibigedit(cfg, model, tok, record: dict) -> dict:
    metrics = eval_fever(cfg, model, tok, record)
    metrics |= score_scope(model, tok, record.get("persona_prompts", []),
                           [record["target_new"]], "persona_prompts")
    metrics |= score_scope(model, tok, record.get("multihop_prompts", []),
                           record.get("multihop_answers", []), "multihop_prompts")
    return metrics
