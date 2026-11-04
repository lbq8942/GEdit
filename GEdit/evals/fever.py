"""FEVER-specific evaluation."""

from evals.exact_match import score_scope


def eval_fever(cfg, model, tok, record: dict) -> dict:
    metrics = score_scope(model, tok, [record["prompt"].format(record["subject"])],
                          [record["target_new"]], "rewrite_prompts")
    metrics |= score_scope(model, tok, record.get("paraphrase_prompts", []),
                           [record["target_new"]], "paraphrase_prompts")
    metrics |= score_scope(model, tok, record.get("neighborhood_prompts", []),
                           record.get("neighborhood_prompts_answers", []),
                           "neighborhood_prompts")
    return metrics
