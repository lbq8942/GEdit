"""Dataset evaluator registry."""

from evals.counterfact import eval_counterfact
from evals.mquake_cf import eval_mquake_cf
from evals.fever import eval_fever
from evals.ultraeditbench import eval_ultraeditbench
from evals.wikibigedit import eval_wikibigedit
from evals.wiki_cf import eval_wiki_cf
from evals.zsre import eval_zsre


EVALUATORS = {
    "counterfact": eval_counterfact,
    "zsre_legacy": eval_zsre,
    "zsre": eval_zsre,
    "wiki_cf": eval_wiki_cf,
    "mquake_cf": eval_mquake_cf,
    "fever": eval_fever,
    "wikibigedit": eval_wikibigedit,
    "ultraeditbench": eval_ultraeditbench,
}


def evaluate_record(cfg, model, tok, record: dict) -> dict:
    dataset = record.get("dataset")
    try:
        evaluator = EVALUATORS[dataset]
    except KeyError as error:
        raise ValueError(f"no evaluator registered for dataset {dataset!r}") from error
    return evaluator(cfg, model, tok, record)
