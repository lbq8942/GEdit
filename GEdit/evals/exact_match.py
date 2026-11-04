"""Capability-based exact-match evaluation for heterogeneous datasets."""

from __future__ import annotations

import numpy as np

from evals.zsre import get_prompt_target_pairs, test_batch_prediction_acc


def score_scope(model, tok, prompts: list[str], targets: list[str], name: str) -> dict:
    if not prompts:
        return {}
    if len(targets) == 1 and len(prompts) > 1:
        targets = targets * len(prompts)
    if len(prompts) != len(targets):
        raise ValueError(f"{name}: {len(prompts)} prompts but {len(targets)} targets")

    expanded_prompts, expanded_targets, sizes = [], [], []
    for prompt, target in zip(prompts, targets):
        item_prompts, item_targets = get_prompt_target_pairs(tok, model, prompt, target)
        expanded_prompts.extend(item_prompts)
        expanded_targets.extend(item_targets)
        sizes.append(len(item_targets))

    correct = test_batch_prediction_acc(model, tok, expanded_prompts, expanded_targets)
    offsets = np.cumsum([0, *sizes])
    per_prompt = [correct[offsets[i]:offsets[i + 1]] for i in range(len(sizes))]
    flat = [value for values in per_prompt for value in values]
    strict = [int(all(values)) for values in per_prompt]
    return {
        f"{name}_correct": float(np.mean(flat)),
        f"{name}_strict_correct": float(np.mean(strict)),
    }


