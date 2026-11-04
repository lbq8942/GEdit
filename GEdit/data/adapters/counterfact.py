"""Adapter for the four already-normalized legacy datasets."""

from copy import deepcopy


def adapt_normalized(record: dict, index: int, dataset: str) -> dict:
    result = deepcopy(record)
    result.setdefault("id", record.get("case_id", index))
    result.setdefault("target_true", "")
    result["dataset"] = dataset
    return result
