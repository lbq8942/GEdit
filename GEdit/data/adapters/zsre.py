"""Adapter for native ZSRE records."""

from data.validation import make_record


def adapt_zsre(record: dict, index: int) -> dict:
    result = make_record(
        record, index, "zsre", subject=record["subject"], prompt=record["src"],
        target_new=record["ans"], target_true=record.get("alt", ""),
    )
    result["paraphrase_prompts"] = [record["rephrase"]]
    loc = record["loc"]
    result["neighborhood_prompts"] = [loc + ("" if loc.endswith("?") else "?")]
    result["neighborhood_prompts_answers"] = [record["loc_ans"]]
    return result
