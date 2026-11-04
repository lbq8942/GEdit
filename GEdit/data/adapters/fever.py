"""Adapter for native FEVER records."""

from data.validation import make_record


def adapt_fever(record: dict, index: int) -> dict:
    claim = record["prompt"]
    result = make_record(
        record, index, "fever", subject=claim, prompt="{}",
        target_new=record["alt"], target_true=record.get("ans", ""),
    )
    result["paraphrase_prompts"] = list(record.get("equiv_prompt", []))
    result["neighborhood_prompts"] = [record["unrel_prompt"]]
    result["neighborhood_prompts_answers"] = [record["unrel_ans"]]
    result["subject_strategy"] = "full_prompt"
    return result
