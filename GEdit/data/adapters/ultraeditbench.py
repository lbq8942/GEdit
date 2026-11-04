"""Adapter for native UltraEditBench records."""

from data.validation import make_record


def adapt_ultraeditbench(record: dict, index: int) -> dict:
    result = make_record(
        record, index, "ultraeditbench", subject=record["subject"],
        prompt=record["prompt"], target_new=record["ans"], target_true="",
    )
    result["paraphrase_prompts"] = [record["rephrase_prompt"]]
    result["neighborhood_prompts"] = [record["loc"]]
    result["neighborhood_prompts_answers"] = [record["loc_ans"]]
    return result
