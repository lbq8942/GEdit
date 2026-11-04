"""Adapter for native WikiBigEdit records."""

from data.validation import make_record


def adapt_wikibigedit(record: dict, index: int) -> dict:
    result = make_record(
        record, index, "wikibigedit", subject=record["subject"],
        prompt=record["update"], target_new=record["ans"], target_true="",
    )
    result["paraphrase_prompts"] = [record["rephrase"]]
    result["neighborhood_prompts"] = [record["loc"]]
    result["neighborhood_prompts_answers"] = [record["loc_ans"]]
    result["persona_prompts"] = [record["personas"]]
    result["multihop_prompts"] = [record["mhop"]] if record.get("mhop") else []
    result["multihop_answers"] = [record["mhop_ans"]] if record.get("mhop_ans") else []
    return result
