"""Shared schema helpers and validation for in-memory edit records."""

from __future__ import annotations

import re
from copy import deepcopy


def prompt_template(prompt: str, subject: str) -> str:
    if "{}" in prompt:
        return prompt
    if subject:
        match = re.search(re.escape(subject), prompt, flags=re.IGNORECASE)
        if match:
            return prompt[:match.start()] + "{}" + prompt[match.end():]
    raise ValueError(f"subject {subject!r} is not present in prompt {prompt!r}")


def make_record(record: dict, index: int, dataset: str, *, subject: str,
                prompt: str, target_new: str, target_true: str | None) -> dict:
    result = deepcopy(record)
    result.update(
        id=record.get("id", record.get("case_id", index)),
        dataset=dataset,
        subject=subject,
        prompt=prompt_template(prompt, subject),
        target_new=target_new,
        target_true=target_true,
    )
    return result


def validate_record(record: dict) -> None:
    required = ("id", "dataset", "subject", "prompt", "target_new", "target_true")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"record {record.get('id', '?')} is missing fields: {missing}")
    if not isinstance(record["subject"], str) or not record["subject"]:
        raise ValueError(f"record {record['id']} has an empty subject")
    if not isinstance(record["target_new"], str) or not record["target_new"]:
        raise ValueError(f"record {record['id']} has an empty target_new")
    if not isinstance(record["target_true"], (str, type(None))):
        raise ValueError(f"record {record['id']} target_true must be a string or null")
    if record["prompt"].count("{}") != 1:
        raise ValueError(f"record {record['id']} prompt must contain exactly one {{}}")
