"""Batched teacher-forced locality evaluation using in-memory references."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


_ORIGINAL_LOCALITY_PREDICTIONS = None


def set_loc_reference(predictions) -> None:
    """Keep pre-edit locality predictions in memory for this process."""
    global _ORIGINAL_LOCALITY_PREDICTIONS
    _ORIGINAL_LOCALITY_PREDICTIONS = predictions


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _locality_pairs(record: dict) -> list[tuple[str, str]]:
    prompts = _as_list(record.get("neighborhood_prompts"))
    answers = _as_list(record.get("neighborhood_prompts_answers"))
    if not answers:
        answers = _as_list(record.get("target_true"))
    if len(answers) == 1 and len(prompts) > 1:
        answers *= len(prompts)
    if len(prompts) != len(answers):
        raise ValueError(
            "Locality prompt/answer count mismatch: "
            f"{len(prompts)} prompts and {len(answers)} answers"
        )
    return [(str(prompt), str(answer)) for prompt, answer in zip(prompts, answers)]


def _prepare_example(prompt: str, answer: str, tokenizer):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    answer_ids = tokenizer(
        " " + answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    input_ids = torch.cat((prompt_ids, answer_ids[:-1]))
    prediction_start = prompt_ids.numel() - 1
    return input_ids, prediction_start, answer_ids.numel()


def _batched_teacher_forced_top1(pairs, model, tokenizer, batch_size: int):
    prepared = [_prepare_example(prompt, answer, tokenizer) for prompt, answer in pairs]
    predictions = []
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        sequences = [item[0] for item in batch]
        input_ids = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        attention_mask = pad_sequence(
            [torch.ones_like(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=0,
        ).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for row, (_, prediction_start, answer_length) in enumerate(batch):
            if answer_length == 0:
                predictions.append([])
                continue
            answer_logits = logits[
                row,
                prediction_start : prediction_start + answer_length,
            ]
            predictions.append(answer_logits.argmax(dim=-1).cpu().tolist())
        del logits, input_ids, attention_mask
    return predictions


def get_loc_answer(cfg, data, model, tok):
    """Collect batched top-1 answer-position predictions for locality data."""
    pairs = []
    per_record_counts = []
    for record in data:
        record_pairs = _locality_pairs(record)
        per_record_counts.append(len(record_pairs))
        pairs.extend(record_pairs)

    model_was_training = model.training
    model.eval()
    flat_predictions = _batched_teacher_forced_top1(
        pairs,
        model,
        tok,
        int(cfg.get("loc_eval_batch_size", 16)),
    )
    model.train(model_was_training)

    nested_predictions = []
    offset = 0
    for count in per_record_counts:
        nested_predictions.append(flat_predictions[offset : offset + count])
        offset += count
    return nested_predictions


def get_or_compute_loc_reference(cfg, data, model, tok):
    """Load or atomically create the base-model locality predictions."""
    identity = {
        "format": 1,
        "model": cfg.llms.name,
        "dataset": str(cfg.data),
        "records": [
            {
                "pairs": _locality_pairs(record),
            }
            for record in data
        ],
    }
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    path = (
        Path(cfg.cache_dir) / "locality" / cfg.llms.name.replace("/", "-")
        / str(cfg.data) / f"{digest}.json"
    )
    if path.exists():
        print(f"Loading cached base locality reference: {path}")
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    predictions = get_loc_answer(cfg, data, model, tok)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)
    os.replace(temporary, path)
    print(f"Saved base locality reference: {path}")
    return predictions


def get_loc_eval(
    cfg,
    data,
    model,
    tok,
    original_loc_answer=None,
    original_indices=None,
):
    """Return mean per-sample agreement of aligned pre/post top-1 tokens."""
    if original_loc_answer is None:
        original_loc_answer = _ORIGINAL_LOCALITY_PREDICTIONS
    if original_loc_answer is None:
        raise RuntimeError("Pre-edit locality predictions have not been initialized")

    edited_predictions = get_loc_answer(cfg, data, model, tok)
    if original_indices is None:
        original_loc_answer = original_loc_answer[: len(edited_predictions)]
    else:
        original_indices = list(original_indices)
        if len(original_indices) != len(edited_predictions):
            raise ValueError(
                "Locality reference index count must match edited record count"
            )
        original_loc_answer = [
            original_loc_answer[index] for index in original_indices
        ]
    if len(edited_predictions) != len(original_loc_answer):
        raise ValueError("Pre-edit and post-edit locality record counts differ")

    sample_scores = []
    for edited_record, original_record in zip(edited_predictions, original_loc_answer):
        if len(edited_record) != len(original_record):
            raise ValueError("Pre-edit and post-edit locality sample counts differ")
        for edited_tokens, original_tokens in zip(edited_record, original_record):
            if len(edited_tokens) != len(original_tokens):
                raise ValueError("Pre-edit and post-edit locality token counts differ")
            if edited_tokens:
                sample_scores.append(
                    float(np.mean(np.asarray(edited_tokens) == np.asarray(original_tokens)))
                )

    accuracy = float(np.mean(sample_scores)) if sample_scores else 0.0
    return {"locality_sample_accuracy": round(accuracy, 3)}
