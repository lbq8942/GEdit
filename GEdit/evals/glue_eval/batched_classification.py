import time

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef


def _candidate_token_ids(tokenizer, text, model_name):
    ids = tokenizer(f" {text}")["input_ids"]
    if "llama" in model_name.lower() and ids:
        ids = ids[1:]
    return ids


def _batched_generate(model, tokenizer, prompts, gen_len, device):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**batch, max_new_tokens=gen_len, do_sample=False)
        return tokenizer.batch_decode(output, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = old_padding_side


def _batched_candidate_scores(model, tokenizer, prompts, candidates, device):
    """Return exp(mean token log-probability), matching the legacy scorer."""
    model_name = model.config._name_or_path
    candidate_ids = [
        _candidate_token_ids(tokenizer, candidate, model_name)
        for candidate in candidates
    ]
    rows = []
    answer_spans = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for prompt in prompts:
        prompt_ids = tokenizer(prompt)["input_ids"]
        if "llama" in model_name.lower() and prompt_ids:
            prompt_ids = prompt_ids[1:]
        for ids in candidate_ids:
            rows.append(prompt_ids + ids)
            answer_spans.append((len(prompt_ids), ids))

    max_len = max(map(len, rows))
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        input_ids[row_index, :len(row)] = torch.tensor(row)
        attention_mask[row_index, :len(row)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    scores = []
    for row_index, (start, ids) in enumerate(answer_spans):
        positions = torch.arange(start - 1, start + len(ids) - 1, device=device)
        targets = torch.tensor(ids, device=device)
        mean_log_prob = log_probs[row_index, positions, targets].mean()
        scores.append(mean_log_prob.exp().item())
    return np.asarray(scores).reshape(len(prompts), len(candidates))


def evaluate_batched(evaluator, gen_len, candidates, task, print_logs=False):
    batch_size = int(evaluator.cfg.get("glue_eval_batch_size", 4))
    device = next(evaluator.model.parameters()).device
    prepared = [evaluator._create_prompt(example, gen_len) for example in evaluator.eval_dataset]
    predictions = []
    predictions_new = []
    labels = []
    stored_generations = []
    start_time = time.time()

    for batch_start in range(0, len(prepared), batch_size):
        batch_items = prepared[batch_start:batch_start + batch_size]
        prompts = [item[0] for item in batch_items]
        generated = _batched_generate(
            evaluator.model, evaluator.tokenizer, prompts, gen_len, device
        )
        scores = _batched_candidate_scores(
            evaluator.model, evaluator.tokenizer, prompts, candidates, device
        )

        for item, prompt, generated_text, candidate_scores in zip(
            batch_items, prompts, generated, scores
        ):
            if task == "nli":
                _, sentence1, sentence2, label = item
            else:
                _, sentence, label = item
            prompt_text = evaluator.tokenizer.decode(
                evaluator.tokenizer(prompt)["input_ids"], skip_special_tokens=True
            )
            generated_suffix = generated_text[len(prompt_text):]
            answer = evaluator._get_answer(
                generated_suffix if task == "mmlu" else generated_text
            )
            candidate_index = int(np.argmax(candidate_scores))
            answer_new = candidate_index if task == "mmlu" else (1 if candidate_index == 0 else 0)
            predictions.append(answer)
            predictions_new.append(answer_new)
            labels.append(label)

            record = {
                "input_prompt": prompt_text,
                "generated_text": generated_suffix,
                "answer": answer,
                "correct": answer == label,
                "highest_probability_answer": candidates[candidate_index],
                "correct_new": answer_new == label,
            }
            if task == "nli":
                record.update(sentence1=sentence1, sentence2=sentence2,
                              true_answer=candidates[0 if label == 1 else 1],
                              prob_true=candidate_scores[0], prob_false=candidate_scores[1])
            elif task == "sst":
                record.update(sentence=sentence,
                              true_answer=candidates[0 if label == 1 else 1],
                              prob_positive=candidate_scores[0], prob_negative=candidate_scores[1])
            elif task == "cola":
                record.update(sentence=sentence,
                              true_answer=candidates[0 if label == 1 else 1],
                              prob_yes=candidate_scores[0], prob_no=candidate_scores[1])
            else:
                record.update(sentence=sentence, true_answer=evaluator._get_label(label),
                              prob_a=candidate_scores[0], prob_b=candidate_scores[1],
                              prob_c=candidate_scores[2], prob_d=candidate_scores[3])
            stored_generations.append(record)

            if print_logs:
                print(generated_text)
                print(f"{len(stored_generations)}/{len(prepared)} correct_new={record['correct_new']}")

    correct = sum(pred == label for pred, label in zip(predictions, labels))
    invalid = sum(pred == -1 for pred in predictions)
    incorrect = len(labels) - correct - invalid
    result = {
        "correct": correct,
        "incorrect": incorrect,
        "invalid": invalid,
        "total": len(labels),
        "f1": f1_score(labels, predictions, average="weighted"),
        "f1_new": f1_score(labels, predictions_new, average="weighted"),
        "mcc": matthews_corrcoef(labels, predictions),
        "time": time.time() - start_time,
    }
    if task == "mmlu":
        correct_new = sum(pred == label for pred, label in zip(predictions_new, labels))
        result["correct_new"] = correct_new
        result["incorrect_new"] = len(labels) - correct_new
    return result, stored_generations
