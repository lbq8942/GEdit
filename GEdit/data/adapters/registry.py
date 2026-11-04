"""In-memory adapters for heterogeneous knowledge-editing datasets."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable
from data.adapters.counterfact import adapt_normalized
from data.adapters.fever import adapt_fever
from data.adapters.ultraeditbench import adapt_ultraeditbench
from data.adapters.wikibigedit import adapt_wikibigedit
from data.adapters.zsre import adapt_zsre
from data.validation import validate_record

Adapter = Callable[[dict, int], dict]



DATASETS: dict[str, tuple[str, Adapter]] = {
    "multi_counterfact_20877": ("multi_counterfact_20877.json", lambda r, i: adapt_normalized(r, i, "counterfact")),
    "zsre_mend_eval_19086": ("zsre_mend_eval_19086.json", lambda r, i: adapt_normalized(r, i, "zsre_legacy")),
    "wiki_cf_2266": ("wiki_cf_2266.json", lambda r, i: adapt_normalized(r, i, "wiki_cf")),
    "mquake_cf_9218": ("mquake_cf_9218.json", lambda r, i: adapt_normalized(r, i, "mquake_cf")),
    "zsre": ("zsre/zsre_eval_20k.json", adapt_zsre),
    "zsre_eval_20k": ("zsre/zsre_eval_20k.json", adapt_zsre),
    "fever": ("fever/fever_eval_20k.json", adapt_fever),
    "fever_eval_20k": ("fever/fever_eval_20k.json", adapt_fever),
    "fever_eval_100k": ("fever/fever_eval_100k.json", adapt_fever),
    "wikibigedit": ("wikibigedit/wikibigedit_eval_17k.json", adapt_wikibigedit),
    "wikibigedit_eval_17k": ("wikibigedit/wikibigedit_eval_17k.json", adapt_wikibigedit),
    "ultraeditbench": ("ultraeditbench/UltraEditBench_train_20k.json", adapt_ultraeditbench),
    "UltraEditBench_train_20k": ("ultraeditbench/UltraEditBench_train_20k.json", adapt_ultraeditbench),
}




def load_and_adapt(data_name: str, data_dir: str | Path,
                   limit: int | None = None) -> list[dict]:
    name = data_name[:-5] if data_name.endswith(".json") else data_name
    if name not in DATASETS:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {data_name!r}; choices: {choices}")
    relative_path, adapter = DATASETS[name]
    path = Path(data_dir) / relative_path
    with path.open(encoding="utf-8") as handle:
        raw_records = json.load(handle)
    records = []
    invalid = []
    for index, raw_record in enumerate(raw_records):
        try:
            record = adapter(raw_record, index)
            validate_record(record)
        except (KeyError, TypeError, ValueError) as error:
            case_id = raw_record.get("id", raw_record.get("case_id", index))
            invalid.append((case_id, str(error)))
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if invalid:
        preview = "; ".join(f"{case_id}: {reason}" for case_id, reason in invalid[:3])
        warnings.warn(f"{name}: skipped {len(invalid)} invalid records ({preview})")
    return records
