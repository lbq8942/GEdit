"""Compute multiple layers' second moments in one model forward pass.

This opt-in path writes the same per-layer cache files as layer_stats.py.
"""

import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from util.nethook import TraceDict
from util.runningstats import CombinedStat, Mean, SecondMoment, make_loader, save_cached_state

from .tok_dataset import TokenizedDataset, dict_to_, flatten_masked_batch, length_collation


def _cache_path(cfg, layer):
    model_name = cfg.llms.name.replace("/", "-")
    return os.path.join(cfg.cache_dir, "stats", model_name, f"layer-{layer}.npz")


def _mean_cache_path(cfg, layer):
    model_name = cfg.llms.name.replace("/", "-")
    return os.path.join(cfg.cache_dir, "stats", model_name, f"layer-{layer}-mean.npz")


def _tokenized_dataset(name, tokenizer, maxlen):
    config = {"wikitext": "wikitext-103-raw-v1", "wikipedia": "20220301.en"}[name]
    raw_dataset = load_dataset(name, config)
    return TokenizedDataset(raw_dataset["train"], tokenizer, maxlen=maxlen)


def multi_layer_stats(
    cfg,
    model,
    tokenizer,
    layers,
    ds_name,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
):
    """Collect and cache several layers' second moments in one pass."""
    layers = [int(layer) for layer in layers]
    missing_layers = [layer for layer in layers if not os.path.exists(_cache_path(cfg, layer))]
    if not missing_layers:
        print("All requested multi-layer covariance statistics are already cached.")
        return

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    if not hasattr(model.config, "max_position_embeddings"):
        raise AssertionError(
            "The model config has no max_position_embeddings; set mom2_maxseqlen explicitly."
        )
    npos = model.config.max_position_embeddings
    if batch_tokens is not None and batch_tokens < npos:
        npos = batch_tokens
    if batch_tokens is None:
        batch_tokens = npos * 3
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    dataset = _tokenized_dataset(ds_name, tokenizer, npos)
    loader = make_loader(
        dataset,
        sample_size=sample_size,
        batch_size=100,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    stats = {layer: CombinedStat(mom2=SecondMoment()) for layer in missing_layers}
    module_names = {
        layer: cfg.llms.rewrite_module_tmp.format(layer) for layer in missing_layers
    }
    batch_count = -(-(sample_size or len(dataset)) // 100)

    print(
        f"Computing covariance for layers {missing_layers} in one forward pass; "
        "existing layer caches are skipped."
    )
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with TraceDict(
                    model,
                    list(module_names.values()),
                    retain_input=True,
                    retain_output=False,
                    detach=True,
                    stop=True,
                ) as traces:
                    model(**batch)
                for layer in missing_layers:
                    feats = flatten_masked_batch(
                        traces[module_names[layer]].input, batch["attention_mask"]
                    ).to(dtype=dtype)
                    stats[layer].add(feats)

    cache_args = {"sample_size": sample_size}
    for layer, stat in stats.items():
        stat.to_(device="cpu")
        filename = _cache_path(cfg, layer)
        save_cached_state(filename, stat, cache_args)
        print(f"Saved multi-layer covariance statistics to {filename}")


def get_covs_multi(cfg, model, tokenizer, layers):
    """Populate missing standard covariance cache files for layers."""
    return multi_layer_stats(
        cfg,
        model,
        tokenizer,
        layers,
        cfg.llms.mom2_dataset,
        sample_size=cfg.llms.mom2_n_samples,
        precision=cfg.llms.mom2_dtype,
        batch_tokens=cfg.llms.mom2_maxseqlen,
    )

def multi_layer_means(
    cfg,
    model,
    tokenizer,
    layers,
    ds_name,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
):
    """Collect and cache several layer-input means in one model forward pass."""
    layers = [int(layer) for layer in layers]
    missing_layers = [
        layer for layer in layers if not os.path.exists(_mean_cache_path(cfg, layer))
    ]
    if not missing_layers:
        print("All requested multi-layer mean statistics are already cached.")
        return

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    if not hasattr(model.config, "max_position_embeddings"):
        raise AssertionError(
            "The model config has no max_position_embeddings; set mom2_maxseqlen explicitly."
        )
    npos = model.config.max_position_embeddings
    if batch_tokens is not None and batch_tokens < npos:
        npos = batch_tokens
    if batch_tokens is None:
        batch_tokens = npos * 3
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    dataset = _tokenized_dataset(ds_name, tokenizer, npos)
    loader = make_loader(
        dataset,
        sample_size=sample_size,
        batch_size=100,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    stats = {layer: CombinedStat(mean=Mean()) for layer in missing_layers}
    module_names = {
        layer: cfg.llms.rewrite_module_tmp.format(layer) for layer in missing_layers
    }
    batch_count = -(-(sample_size or len(dataset)) // 100)

    print(
        f"Computing means for layers {missing_layers} in one forward pass; "
        "existing layer mean caches are skipped."
    )
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with TraceDict(
                    model,
                    list(module_names.values()),
                    retain_input=True,
                    retain_output=False,
                    detach=True,
                    stop=True,
                ) as traces:
                    model(**batch)
                for layer in missing_layers:
                    feats = flatten_masked_batch(
                        traces[module_names[layer]].input, batch["attention_mask"]
                    ).to(dtype=dtype)
                    stats[layer].add(feats)

    cache_args = {"sample_size": sample_size}
    for layer, stat in stats.items():
        stat.to_(device="cpu")
        filename = _mean_cache_path(cfg, layer)
        save_cached_state(filename, stat, cache_args)
        print(f"Saved multi-layer mean statistics to {filename}")


def get_means_multi(cfg, model, tokenizer, layers):
    """Populate missing per-layer mean cache files in one forward pass."""
    return multi_layer_means(
        cfg,
        model,
        tokenizer,
        layers,
        cfg.llms.mom2_dataset,
        sample_size=cfg.llms.mom2_n_samples,
        precision=cfg.llms.mom2_dtype,
        batch_tokens=cfg.llms.mom2_maxseqlen,
    )

