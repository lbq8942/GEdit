"""Precompute multi-layer input means used by AlphaEditF."""

import argparse
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.multi_layer_stats import get_means_multi


MODEL_PATHS = {
    "gpt2-xl": "gpt2-xl",
    "gpt-j-6b": "gpt-j-6B",
    "llama3.1-8b": "Meta-Llama-3.1-8B-Instruct",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=MODEL_PATHS)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    project_root = root.parent
    base_cfg = OmegaConf.load(root / "configs/config.yaml")

    for config_name in args.models:
        llm_cfg = OmegaConf.load(root / f"configs/llms/{config_name}.yaml")
        cfg = OmegaConf.merge(base_cfg, {"llms": llm_cfg, "gpu": args.gpu})
        model_path = project_root / "llms" / MODEL_PATHS[config_name]
        dtype = getattr(torch, str(cfg.model_dtype))
        trust_remote_code = bool(cfg.llms.get("trust_remote_code", False))

        print(f"Loading {cfg.llms.name} from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(torch.device(f"cuda:{args.gpu}"))
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        tokenizer.pad_token = tokenizer.eos_token

        get_means_multi(cfg, model, tokenizer, cfg.llms.layers)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
