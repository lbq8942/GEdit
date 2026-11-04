import os
import sys
from pathlib import Path

# def _parse_gpu_from_argv():
#     for arg in sys.argv:
#         if arg.startswith("gpu="):
#             return arg.split("=", 1)[1]
#     return "0"  #  GPU 0
#
# gpu_id = _parse_gpu_from_argv()
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import torch
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from evals.evaluation import eval_algo, get_last_eval_metrics
from evals.loc_eval import get_or_compute_loc_reference, set_loc_reference
from datetime import datetime

from load import load_model,load_data,save_model
from util.utility import ensure_file_directory,print_dict
from util.notify import notify_finished
import numpy as np
import time
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from huggingface_hub import login


import importlib

def apply_method_to_model(method):
    module_name = f"algs.{method}"
    function_name = f"apply_{method}_to_model"

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except ImportError:
        raise ImportError(f"Module {module_name} not found.")
    except AttributeError:
        raise AttributeError(f"Function {function_name} not found in {module_name}.")

LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "llms"
MODEL_PATH = {
    "EleutherAI/gpt-j-6B": str(LOCAL_MODEL_PATH / "gpt-j-6B"),
    "gpt2-xl": str(LOCAL_MODEL_PATH / "gpt2-xl"),
    "meta-llama/Llama-3.1-8B-Instruct": str(
        LOCAL_MODEL_PATH / "Meta-Llama-3.1-8B-Instruct"
    ),
    "Qwen/Qwen3-8B": str(LOCAL_MODEL_PATH / "Qwen3-8B"),
    "XHToken/Spark-X2.5-4B": str(LOCAL_MODEL_PATH / "Spark-X2.5-4B"),
    "meta-llama/Llama-3-8B-Instruct": str(
        LOCAL_MODEL_PATH / "Meta-Llama-3-8B-Instruct"
    ),
    "meta-llama/Llama-3.2-3B-Instruct": str(
        LOCAL_MODEL_PATH / "Llama-3.2-3B-Instruct"
    ),
}


def set_random_seed(seed=42):
    torch.manual_seed(seed)  # torchcpu
    torch.cuda.manual_seed_all(seed)  # torchgpu
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True  
    np.random.seed(seed)  
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  # python

import hydra
from omegaconf import DictConfig, OmegaConf
# D:\lbq\project\python\MI\LLMEdit
#D:/lbq/project/python/MI/LLMEdit


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_start_time = time.time()
    final_results_file = None
    final_metrics = None
    print(OmegaConf.to_yaml(cfg))
    # cfg.gpu=0
    set_random_seed(cfg.seed)
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() and cfg.gpu>=0 else "cpu")
    print("Start Loading model")
    model_name=cfg.llms.name
    local_model_path = MODEL_PATH.get(model_name)
    model_name_or_path = (
        local_model_path if local_model_path and Path(local_model_path).exists() else model_name
    )
    trust_remote_code = bool(cfg.llms.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=cfg.model_dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    print("Loading model successfully")
    tok.pad_token = tok.eos_token
    apply_algo = apply_method_to_model(cfg.algs.name)
    if float(cfg.algs.get("cuda_reserve_gib", 0) or 0) > 0:
        # Protect long runs before locality/reference preparation starts.
        # This allocator-cache reservation is algorithm-independent; methods
        # that need it opt in through their algorithm config/CLI overrides.
        from algs.gedit.gedit_main import _reserve_cuda_cache
        _reserve_cuda_cache(cfg, device)
    # apply_algo = ALG_DICT[cfg.algs.name]
    data=load_data(cfg)
    selection_manifest = cfg.get("selection_manifest")
    if selection_manifest is not None:
        manifest_path = Path(str(selection_manifest))
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("dataset") != cfg.data:
            raise ValueError(
                f"Selection manifest dataset {manifest.get('dataset')!r} "
                f"does not match configured dataset {cfg.data!r}."
            )
        safe_record_ids = manifest.get("safe_record_ids", [])
        record_by_id = {record["id"]: record for record in data}
        missing_ids = [record_id for record_id in safe_record_ids if record_id not in record_by_id]
        if missing_ids:
            raise ValueError(
                f"Selection manifest contains {len(missing_ids)} IDs missing from "
                f"the loaded pool; first missing ID: {missing_ids[0]!r}."
            )
        requested = int(cfg.num_edits)
        if len(safe_record_ids) < requested:
            raise ValueError(
                f"Selection manifest has only {len(safe_record_ids)} safe records; "
                f"requested {requested}."
            )
        data = [record_by_id[record_id] for record_id in safe_record_ids[:requested]]
        print(
            f"Selected {len(data)} edits from shared manifest {manifest_path} "
            f"(reference model: {manifest.get('reference_model', 'unknown')})."
        )
    elif bool(cfg.get("skip_first_token_edits", False)):
        from algs.gedit.compute_z import find_fact_lookup_idx

        selected = []
        selected_source_indices = []
        for source_index, record in enumerate(data):
            lookup_index = find_fact_lookup_idx(
                record["prompt"],
                record["subject"],
                tok,
                cfg.llms.fact_token,
                verbose=False,
            )
            if lookup_index > 0:
                selected.append(record)
                selected_source_indices.append(source_index)
                if len(selected) == int(cfg.num_edits):
                    break
        if len(selected) < int(cfg.num_edits):
            raise ValueError(
                f"Only found {len(selected)} non-first-token edits in "
                f"a pool of {len(data)}; requested {cfg.num_edits}."
            )
        data = selected
        print(
            f"Selected {len(data)} non-first-token edits from {len(selected_source_indices) and selected_source_indices[-1] + 1} "
            f"source records (last source index {selected_source_indices[-1]})."
        )
    if cfg.loc_eval:
        set_loc_reference(get_or_compute_loc_reference(cfg, data, model, tok))

    if cfg.get("baseline_only", False):
        print("Start Evaluating the Unedited Base Model")
        baseline_metrics = eval_algo(cfg, model, tok, data)
        baseline_results_file = cfg.results_dir + "/{}/{}/base-{}".format(
            cfg.data,
            cfg.llms.name.replace("/", "-"),
            cfg.get("baseline_save_name", cfg.save_name),
        )
        ensure_file_directory(baseline_results_file)
        with open(baseline_results_file, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg) + "\n\n")
            f.write("The Evaluation Results before Editing:\n\n")
            json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print_dict(baseline_metrics)
        notify_finished(
            cfg,
            time.time() - run_start_time,
            metrics=baseline_metrics,
            result_file=baseline_results_file,
        )
        return


    # if cfg.large:
    #     data=data[6:7]
    if cfg.test_only:
        from evals.lweval import predicts
        pre_results_file = cfg.results_dir + "/{}/{}/{}".format(cfg.data, cfg.llms.name.replace("/", "-"), cfg.save_name)
        ensure_file_directory(pre_results_file)
        if not os.path.exists(pre_results_file) and cfg.pre_eval:
            start_time = time.time()
            print("Start Evaluating the Original Model")
            pre_metrics=eval_algo(cfg, model, tok, data)
            end_time = time.time()
            hours = np.round((end_time - start_time) / 3600, 3)
            with open(pre_results_file, "w", encoding="utf-8") as f:
                f.write("\n\nEvaluation Took {} Hours".format(hours))
                f.write("\n\n")
                f.write("The Evaluation Results before Editing:")
                f.write("\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
            print("End Evaluating the Original Model")
            if cfg.lw_eval:
                file="lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/pred_lw_eval.npy"
                ensure_file_directory(file)
                np.save(file,np.array(predicts))
                predicts.clear()
        edited_model=load_model(model,cfg)
        print("Start Evaluating the Edited Model")
        post_metrics = eval_algo(cfg, edited_model, tok, data)
        # formatted_time = datetime.now().strftime("%d_%H_%M_%S")
        # post_results_file = cfg.results_dir + "/{}/{}/{}-{}-{}".format(cfg.data, cfg.llms.name.replace("/","-"),cfg.algs.name, cfg.num_edits,formatted_time)
        post_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(cfg.data, cfg.llms.name.replace("/","-"),cfg.algs.name, cfg.save_name)

        ensure_file_directory(post_results_file)
        with open(post_results_file, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg) + "\n\n")  
            f.write("The Evaluation Results after Editing:")
            f.write("\n\n")
            json.dump(post_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
        final_results_file = post_results_file
        final_metrics = post_metrics
        print("End Evaluating the Edited Model")
        if cfg.lw_eval:
            file = "lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/pred_lw_eval.npy"
            ensure_file_directory(file)
            np.save(file, np.array(predicts))
    else:
        pre_metrics={}
        if cfg.pre_eval:
            print("Start Evaluating the Unedited Base Model")
            pre_metrics = eval_algo(cfg, model, tok, data)
            pre_results_file = cfg.results_dir + "/{}/{}/base-{}".format(
                cfg.data,
                cfg.llms.name.replace("/", "-"),
                cfg.get("baseline_save_name", cfg.save_name),
            )
            ensure_file_directory(pre_results_file)
            with open(pre_results_file, "w", encoding="utf-8") as f:
                f.write(OmegaConf.to_yaml(cfg) + "\n\n")
                f.write("The Evaluation Results before Editing:\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n")
        elif cfg.debug_mode and cfg.num_edits<=500:
            pre_metrics = eval_algo(cfg, model, tok, data)  
        # if cfg.c_edit:
        #     model = load_model(model, cfg)
        #     data=data[2000:cfg.num_edits]
        edited_model = apply_algo(model,tok,data,cfg)
        final_metrics = get_last_eval_metrics()
        if final_metrics is not None:
            final_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(cfg.data, cfg.llms.name.replace("/", "-"), cfg.algs.name, cfg.save_name)

        if cfg.debug_mode:
            post_metrics = eval_algo(cfg, edited_model, tok, data)
            print("The Evaluation Results before Editing:")
            print_dict(pre_metrics)
            print("\n\n")
            print("The Evaluation Results after Editing:")
            print_dict(post_metrics)
            final_metrics = post_metrics
        else:
            if cfg.save_model:
                save_model(edited_model,cfg)

    notify_finished(cfg, time.time() - run_start_time, metrics=final_metrics, result_file=final_results_file)

if __name__ == "__main__":
    main()

    # data = [{"subject": "Taylor Swift",
    #          "prompt": "Which country was {} born?",
    #          "target_new": "The United Kingdom",
    #          "target_true": "The United States"},
    #         {"subject": "Watts Humphrey",
    #          "prompt": "What university did {} attend?",
    #          "target_new": "University of Michigan",
    #          "target_true": "Illinois Institute of Technology"},
    #         {"subject": "Danielle Darrieux",
    #          "prompt": "The mother tongue of {} is",
    #          "target_new": "English",
    #          "target_true": "French"},
    #         {
    #             "subject": "Edwin of Northumbria",
    #             "prompt": "The official religion of {} is",
    #             "target_new": "Islam",
    #             "target_true": "Christianity"}
    #         ]
