from evals.counterfact import eval_counterfact
from evals.zsre import eval_zsre
from evals.wiki_cf import eval_wiki_cf
from evals.mquake_cf import eval_mquake_cf
from evals.glue_eval.glue_eval import GLUEEval
from evals.loc_eval import get_loc_eval
from evals.registry import evaluate_record
import numpy as np
from util.utility import ensure_file_directory,print_dict

import json

_last_eval_metrics = None

def get_last_eval_metrics():
    return _last_eval_metrics

def eval_one_edit(cfg, model, tok, edit):
    return evaluate_record(cfg, model, tok, edit)


def eval_algo(cfg, model, tok, data, locality_indices=None):
    avg_metrics = {}
    # if cfg.edit_eval or cfg.algs.name=="emmet" or cfg.algs.name=="memit":
    all_metrics = {}
    
    for edit in data:  
        metrics = eval_one_edit(cfg, model, tok, edit)
        if len(all_metrics) == 0:
            for key, value in metrics.items():
                all_metrics[key] = [value]
        else:
            for key, value in metrics.items():
                all_metrics[key].append(value)
    
    # import torch
    # torch.save(all_metrics,'{}-{}-{}-all_metrics.pt'.format(cfg.llms.name.replace("/","-"),cfg.algs.name,cfg.data))
    for key, value in all_metrics.items():
        avg_metrics[key] = np.round(np.mean(value), 3).item()
    if cfg.glue_eval:
        glue_results = {}
        shots=0
        glue_eval = GLUEEval(cfg,model, tok, number_of_tests=cfg.get("glue_number_of_tests", 200),nli_number_of_few_shots=shots,dialogue_number_of_few_shots=shots,
                             sentiment_analysis_number_of_few_shots=shots,sst_number_of_few_shots=shots,mrpc_number_of_few_shots=shots,
                             cola_number_of_few_shots=shots,rte_number_of_few_shots=shots,mmlu_number_of_few_shots=shots
                             )
        out_file = "glue_generation.json"
        # glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag=True, sst_flag=True, cola_flag=True,
        #                                   rte_flag=True, mmlu_flag=True, mrpc_flag=True,sentiment_analysis_flag=True,
        #                                   dialogue_flag=True)
        glue_results = glue_eval.evaluate(
            glue_results,
            out_file,
            nli_flag=True,
            sst_flag=True,
            mmlu_flag=True,
            cola_flag=True,
        )
        avg_metrics = avg_metrics | glue_results
    # glue_results = {}
    # shots = 0
    # glue_eval = GLUEEval(cfg, model, tok, number_of_tests=200, nli_number_of_few_shots=shots,
    #                      dialogue_number_of_few_shots=shots,
    #                      sentiment_analysis_number_of_few_shots=shots, sst_number_of_few_shots=shots,
    #                      mrpc_number_of_few_shots=shots,
    #                      cola_number_of_few_shots=shots, rte_number_of_few_shots=shots, mmlu_number_of_few_shots=shots
    #                      )
    # out_file = "glue_generation.json"
    # glue_results = glue_eval.evaluate(glue_results, out_file, sst_flag=True)
    # avg_metrics = avg_metrics | glue_results
    if cfg.loc_eval:
        loc_metric=get_loc_eval(
            cfg, data, model, tok, original_indices=locality_indices
        )
        avg_metrics = avg_metrics | loc_metric
    
    return avg_metrics



def eval_every(
    cfg, model, tok, data, count, file=True, locality_indices=None
):
    """Evaluate a checkpoint and always persist it locally.

    The legacy ``file`` argument is retained for call compatibility, but local
    persistence is intentionally unconditional. Network notification is handled
    once, at the end of ``main.py``.
    """
    global _last_eval_metrics
    metrics = eval_algo(
        cfg, model, tok, data, locality_indices=locality_indices
    )
    post_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(
        cfg.data, cfg.llms.name.replace("/", "-"), cfg.algs.name, cfg.save_name
    )
    ensure_file_directory(post_results_file)
    write_type = "w" if count <= cfg.eval_every else "a"
    with open(post_results_file, write_type, encoding="utf-8") as f:
        f.write(f"The Evaluation Results after Editing {count} Edits:\n\n")
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n\n")
        f.flush()

    if cfg.debug_mode:
        print_dict(metrics)

    _last_eval_metrics = metrics
    return metrics


