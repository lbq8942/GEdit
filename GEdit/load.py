import json
import torch
from util import nethook
from util.utility import ensure_file_directory
from data.adapters.registry import load_and_adapt
def load_data(cfg):
    """Read raw JSON and normalize records once, in memory."""
    uses_selection_pool = (
        bool(cfg.get("skip_first_token_edits", False))
        or cfg.get("selection_manifest") is not None
    )
    limit = (
        int(cfg.get("selection_pool_size"))
        if uses_selection_pool and cfg.get("selection_pool_size") is not None
        else cfg.num_edits
    )
    return load_and_adapt(cfg.data, cfg.data_dir, limit)

def load_model(model,cfg):
    weights_dir = cfg.cache_dir + "/saved_weights"
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(cfg.algs.name, cfg.data, cfg.load_name,
                                                          cfg.llms.name.replace("/", "-"))
    device=torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    weights=torch.load(weights_file)
    with torch.no_grad():
        for key, value in weights.items():
            weight=nethook.get_parameter(model,key)
            weight[...]=value.to(device)
    return model

def save_model(model,cfg):
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        ).cpu()
        for layer in cfg.llms.layers
    }
    weights_dir = cfg.cache_dir + "/saved_weights"
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(cfg.algs.name, cfg.data, cfg.save_name,
                                                          cfg.llms.name.replace("/", "-"))
    ensure_file_directory(weights_file)
    torch.save(weights, weights_file)  
