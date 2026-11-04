import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov, get_covariance_and_mean
from util import nethook
from util.generate import generate_fast
from omegaconf import DictConfig
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .v_cache import SequentialZCache, resolve_gedit_hyperparameters
from evals.evaluation import eval_every

# Cache variable(s)

CONTEXT_TEMPLATES_CACHE = None
# priors=[]
covs=[]
means=[]
from util.utility import ensure_file_directory


def _empty_cuda_cache_if_enabled(cfg) -> None:
    """Return cached blocks to CUDA only when explicitly requested."""
    if (torch.cuda.is_available()
            and bool(cfg.algs.get("empty_cache_between_layers", False))):
        torch.cuda.empty_cache()


def _reserve_cuda_cache(cfg, device: torch.device) -> None:
    """Warm the PyTorch caching allocator up to a configured target."""
    target_gib = float(cfg.algs.get("cuda_reserve_gib", 0) or 0)
    if device.type != "cuda" or target_gib <= 0:
        return
    if bool(cfg.algs.get("empty_cache_between_layers", False)):
        print(
            "[GEdit CUDA reserve] disabled because "
            "empty_cache_between_layers=true"
        )
        return

    gib = 1024 ** 3
    target = int(target_gib * gib)
    safety = int(float(cfg.algs.get("cuda_reserve_safety_gib", 2.0)) * gib)
    current = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    wanted = max(0, target - current)
    allocatable = max(0, free - safety)
    to_allocate = min(wanted, allocatable)

    # One-GiB blocks can be split and reused by the medium/large allocations.
    blocks = []
    remaining = to_allocate
    try:
        while remaining >= 64 * 1024 ** 2:
            block_size = min(gib, remaining)
            blocks.append(
                torch.empty(block_size, dtype=torch.uint8, device=device)
            )
            remaining -= block_size
    except torch.OutOfMemoryError:
        print("[GEdit CUDA reserve] concurrent allocation limited reservation")
    finally:
        del blocks

    reserved = torch.cuda.memory_reserved(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    print(
        "[GEdit CUDA reserve] "
        f"target={target_gib:.2f} GiB, "
        f"reserved={reserved / gib:.2f} GiB, "
        f"device_free={free_after / gib:.2f}/{total / gib:.2f} GiB"
    )


def load_cov(cfg,model,tok):
    covs.clear()
    means.clear()
    layers=cfg.llms.layers
    for i, layer in enumerate(layers):
        cov, mean = get_covariance_and_mean(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        )
        covs.append(cov)
        means.append(mean)

# def load_project(cfg):
#     for i, layer in enumerate(cfg.llms.layers):
#         print(f"\n\nLAYER {layer}\n")
#         Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) + ".pt"

#         Ps.append(Pi)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim

norms=[]
Us=[]

import torch

def compute_v(cfg, C):
    """
    Build an orthonormal basis for the complement of the principal covariance
    subspace and the diagonal residual D = diag(C - U U^T).

    Args:
        cfg: Configuration containing prior_rank in (0, 1).
        C: Centered key covariance with shape [d, d].

    Returns:
        V: Complement basis M with shape [d, d-r].
        D_diag: Clamped diagonal residual with shape [d].
    """
    d = C.shape[0]
    # device = C.device

    
    eigvals, eigvecs = torch.linalg.eigh(C)  

    
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # 3. 
    eigvals_clamped = torch.clamp(eigvals, min=0.0)

    
    total_energy = eigvals_clamped.sum()
    if total_energy <= 0:
        r = 1
    else:
        cumsum_energy = torch.cumsum(eigvals_clamped, dim=0)
        energy_threshold = cfg.algs.prior_rank

        cond_num = torch.abs(eigvals).max() / (torch.abs(eigvals).min() + 1e-12)
        if energy_threshold >= 0.98 and cond_num > 4e4:
            energy_threshold = 0.98
            print("Warning: large condition number detected. Using energy threshold = 0.98 for stability.")

        mask = cumsum_energy >= energy_threshold * total_energy
        r = torch.where(mask)[0][0].item() + 1 if mask.any() else len(eigvals)
        r = max(1, r)

    
    Lambda_r = eigvals_clamped[:r]      # [r]
    U_r = eigvecs[:, :r]                
    sqrt_Lambda_r = torch.sqrt(Lambda_r)
    U = U_r * sqrt_Lambda_r.unsqueeze(0)  # [d, r]

    
    diag_C = torch.diag(C)
    diag_UUT = torch.sum(U * U, dim=1)
    D_diag = diag_C - diag_UUT
    D_diag = torch.clamp(D_diag, min=1e-8)

    # 7. U = Q_r sqrt(Lambda_r), so its orthogonal-complement basis is
    #    M = Q[:, r:]. The caller historically names this matrix V.
    V = eigvecs[:, r:]  # [d, r_v], r_v = d - r

    return V, D_diag

def compute_p(cfg, C):
    """
    Build the projector onto the complement of the principal covariance
    subspace and return the diagonal residual.

    Args:
        cfg: Configuration containing prior_rank in (0, 1).
        C: Centered key covariance with shape [d, d].

    Returns:
        P: Orthogonal complement projector with shape [d, d].
        D_diag: Clamped diagonal residual with shape [d].
    """
    d = C.shape[0]
    device = C.device

    
    eigvals, eigvecs = torch.linalg.eigh(C)  

    
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    
    eigvals_clamped = torch.clamp(eigvals, min=0.0)

    
    total_energy = eigvals_clamped.sum()
    if total_energy <= 0:
        
        r = 1
    else:
        cumsum_energy = torch.cumsum(eigvals_clamped, dim=0)
        energy_threshold = cfg.algs.prior_rank

        
        cond_num = torch.abs(eigvals).max() / (torch.abs(eigvals).min() + 1e-12)
        if energy_threshold >= 0.98 and cond_num > 4e4:
            energy_threshold = 0.98
            print("Warning: large condition number detected. Using energy threshold = 0.98 for stability.")

        mask = cumsum_energy >= energy_threshold * total_energy
        if mask.any():
            r_eff = torch.where(mask)[0][0].item() + 1
        else:
            r_eff = len(eigvals)
        r = max(1, r_eff)

    
    Lambda_r = eigvals_clamped[:r]  # [r], 
    U_r = eigvecs[:, :r]            # [d, r]

    
    sqrt_Lambda_r = torch.sqrt(Lambda_r)  # [r]
    U = U_r * sqrt_Lambda_r.unsqueeze(0)  

    # 6.  D_diag = diag(C) - diag(UU^T)
    diag_C = torch.diag(C)                  # [d]
    diag_UUT = torch.sum(U * U, dim=1)      # [d],  diag(UU^T)
    D_diag = diag_C - diag_UUT
    D_diag = torch.clamp(D_diag, min=1e-8)  

    # 7.  P = I - Proj_{range(U)}
    
    if r == 0 or U.abs().max() < 1e-12:
        
        P = torch.eye(d, device=device)
    else:
        #  U 
        # Proj_U = U @ torch.linalg.pinv(U)  # shape [d, d]
        
        try:
            U_pinv = torch.linalg.pinv(U)  # [r, d]
            Proj_U = U @ U_pinv            # [d, d]
        except Exception:
            
            G = U.T @ U                    # [r, r]
            G_inv = torch.linalg.pinv(G)   # 
            Proj_U = U @ G_inv @ U.T
        P = torch.eye(d, device=device) - Proj_U

    return P, D_diag
VDs=[]
A_invs=[]
def apply_gedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """Apply GEdit sequentially using its adaptive geometry coefficient."""

    # Update target and print info
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    # Reserve immediately on entering GEdit. A second call after geometry
    # initialization tops the cache back up if temporary work changed it.
    _reserve_cuda_cache(cfg, device)

    resolved_rank, resolved_c = resolve_gedit_hyperparameters(cfg)
    # Resolve once so geometry, updates, cache identity and saved config agree.
    cfg.algs.prior_rank = resolved_rank
    cfg.algs.c = resolved_c
    print(
        f"GEdit hyperparameters: model={cfg.llms.name}, "
        f"prior_rank={resolved_rank:g}, c={resolved_c:g}"
    )

    requests_copy=deepcopy(requests)
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    # load_project(cfg)
    # llmname={"gpt2":0,"gpt-j":1,"llama":2}
    # dataname={"counter":0,"zsre":1}
    # pranks=[[0.7,0.8],[0.8,0.8],[0.5,0.5]]
    # prank=0,0
    # for key1,value1 in llmname.items():
    #     for key2,value2 in dataname.items():
    #         if key1 in cfg.llms.name and key2 in cfg.data:
    #             prank=value1,value2
    # cfg.algs.prior_rank=pranks[prank[0]][prank[1]]
    # global Us
    
    global VDs
    VDs=[]
    projected_regularizers=[]
    geometry_gpu = bool(cfg.algs.get("geometry_gpu", True))
    geometry_device = device if geometry_gpu else torch.device("cpu")
    # Build one layer at a time. When geometry_gpu is false, only the layer
    # currently being decomposed occupies GPU memory; its M basis and D_bar are
    # immediately offloaded to CPU.
    for i, layer in enumerate(layers):
        # Stream one cached covariance at a time. A GPT-J covariance is about
        # 1 GiB (16384 x 16384 float32), so loading all six layers first adds
        # an avoidable ~6 GiB host-memory peak.
        cov, layer_mean = get_covariance_and_mean(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        )
        V, D_diag = compute_v(cfg, cov.to(device))
        mean = layer_mean.to(device=device, dtype=V.dtype)
        projected_mean = V.T @ mean
        # bar_D = b b^T + D, projected into the editable M-subspace.
        D_bar = V.T @ (D_diag.unsqueeze(1) * V)
        D_bar = D_bar + torch.outer(projected_mean, projected_mean)
        VDs.append(V.detach().to(geometry_device))
        projected_regularizers.append(D_bar.detach().to(geometry_device))
        if not geometry_gpu:
            del V, D_diag, mean, projected_mean, D_bar
            _empty_cuda_cache_if_enabled(cfg)
        del cov, layer_mean
    # Keep previous edit keys in the M-subspace. This is substantially
    # smaller than caching full d x d key covariances.
    kp_device = device if cfg.get("kp_gpu", True) else torch.device("cpu")
    adaptive_c = float(cfg.algs.get("c", 1.0))
    if adaptive_c <= 0:
        raise ValueError(f"algs.c must be positive, got {adaptive_c}")

    # GEdit measures current keys only against pretrained geometry D_bar^-1
    # and stores the historical projected second moment, never individual keys.
    pretrained_inverse_cache = []
    for D_bar in projected_regularizers:
        D_bar_on_device = D_bar.to(device)
        cached_inverse = torch.linalg.inv(D_bar_on_device).detach().to(kp_device)
        pretrained_inverse_cache.append(cached_inverse)
        if kp_device.type == "cpu":
            del D_bar_on_device, cached_inverse
            _empty_cuda_cache_if_enabled(cfg)
    projected_history_moments = [
        torch.zeros_like(D_bar, device=kp_device)
        for D_bar in projected_regularizers
    ]

    import time
    start=time.time()
    # upds=[]
    count=0
    context_templates = get_context_templates(model, tok)
    hidden_size = getattr(model.config, "n_embd", None) or model.config.hidden_size
    value_cache = SequentialZCache(
        cfg, requests, context_templates, hidden_size
    )
    # Resume with the exact contexts that produced the cached values.
    context_templates = value_cache.contexts


    # Keep reusable CUDA blocks owned by this process throughout the long
    # sequential run, preventing a later process from taking peak-work memory.
    _reserve_cuda_cache(cfg, device)


    stopped_early = False
    last_eval_count = 0
    for requests_chunks in chunks(requests, cfg.bs):
        chunk_size = len(requests_chunks)
        corruption = batch_edit(
            cfg,
            model,
            tok,
            requests_chunks,
            device,
            projected_regularizers,
            pretrained_inverse_cache,
            projected_history_moments,
            value_cache,
            count,
            context_templates,
        )
        count += chunk_size
        if count % cfg.eval_every == 0 or count == len(requests):
            print()
            print("##########Evaluating#########")
            metrics=eval_every(cfg,model,tok,requests_copy[:count],count)
            last_eval_count = count
            threshold=0.01
            break_count = sum(
                1
                for value in metrics.values()
                if isinstance(value, (int, float)) and value < threshold
            )
            if break_count >= 3:
                stopped_early = True
                break
        if corruption:
            print(f"[GEDIT EARLY ABORT] unsafe update detected after {count} edits")
            stopped_early = True
            break
    if stopped_early and last_eval_count != count:
        print("##########Evaluating early-stop checkpoint#########")
        eval_every(cfg, model, tok, requests_copy[:count], count)
    total_time=time.time()-start
    zzz_file=cfg.cache_dir+"/deltau/{}-{}-{}-{}-{}.tensor".format(cfg.algs.name,cfg.llms.name.replace("/","-"),cfg.data,cfg.save_name,cfg.num_edits)
    ensure_file_directory(zzz_file)
    torch.save(norms,zzz_file)
    return model


def batch_edit(
    cfg,
    model,
    tok,
    requests,
    device,
    projected_regularizers,
    pretrained_inverse_cache,
    projected_history_moments,
    value_cache,
    request_offset,
    context_templates,
):
    # deltas = {}
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    # Compute z for final layer
    z_layer = cfg.llms.layers[-1]
    # In strict sequential editing, z_t must be optimized on the model after
    # edits 1..t-1. It therefore cannot be precomputed on the base model.
    z_list = []
    for local_index, request in enumerate(requests):
        cache_index = request_offset + local_index
        cur_z = value_cache.get(cache_index, device)
        if cur_z is None:
            cur_z = compute_z(
                model, tok, request, cfg, z_layer, context_templates,
            )
            value_cache.store(cache_index, cur_z)
        else:
            print(f"Replaying cached value {cache_index + 1}/{value_cache.completed}")
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        
        # print(layer_ks.sum(),layer_ks.mean(),layer_ks.std())
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=cfg.llms.layer_module_tmp,
            fact_token_strategy=cfg.llms.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers
        upd_type=torch.float
        # import time
        # start=time.time()
        # upd_matrix = torch.linalg.solve(
        #         Ps[i].to(device) @ (layer_ks @ layer_ks.T + cache_c[i,:,:].to(device)) +
        #         cfg.algs.L2*torch.eye(layer_ks.shape[0], dtype=upd_type,device=device),
        #     Ps[i].to(device) @ layer_ks.to(upd_type) @ resid.T
        # )
        # upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        R=resid
        # # # print("time:",time.time()-start)
        
        # 
        V = VDs[i].to(device)
        K1_bar = V.T @ layer_ks.to(upd_type)  # [r_v, k]

        D_bar = projected_regularizers[i].to(device)
        num_edit_keys = K1_bar.shape[1]
        if num_edit_keys < 1:
            raise ValueError("GEdit received no edit keys")
        adaptive_c = float(cfg.algs.get("c", 1.0))

        pretrained_inverse = pretrained_inverse_cache[i].to(device)
        pretrained_inverse_times_keys = pretrained_inverse @ K1_bar
        pretrained_gram = K1_bar.T @ pretrained_inverse_times_keys
        mu_e = torch.trace(pretrained_gram) / (adaptive_c * num_edit_keys)
        if not bool(torch.isfinite(mu_e).item()) or mu_e.item() <= 0:
            raise RuntimeError(
                f"Invalid adaptive mu at layer {layer}: {mu_e.item()}"
            )
        print(f"adaptive mu {mu_e.item():.8g}")

        history_moment = projected_history_moments[i]
        history_moment_on_device = history_moment.to(device)
        current_key_cov = K1_bar @ K1_bar.T
        system = current_key_cov + history_moment_on_device + mu_e * D_bar
        solved_keys = torch.linalg.solve(system, K1_bar)
        pending_history_moment = None
        if cfg.algs.add_old_keys:
            pending_history_moment = (
                history_moment_on_device + current_key_cov
            ).detach().to(history_moment.device)

        upd_matrix = R @ solved_keys.T @ V.T  # [m, d]
        # upd_matrix = resid@ trackers[i].get_KT_Ainv(layer_ks.to(upd_type))

        # if upd_matrix.shape[1]==Us[i][0].shape[0]:
        
        
        #     norms.append(torch.trace(upd_matrix @ covs[i].to(device)@upd_matrix.T).item())#
        # else:
        #     # norms.append(torch.norm(upd_matrix.T @ Us[i][0].to(device)).item())
        #     # norms.append(torch.norm(upd_matrix.T @ Us[i][1].to(device)).item())
        #     norms.append(torch.trace(upd_matrix.T @ covs[i].to(device)@upd_matrix).item())

        # Adjust update matrix shape
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            update_norm = torch.linalg.norm(upd_matrix)
            weight_norm = torch.linalg.norm(weights[weight_name])
            if (not torch.isfinite(update_norm) or not torch.isfinite(weight_norm)
                    or update_norm > 100 or weight_norm > 1000):
                return True
            else:
                weights[weight_name][...] = weights[weight_name] + upd_matrix
                # Commit history only after this layer's weight update has
                # passed the safety check and has actually been applied.
                if pending_history_moment is not None:
                    projected_history_moments[i] = pending_history_moment
        if (not bool(cfg.algs.get("geometry_gpu", True))
                and bool(cfg.algs.get("empty_cache_between_layers", True))):
            # Drop every large per-layer GPU tensor before materializing the
            # next layer. Persistent geometry/history copies remain on CPU.
            layer_ks = cur_zs = targets = resid = R = None
            K1_bar = D_bar = current_key_cov = None
            solved_keys = upd_matrix = None
            pending_history_moment = None
            pretrained_inverse = pretrained_inverse_times_keys = None
            pretrained_gram = history_moment_on_device = system = None
            V = None
            torch.cuda.empty_cache()
            # deltas[weight_name] = upd_matrix
            # if len(upds)<=i:
            #     upds.append(upd_matrix)
            # else:
            #     upds[i]=upds[i].to(device)+upd_matrix
            # norms.append(torch.norm(upds[i]).item())  # 
            # if upds[i].shape[1] == covs[i].shape[0]:
            
            
            
            # else:
            #     # norms.append(torch.norm(upds[i].T @ Us[i][0].to(device)).item())
            #     # norms.append(torch.norm(upds[i].T @ Us[i][1].to(device)).item())
            #     norms.append(torch.trace(upds[i].T @ covs[i].to(device)@upds[i]).item())
            # upds[i]=upds[i].cpu()
        # Pi.cpu()
        # for x in [layer_ks, cur_zs, targets]:
        #     x.cpu()
        #     del x
        # torch.cuda.empty_cache()

    # if cfg.algs.add_old_keys:
    #     for i, layer in enumerate(cfg.llms.layers):
    #         layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
    #         cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()



def update_model(model,deltas,recover=False):
    with torch.no_grad():
        for weight_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, weight_name)
            if recover:
                w[...] -= upd_matrix
            else:
                w[...] += upd_matrix
    return model

def get_project(model, tok, layer, cfg):
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    force_recompute = False
    cov = get_cov(
        cfg,
        model,
        tok,
        layer,
        cfg.llms.mom2_dataset,
        cfg.llms.mom2_n_samples,
        cfg.llms.mom2_dtype,
        force_recompute=force_recompute,
    )
    U, S, _ = torch.linalg.svd(cov.to(device), full_matrices=False)
    threshold = cfg.algs.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print("the number of small singular values",len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


    return False

def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
