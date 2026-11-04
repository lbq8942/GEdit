"""Durable value cache for exact replay of sequential GEdit runs."""
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch

def _hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()

def resolve_gedit_hyperparameters(cfg):
    """Resolve scalar overrides, model mappings, then global defaults."""
    model_name = str(cfg.llms.name)
    rank_override = cfg.algs.get("prior_rank")
    c_override = cfg.algs.get("c")
    rank_by_model = cfg.algs.get("prior_rank_by_model", {})
    c_by_model = cfg.algs.get("c_by_model", {})

    rank = float(
        rank_override if rank_override is not None
        else rank_by_model.get(model_name, cfg.algs.get("prior_rank_default", 0.85))
    )
    adaptive_c = float(
        c_override if c_override is not None
        else c_by_model.get(model_name, cfg.algs.get("c_default", 1.25))
    )
    if not 0 < rank < 1:
        raise ValueError(f"GEdit prior_rank must be in (0, 1), got {rank}")
    if adaptive_c <= 0:
        raise ValueError(f"GEdit c must be positive, got {adaptive_c}")
    return rank, adaptive_c

class SequentialZCache:
    def __init__(self, cfg, requests, contexts, hidden_size):
        self.enabled = bool(cfg.algs.get("value_cache", True))
        self.contexts = contexts
        self.requests, self.hidden_size = requests, int(hidden_size)
        self.hashes = [_hash({k:r.get(k) for k in ("id","subject","prompt","target_new")}) for r in requests]
        self.completed, self.dirty, self.values = 0, 0, None
        if not self.enabled: return
        resolved_rank, resolved_c = resolve_gedit_hyperparameters(cfg)
        core_identity = {
            "format":2, "model":cfg.llms.name, "dataset":str(cfg.data),
            "rank":resolved_rank, "c":resolved_c,
            "layers":list(cfg.llms.layers), "hidden":self.hidden_size,
            "fact_token":cfg.llms.fact_token, "v_steps":int(cfg.llms.v_num_grad_steps),
            "v_lr":float(cfg.llms.v_lr), "v_loss_layer":int(cfg.llms.v_loss_layer),
            "v_decay":float(cfg.llms.v_weight_decay), "kl":float(cfg.llms.kl_factor),
        }
        requested_identity = {**core_identity, "contexts": contexts}
        base_directory = (Path(cfg.cache_dir) / "gedit_values"
                          / cfg.llms.name.replace("/", "-") / str(cfg.data))
        requested_directory = base_directory / _hash(requested_identity)[:20]

        # Context templates are sampled and can differ after a restart. Search
        # caches for the same deterministic experiment, validate the request
        # prefix/value file, and restore the contexts that produced the values.
        row_bytes = self.hidden_size * 4
        best = None
        candidates = list(base_directory.iterdir()) if base_directory.exists() else []
        for directory in candidates:
            meta = directory / "metadata.json"
            path = directory / "values.f32"
            if not directory.is_dir() or not meta.exists() or not path.exists():
                continue
            try:
                candidate_metadata = json.load(meta.open())
                cached_identity = candidate_metadata.get("config", {})
                cached_core = {k: v for k, v in cached_identity.items() if k != "contexts"}
                if cached_core != core_identity or "contexts" not in cached_identity:
                    continue
                cached_hashes = candidate_metadata.get("request_hashes", [])
                candidate_reusable = min(
                    int(candidate_metadata.get("completed", 0)),
                    len(requests), len(cached_hashes),
                )
                for i in range(candidate_reusable):
                    if cached_hashes[i] != self.hashes[i]:
                        candidate_reusable = i
                        break
                if path.stat().st_size < candidate_reusable * row_bytes:
                    candidate_reusable = 0
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            candidate = (
                candidate_reusable, directory == requested_directory,
                directory, candidate_metadata, cached_identity,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is not None and best[0] > 0:
            reusable, _, self.directory, metadata, self.identity = best
            self.contexts = self.identity["contexts"]
            if self.directory != requested_directory:
                print(f"Restored cached context templates from {self.directory}")
        else:
            reusable, metadata = 0, {}
            self.directory, self.identity = requested_directory, requested_identity

        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta, self.path = self.directory/"metadata.json", self.directory/"values.f32"
        self.path.touch(exist_ok=True)
        with self.path.open("r+b") as f:
            if self.path.stat().st_size < len(requests)*row_bytes: f.truncate(len(requests)*row_bytes)
        self.values=np.memmap(self.path,dtype=np.float32,mode="r+",shape=(len(requests),self.hidden_size))
        self.completed=reusable
        print(f"GEdit value cache: {self.directory} (replay {reusable}/{len(requests)})")

    def get(self,index,device):
        if not self.enabled or index>=self.completed: return None
        return torch.from_numpy(np.array(self.values[index],copy=True)).to(device)

    def store(self,index,value):
        if not self.enabled or index<self.completed: return
        if index!=self.completed: raise RuntimeError(f"nonsequential value cache write {index}!={self.completed}")
        self.values[index]=value.detach().float().cpu().numpy(); self.completed+=1; self.dirty+=1
        self.flush()

    def flush(self):
        if not self.enabled or not self.dirty: return
        self.values.flush()
        data={"config":self.identity,"completed":self.completed,"request_hashes":self.hashes[:self.completed],"shape":[len(self.requests),self.hidden_size]}
        tmp=self.meta.with_suffix(".tmp")
        with tmp.open("w") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp,self.meta); self.dirty=0
        print(f"GEdit value checkpoint: {self.completed}/{len(self.requests)}")
