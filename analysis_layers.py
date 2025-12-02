import os, json, time, collections, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from config import CHECKPOINTS_DIR
from utils import (
    set_seed,
    ensure_flash_attn,
    build_model_for_mode,
    load_checkpoint_if_exists,
)
from modeling import MoELayer
from transformers import GPT2Config

@torch.no_grad()
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float(); b = b.float()
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (a * b).sum(dim=-1)

@torch.no_grad()
def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)

@torch.no_grad()
def _cka_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xc = _center(X); Yc = _center(Y)
    XT_Y = Xc.T @ Yc
    num = (XT_Y ** 2).sum()
    den = torch.sqrt(((Xc.T @ Xc) ** 2).sum() + 1e-12) * torch.sqrt(((Yc.T @ Yc) ** 2).sum() + 1e-12)
    return num / (den + 1e-12)

class _Recorder:
    def __init__(self, max_tokens:int=16384):
        self.pre_mlp: Dict[int, List[torch.Tensor]]  = collections.defaultdict(list)
        self.post_mlp: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.expert_indices: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.token_counts: Dict[int, int] = collections.defaultdict(int)
        self.layers: List[int] = []
        self.max_tokens = int(max_tokens)

    def is_full(self) -> bool:
        if not self.layers: return False
        return all(self.token_counts[l] >= self.max_tokens for l in self.layers)

    def _trimcat(self, lst: List[torch.Tensor], dtype=torch.float32) -> Optional[torch.Tensor]:
        if not lst: return None
        X = torch.cat(lst, dim=0)
        if X.size(0) > self.max_tokens:
            X = X[: self.max_tokens]
        return X.to(dtype)

    def get_pair(self, l:int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._trimcat(self.pre_mlp[l]), self._trimcat(self.post_mlp[l])
    
    def get_indices(self, l:int) -> Optional[torch.Tensor]:
        return self._trimcat(self.expert_indices[l], dtype=torch.long)

def _register_hooks(model: nn.Module, rec: _Recorder):
    handles = []
    pat = re.compile(r"^transformer\.h\.(\d+)\.mlp$")

    def pre_hook(module, inputs):
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None and rec.token_counts[lidx] >= rec.max_tokens:
            return

        x = inputs[0]
        if not isinstance(x, torch.Tensor): return
        B, T, H = x.shape
        needed = rec.max_tokens - rec.token_counts[lidx]
        flat = x.reshape(B*T, H)
        if flat.size(0) > needed:
            flat = flat[:needed]
        
        flat_cpu = flat.detach().to(torch.float32).cpu()
        if lidx is not None:
            rec.pre_mlp[lidx].append(flat_cpu)

    def fwd_hook(module, inputs, outputs):
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None and rec.token_counts[lidx] >= rec.max_tokens:
            return

        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        B, T, H = out.shape
        
        needed = rec.max_tokens - rec.token_counts[lidx]
        flat = out.reshape(B*T, H)
        if flat.size(0) > needed:
            flat = flat[:needed]
        
        flat_cpu = flat.detach().to(torch.float32).cpu()
        
        if lidx is not None:
            rec.post_mlp[lidx].append(flat_cpu)
            rec.token_counts[lidx] += flat.size(0)

            moe = getattr(module, "moe", None)
            scores = None
            if moe is not None:
                if getattr(moe, "last_scores", None) is not None:
                    scores = moe.last_scores
                elif hasattr(moe, "router") and getattr(moe.router, "last_scores", None) is not None:
                    scores = moe.router.last_scores
                elif hasattr(moe, "xmoe_router") and getattr(moe.xmoe_router, "last_scores", None) is not None:
                    scores = moe.xmoe_router.last_scores
            
            if scores is not None:
                top1 = scores.detach().argmax(dim=-1).view(-1)
                if top1.size(0) > needed:
                    top1 = top1[:needed]
                rec.expert_indices[lidx].append(top1.cpu())

    for name, module in model.named_modules():
        m = pat.match(name)
        if m:
            lidx = int(m.group(1))
            module._layer_idx = lidx
            rec.layers.append(lidx)
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(fwd_hook))
    return handles

@torch.no_grad()
def compute_same_token_intra_expert_cka(model: nn.Module, rec: _Recorder, device: torch.device) -> Dict[int, float]:
    out = {}
    print("Computing Intra-Expert CKA (Same Token Similarity)...")
    
    layers = sorted(list(set(rec.layers)))
    
    for l in layers:
        inputs_list = rec.pre_mlp[l]
        if not inputs_list:
            continue
        
        X_cpu = torch.cat(inputs_list, dim=0)
        if X_cpu.size(0) > 2048:
            X_cpu = X_cpu[:2048]
        
        X = X_cpu.to(device)
        
        if l >= len(model.transformer.h): continue
        block = model.transformer.h[l]
        if not hasattr(block, "mlp"): continue
        mlp = block.mlp
        
        moe = None
        if hasattr(mlp, "moe"):
            moe = mlp.moe
        elif isinstance(mlp, MoELayer):
            moe = mlp
            
        if moe is None:
            continue
            
        experts_to_compare = []
        if hasattr(moe, "experts") and moe.experts is not None:
            experts_to_compare.extend([e for e in moe.experts])
        if hasattr(moe, "global_experts") and moe.global_experts is not None:
            experts_to_compare.extend([e for e in moe.global_experts])
        if hasattr(moe, "experts_hypermoe") and moe.experts_hypermoe is not None:
             experts_to_compare.extend([e for e in moe.experts_hypermoe])
             
        num_experts = len(experts_to_compare)
        if num_experts < 2:
            out[l] = float("nan")
            continue
            
        expert_outputs = []
        for exp in experts_to_compare:
            y = exp(X)
            expert_outputs.append(y)
        
        cka_vals = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                val = _cka_linear(expert_outputs[i], expert_outputs[j]).item()
                cka_vals.append(val)
        
        if cka_vals:
            out[l] = float(np.mean(cka_vals))
        else:
            out[l] = 0.0
            
    return out

@torch.no_grad()
def compute_trajectory_metrics(rec: _Recorder) -> Dict:
    print("Computing Trajectory Metrics (Expert Reuse)...")
    layers = sorted(set(rec.layers))
    out = {}
    
    reuse_rates = {}
    
    for i in range(len(layers) - 1):
        l_curr = layers[i]
        l_next = layers[i+1]
        
        idx_curr = rec.get_indices(l_curr)
        idx_next = rec.get_indices(l_next)
        
        if idx_curr is None or idx_next is None:
            continue
            
        min_len = min(idx_curr.size(0), idx_next.size(0))
        if min_len == 0:
            continue
            
        ic = idx_curr[:min_len]
        in_ = idx_next[:min_len]
        
        matches = (ic == in_).float().mean().item()
        reuse_rates[f"{l_curr}->{l_next}"] = matches
        
    out["layer_reuse_rate"] = reuse_rates
    if reuse_rates:
        out["avg_reuse_rate"] = float(np.mean(list(reuse_rates.values())))
    else:
        out["avg_reuse_rate"] = 0.0
        
    return out

@torch.no_grad()
def compute_A_metrics(rec: _Recorder) -> Dict:
    layers = sorted(set(rec.layers))
    out = {"LEI": {}, "InterCKA": {}, "InterCKA_by_delta": {}}
    
    print("Computing metrics based on collected activations...")
    for l in layers:
        X, Y = rec.get_pair(l)
        if X is None or Y is None or X.size(0) != Y.size(0):
            continue
        cos = _cosine_similarity(X, Y)
        out["LEI"][l] = float((1.0 - cos.mean()).item())
        
    reps = {l: rec.get_pair(l)[1] for l in layers}
    valid_layers = [l for l in layers if reps.get(l) is not None]
    
    inter_pairs = {}
    for i, l1 in enumerate(valid_layers):
        for l2 in valid_layers[i+1:]:
            X, Y = reps[l1], reps[l2]
            n = min(X.size(0), Y.size(0))
            if n > 10000:
                indices = torch.randperm(n)[:10000]
                X_sub = X[indices]
                Y_sub = Y[indices]
                cka = _cka_linear(X_sub, Y_sub)
            else:
                cka = _cka_linear(X[:n], Y[:n])
            key = f"{l1}-{l2}"
            inter_pairs[key] = float(cka.item())
            
    out["InterCKA"] = inter_pairs
    delta_acc = collections.defaultdict(list)
    for k, v in inter_pairs.items():
        a, b = k.split("-")
        d = abs(int(a) - int(b))
        delta_acc[d].append(v)
    out["InterCKA_by_delta"] = {int(d): float(np.mean(v)) for d, v in delta_acc.items()}
        
    return out

@torch.no_grad()
def run_analysis_A(mode: str = "ours_refine",
                   num_experts: int = 16,
                   batch_size: int = 64,
                   seq_len: int = 1024,
                   max_batches: Optional[int] = None,
                   save_json: Optional[str] = None,
                   use_flash_attn: bool = True,
                   verbose: bool = True):
    assert mode in {"dense","switch","gshard","hash","ours_refine"}, f"Unsupported mode: {mode}"
    set_seed(42)
    if use_flash_attn:
        ensure_flash_attn()
    if verbose:
        print(f"\n{'='*70}")
        print(f"Layer-Level Analysis (Part A + Trajectory) - Mode: {mode}")
        print(f"{'='*70}")
    
    from data import load_pile_test, worker_init_fn, get_dataloader_generator
    from torch.utils.data import DataLoader
    
    pile_test = load_pile_test(verbose=verbose)
    pile_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_ds = pile_test
    
    num_workers = min(16, max(4, (os.cpu_count() or 8)//2))
    loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )
    total_batches = len(loader)
    
    if max_batches is None:
        effective_batches = total_batches
    else:
        effective_batches = min(max_batches, total_batches)

    from utils import find_checkpoint_path
    ckpt_path = find_checkpoint_path(mode, CHECKPOINTS_DIR)
    
    if ckpt_path and os.path.exists(os.path.join(os.path.dirname(ckpt_path), "config.json")):
        config = GPT2Config.from_pretrained(os.path.dirname(ckpt_path))
    else:
        config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)

    model = build_model_for_mode(mode, num_experts=num_experts, config=config)
    load_checkpoint_if_exists(model, mode, CHECKPOINTS_DIR, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    rec = _Recorder(max_tokens=16384)
    handles = _register_hooks(model, rec)
    
    processed = 0
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        if i >= effective_batches: break
        if rec.is_full():
            if verbose: print("✅ Sufficient data collected. Stopping early.")
            break
            
        input_ids = batch["input_ids"][:, :seq_len].to(device, non_blocking=True)
        attn      = batch["attention_mask"][:, :seq_len].to(device, non_blocking=True)
        
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            else:
                _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        processed += 1
        
        if verbose and processed % 10 == 0:
            print(f"Processed {processed} batches...")
            
    resA = compute_A_metrics(rec)
    
    intra_cka = compute_same_token_intra_expert_cka(model, rec, device)
    resA["IntraExpertCKA"] = intra_cka
    
    traj = compute_trajectory_metrics(rec)
    resA["Trajectory"] = traj
    
    result = {
        "mode": mode,
        "num_experts": num_experts,
        "batches": processed,
        "batch_size": batch_size,
        "runtime_sec": float(time.time() - t0),
        "A": resA,
    }
    
    if save_json is None:
        save_json = os.path.join(CHECKPOINTS_DIR, f"analysis_A_{mode}.json")
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    for h in handles:
        try: h.remove()
        except Exception: pass
    del model
    torch.cuda.empty_cache()
    return result

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense","switch","gshard","hash","ours_refine"], default="ours_refine")
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--no_flash", action="store_true")
    args = ap.parse_args()
    run_analysis_A(
        mode=args.mode,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        save_json=args.save_json,
        use_flash_attn=not args.no_flash,
    )