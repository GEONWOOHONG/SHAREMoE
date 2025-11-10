import os, json, math, time, collections, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHECKPOINTS_DIR
from utils import (
    set_seed,
    ensure_flash_attn,
    build_model_for_mode,
    load_checkpoint_if_exists,
)
from data import make_validation_dataloader
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

@torch.no_grad()
def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1)

class _Recorder:
    def __init__(self, max_tokens:int=8192, save_routes:bool=True):
        self.pre_mlp: Dict[int, List[torch.Tensor]]  = collections.defaultdict(list)
        self.post_mlp: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.route_dist: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.layers: List[int] = []
        self.max_tokens = int(max_tokens)
        self.save_routes = save_routes

    def _trimcat(self, lst: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not lst: return None
        X = torch.cat(lst, dim=0)
        if X.size(0) > self.max_tokens:
            X = X[: self.max_tokens]
        return X

    def get_pair(self, l:int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._trimcat(self.pre_mlp[l]), self._trimcat(self.post_mlp[l])

    def get_routes(self, l:int) -> Optional[torch.Tensor]:
        if not self.save_routes: return None
        return self._trimcat(self.route_dist[l])

def _register_hooks(model: nn.Module, rec: _Recorder):
    handles = []
    pat = re.compile(r"^transformer\.h\.(\d+)\.mlp$")

    def pre_hook(module, inputs):
        x = inputs[0]
        if not isinstance(x, torch.Tensor): return
        B, T, H = x.shape
        flat = x.reshape(B*T, H).detach().to(torch.float32).cpu()
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None:
            rec.pre_mlp[lidx].append(flat)

    def fwd_hook(module, inputs, outputs):
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        B, T, H = out.shape
        flat = out.reshape(B*T, H).detach().to(torch.float32).cpu()
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None:
            rec.post_mlp[lidx].append(flat)

        for m in module.modules():
            if isinstance(m, MoELayer):
                scores = None
                if hasattr(m, "router") and getattr(m.router, "last_scores", None) is not None:
                    scores = m.router.last_scores
                elif hasattr(m, "xmoe_router") and getattr(m.xmoe_router, "last_scores", None) is not None:
                    scores = m.xmoe_router.last_scores
                elif getattr(m, "last_scores", None) is not None:
                    scores = m.last_scores
                if scores is not None and rec.save_routes:
                    rec.route_dist[lidx].append(scores.detach().to(torch.float32).cpu())

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
def compute_A_metrics(rec: _Recorder) -> Dict:
    layers = sorted(set(rec.layers))
    out = {"LEI": {}, "InterCKA": {}, "IntraRedundancyProxy": {}}

    for l in layers:
        X, Y = rec.get_pair(l)
        if X is None or Y is None or X.size(0) != Y.size(0):
            continue
        cos = _cosine_similarity(X, Y)
        out["LEI"][l] = float((1.0 - cos.mean()).item())

    reps = {l: rec.get_pair(l)[1] for l in layers}
    valid_layers = [l for l in layers if reps.get(l) is not None]
    for i, l1 in enumerate(valid_layers):
        for l2 in valid_layers[i+1:]:
            X, Y = reps[l1], reps[l2]
            n = min(X.size(0), Y.size(0))
            cka = _cka_linear(X[:n], Y[:n])
            out["InterCKA"][f"{l1}-{l2}"] = float(cka.item())

    for l in layers:
        P = rec.get_routes(l)
        if P is None: 
            continue
        E = P.shape[1]
        if E < 2: 
            continue
        Q = P / (P.norm(dim=0, keepdim=True) + 1e-9)
        S = torch.einsum("ne,nk->ek", Q, Q)
        mask = (~torch.eye(E, dtype=torch.bool))
        out["IntraRedundancyProxy"][l] = float(S[mask].mean().item())

    return out

@torch.no_grad()
def run_analysis_A(mode: str = "ours_refine",
                   num_experts: int = 16,
                   batch_size: int = 32,
                   seq_len: int = 1024,
                   max_batches: Optional[int] = None,
                   save_json: Optional[str] = None,
                   sample_fraction: float = 0.10,
                   use_flash_attn: bool = True,
                   verbose: bool = True):
    """
    Computes Part A only and saves JSON.
    
    Args:
        mode: Model mode (dense, switch, gshard, hash, ours_refine)
        num_experts: Number of experts
        batch_size: Batch size
        seq_len: Sequence length
        max_batches: Max batches to process (None = use sample_fraction)
        save_json: Path to save JSON results (None = auto-generate)
        sample_fraction: Fraction of validation data when max_batches is None
        use_flash_attn: Whether to use Flash Attention
        verbose: Whether to print progress
    """
    assert mode in {"dense","switch","gshard","hash","ours_refine"}, f"Unsupported mode: {mode}"

    set_seed(42)
    if use_flash_attn:
        ensure_flash_attn()

    if verbose:
        print(f"\n{'='*70}")
        print(f"🔎 Layer-Level Analysis (Part A) - Mode: {mode}")
        print(f"{'='*70}")

    loader, total_batches, effective_batches = make_validation_dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        max_batches=max_batches,
        sample_fraction=sample_fraction,
        dataset_name="pile",
        verbose=verbose
    )

    config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=1024, n_layer=8, n_head=8)
    model = build_model_for_mode(mode, num_experts=num_experts, config=config)

    load_checkpoint_if_exists(model, mode, CHECKPOINTS_DIR, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rec = _Recorder(max_tokens=8192, save_routes=True)
    handles = _register_hooks(model, rec)

    processed = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= effective_batches: break
        input_ids = batch["input_ids"][:, :seq_len].to(device, non_blocking=True)
        attn      = batch["attention_mask"][:, :seq_len].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        processed += 1
        if verbose and processed % 10 == 0:
            print(f"  • Processed {processed}/{effective_batches} batches...")

    resA = compute_A_metrics(rec)
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
    
    if verbose:
        print(f"📄 Saved Part-A metrics to: {save_json}")
        print(f"⏱️  Runtime: {result['runtime_sec']:.2f}s")

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
    ap.add_argument("--sample_fraction", type=float, default=0.10)
    ap.add_argument("--no_flash", action="store_true")
    args = ap.parse_args()

    run_analysis_A(
        mode=args.mode,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        save_json=args.save_json,
        sample_fraction=args.sample_fraction,
        use_flash_attn=not args.no_flash,
    )