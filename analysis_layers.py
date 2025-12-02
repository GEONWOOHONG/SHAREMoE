# analysis_layers.py (수정됨)
import os, json, time, collections, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR
from utils import (
    set_seed,
    ensure_flash_attn,
    build_model_for_mode,
    load_checkpoint_if_exists,
)
from modeling import MoELayer
from transformers import GPT2Config

# -------------------------------------------------------------------------
# 기존 함수들 (_cosine_similarity, _center, _cka_linear) 그대로 유지
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# [수정] Recorder: 소스 정보(src_ids)와 토큰 정보(token_ids)도 수집하도록 변경
# -------------------------------------------------------------------------
class _Recorder:
    def __init__(self, max_tokens: int = 500000):
        self.pre_mlp: Dict[int, List[torch.Tensor]]  = collections.defaultdict(list)
        self.post_mlp: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.expert_indices: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        
        # [New] 메타데이터 수집용 리스트
        self.batch_src_ids: List[torch.Tensor] = []    # 배치별 소스 ID
        self.batch_token_ids: List[torch.Tensor] = []  # 배치별 토큰 ID
        
        self.token_counts: Dict[int, int] = collections.defaultdict(int)
        self.layers: List[int] = []
        self.max_tokens = int(max_tokens)
        self.captured_tokens = 0  # 0번 레이어 기준 수집된 총 토큰 수

    def is_full(self) -> bool:
        if not self.layers: return False
        # 모든 레이어 데이터 + 메타데이터가 충분히 모였는지 확인
        return all(self.token_counts[l] >= self.max_tokens for l in self.layers)

    def _trimcat(self, lst: List[torch.Tensor], dtype=torch.float32) -> Optional[torch.Tensor]:
        if not lst: return None
        X = torch.cat(lst, dim=0)
        if X.size(0) > self.max_tokens:
            X = X[: self.max_tokens]
        return X.to(dtype)

    def get_pair(self, l:int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._trimcat(self.pre_mlp[l]), self._trimcat(self.post_mlp[l])
    
    # 메타데이터 기록 함수 추가
    def record_metadata(self, input_ids: torch.Tensor, src_ids: torch.Tensor):
        # input_ids: [B, T], src_ids: [B, T]
        B, T = input_ids.shape
        needed = self.max_tokens - self.captured_tokens
        if needed <= 0: return

        flat_ids = input_ids.reshape(-1)
        flat_src = src_ids.reshape(-1)
        
        if flat_ids.size(0) > needed:
            flat_ids = flat_ids[:needed]
            flat_src = flat_src[:needed]
            
        self.batch_token_ids.append(flat_ids.cpu())
        self.batch_src_ids.append(flat_src.cpu())
        self.captured_tokens += flat_ids.size(0)

# Hook 함수들은 기존 로직 유지 (메타데이터는 메인 루프에서 수집)
def _register_hooks(model: nn.Module, rec: _Recorder):
    handles = []
    pat = re.compile(r"^transformer\.h\.(\d+)\.mlp$")

    def pre_hook(module, inputs):
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None and rec.token_counts[lidx] >= rec.max_tokens: return
        x = inputs[0]
        if not isinstance(x, torch.Tensor): return
        B, T, H = x.shape
        needed = rec.max_tokens - rec.token_counts[lidx]
        flat = x.reshape(B*T, H)
        if flat.size(0) > needed: flat = flat[:needed]
        rec.pre_mlp[lidx].append(flat.detach().to(torch.float32).cpu())

    def fwd_hook(module, inputs, outputs):
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None and rec.token_counts[lidx] >= rec.max_tokens: return
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        B, T, H = out.shape
        needed = rec.max_tokens - rec.token_counts[lidx]
        flat = out.reshape(B*T, H)
        if flat.size(0) > needed: flat = flat[:needed]
        
        rec.post_mlp[lidx].append(flat.detach().to(torch.float32).cpu())
        rec.token_counts[lidx] += flat.size(0)

        moe = getattr(module, "moe", None)
        scores = None
        if moe is not None:
            if getattr(moe, "last_scores", None) is not None: scores = moe.last_scores
            elif hasattr(moe, "router") and getattr(moe.router, "last_scores", None) is not None: scores = moe.router.last_scores
            elif hasattr(moe, "xmoe_router") and getattr(moe.xmoe_router, "last_scores", None) is not None: scores = moe.xmoe_router.last_scores
        
        if scores is not None:
            top1 = scores.detach().argmax(dim=-1).view(-1)
            if top1.size(0) > needed: top1 = top1[:needed]
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

# -------------------------------------------------------------------------
# [수정] save_raw_paths: 메타데이터(Source ID, Token ID)도 함께 저장
# -------------------------------------------------------------------------
@torch.no_grad()
def save_raw_paths(rec: _Recorder, mode: str, idx_to_source: Dict[int, str]) -> str:
    print("💾 Saving RAW Trajectory Paths with Metadata...")
    layers = sorted(set(rec.layers))
    if not layers: return ""
    
    # 1. Expert Indices 병합
    layer_tensors = {}
    sizes = []
    for l in layers:
        indices_list = rec.expert_indices[l]
        if not indices_list: continue
        t = torch.cat(indices_list, dim=0)
        layer_tensors[l] = t
        sizes.append(t.size(0))
    
    # 2. 메타데이터 병합
    if not rec.batch_token_ids:
        print("⚠️ No metadata collected.")
        return ""
    all_tokens = torch.cat(rec.batch_token_ids, dim=0)
    all_srcs = torch.cat(rec.batch_src_ids, dim=0)
    
    # 3. 길이 동기화 (가장 짧은 길이에 맞춤)
    min_len = min(sizes + [all_tokens.size(0), all_srcs.size(0)])
    print(f"🔹 Aligning tokens across {len(layers)} layers & metadata. Count: {min_len:,}")

    aligned_paths = []
    for l in layers:
        t = layer_tensors[l][:min_len].unsqueeze(1)
        aligned_paths.append(t)
    
    raw_paths = torch.cat(aligned_paths, dim=1).to(torch.int16) # [N, Layers]
    saved_tokens = all_tokens[:min_len].to(torch.int32)         # [N]
    saved_srcs = all_srcs[:min_len].to(torch.int16)             # [N]
    
    # 4. 저장
    filename = f"raw_trajectory_{mode}.pt"
    save_path = os.path.join(CHECKPOINTS_DIR, filename)
    
    torch.save({
        "paths": raw_paths,      # Tensor [N, L] (Expert IDs)
        "tokens": saved_tokens,  # Tensor [N] (Token IDs - 단어 확인용)
        "sources": saved_srcs,   # Tensor [N] (Source IDs - 출처 확인용)
        "source_map": idx_to_source, # Dict[int, str] (ID -> 이름 매핑)
        "layers": layers,
        "mode": mode
    }, save_path)
    
    print(f"✅ Saved raw paths & metadata to: {save_path}")
    return save_path

# compute_same_token_intra_expert_cka, compute_A_metrics 등 기존 분석 함수 유지
@torch.no_grad()
def compute_same_token_intra_expert_cka(model: nn.Module, rec: _Recorder, device: torch.device) -> Dict[int, float]:
    out = {}
    print("Computing Intra-Expert CKA (Same Token Similarity)...")
    layers = sorted(list(set(rec.layers)))
    for l in layers:
        inputs_list = rec.pre_mlp[l]
        if not inputs_list: continue
        X_cpu = torch.cat(inputs_list, dim=0)
        if X_cpu.size(0) > 2048: X_cpu = X_cpu[:2048]
        X = X_cpu.to(device)
        
        if l >= len(model.transformer.h): continue
        block = model.transformer.h[l]
        if not hasattr(block, "mlp"): continue
        mlp = block.mlp
        moe = getattr(mlp, "moe", None)
        if isinstance(mlp, MoELayer): moe = mlp
        if moe is None: continue
            
        experts_to_compare = []
        if hasattr(moe, "experts") and moe.experts is not None:
            experts_to_compare.extend([e for e in moe.experts])
        if hasattr(moe, "global_experts") and moe.global_experts is not None:
            experts_to_compare.extend([e for e in moe.global_experts])
             
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
        if cka_vals: out[l] = float(np.mean(cka_vals))
        else: out[l] = 0.0
    return out

@torch.no_grad()
def compute_A_metrics(rec: _Recorder) -> Dict:
    layers = sorted(set(rec.layers))
    out = {"LEI": {}, "InterCKA": {}, "InterCKA_by_delta": {}}
    
    print("Computing metrics based on collected activations (LEI, CKA)...")
    for l in layers:
        X, Y = rec.get_pair(l)
        if X is None or Y is None or X.size(0) != Y.size(0): continue
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
                X_sub = X[indices]; Y_sub = Y[indices]
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
    if use_flash_attn: ensure_flash_attn()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Layer-Level Analysis (Trajectory + Metadata) - Mode: {mode}")
        print(f"{'='*70}")
    
    from data import load_pile_test, worker_init_fn, get_dataloader_generator
    
    # [수정] 메타데이터 로딩을 위해 pile_set_name 매핑 준비
    pile_test = load_pile_test(verbose=verbose)
    all_sources = set([m for m in pile_test["meta"]])
    source_to_idx = {name: i for i, name in enumerate(sorted(all_sources))}
    idx_to_source = {i: n for n, i in source_to_idx.items()}
    
    def add_id(example):
        return {"pile_set_id": source_to_idx[example["meta"]]}
    
    pile_test = pile_test.map(add_id, num_proc=max(1, os.cpu_count()//2))
    # [수정] pile_set_id 포함
    pile_test.set_format(type="torch", columns=["input_ids", "attention_mask", "pile_set_id"])
    
    rec_max = 500_000 if max_batches is None else (max_batches * batch_size * seq_len)
    rec = _Recorder(max_tokens=rec_max)
    
    num_workers = min(16, max(4, (os.cpu_count() or 8)//2))
    loader = DataLoader(
        pile_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )
    total_batches = len(loader)
    effective_batches = total_batches if max_batches is None else min(max_batches, total_batches)

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
        # [수정] 소스 ID 준비
        src_ids_scalar = batch["pile_set_id"].view(-1, 1) # [B, 1]
        # [B, T]로 확장
        current_seq = input_ids.shape[1]
        src_ids = src_ids_scalar.expand(-1, current_seq).to(device, non_blocking=True)

        # [수정] 메타데이터 기록 (Forward 전/후 상관없이 배치 단위 기록)
        rec.record_metadata(input_ids, src_ids)
        
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            else:
                _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        processed += 1
        if verbose and processed % 10 == 0:
            print(f"Processed {processed} batches...")
            
    # [수정] 메타데이터 맵도 함께 전달
    raw_path_file = save_raw_paths(rec, mode, idx_to_source)
    
    resA = compute_A_metrics(rec)
    resA["IntraExpertCKA"] = compute_same_token_intra_expert_cka(model, rec, device)
    
    result = {
        "mode": mode,
        "num_experts": num_experts,
        "batches": processed,
        "runtime_sec": float(time.time() - t0),
        "raw_path_file": raw_path_file,
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