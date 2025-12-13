import os, sys, math, json, random, importlib.util, subprocess, warnings
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import safe_open

from config import CHECKPOINTS_DIR, HF_DATASETS_CACHE, HF_HOME, HASH_TABLE_PATH, WORKSPACE_ROOT, MODEL_SPECS
from modeling import MoELayer
from data import load_or_prepare_pile, worker_init_fn, get_dataloader_generator
from utils import (
    ensure_flash_attn,
    build_model_for_mode,
    load_checkpoint_if_exists,
)

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

def pick_ckpt_path(mode: str) -> Optional[str]:
    from utils import find_checkpoint_path
    return find_checkpoint_path(mode, CHECKPOINTS_DIR)

def load_pile_validation_with_source_labels() -> Tuple["datasets.Dataset", Dict[str,int], Dict[int,str]]:
    from data import load_pile_test
    
    ds = load_pile_test(verbose=True)
    print(f"Building source label mapping from meta.pile_set_name (using test set)")
    
    all_sources = set([m for m in ds["meta"]]) 
    source_to_idx = {name: i for i, name in enumerate(sorted(all_sources))}
    idx_to_source = {i: n for n, i in source_to_idx.items()}
    
    def add_id(example):
        return {"pile_set_id": source_to_idx[example["meta"]]}
        
    ds = ds.map(add_id, num_proc=max(1, os.cpu_count()//2))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "pile_set_id"])
    print(f"Pile validation prepared: {len(ds)} samples, {len(source_to_idx)} unique sources")
    return ds, source_to_idx, idx_to_source

@torch.no_grad()
def analyze_batch_collect_specialization(model: nn.Module,
                                         input_ids: torch.Tensor,
                                         attention_mask: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
    """
    신뢰도 통계(Entropy, Margin 등)를 모두 제거하고,
    오직 '어떤 전문가가 선택되었는지(Top-1 Expert ID)'만 수집합니다.
    """
    model.eval()
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    per_layer = []
    
    for module in model.modules():
        if isinstance(module, MoELayer):
            scores = None
            if hasattr(module, "router") and getattr(module.router, "last_scores", None) is not None:
                scores = module.router.last_scores
            elif hasattr(module, "xmoe_router") and getattr(module.xmoe_router, "last_scores", None) is not None:
                scores = module.xmoe_router.last_scores
            elif getattr(module, "last_scores", None) is not None:
                scores = module.last_scores
            
            if scores is not None:
                # 확률 계산 없이 바로 가장 점수가 높은 전문가 인덱스만 추출
                top1 = torch.argmax(scores.detach(), dim=-1)
                per_layer.append({"top1": top1})
            else:
                per_layer.append({"top1": None})
    return per_layer

def accumulate_specialization(per_layer: List[Dict[str, torch.Tensor]],
                              pile_set_ids: torch.Tensor,
                              stats_spec: Dict):
    """
    NLL 및 신뢰도 관련 누적 로직 제거.
    오직 (Layer, Expert, Source) 조합의 빈도수만 카운트합니다.
    """
    token_sources_all = pile_set_ids.reshape(-1).cpu().tolist()

    layer_idx = -1
    for entry in per_layer:
        layer_idx += 1
        top1 = entry.get("top1", None)
        
        if top1 is None:
            continue

        # [B, S] -> [B*S]
        expert_ids_all = top1.view(-1).cpu().tolist()

        # 길이 맞추기 (배치 내 유효 토큰 수 등)
        N_all = min(len(token_sources_all), len(expert_ids_all))
        expert_ids_all = expert_ids_all[:N_all]
        token_sources_all_use = token_sources_all[:N_all]

        # 카운팅
        c = stats_spec[layer_idx]
        for eid, sid in zip(expert_ids_all, token_sources_all_use):
            c[eid][sid] += 1

def finalize_specialization_tables(stats_spec: Dict,
                                   idx_to_source: Dict[int, str]) -> pd.DataFrame:
    """
    신뢰도(Confidence) 관련 컬럼 병합 로직 제거.
    특화도(Specialization Index)와 엔트로피(Source Entropy)만 계산합니다.
    """
    rows = []
    for layer, exp_dict in stats_spec.items():
        for eid, src_counter in exp_dict.items():
            for sid, cnt in src_counter.items():
                rows.append({
                    "Layer": layer,
                    "Expert_ID": eid,
                    "Pile_Set_ID": sid,
                    "Pile_Set_Name": idx_to_source[sid],
                    "Activation_Count": int(cnt)
                })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # 1. 전문가별 총 활성화 횟수
    df["Expert_Total_Activation"] = df.groupby(["Layer","Expert_ID"])["Activation_Count"].transform("sum")
    
    # 2. P(Source | Expert)
    df["P_Source_Given_Expert"] = df["Activation_Count"] / (df["Expert_Total_Activation"] + 1e-12)
    
    # 3. P(Source)_Global (전체 데이터 분포)
    total_act = df["Activation_Count"].sum()
    gsrc = df.groupby("Pile_Set_Name")["Activation_Count"].sum() / float(total_act)
    gdf = gsrc.rename("P_Source_Global").reset_index()
    df = df.merge(gdf, on="Pile_Set_Name", how="left")
    
    # 4. Specialization Index (특화 지수)
    df["Specialization_Index"] = df["P_Source_Given_Expert"] / (df["P_Source_Global"] + 1e-10)
    
    # 5. 각 전문가가 가장 특화된 소스 찾기
    idxmax = df.groupby(["Layer","Expert_ID"])["Specialization_Index"].idxmax()
    max_df = df.loc[idxmax, ["Layer","Expert_ID","Pile_Set_Name","Specialization_Index"]].copy()
    max_df = max_df.rename(columns={
        "Pile_Set_Name":"Max_Specialized_Source",
        "Specialization_Index":"Max_Specialization_Index"
    })
    df = df.merge(max_df, on=["Layer","Expert_ID"], how="left")
    
    # 6. Source Entropy (해당 전문가가 얼마나 다양한 소스를 처리하는지)
    ent = df.groupby(["Layer","Expert_ID"])["P_Source_Given_Expert"].apply(lambda s: entropy(s.values, base=2))
    ent = ent.rename("Source_Entropy").reset_index()
    df = df.merge(ent, on=["Layer","Expert_ID"], how="left")
    
    preferred_cols = [
        "Layer","Expert_ID","Pile_Set_Name","Activation_Count",
        "Expert_Total_Activation","P_Source_Global","P_Source_Given_Expert",
        "Specialization_Index","Max_Specialized_Source","Max_Specialization_Index",
        "Source_Entropy"
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols]
    return df

def extract_model_metadata(model: nn.Module, mode: str, num_experts: int) -> Dict:
    active_k = 1
    capacity_factor = None
    aux_alpha = None
    
    for module in model.modules():
        if isinstance(module, MoELayer):
            active_k = getattr(module, "top_k", getattr(module, "k", 1))
            capacity_factor = getattr(module, "capacity_factor", None)
            if hasattr(module, "router"):
                aux_alpha = getattr(module.router, "aux_alpha", None)
            elif hasattr(module, "xmoe_router"):
                aux_alpha = getattr(module.xmoe_router, "aux_alpha", None)
            break
    
    if mode == "gshard": active_k = 2
    elif mode == "switch": active_k = 1
    elif mode == "hash": active_k = 1
    
    return {
        "Model": mode,
        "VisibleExperts": num_experts,
        "ActiveK": active_k,
        "CapacityFactor": capacity_factor if capacity_factor is not None else "N/A",
        "DroplessEval": True,
        "AuxAlpha": aux_alpha if aux_alpha is not None else 0.0,
    }

def run_specialization_only(
    modes: List[str] = ("switch","gshard","hash","ours_refine"),
    model_size: str = "base",
    num_experts: int = 16,
    batch_size: int = 44,
    seq_len: int = 1024,
    max_batches: Optional[int] = None,
    use_flash_attn: bool = True,
    sample_fraction: float = 0.10,
    verbose: bool = True
):
    set_seed(42)
    if use_flash_attn:
        ensure_flash_attn()
    if verbose:
        print("="*70)
        print("Specialization Analysis Only (No Confidence/NLL Metrics)")
        print("="*70)
        
    ds, source_to_idx, idx_to_source = load_pile_validation_with_source_labels()
    
    nworkers = min(8, max(2, (os.cpu_count() or 8)//2))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=nworkers, pin_memory=True, prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
        generator=get_dataloader_generator(0),
    )
    
    total_batches = len(loader)
    if max_batches is None:
        max_batches = max(1, int(total_batches * sample_fraction))
        if verbose:
            print(f"Using approximately {sample_fraction*100:.0f}% of validation: {max_batches}/{total_batches} batches")
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_spec_tables = []
    meta_rows = []
    
    for mode in modes:
        if verbose:
            print("="*70)
            print(f"Analyzing mode = {mode}")
            print("="*70)
        
        ckpt_path = pick_ckpt_path(mode)
        if ckpt_path and os.path.exists(os.path.join(os.path.dirname(ckpt_path), "config.json")):
             config = GPT2Config.from_pretrained(os.path.dirname(ckpt_path))
             if verbose: print(f"🔹 Loaded config from {os.path.dirname(ckpt_path)}")
        else:
             print(f"⚠️ Config not found. Falling back to --{model_size}")
             spec = MODEL_SPECS.get(model_size, MODEL_SPECS["base"])
             config = GPT2Config(
                 vocab_size=50257, 
                 n_positions=1024, 
                 n_ctx=1024, 
                 n_embd=spec["n_embd"], 
                 n_layer=spec["n_layer"], 
                 n_head=spec["n_head"],
                 n_inner=spec["d_ff"]
             )

        model = build_model_for_mode(mode, num_experts=num_experts, config=config)
        load_checkpoint_if_exists(model, mode, CHECKPOINTS_DIR, strict=False)
        model.to(device).eval()
        
        meta = extract_model_metadata(model, mode, num_experts)
        meta_rows.append(meta)
        
        # 신뢰도 관련 stats_conf 제거됨
        stats_spec = defaultdict(lambda: defaultdict(Counter))
        
        processed = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"][:, :seq_len].to(device, non_blocking=True)
            mask = batch["attention_mask"][:, :seq_len].to(device, non_blocking=True)
            
            current_bsz, current_seq = input_ids.shape
            pile_set_ids_scalar = batch["pile_set_id"].view(current_bsz, 1)
            pile_set_ids = pile_set_ids_scalar.expand(current_bsz, current_seq)

            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        # Token NLL 계산(forward_with_token_nll) 제거됨
                        per_layer = analyze_batch_collect_specialization(model, input_ids, mask)
                else:
                    per_layer = analyze_batch_collect_specialization(model, input_ids, mask)
            
            accumulate_specialization(per_layer, pile_set_ids, stats_spec)
            
            processed += 1
            if verbose and processed % 10 == 0:
                print(f"Processed {processed}/{max_batches} batches")
        
        # 신뢰도 테이블 생성 함수 호출 제거
        spec_df = finalize_specialization_tables(stats_spec, idx_to_source)
        
        if not spec_df.empty:
            spec_df.insert(0, "Model", mode)
            all_spec_tables.append(spec_df)
        
        del model
        torch.cuda.empty_cache()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    outputs = {}
    
    if all_spec_tables:
        spec_out = os.path.join(CHECKPOINTS_DIR, "expert_source_mapping_with_specialization.csv")
        pd.concat(all_spec_tables, ignore_index=True).to_csv(spec_out, index=False)
        if verbose:
            print(f"Specialization CSV saved: {spec_out}")
        outputs["specialization_csv"] = spec_out
    else:
        if verbose:
            print("No specialization table produced")
            
    if meta_rows:
        meta_out = os.path.join(CHECKPOINTS_DIR, "routing_experiment_meta.csv")
        pd.DataFrame(meta_rows).to_csv(meta_out, index=False)
        outputs["experiment_meta_csv"] = meta_out
    
    summary_path = os.path.join(CHECKPOINTS_DIR, "analysis_specialization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    
    return outputs

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, default="switch,gshard,hash,ours_refine")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=44)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--no_flash", action="store_true")
    ap.add_argument("--sample_fraction", type=float, default=0.10)
    args = ap.parse_args()
    
    ms = "base"
    if args.small: ms = "small"
    elif args.large: ms = "large"
    
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_specialization_only(
        modes=modes,
        model_size=ms,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        use_flash_attn=not args.no_flash,
        sample_fraction=args.sample_fraction,
    )