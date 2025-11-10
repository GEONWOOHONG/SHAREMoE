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

from config import CHECKPOINTS_DIR, HF_DATASETS_CACHE, HF_HOME, HASH_TABLE_PATH
from modeling import MoELayer
from data import load_or_prepare_pile, worker_init_fn, get_dataloader_generator
from utils import (
    ensure_flash_attn,
    build_model_for_mode,
    load_checkpoint_if_exists,
)

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", HF_HOME if HF_HOME else "/workspace/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE if HF_DATASETS_CACHE else "/workspace/hf_cache/datasets")
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

def load_model_safetensors(model: nn.Module, ckpt_path: str, strict: bool=False):
    from utils import load_safetensors
    load_safetensors(model, ckpt_path, strict=strict)
    print(f"Loaded tensors from: {ckpt_path}")

def load_pile_validation_with_source_labels() -> Tuple["datasets.Dataset", Dict[str,int], Dict[int,str]]:
    train, valid = load_or_prepare_pile(verbose=True)
    ds = valid
    print("Building source label mapping from meta.pile_set_name")
    all_sources = set([m["pile_set_name"] for m in ds["meta"]])
    source_to_idx = {name: i for i, name in enumerate(sorted(all_sources))}
    idx_to_source = {i: n for n, i in source_to_idx.items()}
    def add_id(example):
        return {"pile_set_id": source_to_idx[example["meta"]["pile_set_name"]]}
    ds = ds.map(add_id, num_proc=max(1, os.cpu_count()//2))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "pile_set_id"])
    print(f"Pile validation prepared: {len(ds)} samples, {len(source_to_idx)} unique sources")
    return ds, source_to_idx, idx_to_source

@torch.no_grad()
def analyze_batch_collect(model: nn.Module,
                          input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
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
                probs = torch.softmax(scores.detach(), dim=-1)
                top1 = torch.argmax(probs, dim=-1)
                top1_prob = probs.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
                per_layer.append({"probs": probs, "top1": top1, "top1_prob": top1_prob})
            else:
                per_layer.append({"probs": None, "top1": None, "top1_prob": None})
    return per_layer

def accumulate_specialization_and_confidence(per_layer: List[Dict[str, torch.Tensor]],
                                             pile_set_ids: torch.Tensor,
                                             stats_spec: Dict,
                                             stats_conf: Dict,
                                             stats_conf_by_source: Dict):
    B, S = pile_set_ids.shape
    token_sources = pile_set_ids.reshape(-1).cpu().tolist()
    layer_idx = -1
    for entry in per_layer:
        layer_idx += 1
        probs = entry.get("probs", None)
        top1 = entry.get("top1", None)
        top1_prob = entry.get("top1_prob", None)
        if probs is None and top1 is None:
            continue
        if probs is not None:
            N = probs.size(0)
            assert N == B*S, "Mismatch: routing probs vs input tokens"
            expert_ids = torch.argmax(probs, dim=-1).cpu().tolist()
            max_probs = top1_prob.cpu().tolist()
        else:
            expert_ids = top1.view(-1).cpu().tolist()
            max_probs = [None for _ in expert_ids]
        c = stats_spec[layer_idx]
        for eid, sid in zip(expert_ids, token_sources):
            c[eid][sid] += 1
        cc = stats_conf[layer_idx]
        for p, eid in zip(max_probs, expert_ids):
            if p is not None:
                cc["top1_probs"].append(float(p))
            cc["expert_choices"].append(int(eid))
        cs = stats_conf_by_source[layer_idx]
        for p, eid, sid in zip(max_probs, expert_ids, token_sources):
            if p is not None:
                cs[eid][sid].append(float(p))

def finalize_specialization_tables(stats_spec: Dict,
                                   idx_to_source: Dict[int, str],
                                   stats_conf_by_source: Dict,
                                   source_cardinality: int) -> pd.DataFrame:
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
    df["Expert_Total_Activation"] = df.groupby(["Layer","Expert_ID"])["Activation_Count"].transform("sum")
    df["P_Source_Given_Expert"] = df["Activation_Count"] / (df["Expert_Total_Activation"] + 1e-12)
    total_act = df["Activation_Count"].sum()
    gsrc = df.groupby("Pile_Set_Name")["Activation_Count"].sum() / float(total_act)
    gdf = gsrc.rename("P_Source_Global").reset_index()
    df = df.merge(gdf, on="Pile_Set_Name", how="left")
    df["Specialization_Index"] = df["P_Source_Given_Expert"] / (df["P_Source_Global"] + 1e-10)
    idxmax = df.groupby(["Layer","Expert_ID"])["Specialization_Index"].idxmax()
    max_df = df.loc[idxmax, ["Layer","Expert_ID","Pile_Set_Name","Specialization_Index"]].copy()
    max_df = max_df.rename(columns={
        "Pile_Set_Name":"Max_Specialized_Source",
        "Specialization_Index":"Max_Specialization_Index"
    })
    df = df.merge(max_df, on=["Layer","Expert_ID"], how="left")
    ent = df.groupby(["Layer","Expert_ID"])["P_Source_Given_Expert"].apply(lambda s: entropy(s.values, base=2))
    ent = ent.rename("Source_Entropy").reset_index()
    df = df.merge(ent, on=["Layer","Expert_ID"], how="left")
    rows_conf = []
    for layer, exp_dict in stats_conf_by_source.items():
        for eid, sdict in exp_dict.items():
            for sid, probs in sdict.items():
                if not probs:
                    continue
                rows_conf.append({
                    "Layer": layer, "Expert_ID": eid, "Pile_Set_ID": sid,
                    "Pile_Set_Name": idx_to_source[sid],
                    "Avg_Confidence_for_Source": float(np.mean(probs)),
                    "Std_Confidence_for_Source": float(np.std(probs))
                })
    if rows_conf:
        cdf = pd.DataFrame(rows_conf)
        df = df.merge(cdf, on=["Layer","Expert_ID","Pile_Set_ID","Pile_Set_Name"], how="left")
    preferred_cols = [
        "Layer","Expert_ID","Pile_Set_Name","Activation_Count",
        "Expert_Total_Activation","P_Source_Global","P_Source_Given_Expert",
        "Specialization_Index","Max_Specialized_Source","Max_Specialization_Index",
        "Source_Entropy","Avg_Confidence_for_Source","Std_Confidence_for_Source"
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    return df

def finalize_confidence_tables(stats_conf: Dict, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conf_rows = []
    hist_rows = []
    for layer, d in stats_conf.items():
        probs = np.array(d.get("top1_probs", []), dtype=np.float32)
        choices = np.array(d["expert_choices"], dtype=np.int32)
        if choices.size == 0:
            continue
        vals, cnts = np.unique(choices, return_counts=True)
        freqs = cnts / cnts.sum()
        ent = float(-np.sum(freqs * np.log(freqs + 1e-12))) if freqs.size > 0 else 0.0
        if probs.size > 0:
            very = float(np.mean(probs > 0.9))
            mid  = float(np.mean((probs > 0.75) & (probs <= 0.9)))
            low  = float(np.mean((probs > 0.5) & (probs <= 0.75)))
            unc  = float(np.mean(probs <= 0.5))
            avgp, stdp = float(probs.mean()), float(probs.std())
            pmin, pmax = float(probs.min()), float(probs.max())
        else:
            very = mid = low = unc = 0.0
            avgp = stdp = pmin = pmax = 0.0
        conf_rows.append({
            "Model": model_name,
            "Layer": int(layer),
            "Entropy": ent,
            "Avg_Top1_Prob": avgp,
            "Std_Top1_Prob": stdp,
            "Min_Top1_Prob": pmin,
            "Max_Top1_Prob": pmax,
            "Very_Confident_Ratio_P_gt_0.9": very,
            "Moderate_Confident_Ratio_0.75_lt_P_le_0.9": mid,
            "Low_Confidence_Ratio_0.5_lt_P_le_0.75": low,
            "Uncertain_Ratio_P_le_0.5": unc,
            "Num_Active_Experts_gt_1pct": float((freqs > 0.01).sum()),
            "Total_Decisions": int(choices.size)
        })
        if probs.size > 0:
            counts, bins = np.histogram(probs, bins=20, range=(0.0, 1.0))
            for i in range(len(counts)):
                bin_center = (bins[i] + bins[i+1]) * 0.5
                bin_width  = (bins[i+1] - bins[i])
                hist_rows.append({
                    "Model": model_name,
                    "Layer": int(layer),
                    "Bin_Index": i,
                    "Bin_Start": float(bins[i]),
                    "Bin_End": float(bins[i+1]),
                    "Bin_Center": float(bin_center),
                    "Bin_Width": float(bin_width),
                    "Count": int(counts[i]),
                    "Density": float(counts[i] / (max(1, probs.size) * bin_width))
                })
    cdf = pd.DataFrame(conf_rows) if conf_rows else pd.DataFrame()
    hdf = pd.DataFrame(hist_rows) if hist_rows else pd.DataFrame()
    return cdf, hdf

def run_specialization_confidence(
    modes: List[str] = ("switch","gshard","hash","ours_refine"),
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
        print("Specialization and Confidence Analysis")
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
    all_conf_tables = []
    all_hist_tables = []
    for mode in modes:
        assert mode in {"dense","switch","gshard","hash","ours_refine"}, f"Unsupported mode: {mode}"
        if verbose:
            print("="*70)
            print(f"Analyzing mode = {mode}")
            print("="*70)
        config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=1024, n_layer=8, n_head=8)
        model = build_model_for_mode(mode, num_experts=num_experts, config=config)
        load_checkpoint_if_exists(model, mode, CHECKPOINTS_DIR, strict=False)
        model.to(device).eval()
        stats_spec = defaultdict(lambda: defaultdict(Counter))
        stats_conf = defaultdict(lambda: {"top1_probs": [], "expert_choices": []})
        stats_conf_by_source = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        processed = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"][:, :seq_len].to(device, non_blocking=True)
            mask = batch["attention_mask"][:, :seq_len].to(device, non_blocking=True)
            pile_set_ids = batch["pile_set_id"][:, :seq_len]
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        per_layer = analyze_batch_collect(model, input_ids, mask)
                else:
                    per_layer = analyze_batch_collect(model, input_ids, mask)
            accumulate_specialization_and_confidence(per_layer, pile_set_ids, stats_spec, stats_conf, stats_conf_by_source)
            processed += 1
            if verbose and processed % 10 == 0:
                print(f"Processed {processed}/{max_batches} batches")
        spec_df = finalize_specialization_tables(stats_spec, idx_to_source, stats_conf_by_source, source_cardinality=len(source_to_idx))
        if not spec_df.empty:
            spec_df.insert(0, "Model", mode)
            all_spec_tables.append(spec_df)
        conf_df, hist_df = finalize_confidence_tables(stats_conf, model_name=mode)
        if not conf_df.empty:
            all_conf_tables.append(conf_df)
        if not hist_df.empty:
            all_hist_tables.append(hist_df)
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
    if all_conf_tables:
        conf_out = os.path.join(CHECKPOINTS_DIR, "routing_confidence_analysis.csv")
        pd.concat(all_conf_tables, ignore_index=True).to_csv(conf_out, index=False)
        if verbose:
            print(f"Confidence CSV saved: {conf_out}")
        outputs["confidence_csv"] = conf_out
    else:
        if verbose:
            print("No confidence table produced")
    if all_hist_tables:
        hist_out = os.path.join(CHECKPOINTS_DIR, "routing_confidence_histogram.csv")
        pd.concat(all_hist_tables, ignore_index=True).to_csv(hist_out, index=False)
        if verbose:
            print(f"Confidence histogram CSV saved: {hist_out}")
        outputs["confidence_histogram_csv"] = hist_out
    summary_path = os.path.join(CHECKPOINTS_DIR, "analysis_specialization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    if verbose:
        print(f"Summary JSON: {summary_path}")
    return outputs

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, default="switch,gshard,hash,ours_refine")
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=44)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--no_flash", action="store_true")
    ap.add_argument("--sample_fraction", type=float, default=0.10)
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_specialization_confidence(
        modes=modes,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        use_flash_attn=not args.no_flash,
        sample_fraction=args.sample_fraction,
    )