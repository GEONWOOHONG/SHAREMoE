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
    from data import load_pile_test
    ds = load_pile_test(verbose=True)
    print(f"Building source label mapping from meta.pile_set_name (using test set)")
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
                top2_vals, top2_idx = torch.topk(scores.detach(), k=2, dim=-1)
                logit_margin = (top2_vals[..., 0] - top2_vals[..., 1])
                logit_std = scores.detach().float().std(dim=-1, unbiased=False)
                std_margin = logit_margin / (logit_std + 1e-9)
                
                token_entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
                norm_entropy = token_entropy / math.log(probs.size(-1) + 1e-12)
                
                scores_f = scores.detach().float()
                mu = scores_f.mean(dim=-1, keepdim=True)
                sd = scores_f.std(dim=-1, keepdim=True, unbiased=False)
                z = (scores_f - mu) / (sd + 1e-9)
                pz = torch.softmax(z, dim=-1)
                z_top1 = torch.gather(pz, -1, top2_idx[..., 0:1]).squeeze(-1)
                
                top1 = torch.argmax(probs, dim=-1)
                top1_prob = probs.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
                per_layer.append({
                    "probs": probs,
                    "top1": top1,
                    "top1_prob": top1_prob,
                    "std_margin": std_margin,
                    "norm_entropy": norm_entropy,
                    "z_top1_prob": z_top1,
                })
            else:
                per_layer.append({
                    "probs": None, "top1": None, "top1_prob": None,
                    "std_margin": None, "norm_entropy": None, "z_top1_prob": None
                })
    return per_layer

@torch.no_grad()
def forward_with_token_nll(model: nn.Module,
                           input_ids: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
    else:
        mask = torch.ones_like(labels, dtype=torch.float)
    logp = torch.log_softmax(logits, dim=-1)
    nll = -(logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
    nll = nll * mask
    return nll

def accumulate_specialization_and_confidence(per_layer: List[Dict[str, torch.Tensor]],
                                             pile_set_ids: torch.Tensor,
                                             stats_spec: Dict,
                                             stats_conf: Dict,
                                             stats_conf_by_source: Dict,
                                             token_nll: Optional[torch.Tensor] = None):
    B, S = pile_set_ids.shape
    token_sources = pile_set_ids[:, :-1].reshape(-1).cpu().tolist()
    nll_flat = None
    if token_nll is not None:
        nll_flat = token_nll.reshape(-1).float().cpu().numpy()
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
            expert_ids = torch.argmax(probs, dim=-1).cpu().tolist()
            max_probs = top1_prob.cpu().numpy()
            std_m = entry.get("std_margin", None)
            nH = entry.get("norm_entropy", None)
            z_p1 = entry.get("z_top1_prob", None)
            if std_m is not None:
                std_m = std_m.view(-1).cpu().numpy()
            if nH is not None:
                nH = nH.view(-1).cpu().numpy()
            if z_p1 is not None:
                z_p1 = z_p1.view(-1).cpu().numpy()
        else:
            expert_ids = top1.view(-1).cpu().tolist()
            max_probs = np.array([np.nan for _ in expert_ids], dtype=np.float32)
            std_m = nH = z_p1 = None
        c = stats_spec[layer_idx]
        for eid, sid in zip(expert_ids, token_sources):
            c[eid][sid] += 1
        cc = stats_conf[layer_idx]
        cc.setdefault("top1_probs", []).extend(max_probs.tolist())
        cc.setdefault("expert_choices", []).extend(expert_ids)
        if std_m is not None:
            cc.setdefault("std_margin", []).extend(std_m.tolist())
        if nH is not None:
            cc.setdefault("norm_entropy", []).extend(nH.tolist())
        if z_p1 is not None:
            cc.setdefault("z_top1_prob", []).extend(z_p1.tolist())
        if nll_flat is not None:
            cc.setdefault("token_nll", []).extend(nll_flat.tolist())
        cs = stats_conf_by_source[layer_idx]
        for p, eid, sid in zip(max_probs, expert_ids, token_sources):
            if not np.isnan(p):
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
        probs = probs[~np.isnan(probs)]
        choices = np.array(d["expert_choices"], dtype=np.int32)
        std_m = np.array(d.get("std_margin", []), dtype=np.float32) if "std_margin" in d else None
        nH = np.array(d.get("norm_entropy", []), dtype=np.float32) if "norm_entropy" in d else None
        z_p1 = np.array(d.get("z_top1_prob", []), dtype=np.float32) if "z_top1_prob" in d else None
        
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
        
        row = {
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
        }
        if std_m is not None and std_m.size > 0:
            std_m = std_m[~np.isnan(std_m)]
            row["Avg_StdMargin"] = float(std_m.mean()) if std_m.size > 0 else 0.0
            row["Std_StdMargin"] = float(std_m.std()) if std_m.size > 0 else 0.0
        if nH is not None and nH.size > 0:
            nH = nH[~np.isnan(nH)]
            row["Avg_NormEntropy"] = float(nH.mean()) if nH.size > 0 else 0.0
            row["Std_NormEntropy"] = float(nH.std()) if nH.size > 0 else 0.0
        if z_p1 is not None and z_p1.size > 0:
            z_p1 = z_p1[~np.isnan(z_p1)]
            row["Avg_ZTop1Prob"] = float(z_p1.mean()) if z_p1.size > 0 else 0.0
            row["Std_ZTop1Prob"] = float(z_p1.std()) if z_p1.size > 0 else 0.0
        conf_rows.append(row)
        
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

def summarize_confidence_quality(stats_conf: Dict, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr, pearsonr
    
    rows = []
    bin_rows = []
    bins = np.linspace(0.0, 1.0, 21)
    
    for layer, d in stats_conf.items():
        if not d.get("token_nll"):
            continue
        nll = np.array(d["token_nll"], dtype=np.float32)
        z_p = np.array(d.get("z_top1_prob", []), dtype=np.float32) if "z_top1_prob" in d else None
        m = np.array(d.get("std_margin", []), dtype=np.float32) if "std_margin" in d else None
        nH = np.array(d.get("norm_entropy", []), dtype=np.float32) if "norm_entropy" in d else None
        
        def corr(a, b):
            if a is None or b is None:
                return np.nan, np.nan
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 100:
                return np.nan, np.nan
            s = spearmanr(a[ok], b[ok]).correlation
            p = pearsonr(a[ok], b[ok])[0]
            return s, p
        
        s_z, p_z = corr(-nll, z_p) if z_p is not None else (np.nan, np.nan)
        s_m, p_m = corr(-nll, m) if m is not None else (np.nan, np.nan)
        s_h, p_h = corr(nll, nH) if nH is not None else (np.nan, np.nan)
        
        if z_p is not None and z_p.size > 0:
            dig = np.digitize(z_p, bins) - 1
            for bi in range(len(bins)-1):
                sel = dig == bi
                if sel.sum() == 0:
                    continue
                bin_rows.append({
                    "Model": model_name,
                    "Layer": int(layer),
                    "BinStart": float(bins[bi]),
                    "BinEnd": float(bins[bi+1]),
                    "Count": int(sel.sum()),
                    "AvgNLL": float(nll[sel].mean()),
                    "StdNLL": float(nll[sel].std()),
                    "AvgZTop1": float(z_p[sel].mean()),
                })
        
        rows.append({
            "Model": model_name,
            "Layer": int(layer),
            "Spearman_negNLL_vs_ZTop1": float(s_z),
            "Pearson_negNLL_vs_ZTop1": float(p_z),
            "Spearman_negNLL_vs_StdMargin": float(s_m),
            "Pearson_negNLL_vs_StdMargin": float(p_m),
            "Spearman_NLL_vs_NormEntropy": float(s_h),
            "Pearson_NLL_vs_NormEntropy": float(p_h),
        })
    
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)

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
    
    if mode == "gshard":
        active_k = 2
    elif mode == "switch":
        active_k = 1
    elif mode == "hash":
        active_k = 1
    
    return {
        "Model": mode,
        "VisibleExperts": num_experts,
        "ActiveK": active_k,
        "CapacityFactor": capacity_factor if capacity_factor is not None else "N/A",
        "DroplessEval": True,
        "AuxAlpha": aux_alpha if aux_alpha is not None else 0.0,
        "HasAux": bool(aux_alpha not in (0.0, None)) if aux_alpha is not None else False,
    }

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
    meta_rows = []
    
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
        
        meta = extract_model_metadata(model, mode, num_experts)
        meta_rows.append(meta)
        
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
                    with torch.autocast("cuda", enabled=False):
                        token_nll = forward_with_token_nll(model, input_ids, mask)
                else:
                    per_layer = analyze_batch_collect(model, input_ids, mask)
                    token_nll = forward_with_token_nll(model, input_ids, mask)
            
            for entry in per_layer:
                if entry.get("probs") is None:
                    continue
                for k in ["probs", "top1", "top1_prob", "std_margin", "norm_entropy", "z_top1_prob"]:
                    if entry.get(k) is not None:
                        t = entry[k]
                        if t.dim() == 2:
                            entry[k] = t[:, :-1]
                        elif t.dim() == 3:
                            entry[k] = t[:, :-1, :]
            
            accumulate_specialization_and_confidence(per_layer, pile_set_ids, stats_spec, stats_conf, stats_conf_by_source, token_nll)
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
        
        qual_df, bin_df = summarize_confidence_quality(stats_conf, model_name=mode)
        if not qual_df.empty:
            if "quality_tables" not in locals():
                quality_tables = []
            quality_tables.append(qual_df)
        if not bin_df.empty:
            if "bin_tables" not in locals():
                bin_tables = []
            bin_tables.append(bin_df)
        
        del model
        torch.cuda.empty_cache()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    outputs = {}
    quality_tables = locals().get("quality_tables", [])
    bin_tables = locals().get("bin_tables", [])
    
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
    
    if quality_tables:
        qual_out = os.path.join(CHECKPOINTS_DIR, "routing_confidence_quality_correlation.csv")
        pd.concat(quality_tables, ignore_index=True).to_csv(qual_out, index=False)
        if verbose:
            print(f"Confidence quality correlation CSV saved: {qual_out}")
        outputs["quality_correlation_csv"] = qual_out
    
    if bin_tables:
        bin_out = os.path.join(CHECKPOINTS_DIR, "routing_confidence_binned_nll.csv")
        pd.concat(bin_tables, ignore_index=True).to_csv(bin_out, index=False)
        if verbose:
            print(f"Binned NLL CSV saved: {bin_out}")
        outputs["binned_nll_csv"] = bin_out
    
    if meta_rows:
        meta_out = os.path.join(CHECKPOINTS_DIR, "routing_experiment_meta.csv")
        pd.DataFrame(meta_rows).to_csv(meta_out, index=False)
        if verbose:
            print(f"Experiment metadata CSV saved: {meta_out}")
        outputs["experiment_meta_csv"] = meta_out
    
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