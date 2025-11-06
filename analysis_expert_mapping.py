import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import tiktoken
import warnings
from scipy.stats import entropy as shannon_entropy
from config import HF_DATASETS_CACHE, HASH_TABLE_PATH, CHECKPOINTS_DIR

from modeling import (
    convert_gpt2_to_moe,
    GPT2LayerMoE,
    HashRouter,
    Expert,
    Router,
    MoELayer,
)
from patches import (
    patch_model_basic,
    patch_model_for_hash_moe,
    patch_model_for_ours_com,
    block_moe_forward_patch,
)

from utils import set_seed, load_safetensors

warnings.filterwarnings("ignore")

import numpy as np

def _entropy_from_counts(cnt):
    p = cnt / (cnt.sum() + 1e-12)
    return float(-(p * np.log2(p + 1e-12)).sum())

def _mi_from_labels(x, y):
    x = x.astype(np.int64); y = y.astype(np.int64)
    Hx = _entropy_from_counts(np.bincount(x))
    Hy = _entropy_from_counts(np.bincount(y))
    base = int(max(x.max(), y.max())) + 1
    Hxy = _entropy_from_counts(np.bincount(x * base + y))
    return Hx + Hy - Hxy

def _gini(counts):
    x = np.sort(counts.astype(np.float64))
    if x.sum() <= 0: return 0.0
    n = len(x); cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n

def _ece_top1(probs, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if not mask.any(): 
            continue
        conf = float(probs[mask].mean())
        acc  = 1.0
        ece += mask.mean() * abs(acc - conf)
    return float(ece)

def _brier_top1(probs):
    # top-1 사건의 확률 예측에 대한 Brier score proxy
    return float(((1.0 - probs) ** 2).mean())

# === Top-k invariant routing-confidence helpers ===
def _selected_set_metrics_from_scores(scores: torch.Tensor, k: int):
    """
    scores: [N, E] softmax 직후 확률 (또는 normalize된 점수)
    k: 선택 집합 크기 (Switch/top-1=1, GShard/top-2=2)
    Returns: dict of numpy arrays (cpu)
    """
    eps = 1e-12
    topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)           # [N,k]
    C = topk_vals.sum(dim=-1)                                       # [N]
    if k == 1:
        D = torch.ones_like(C)                                      # top-1이면 내부 마진=1
        H_sel = torch.zeros_like(C)
        EEC = torch.ones_like(C)
        S1 = torch.ones_like(C)                                     # 집합 내 점유율=1
    else:
        pnorm = topk_vals / (C.unsqueeze(-1) + eps)                 # [N,k]
        # D: (p1 - p2) / sum; k>1 가정
        D = (topk_vals[:, 0] - topk_vals[:, 1]) / (C + eps)
        H_sel = -(pnorm * (pnorm + eps).log()).sum(dim=-1) / math.log(k)
        EEC = 1.0 / (pnorm.pow(2).sum(dim=-1) + eps)
        S1 = topk_vals[:, 0] / (C + eps)

    return {
        "C": C.detach().float().cpu().numpy(),
        "D": D.detach().float().cpu().numpy(),
        "H_sel": H_sel.detach().float().cpu().numpy(),
        "EEC": EEC.detach().float().cpu().numpy(),
        "S1": S1.detach().float().cpu().numpy(),
        "topk_idx_cpu": topk_idx.detach().int().cpu().numpy(),      # specialization(top-2)용
        "top1_idx_cpu": topk_idx[:, 0].detach().int().cpu().numpy()
    }


enc = tiktoken.get_encoding("gpt2")

set_seed(42)

def load_pile_validation_dataset():
    """Pile validation 데이터셋 로드 (HF 허브에서 캐시 사용)"""
    from data import load_or_prepare_pile

    # HF 캐시 경로는 config.HF_DATASETS_CACHE에 이미 들어있음
    train_ds, valid_ds = load_or_prepare_pile()
    print(f"✅ Loaded validation split: {len(valid_ds)} samples")

    # meta.pile_set_name이 들어있는지 확인(토크나이즈 버전이면 meta가 그대로 들어있음)
    if "meta" not in valid_ds.column_names:
        raise KeyError("❌ validation split has no 'meta' column (pile_set_name required)")

    return valid_ds

@torch.no_grad()
def analyze_expert_specialization(
    model, dataloader, device, mode, pile_set_map,
    max_batches=None, run_specialization=True, run_confidence=True, run_routes=True
):
    """
    Optimized: 
    - GPU에서 argmax/max/topk까지 처리 → CPU 전송량 최소화
    - route 시퀀스는 배치 내에서 np.unique로 벡터화 집계 후 누적
    - MI 계산은 라우트 행렬만 CPU로 내린 뒤 벡터화
    """
    model.eval()

    # === Pre-cache MoE layers ===
    moe_layers = []
    for h in model.transformer.h:
        if isinstance(h.mlp, GPT2LayerMoE):
            moe_layers.append(h.mlp.moe)
    print(f"✅ Found {len(moe_layers)} MoE layers")

    reverse_pile_set_map = {v: k for k, v in pile_set_map.items()}

    # accumulators
    specialization_stats = defaultdict(lambda: defaultdict(Counter))
    confidence_stats = defaultdict(lambda: {
        'selected_mass': [],     # C
        'decisiveness': [],      # D
        'h_sel': [],             # H_sel
        'eec': [],               # EEC
        'top1_share': [],        # S1
        'expert_choices': []     # top-1 선택 eid (route/빈도용)
    })
    # per-source는 '확률' 대신 Selected-Mass(C) 평균으로 전환
    confidence_per_source_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # [layer][expert][source] -> list of C

    domain_route_counter = defaultdict(Counter)  # domain -> Counter(route_id)  (압축키 사용)
    total_routes_per_domain = Counter()

    # fairness-safe metrics
    smoothness_vals = []
    interlayer_mi_by_dist = defaultdict(list)
    load_counts_acc = dict()
    calibration_rows = []
    loadbalance_rows = []

    # 루프
    total_batches = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    progress = tqdm(enumerate(dataloader), total=total_batches, desc=f"[{mode}] Analyzing Data")

    for i, batch in progress:
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True).bool()
        pile_set_ids = batch["pile_set_id"]                 # CPU tensor
        B, S = input_ids.shape

        # GPU에서 한 번에 실행
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(input_ids=input_ids, attention_mask=attn_mask)

        # GPU boolean mask (flatten)
        valid_t = attn_mask.view(-1)                         # [B*T] (cuda)
        # CPU 측에서 쓰는 라벨/도메인
        pile_set_ids_exp_cpu = pile_set_ids.unsqueeze(1).expand(B, S).reshape(-1).cpu().numpy()

        # 이 배치에서 레이어별 argmax expert id (유효 토큰만) 모아두기
        batch_layer_eids_valid = []

        for layer_idx, moe_layer in enumerate(moe_layers):
            scores = None
            eids_flat = None           # torch.long [B*T] on GPU

            if mode == "hash":
                # 이미 GPU long tensor로 계산됨
                eids_flat = moe_layer.hash_router.route(input_ids).view(-1)
            else:
                # ours_com / switch / gshard / expert_choice : last_scores 또는 router.last_scores
                if getattr(moe_layer, 'last_scores', None) is not None:
                    scores = moe_layer.last_scores            # [B*T, E] or [B,T,E] → 아래에서 reshape
                elif hasattr(moe_layer, 'router') and getattr(moe_layer.router, 'last_scores', None) is not None:
                    scores = moe_layer.router.last_scores
                # shape 정규화
                if scores is not None and scores.dim() == 3:
                    scores = scores.view(-1, scores.size(-1))  # [B*T, E]

            # --- 공정지표용 top-k 설정 ---
            k_sel = 2 if mode == "gshard" else 1

            if scores is not None:
                with torch.no_grad():
                    row_sum = scores.sum(dim=-1, keepdim=True)
                need_norm = (scores.min() < 0) or (scores.max() > 1.0) or (
                    not torch.allclose(row_sum.mean(), torch.tensor(1.0, device=row_sum.device), atol=1e-3)
                )
                if need_norm:
                    scores = torch.softmax(scores.float(), dim=-1)

                met = _selected_set_metrics_from_scores(scores, k=k_sel)
                top1_gpu = torch.as_tensor(met["top1_idx_cpu"], device=scores.device, dtype=torch.long)  # 편의상 gpu로
                eids_flat = top1_gpu
                if mode == "gshard":
                    topk_idx_cpu = met["topk_idx_cpu"][valid_t.cpu().numpy()]  # specialization에서 top-2 카운트용

            if eids_flat is None:
                batch_layer_eids_valid.append(None)
                continue

            eids_valid_cpu = eids_flat[valid_t].int().cpu().numpy()              # [N_valid]
            batch_layer_eids_valid.append(eids_valid_cpu)

            # ---- Load-balance (per-layer counts) 누적 (CPU 벡터화) ----
            E = int(scores.size(-1)) if scores is not None else int(eids_valid_cpu.max()) + 1
            if layer_idx not in load_counts_acc:
                load_counts_acc[layer_idx] = np.zeros(E, dtype=np.float64)
            elif load_counts_acc[layer_idx].shape[0] < E:
                tmp = np.zeros(E, dtype=np.float64)
                tmp[:load_counts_acc[layer_idx].shape[0]] = load_counts_acc[layer_idx]
                load_counts_acc[layer_idx] = tmp
            load_counts_acc[layer_idx][:E] += np.bincount(eids_valid_cpu, minlength=E).astype(np.float64)

            # ---- Specialization (도메인-전문가 카운트) ----
            if run_specialization:
                sids_valid = pile_set_ids_exp_cpu[valid_t.cpu().numpy()]   # [N_valid]
                if mode == "gshard" and scores is not None:
                    # top-2만 CPU로 이미 내려와 있음
                    for (e1, e2), sid in zip(topk_idx_cpu, sids_valid):
                        src = reverse_pile_set_map[int(sid)]
                        specialization_stats[layer_idx][int(e1)][src] += 1
                        specialization_stats[layer_idx][int(e2)][src] += 1
                elif mode == "gshard" and scores is None:
                    for eid, sid in zip(eids_valid_cpu, sids_valid):
                        src = reverse_pile_set_map[int(sid)]
                        specialization_stats[layer_idx][int(eid)][src] += 1

            # ---- Confidence (Selected-Mass 기반) ----
            if run_confidence and (scores is not None):
                vt = valid_t.cpu().numpy()
                C_valid  = met["C"][vt]
                D_valid  = met["D"][vt]
                H_valid  = met["H_sel"][vt]
                EEC_valid= met["EEC"][vt]
                S1_valid = met["S1"][vt]

                cs = confidence_stats[layer_idx]
                cs['selected_mass'].extend(C_valid.tolist())
                cs['decisiveness'].extend(D_valid.tolist())
                cs['h_sel'].extend(H_valid.tolist())
                cs['eec'].extend(EEC_valid.tolist())
                cs['top1_share'].extend(S1_valid.tolist())
                cs['expert_choices'].extend(eids_valid_cpu.tolist())

                # 도메인별 평균: 이제 C(Selected-Mass)를 저장
                sids_valid = pile_set_ids_exp_cpu[vt]
                for c, eid, sid in zip(C_valid, eids_valid_cpu, sids_valid):
                    src = reverse_pile_set_map[int(sid)]
                    confidence_per_source_stats[layer_idx][int(eid)][src].append(float(c))

        # ---- Route 시퀀스 (배치 내 벡터화) ----
        if run_routes:
            # None 레이어 제외
            stacked = [e for e in batch_layer_eids_valid if e is not None]
            if stacked:
                routes_mat = np.stack(stacked, axis=0).astype(np.int16, copy=False)  # [L_eff, N_valid]
                routes_tok = routes_mat.T                                            # [N_valid, L_eff]

                # Smoothness: 인접 토큰간 해밍거리 평균
                if routes_tok.shape[0] > 1:
                    pair_diff = (routes_tok[1:] != routes_tok[:-1]).mean(axis=1)
                    smoothness_vals.append(float(1.0 - pair_diff.mean()))

                # Inter-layer MI: 벡터화 루프(작은 L이라 가볍지만 numpy만 사용)
                L_eff = routes_tok.shape[1]
                for d in range(1, L_eff):
                    # (i, i+d) 쌍만 모아 평균
                    mi_vals = []
                    for i0 in range(L_eff - d):
                        xi = routes_tok[:, i0]
                        xj = routes_tok[:, i0 + d]
                        # MI 계산(벡터화된 카운트)
                        base = int(max(xi.max(), xj.max())) + 1
                        joint = np.bincount(xi.astype(np.int64) * base + xj.astype(np.int64))
                        joint = joint[joint > 0].astype(np.float64)
                        pxy = joint / joint.sum()
                        px = np.bincount(xi, minlength=base).astype(np.float64); px = px[px > 0]; px /= px.sum()
                        py = np.bincount(xj, minlength=base).astype(np.float64); py = py[py > 0]; py /= py.sum()
                        Hx = -(px * np.log(px)).sum(); Hy = -(py * np.log(py)).sum()
                        Hxy = -(pxy * np.log(pxy)).sum()
                        mi_vals.append(float(Hx + Hy - Hxy))
                    interlayer_mi_by_dist[d].append(float(np.mean(mi_vals)))

                sids_valid = pile_set_ids_exp_cpu[valid_t.cpu().numpy()]
                for dom in np.unique(sids_valid):
                    mask = (sids_valid == dom)
                    if not np.any(mask): 
                        continue
                    uniq_d, cnt_d = np.unique(routes_tok[mask], axis=0, return_counts=True)
                    for r, c in zip(uniq_d, cnt_d):
                        route_key = tuple(int(x) for x in r)  # tuple 생성은 unique 후 소수만
                        domain_route_counter[reverse_pile_set_map[int(dom)]][route_key] += int(c)
                        total_routes_per_domain[reverse_pile_set_map[int(dom)]] += int(c)

    specialization_results = []
    if run_specialization:
        for layer, expert_data in specialization_stats.items():
            for expert, source_counts in expert_data.items():
                for source, count in source_counts.items():
                    specialization_results.append({
                        "Model": mode, "Layer": layer, "Expert_ID": expert,
                        "Pile_Set_Name": source, "Activation_Count": count
                    })

    confidence_results = []
    confidence_histogram_results = []
    if run_confidence:
        for layer_idx, data in confidence_stats.items():
            if not data['selected_mass']:
                continue
            C = np.asarray(data['selected_mass'], dtype=np.float64)
            D = np.asarray(data['decisiveness'], dtype=np.float64)
            H = np.asarray(data['h_sel'], dtype=np.float64)
            E = np.asarray(data['eec'], dtype=np.float64)
            S1 = np.asarray(data['top1_share'], dtype=np.float64)
            choices = np.asarray(data['expert_choices'], dtype=np.int32)

            if C.size == 0:
                continue

            unique_experts, counts = np.unique(choices, return_counts=True)
            freqs = (counts / counts.sum()).astype(np.float64)
            entropy = float(-(freqs * np.log(freqs + 1e-12)).sum()) if len(freqs) > 0 else 0.0

            # 선택집합 질량 C 기반 binning (공정 비교)
            hist_counts, bin_edges = np.histogram(C, bins=20, range=(0.0, 1.0))

            confidence_results.append({
                'Model': mode, 'Layer': layer_idx, 'Entropy_Load': entropy,
                'Avg_Selected_Mass': float(C.mean()), 'Std_Selected_Mass': float(C.std()),
                'Avg_Decisiveness': float(D.mean()), 'Avg_Hsel': float(H.mean()),
                'Avg_EEC': float(E.mean()), 'Avg_Top1Share': float(S1.mean()),
                'Total_Decisions': int(C.size)
            })

            for i in range(len(hist_counts)):
                confidence_histogram_results.append({
                    'Model': mode, 'Layer': layer_idx, 'Bin_Index': i,
                    'Bin_Start': float(bin_edges[i]), 'Bin_End': float(bin_edges[i+1]),
                    'Bin_Center': float((bin_edges[i]+bin_edges[i+1])/2),
                    'Count': int(hist_counts[i])
                })

    # Confidence per Source
    source_confidence_results = []
    if run_confidence and confidence_per_source_stats:
        for layer, expert_data in confidence_per_source_stats.items():
            for expert, source_data in expert_data.items():
                for source, Cs in source_data.items():
                    if Cs:
                        arr = np.asarray(Cs, dtype=np.float64)
                        source_confidence_results.append({
                            "Model": mode, "Layer": layer, "Expert_ID": expert,
                            "Pile_Set_Name": source,
                            "Avg_SelectedMass_for_Source": float(arr.mean()),
                            "Std_SelectedMass_for_Source": float(arr.std())
                        })
    if run_specialization and run_confidence and specialization_results and source_confidence_results:
        df_spec = pd.DataFrame(specialization_results)
        df_conf_src = pd.DataFrame(source_confidence_results)
        specialization_results = pd.merge(
            df_spec, df_conf_src,
            on=["Model", "Layer", "Expert_ID", "Pile_Set_Name"], how="left"
        ).to_dict('records')

    # Route summary/detail
    route_summary_rows, route_detail_rows = [], []
    if run_routes and len(domain_route_counter) > 0:
        for dom, cnt in domain_route_counter.items():
            total = total_routes_per_domain[dom]
            if total <= 0: 
                continue
            # 상세(상위 1000)
            for route_tup, c in cnt.most_common(1000):
                route_detail_rows.append({
                    "Model": mode, "Domain": dom,
                    "Route": "-".join(map(str, route_tup)),
                    "Count": int(c), "Share": float(c/total),
                })
            # 요약
            counts = np.fromiter(cnt.values(), dtype=np.float64)
            probs = counts / counts.sum()
            route_summary_rows.append({
                "Model": mode, "Domain": dom,
                "RouteConsistency_ModalShare": float(probs.max()),
                "RouteEntropy": float(-(probs * np.log2(probs + 1e-12)).sum()),
                "NumUniqueRoutes": int((counts > 0).sum()),
                "TotalTokens": int(total),
            })

    # Calibration
    for layer_idx, data in confidence_stats.items():
        S1_list = data.get('top1_share', [])
        if not S1_list:
            continue
        probs = np.asarray(S1_list, dtype=np.float64)  # calibration proxy
        calibration_rows.append({
            "Model": mode, "Layer": layer_idx,
            "ECE_S1": float(_ece_top1(probs)), 
            "Brier_S1": float(_brier_top1(probs)),
            "N": int(probs.size)
        })

    # Load-balance
    for layer_idx, counts in load_counts_acc.items():
        if counts.sum() <= 0:
            continue
        gini = _gini(counts)
        cov  = float(counts.std() / (counts.mean() + 1e-12))
        loadbalance_rows.append({
            "Model": mode, "Layer": layer_idx,
            "Gini": float(gini), "CoV": float(cov), "Total": float(counts.sum())
        })

    # Smoothness / Inter-layer MI 집계
    smoothness_rows = []
    if smoothness_vals:
        arr = np.asarray(smoothness_vals, dtype=np.float64)
        smoothness_rows.append({
            "Model": mode, "Smoothness_Mean": float(arr.mean()),
            "Smoothness_Std": float(arr.std()), "Batches": int(arr.size)
        })
    interlayer_mi_rows = []
    for dist, vals in interlayer_mi_by_dist.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size:
            interlayer_mi_rows.append({
                "Model": mode, "LayerDist": int(dist),
                "MI_Mean": float(arr.mean()), "MI_Std": float(arr.std()),
                "Pairs": int(arr.size)
            })

    return (specialization_results, confidence_results, confidence_histogram_results,
            route_summary_rows, route_detail_rows,
            smoothness_rows, interlayer_mi_rows, calibration_rows, loadbalance_rows)

# =====================
# 전체 실험 실행
# =====================
def run_mapping_analysis(
    batch_size=44, base_num_experts=16, max_batches=None,
    run_specialization=True, run_confidence=True, run_routes=True
):
    """Expert-to-Source Mapping 분석 실행"""
    if convert_gpt2_to_moe is None:
        print("❌ MoE functions could not be imported")
        return

    if not run_specialization and not run_confidence:
        print("ℹ️ No analysis mode selected. Exiting run_mapping_analysis.")
        return

    print("=" * 60)
    print("🔬 Expert-to-Source Mapping Analysis")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_layer=8,
        n_head=8
    )

    # === 데이터셋 로드 및 준비 ===
    pile_valid_dataset = load_pile_validation_dataset()

    # 1) 고유 pile_set_name 매핑
    print("📊 Identifying unique data sources from 'meta' column...")
    all_pile_sets = {meta['pile_set_name'] for meta in pile_valid_dataset['meta']}
    source_to_idx = {name: i for i, name in enumerate(sorted(list(all_pile_sets)))}
    print(f"Found {len(source_to_idx)} unique sources: {list(source_to_idx.keys())}")

    # (NEW) 데이터셋에 전역 행 인덱스 부여
    pile_valid_dataset = pile_valid_dataset.map(
        lambda ex, idx: {"__row_id__": idx},
        with_indices=True,
        num_proc=os.cpu_count() // 2
    )

    # 2) pile_set_id 컬럼 추가
    def add_pile_set_id(example):
        return {'pile_set_id': source_to_idx[example['meta']['pile_set_name']]}

    print("🔧 Adding 'pile_set_id' column to the dataset...")
    pile_valid_dataset = pile_valid_dataset.map(add_pile_set_id, num_proc=os.cpu_count() // 2)

    # 3) DataLoader (컬럼에 '__row_id__' 포함!)
    pile_valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pile_set_id", "__row_id__"])
    pile_valid_loader = DataLoader(
        pile_valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # 기본: 전체의 10%만 사용 (명시적 max_batches 지정 시 전체/지정 비율로)
    if max_batches is None:
        num_batches = len(pile_valid_loader)
        limit_batches = max(1, int(num_batches * 0.1))
        print(f"📊 전체 {num_batches}개 배치 중 약 10%인 {limit_batches}개만 사용합니다.")
        max_batches = limit_batches
    elif max_batches == "debug":
        # Debug 모드: 전체의 0.1%만 사용
        num_batches = len(pile_valid_loader)
        limit_batches = max(1, int(num_batches * 0.001))
        print(f"🐛 DEBUG 모드: 전체 {num_batches}개 배치 중 약 0.1%인 {limit_batches}개만 사용합니다.")
        max_batches = limit_batches

    potential_model_paths = {
        "hash":       os.path.join(CHECKPOINTS_DIR, "hash_exp1", "best_checkpoint.safetensors"),
        "ours_com":   os.path.join(CHECKPOINTS_DIR, "ours_com_exp1", "best_checkpoint.safetensors"),
        "gshard":     os.path.join(CHECKPOINTS_DIR, "gshard_exp1", "best_checkpoint.safetensors"),
        "switch":     os.path.join(CHECKPOINTS_DIR, "switch_exp1", "best_checkpoint.safetensors"),
        "stablemoe":  os.path.join(CHECKPOINTS_DIR, "stablemoe_exp1", "best_checkpoint.safetensors"),
        "hypermoe":   os.path.join(CHECKPOINTS_DIR, "hypermoe_exp1", "best_checkpoint.safetensors"),
        "xmoe":       os.path.join(CHECKPOINTS_DIR, "xmoe_exp1", "best_checkpoint.safetensors"),
        "expert_choice": os.path.join(CHECKPOINTS_DIR, "expert_choice_exp1", "best_checkpoint.safetensors"),
    }

    available_model_paths = {
        mode: path for mode, path in potential_model_paths.items() if os.path.exists(path)
    }

    if not available_model_paths:
        print("❌ No model checkpoints found")
        return

    print(f"\n📂 Found {len(available_model_paths)} models:")
    for mode in available_model_paths:
        print(f"   ✓ {mode}")

    # === 분석 ===
    all_specialization_results = []
    all_confidence_results = []
    all_histogram_results = []
    all_route_summary = []
    all_route_detail = []
    all_smoothness = []
    all_interlayer_mi = []
    all_calibration = []
    all_loadbalance = []

    for mode, ckpt_path in available_model_paths.items():
        print(f"\n{'=' * 60}")
        print(f"📊 Analyzing: {mode}")
        print(f"{'=' * 60}")

        # Config
        config_dir = os.path.dirname(ckpt_path)
        config_path = os.path.join(config_dir, "config.json")

        if os.path.exists(config_path):
            config = GPT2Config.from_pretrained(config_dir)
        else:
            config = default_config

        # 모델 초기화
        model = GPT2LMHeadModel(config)

        # MoE 변환
        if mode != "dense":
            freq_dict = None
            current_num_experts = base_num_experts
            if mode == "ours_com":
                current_num_experts += 1  # base(16) + 1 global = 17

            if mode == "hash":
                if not os.path.exists(HASH_TABLE_PATH):
                    print(f"⚠️ Hash table not found: {HASH_TABLE_PATH}")
                    continue
                freq_dict = {'__load_from_file__': HASH_TABLE_PATH}

            model = convert_gpt2_to_moe(
                model, config,
                mode=mode,
                num_experts=current_num_experts,
                alpha=0.01,
                capacity_factor=1.25,
                freq_dict=freq_dict
            )

        if mode == "hash":
            patch_model_for_hash_moe(model)
        elif mode == "ours_com":
            patch_model_for_ours_com(model)
        elif mode != "dense":
            patch_model_basic(model)

        if mode == "ours_com":
            root_router = model.transformer.h[0].mlp.moe.shared_router
            for l in range(1, config.n_layer):
                model.transformer.h[l].mlp.moe.shared_router = root_router

        try:
            load_safetensors(model, ckpt_path, mode=mode, strict=False)
        except Exception as e:
            print(f"❌ Failed to load {mode}: {e}")
            continue

        model.to(device)

        model.lm_head.weight = model.transformer.wte.weight
        if not torch.equal(model.transformer.wte.weight.data, model.lm_head.weight.data):
            print("⚠️ Weight tying broken after GPU move, restoring...")
            model.transformer.wte.weight = model.lm_head.weight
            print("✅ Weight tying restored after GPU move!")

        (spec_list, conf_list, hist_list, route_sum, route_det,
            smooth_rows, mi_rows, calib_rows, lb_rows) = analyze_expert_specialization(
                model=model,
                dataloader=pile_valid_loader,
                device=device,
                mode=mode,
                pile_set_map=source_to_idx,
                max_batches=max_batches,
                run_specialization=run_specialization,
                run_confidence=run_confidence,
                run_routes=run_routes,
            )

        if run_specialization:
            if not spec_list:
                print(f"⚠️ {mode}: No expert selections were recorded!")
            else:
                df_specialization = pd.DataFrame(spec_list)
                df_specialization['Model'] = mode
                all_specialization_results.append(df_specialization)

        if run_confidence and len(conf_list) > 0:
            df_confidence = pd.DataFrame(conf_list)
            df_confidence['Model'] = mode
            all_confidence_results.append(df_confidence)

        if run_confidence and hist_list:
            df_histogram = pd.DataFrame(hist_list)
            df_histogram['Model'] = mode
            all_histogram_results.append(df_histogram)

        if route_sum:
            all_route_summary.extend(route_sum)
        if route_det:
            all_route_detail.extend(route_det)

        if smooth_rows:      all_smoothness.extend(smooth_rows)
        if mi_rows:          all_interlayer_mi.extend(mi_rows)
        if calib_rows:       all_calibration.extend(calib_rows)
        if lb_rows:          all_loadbalance.extend(lb_rows)

        print(f"✅ {mode} analysis complete")
        if run_specialization and spec_list:
            print(f"   📊 Specialization: {len(df_specialization)} combinations, {df_specialization['Activation_Count'].sum():,} selections")
        if run_confidence and len(conf_list) > 0:
            print(f"   🎯 Confidence: {len(df_confidence)} layers analyzed, "
                  f"avg entropy {df_confidence['Entropy_Load'].mean():.3f}")

        del model
        torch.cuda.empty_cache()

    if all_specialization_results or all_confidence_results:
        output_dir = CHECKPOINTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        if run_specialization and all_specialization_results:
            combined_specialization_df = pd.concat(all_specialization_results, ignore_index=True)

            print("\n\n📊 Calculating Global Source Distribution from analyzed batches...")
            total_activations = combined_specialization_df['Activation_Count'].sum()
            global_source_counts = combined_specialization_df.groupby('Pile_Set_Name')['Activation_Count'].sum()
            global_source_distribution = global_source_counts / total_activations
            print(f"✅ Total tokens analyzed across all models: {total_activations:,}")

            print("\n🔬 Calculating Specialization Indices...")
            combined_specialization_df['Expert_Total_Activation'] = combined_specialization_df.groupby(
                ['Model', 'Layer', 'Expert_ID']
            )['Activation_Count'].transform('sum')
            combined_specialization_df['P_Source_Given_Expert'] = (
                combined_specialization_df['Activation_Count'] / combined_specialization_df['Expert_Total_Activation']
            )
            global_df = global_source_distribution.rename("P_Source_Global").reset_index()
            combined_specialization_df = pd.merge(combined_specialization_df, global_df, on='Pile_Set_Name', how='left')

            epsilon = 1e-10
            combined_specialization_df['Specialization_Index'] = (
                combined_specialization_df['P_Source_Given_Expert'] / (combined_specialization_df['P_Source_Global'] + epsilon)
            )

            max_spec_idx = combined_specialization_df.groupby(
                ['Model', 'Layer', 'Expert_ID']
            )['Specialization_Index'].idxmax()
            max_specialization_per_expert = combined_specialization_df.loc[max_spec_idx,
                ['Model', 'Layer', 'Expert_ID', 'Pile_Set_Name', 'Specialization_Index']
            ].rename(columns={
                'Pile_Set_Name': 'Max_Specialized_Source',
                'Specialization_Index': 'Max_Specialization_Index'
            })
            combined_specialization_df = pd.merge(
                combined_specialization_df,
                max_specialization_per_expert,
                on=['Model', 'Layer', 'Expert_ID'],
                how='left'
            )

            print("\n🔬 Calculating Source Entropy for each expert...")
            expert_source_distribution = combined_specialization_df.groupby(
                ['Model', 'Layer', 'Expert_ID']
            )['P_Source_Given_Expert'].apply(list)
            expert_focus = expert_source_distribution.apply(lambda dist: shannon_entropy(dist, base=2)).reset_index()
            expert_focus = expert_focus.rename(columns={'P_Source_Given_Expert': 'Source_Entropy'})
            combined_specialization_df = pd.merge(
                combined_specialization_df, expert_focus,
                on=['Model', 'Layer', 'Expert_ID'],
                how='left'
            )

            ideal_final_columns = [
                'Model', 'Layer', 'Expert_ID', 'Pile_Set_Name',
                'Activation_Count', 'Expert_Total_Activation',
                'P_Source_Global', 'P_Source_Given_Expert', 'Specialization_Index',
                'Max_Specialized_Source', 'Max_Specialization_Index',
                'Source_Entropy',
                # ▼ 새 컬럼들
                'Avg_SelectedMass_for_Source', 'Std_SelectedMass_for_Source'
            ]
            final_columns = [col for col in ideal_final_columns if col in combined_specialization_df.columns]
            combined_specialization_df = combined_specialization_df[final_columns]

            specialization_output_path = os.path.join(output_dir, "expert_source_mapping_with_specialization.csv")
            combined_specialization_df.to_csv(specialization_output_path, index=False)

            print(f"\n{'=' * 60}")
            print(f"✅ Specialization Results saved to:")
            print(f"   {specialization_output_path}")
            print(f"{'=' * 60}")

            print(f"\n📈 Top 10 Expert-Source pairs by Specialization Index:")
            top_specialized = combined_specialization_df.nlargest(10, 'Specialization_Index')[
                ['Model', 'Layer', 'Expert_ID', 'Pile_Set_Name',
                 'Specialization_Index', 'P_Source_Given_Expert', 'P_Source_Global']
            ]
            print(top_specialized.to_string(index=False))

            print(f"\n📊 Specialization Statistics by Model:")
            for m in combined_specialization_df['Model'].unique():
                mode_df = combined_specialization_df[combined_specialization_df['Model'] == m]
                total_selections = mode_df['Activation_Count'].sum()
                avg_specialization = mode_df.groupby(
                    ['Layer', 'Expert_ID']
                )['Specialization_Index'].max().mean()
                print(f"\n  {m}:")
                print(f"    Total selections: {total_selections:,}")
                print(f"    Unique (Layer, Expert, Source) combinations: {len(mode_df)}")
                print(f"    Avg. max specialization index per expert: {avg_specialization:.3f}")

        elif run_specialization:
            print("\n⚠️ Specialization analysis failed or no data collected.")

        if run_confidence and all_confidence_results:
            combined_confidence_df = pd.concat(all_confidence_results, ignore_index=True)
            confidence_output_path = os.path.join(output_dir, "routing_confidence_analysis.csv")
            combined_confidence_df.to_csv(confidence_output_path, index=False)

            print(f"\n{'=' * 60}")
            print(f"✅ Confidence Analysis Results saved to:")
            print(f"   {confidence_output_path}")
            print(f"{'=' * 60}")

            print(f"Total records: {len(combined_confidence_df)}")

            print(f"\n🎯 Routing Confidence Statistics by Model:")
            print(f"\n{'Model':<20} {'Entropy_Load':>12} {'Avg_C':>10} {'Avg_D':>10} {'Avg_Hsel':>10} {'Avg_EEC':>10} {'Avg_S1':>10}")
            print("-" * 80)
            for m in combined_confidence_df['Model'].unique():
                mode_conf_df = combined_confidence_df[combined_confidence_df['Model'] == m]
                print(f"{m:<20} "
                      f"{mode_conf_df['Entropy_Load'].mean():>12.4f} "
                      f"{mode_conf_df['Avg_Selected_Mass'].mean():>10.4f} "
                      f"{mode_conf_df['Avg_Decisiveness'].mean():>10.4f} "
                      f"{mode_conf_df['Avg_Hsel'].mean():>10.4f} "
                      f"{mode_conf_df['Avg_EEC'].mean():>10.4f} "
                      f"{mode_conf_df['Avg_Top1Share'].mean():>10.4f}")

            print("\n💡 Interpretation Guide (Selected-Mass distributions):")
            print("  • Entropy_Load: 전문가 사용 분포의 엔트로피 (낮을수록 집중)")
            print("  • Avg_C: 평균 Selected-Mass (선택집합 확률 합)")
            print("  • Avg_D: 평균 Decisiveness (top-1과 top-2 차이/합)")
            print("  • Avg_Hsel: 평균 선택집합 내부 엔트로피")
            print("  • Avg_EEC: 평균 Effective Expert Count")
            print("  • Avg_S1: 평균 Top-1 점유율 (선택집합 내)")

        if run_confidence and all_histogram_results:
            combined_histogram_df = pd.concat(all_histogram_results, ignore_index=True)
            histogram_output_path = os.path.join(output_dir, "routing_confidence_histogram.csv")
            combined_histogram_df.to_csv(histogram_output_path, index=False)

            print(f"\n{'=' * 60}")
            print(f"✅ Confidence Histogram Results saved to:")
            print(f"   {histogram_output_path}")
            print(f"{'=' * 60}")

            print(f"Total histogram records: {len(combined_histogram_df)}")
            print("💡 Use this for plotting confidence distributions.")

        elif run_confidence:
            print("\n⚠️ Confidence histogram analysis failed or no data collected.")

        if all_route_detail:
            pd.DataFrame(all_route_detail).to_csv(
                os.path.join(output_dir, "route_sequences_detail.csv"), index=False
            )
            print(f"\n✅ Route sequence details saved to {os.path.join(output_dir, 'route_sequences_detail.csv')}")

        if all_route_summary:
            df_rs = pd.DataFrame(all_route_summary)
            df_rs.to_csv(os.path.join(output_dir, "route_sequences_summary.csv"), index=False)
            print(f"✅ Route sequence summary saved to {os.path.join(output_dir, 'route_sequences_summary.csv')}")

            disp = (df_rs.sort_values(["RouteConsistency_ModalShare"], ascending=False)
                        .groupby("Model").head(5))
            print("\n🏁 Route Consistency (top domains by modal share):")
            print(disp.to_string(index=False))

        if all_smoothness:
            pd.DataFrame(all_smoothness).to_csv(
                os.path.join(output_dir, "route_smoothness.csv"), index=False
            )
            print("✅ Saved route_smoothness.csv")

        if all_interlayer_mi:
            pd.DataFrame(all_interlayer_mi).to_csv(
                os.path.join(output_dir, "interlayer_mi.csv"), index=False
            )
            print("✅ Saved interlayer_mi.csv")

        if all_calibration:
            pd.DataFrame(all_calibration).to_csv(
                os.path.join(output_dir, "routing_calibration.csv"), index=False
            )
            print("✅ Saved routing_calibration.csv")
            print("   Columns now use S1 (Top-1 share within selected set) as the probability proxy.")

        if all_loadbalance:
            pd.DataFrame(all_loadbalance).to_csv(
                os.path.join(output_dir, "load_balance.csv"), index=False
            )
            print("✅ Saved load_balance.csv")

    else:
        print("❌ No results collected or no analysis mode selected.")


if __name__ == "__main__":
    BASE_EXPERTS_COUNT = 16
    EVAL_BATCH_SIZE = 44

    run_mapping_analysis(
        batch_size=EVAL_BATCH_SIZE,
        base_num_experts=BASE_EXPERTS_COUNT,
        max_batches=None,              # None이면 전체의 10%만 사용
        run_specialization=True,       # 전문가-소스 매핑 분석
        run_confidence=True,           # 라우팅 신뢰도 분석
        run_routes=True,               # 라우트 시퀀스 분석
    )