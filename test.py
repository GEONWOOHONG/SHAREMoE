# test.py
import os
from config import HF_HOME, HF_DATASETS_CACHE
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block as _GPT2Block
_ORIG_GPT2BLOCK_FORWARD = _GPT2Block.forward
import math, time, re, numpy as np, torch

from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import tiktoken

from data import load_or_prepare_pile, worker_init_fn, get_dataloader_generator
from modeling import convert_gpt2_to_moe, GPT2LayerMoE
from patches import (
    patch_model_basic,
    patch_model_for_hash_moe,
    patch_model_for_ours_com,
    patch_model_for_stablemoe,
)
from tools_hash import evaluate as eval_ppl_only, compute_moe_stats
from utils import ensure_flash_attn, set_seed

CHECKPOINT_ROOT = "/workspace/checkpoints"
HASH_TABLE_PATH = "/workspace/checkpoints/hash_exp1/global_hash_router_table.pt"

enc = tiktoken.get_encoding("gpt2")

def _attn_flops_per_layer(B, T, d, H):
    return (4 * B * T * d * d) + (2 * B * T * T * d)

def _mlp_flops_dense_per_layer(B, T, d, d_ff):
    return 2 * B * T * d * d_ff

def _theoretical_k(mode):
    return {
        "dense":1, "switch":1, "hash":1,
        "gshard":2, "ours_com":2, "ours_refine":2,
        "hypermoe":1,
    }.get(mode, 1)

def _ensure_calflops():
    try:
        import calflops
        from calflops import calculate_flops
        return calculate_flops
    except Exception as e:
        print(f"[calflops] not installed/usable: {e}")
        return None

@torch.no_grad()
def measure_flops_calflops(model, input_ids, attention_mask, print_detailed=False):
    calculate_flops = _ensure_calflops()
    if calculate_flops is None:
        return None

    model.eval()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    try:
        flops, macs, params = calculate_flops(
            model=model,
            kwargs=kwargs,
            print_results=print_detailed,
            print_detailed=print_detailed,
            output_as_string=False,
        )
    except Exception as e:
        print(f"[calflops] measure failed: {e}")
        return None

    B, T = input_ids.shape[:2]
    flops_per_token = float(flops) / max(1, B*T)
    return float(flops), float(macs), int(params), flops_per_token

def _replicate_ours_com_weights(model, f, checkpoint_keys, model_keys, loaded_keys):
    sr_src_layer = None
    sr_pat = re.compile(r"transformer\.h\.(\d+)\.mlp\.moe\.shared_router\.gru\.weight_ih")
    for k in checkpoint_keys:
        m = sr_pat.match(k)
        if m:
            sr_src_layer = int(m.group(1)); break

    if sr_src_layer is not None:
        sr_src_prefix = f"transformer.h.{sr_src_layer}.mlp.moe.shared_router.gru."
        sr_src_keys = [k for k in checkpoint_keys if k.startswith(sr_src_prefix)]
        h_pat = re.compile(r"transformer\.h\.(\d+)\.")
        num_layers = 0
        for mk in model_keys:
            m = h_pat.match(mk)
            if m: num_layers = max(num_layers, int(m.group(1)) + 1)
        copied = 0
        for tgt in range(num_layers):
            if tgt == sr_src_layer: continue
            for src_k in sr_src_keys:
                tgt_k = src_k.replace(f".h.{sr_src_layer}.", f".h.{tgt}.")
                if tgt_k in model_keys:
                    try:
                        model.state_dict()[tgt_k].copy_(f.get_tensor(src_k))
                        loaded_keys.append(tgt_k); copied += 1
                    except Exception: pass
        print(f"✅ ours_com: shared_router params replicated to all layers (copied={copied})")
    ge_src_layer = None
    ge_pat = re.compile(r"transformer\.h\.(\d+)\.mlp\.moe\.global_experts\.0\.w1\.weight")
    for k in checkpoint_keys:
        m = ge_pat.match(k)
        if m:
            ge_src_layer = int(m.group(1)); break
    if ge_src_layer is not None:
        ge_src_prefix = f"transformer.h.{ge_src_layer}.mlp.moe.global_experts."
        ge_src_keys = [k for k in checkpoint_keys if k.startswith(ge_src_prefix)]
        h_pat = re.compile(r"transformer\.h\.(\d+)\.")
        num_layers = 0
        for mk in model_keys:
            m = h_pat.match(mk)
            if m: num_layers = max(num_layers, int(m.group(1)) + 1)
        copied = 0
        for tgt in range(num_layers):
            if tgt == ge_src_layer: continue
            for src_k in ge_src_keys:
                tgt_k = src_k.replace(f".h.{ge_src_layer}.", f".h.{tgt}.")
                if tgt_k in model_keys:
                    try:
                        model.state_dict()[tgt_k].copy_(f.get_tensor(src_k))
                        loaded_keys.append(tgt_k); copied += 1
                    except Exception: pass
        print(f"✅ ours_com: global_experts params replicated to all layers (copied={copied})")

def load_model_from_safetensors(model, path, strict=True, mode="switch"):
    from safetensors.torch import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        ck = set(f.keys()); mk = set(model.state_dict().keys())
        loaded = []
        for k in ck:
            if k in mk:
                model.state_dict()[k].copy_(f.get_tensor(k)); loaded.append(k)
        if mode in ("ours_com", "ours_refine"):
            _replicate_ours_com_weights(model, f, ck, mk, loaded)

        if "transformer.wte.weight" in mk and "transformer.wte.weight" not in loaded and "lm_head.weight" in ck:
            print("🔄 Tying: lm_head.weight → transformer.wte.weight")
            model.state_dict()["transformer.wte.weight"].copy_(f.get_tensor("lm_head.weight"))
            loaded.append("transformer.wte.weight")

        missing = [k for k in mk if k not in loaded]
        extra = [k for k in ck if k not in mk]
        print(f"✅ Loaded {len(loaded)}/{len(mk)} tensors from checkpoint")
        if missing and not strict:
            print(f"⚠️ Missing {len(missing)} keys (kept random init)")
        if extra and strict:
            print(f"⚠️ Unexpected {len(extra)} keys in checkpoint (strict=True)")


def load_or_prepare_wikitext():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    return load_dataset("Geonwoohong/wikitext-103-raw-v1-5m-tokenized-gpt2",
                        cache_dir=cache_dir)

def load_or_prepare_lambada():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    ds = load_dataset("Geonwoohong/lambada-openai-400k-tokenized-gpt2",
                      split="train", cache_dir=cache_dir)
    return ds.select_columns(["context_ids", "target_ids"]).to_list()

@torch.no_grad()
def evaluate_lambada_fast(model, lambada_data, device, batch_size=64):
    model.eval()
    correct, total = 0, 0
    losses = []
    for i in tqdm(range(0, len(lambada_data), batch_size), desc="LAMBADA"):
        batch = lambada_data[i:i+batch_size]
        ctxs = [b["context_ids"] for b in batch]
        tgts = [b["target_ids"]  for b in batch]
        max_ctx = max(len(c) for c in ctxs)
        max_tgt = max(len(t) for t in tgts)
        input_ids, labels, attn = [], [], []
        for c,t in zip(ctxs, tgts):
            ids = c + t
            pad = max_ctx+max_tgt - len(ids)
            input_ids.append(ids + [enc.eot_token]*pad)
            attn.append([1]*len(ids) + [0]*pad)
            labels.append([-100]*len(c) + t + [-100]*pad)
        input_ids = torch.tensor(input_ids, device=device)
        attn = torch.tensor(attn, device=device)
        labels = torch.tensor(labels, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        losses.append(out.loss.item())
        logits = out.logits
        for b_idx, t in enumerate(tgts):
            ctx_len = len(ctxs[b_idx]); tgt_len = len(t)
            pred = torch.argmax(logits[b_idx, ctx_len-1:ctx_len-1+tgt_len, :], dim=-1).tolist()
            if pred == t: correct += 1
            total += 1
    mean_loss = float(np.mean(losses)) if losses else 0.0
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    acc = correct/total if total>0 else 0.0
    return acc, ppl

@torch.no_grad()
def evaluate_lambada_loss_only(model, lambada_data, device, batch_size=64):
    model.eval()
    losses = []
    for i in tqdm(range(0, len(lambada_data), batch_size), desc="LAMBADA Loss"):
        batch = lambada_data[i:i+batch_size]
        ctxs = [b["context_ids"] for b in batch]
        tgts = [b["target_ids"]  for b in batch]
        max_ctx = max(len(c) for c in ctxs)
        max_tgt = max(len(t) for t in tgts)
        input_ids, labels, attn = [], [], []
        for c,t in zip(ctxs, tgts):
            ids = c + t
            pad = max_ctx+max_tgt - len(ids)
            input_ids.append(ids + [enc.eot_token]*pad)
            attn.append([1]*len(ids) + [0]*pad)
            labels.append([-100]*len(c) + t + [-100]*pad)
        input_ids = torch.tensor(input_ids, device=device)
        attn = torch.tensor(attn, device=device)
        labels = torch.tensor(labels, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        losses.append(out.loss.item())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss

def run_all_tests(batch_size=44, base_num_experts=16):
    set_seed(42)
    ensure_flash_attn()

    _, pile_valid = load_or_prepare_pile(verbose=True)
    pile_valid.set_format(type="torch", columns=["input_ids", "attention_mask"])

    wt = load_or_prepare_wikitext()
    wt_test = wt["test"]
    wt_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    lambada = load_or_prepare_lambada()

    num_workers = min(8, max(2, ((os.cpu_count() or 8)//2)))
    wt_loader = DataLoader(wt_test, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           prefetch_factor=2, persistent_workers=True,
                           worker_init_fn=worker_init_fn,
                           generator=get_dataloader_generator(0))
    pile_loader = DataLoader(pile_valid, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             prefetch_factor=2, persistent_workers=True,
                             worker_init_fn=worker_init_fn,
                             generator=get_dataloader_generator(0))

    candidate_modes = ["dense","switch","gshard","hash","stablemoe","xmoe","ours_com","ours_refine","hypermoe"]
    def pick_ckpt(mode):
        d = os.path.join(CHECKPOINT_ROOT, f"{mode}_exp1")
        if not os.path.isdir(d):
            return None
        candidates = [
            "best_checkpoint.safetensors",
            "last_checkpoint.safetensors",
            "best_checkpoint.safetensors.safetensors",
            "last_checkpoint.safetensors.safetensors",
            "checkpoint_best.safetensors",
            "checkpoint_last.safetensors",
        ]
        for name in candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    available = {m: pick_ckpt(m) for m in candidate_modes}
    available = {m:p for m,p in available.items() if p}
    if not available:
        print("❌ 평가할 체크포인트가 없습니다. 먼저 훈련을 실행하세요.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for mode, ckpt_path in available.items():
        print(f"\n=== Evaluating [{mode}] ===")

        per = {"Checkpoint": ckpt_path}

        _GPT2Block.forward = _ORIG_GPT2BLOCK_FORWARD

        cfg_dir = os.path.dirname(ckpt_path)
        cfg_path = os.path.join(cfg_dir, "config.json")
        if os.path.exists(cfg_path):
            config = GPT2Config.from_pretrained(cfg_dir)
        else:
            config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=1024, n_layer=8, n_head=8)

        model = GPT2LMHeadModel(config)

        if mode != "dense":
            if mode in ("ours_com", "ours_refine"):
                eff_num_experts = base_num_experts + 1
            elif mode == "xmoe":
                eff_num_experts = base_num_experts * 4
            else:
                eff_num_experts = base_num_experts

            extra = {}
            if mode == "hash":
                extra["freq_dict"] = {"__load_from_file__": HASH_TABLE_PATH}
            if mode == "stablemoe":
                extra.update(dict(stable_routing_dim=50, stable_balance_alpha=0.3))

            model = convert_gpt2_to_moe(
                model, config, mode=mode, num_experts=eff_num_experts, alpha=0.01, **extra
            )

        if mode == "hash":
            patch_model_for_hash_moe(model)
        elif mode == "ours_com":
            patch_model_for_ours_com(model)
        elif mode == "ours_refine":
            patch_model_for_ours_com(model)
        elif mode == "stablemoe":
            patch_model_for_stablemoe(model)
        elif mode != "dense":
            patch_model_basic(model)

        try:
            strict_flag = (mode not in ("ours_com", "ours_refine"))
            load_model_from_safetensors(model, ckpt_path, strict=strict_flag, mode=mode)
        except Exception as e:
            print(f"⚠️ 로드 실패 ({mode}): {e}")
            continue

        model.to(device)
        try:
            model.lm_head.weight = model.transformer.wte.weight
        except Exception:
            pass

        total_params = sum(p.numel() for p in model.parameters())
        per["Total Params"] = total_params

        sample_batches = 3
        emp_flops_tok_vals, emp_flops_total_vals = [], []
        emp_params = None

        it = iter(pile_loader)
        for _ in range(sample_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            inp = batch["input_ids"].to(device, non_blocking=True)
            msk = batch["attention_mask"].to(device, non_blocking=True)

            r = measure_flops_calflops(model, inp, msk, print_detailed=False)
            if r is None:
                break
            tot_flops, tot_macs, params, flops_tok = r
            emp_flops_tok_vals.append(flops_tok)
            emp_flops_total_vals.append(tot_flops)
            emp_params = params

        if emp_flops_tok_vals:
            per["FLOPs/token (empirical, Pile)"] = float(np.mean(emp_flops_tok_vals))
            per["FLOPs (empirical, per-batch avg)"] = float(np.mean(emp_flops_total_vals))
            per["Params (from calflops)"] = emp_params
        else:
            per["FLOPs/token (empirical, Pile)"] = float("nan")
            per["FLOPs (empirical, per-batch avg)"] = float("nan")
            per["Params (from calflops)"] = total_params  # ← 이제 안전

        T_fix, B_fix = 1024, 1
        d, H, L = config.n_embd, config.n_head, config.n_layer

        attn_fix = L * _attn_flops_per_layer(B_fix, T_fix, d, H)

        d_ff_dense = d * 4

        if mode == "xmoe":
            try:
                d_ff_exp = model.transformer.h[0].mlp.moe.experts[0].w1.out_features
            except Exception:
                d_ff_exp = d_ff_dense
            k_eff = 1.0
            mlp_fix = L * _mlp_flops_dense_per_layer(B_fix, T_fix, d, d_ff_exp) * k_eff
        else:
            mlp_fix = L * _mlp_flops_dense_per_layer(B_fix, T_fix, d, d_ff_dense) * _theoretical_k(mode)

        per["FLOPs/token (theoretical @T=1024)"] = (attn_fix + mlp_fix) / (B_fix * T_fix)

        wt_loss, _ = eval_ppl_only(model, wt_loader, device, show_bar=True, desc=f"WT103 {mode}")
        pile_loss, _ = eval_ppl_only(model, pile_loader, device, show_bar=True, desc=f"Pile {mode}")
        lamb_acc, _ = evaluate_lambada_fast(model, lambada, device, batch_size=batch_size)
        lamb_loss = evaluate_lambada_loss_only(model, lambada, device, batch_size=batch_size)

        stats = compute_moe_stats(model, config, mode)
        per.update({
            "WikiText-103 Loss": wt_loss,
            "Pile Valid Loss": pile_loss,
            "LAMBADA Accuracy": lamb_acc,
            "LAMBADA Loss": lamb_loss,
            "Expert Balance (entropy)": stats.get("balance", 0.0),
        })

        print(f"✅ {mode}: WT-Loss={wt_loss:.4f}, Pile-Loss={pile_loss:.4f}, "
              f"LAMB-Acc={lamb_acc:.3f}, LAMB-Loss={lamb_loss:.4f}, "
              f"Balance={per['Expert Balance (entropy)']:.4f}")

        results[mode] = per

    if results:
        df = pd.DataFrame(results).T
        out = os.path.join(CHECKPOINT_ROOT, "test_results_combined.csv")
        df.to_csv(out, index=True)
        print(f"\n📄 Saved results to: {out}")
        print(df)

if __name__ == "__main__":
    run_all_tests(batch_size=44, base_num_experts=16)