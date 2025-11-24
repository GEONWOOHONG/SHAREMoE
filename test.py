#test.py
import os
from config import HF_HOME, HF_DATASETS_CACHE
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block as _GPT2Block
_ORIG_GPT2BLOCK_FORWARD = _GPT2Block.forward
import time, numpy as np, torch

from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
import pandas as pd
from contextlib import nullcontext

from data import load_or_prepare_pile, load_pile_test, worker_init_fn, get_dataloader_generator
from modeling import convert_gpt2_to_moe
from patches import (
    patch_model_basic,
    patch_model_for_hash_moe,
    patch_model_for_ours_com,
    patch_model_for_stablemoe,
    patch_model_for_ours_refine,
)
from train import evaluate as eval_ppl_only, compute_moe_stats
from utils import ensure_flash_attn, set_seed, load_safetensors

from config import CHECKPOINTS_DIR, get_hash_table_path
from tools_hash import create_global_hash_table

def load_or_prepare_wt103():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    return load_dataset("Geonwoohong/wikitext-103-raw-v1-test-tokenized-gpt2", cache_dir=cache_dir)

def load_or_prepare_cc100():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    return load_dataset("Geonwoohong/cc100-en-test-tokenized-gpt2", cache_dir=cache_dir)

def load_or_prepare_owt1m():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    return load_dataset("Geonwoohong/openwebtext-test-tokenized-gpt2", cache_dir=cache_dir)

@torch.no_grad()
def measure_prefill_throughput(model, batch, dtype=torch.bfloat16, warmup=5, iters=20):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attn = batch["attention_mask"].to(device, non_blocking=True)
    B, T = input_ids.shape
    total_tokens = B * T
    ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    model.eval()
    for _ in range(warmup):
        with ctx:
            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        if device.type == "cuda": torch.cuda.synchronize()
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        if device.type == "cuda": start.record()
        t0 = time.perf_counter()
        with ctx:
            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        if device.type == "cuda":
            end.record(); torch.cuda.synchronize()
            dt_ms = start.elapsed_time(end)
        else:
            dt_ms = (time.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
    avg_ms = sum(times) / len(times)
    toks_per_s = total_tokens / (avg_ms / 1000.0)
    return {
        "prefill_tokens_per_s": float(toks_per_s),
        "prefill_ms_per_iter": float(avg_ms),
        "Prefill_B": int(B),
        "Prefill_T": int(T)
    }

@torch.no_grad()
def measure_decode_throughput(model, prompt_ids, gen_len=50, dtype=torch.bfloat16, warmup=20):
    device = next(model.parameters()).device
    B, T0 = prompt_ids.shape
    ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    model.eval()
    with ctx:
        out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    x = prompt_ids[:, -1:]
    for _ in range(warmup):
        with ctx:
            out = model(input_ids=x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        x = out.logits.argmax(-1)
    if device.type == "cuda": torch.cuda.synchronize()
    step_times = []
    if device.type == "cuda":
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
    t0 = time.perf_counter()
    with ctx:
        out = model(input_ids=x, past_key_values=past, use_cache=True)
    if device.type == "cuda":
        e.record(); torch.cuda.synchronize()
        first_ms = s.elapsed_time(e)
    else:
        first_ms = (time.perf_counter() - t0) * 1000.0
    ttft_ms = float(first_ms)
    past = out.past_key_values
    x = out.logits.argmax(-1)
    for _ in range(max(0, gen_len - 1)):
        if device.type == "cuda":
            s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        t0 = time.perf_counter()
        with ctx:
            out = model(input_ids=x, past_key_values=past, use_cache=True)
        if device.type == "cuda":
            e.record(); torch.cuda.synchronize()
            dt = s.elapsed_time(e)
        else:
            dt = (time.perf_counter() - t0) * 1000.0
        step_times.append(float(dt))
        past = out.past_key_values
        x = out.logits.argmax(-1)
    if step_times:
        avg_ms_per_tok = sum(step_times) / len(step_times)
    else:
        avg_ms_per_tok = ttft_ms
    toks_per_s = 1000.0 / avg_ms_per_tok
    return {
        "decode_tokens_per_s": float(toks_per_s),
        "decode_ms_per_token": float(avg_ms_per_tok),
        "decode_TTFT_ms": float(ttft_ms),
        "Decode_B": int(B),
        "Decode_T0": int(T0),
        "Decode_gen": int(gen_len)
    }

def run_all_tests(batch_size=44, base_num_experts=16,
                  ablate_local: bool = False,
                  ablate_global: bool = False):
    set_seed(42)
    if torch.cuda.is_available():
        ensure_flash_attn()

    train_ds, pile_valid = load_or_prepare_pile(verbose=True)
    
    pile_test = load_pile_test(verbose=True)
    pile_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    small_size = max(1, int(len(pile_test) * 0.01))
    pile_eval = pile_test.select(range(small_size))
    pile_label_name = "Pile Test Loss"

    wt = load_or_prepare_wt103()
    wt_test = wt["test"]
    wt_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    cc100 = load_or_prepare_cc100()
    cc100_test = cc100["test"]
    cc100_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    owt = load_or_prepare_owt1m()
    owt_test = owt["test"]
    owt_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    num_workers = min(8, max(2, ((os.cpu_count() or 8)//2)))

    wt_loader = DataLoader(
        wt_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=False,
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )
    cc100_loader = DataLoader(
        cc100_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=False,
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )
    owt_loader = DataLoader(
        owt_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=False,
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )
    pile_loader = DataLoader(
        pile_eval, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=False,
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )

    candidate_modes = ["dense","switch","gshard","hash","stablemoe","xmoe","ours_com","ours_refine","hypermoe"]

    def pick_ckpt(mode):
        d = os.path.join(CHECKPOINTS_DIR, f"{mode}_exp1")
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
    available = {m: p for m, p in available.items() if p}
    if not available:
        print("No checkpoints available for evaluation. Please run training first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for mode, ckpt_path in available.items():
        inp = msk = batch = None
        it = None
        model = None
        print(f"\n=== Evaluating [{mode}] ===")
        per = {"Checkpoint": ckpt_path}
        try:
            _GPT2Block.forward = _ORIG_GPT2BLOCK_FORWARD

            cfg_dir = os.path.dirname(ckpt_path)
            cfg_path = os.path.join(cfg_dir, "config.json")
            if os.path.exists(cfg_path):
                config = GPT2Config.from_pretrained(cfg_dir)
            else:
                config = GPT2Config(
                    vocab_size=50257, n_positions=1024, n_ctx=1024,
                    n_embd=1024, n_layer=8, n_head=8
                )
                
            cfg_ablate_local = bool(getattr(config, "ablate_local", False))
            cfg_ablate_global = bool(getattr(config, "ablate_global", False))
            
            model = GPT2LMHeadModel(config)

            if mode != "dense":
                if mode in ("ours_com", "ours_refine"):
                    eff_num_experts = base_num_experts + 1
                elif mode == "xmoe":
                    eff_num_experts = base_num_experts * 4
                else:
                    eff_num_experts = base_num_experts

                vsz = int(getattr(config, "vocab_size", 50257))
                hash_path = get_hash_table_path(vsz)
                if mode == "hash" and not os.path.exists(hash_path):
                    print(f"⚠️  Hash table not found at {hash_path}. Auto-building now for eval.")
                    create_global_hash_table(base_num_experts, vocab_size=vsz, save_path=hash_path, mt=(vsz==32000))

                extra = {}
                if mode == "hash":
                    extra["freq_dict"] = {"__load_from_file__": hash_path}
                if mode == "stablemoe":
                    extra.update(dict(stable_routing_dim=50, stable_balance_alpha=0.3))

                use_ablate_local = cfg_ablate_local or (ablate_local and mode == "ours_refine")
                use_ablate_global = cfg_ablate_global or (ablate_global and mode == "ours_refine")

                if use_ablate_local and use_ablate_global:
                    raise ValueError("Both local and global ablation requested for ours_refine.")

                model = convert_gpt2_to_moe(
                    model, config, mode=mode,
                    num_experts=eff_num_experts, alpha=0.01,
                    ablate_local=use_ablate_local,
                    ablate_global=use_ablate_global,
                    **extra,
                )

            if mode == "hash":
                patch_model_for_hash_moe(model)
            elif mode == "ours_com":
                patch_model_for_ours_com(model)
            elif mode == "ours_refine":
                patch_model_for_ours_refine(model)
            elif mode == "stablemoe":
                patch_model_for_stablemoe(model)
            elif mode != "dense":
                patch_model_basic(model)

            strict_flag = (mode not in ("ours_com", "ours_refine"))
            load_safetensors(model, ckpt_path, mode=mode, strict=strict_flag)

            model.to(device)
            model.lm_head.weight = model.transformer.wte.weight
            total_params = sum(p.numel() for p in model.parameters())
            per["Total Params"] = total_params

            sample_batch = next(iter(pile_loader))
            pref = measure_prefill_throughput(model, sample_batch, dtype=torch.bfloat16, warmup=5, iters=20)
            per.update({
                "Prefill tokens/s": pref["prefill_tokens_per_s"],
                "Prefill ms/iter": pref["prefill_ms_per_iter"],
                "Prefill_B": pref["Prefill_B"],
                "Prefill_T": pref["Prefill_T"],
            })

            prompt_ids = sample_batch["input_ids"][:, :128].to(device, non_blocking=True)
            dec = measure_decode_throughput(model, prompt_ids, gen_len=50, dtype=torch.bfloat16, warmup=20)
            per.update({
                "Decode tokens/s": dec["decode_tokens_per_s"],
                "Decode ms/token": dec["decode_ms_per_token"],
                "Decode TTFT (ms)": dec["decode_TTFT_ms"],
                "Decode_B": dec["Decode_B"],
                "Decode_T0": dec["Decode_T0"],
                "Decode_gen": dec["Decode_gen"],
            })

            # Perplexity들 계산
            wt_loss, _    = eval_ppl_only(model, wt_loader,    device, show_bar=True, desc=f"WT103 {mode}")
            cc100_loss, _ = eval_ppl_only(model, cc100_loader, device, show_bar=True, desc=f"CC100 {mode}")
            owt_loss, _   = eval_ppl_only(model, owt_loader,   device, show_bar=True, desc=f"OWT1M {mode}")
            pile_loss, _  = eval_ppl_only(model, pile_loader,  device, show_bar=True, desc=f"Pile {mode}")

            stats = compute_moe_stats(model, config, mode)
            per.update({
                "WikiText-103 Loss": wt_loss,
                "CC100 Loss": cc100_loss,
                "OpenWebText Loss": owt_loss,
                pile_label_name: pile_loss,
                "Expert Balance (entropy)": stats.get("balance", 0.0),
            })

            print(
                f"{mode}: WT103-Loss={wt_loss:.4f}, CC100-Loss={cc100_loss:.4f}, "
                f"OWT1M-Loss={owt_loss:.4f}, {pile_label_name}={pile_loss:.4f}, "
                f"Balance={per['Expert Balance (entropy)']:.4f}, "
                f"Prefill tok/s={per['Prefill tokens/s']:.1f}, Decode tok/s={per['Decode tokens/s']:.1f}, "
                f"Decode ms/tok={per['Decode ms/token']:.2f}, TTFT ms={per['Decode TTFT (ms)']:.1f}"
            )

            results[mode] = per

        finally:
            if torch.cuda.is_available():
                try: torch.cuda.synchronize()
                except Exception: pass
            inp = None; msk = None; batch = None; it = None
            if model is not None:
                try: model.to('cpu')
                except Exception: pass
                del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    if results:
        df = pd.DataFrame(results).T
        out = os.path.join(CHECKPOINTS_DIR, "test_results_wt103_cc100_owt_pile.csv")
        df.to_csv(out, index=True)
        print(f"\nSaved results to: {out}")
        print(df)

if __name__ == "__main__":
    run_all_tests(batch_size=44, base_num_experts=16)