#utils.py
import os, math, random, numpy as np, torch
from safetensors.torch import save_model, safe_open, load_file, save_file
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
import torch.nn.functional as F
import re
from config import get_hash_table_path

def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"

@torch.no_grad()
def load_safetensors(model, path: str, *, mode: str = "switch", strict: bool = True) -> None:
    device_ctx = "cpu"
    sd = model.state_dict()
    mk = set(sd.keys())
    loaded = []
    loaded_set = set()

    with safe_open(path, framework="pt", device=device_ctx) as f:
        ck = set(f.keys())

        for k in ck:
            if k in mk:
                try:
                    sd[k].copy_(f.get_tensor(k))
                    if k not in loaded_set:
                        loaded.append(k)
                        loaded_set.add(k)
                except Exception:
                    pass

        if mode in ("ours_com", "ours_refine"):
            sr_src_layer = None
            sr_pat = re.compile(r"^transformer\.h\.(\d+)\.mlp\.moe\.shared_router\.gru\.")
            for k in ck:
                m = sr_pat.match(k)
                if m:
                    sr_src_layer = int(m.group(1))
                    break

            if sr_src_layer is not None:
                sr_src_prefix = f"transformer.h.{sr_src_layer}.mlp.moe.shared_router.gru."
                sr_src_keys = [k for k in ck if k.startswith(sr_src_prefix)]

                h_pat = re.compile(r"^transformer\.h\.(\d+)\.")
                num_layers = 0
                for mk_ in mk:
                    m = h_pat.match(mk_)
                    if m:
                        num_layers = max(num_layers, int(m.group(1)) + 1)

                copied = 0
                for tgt in range(num_layers):
                    if tgt == sr_src_layer:
                        continue
                    for src_k in sr_src_keys:
                        tgt_k = src_k.replace(f".h.{sr_src_layer}.", f".h.{tgt}.")
                        if tgt_k in mk:
                            try:
                                sd[tgt_k].copy_(f.get_tensor(src_k))
                                if tgt_k not in loaded_set:
                                    loaded.append(tgt_k)
                                    loaded_set.add(tgt_k)
                                copied += 1
                            except Exception:
                                pass
                if _is_rank0():
                    print(f"ours_com/refine: shared_router params replicated to all layers (copied={copied})")

            ge_src_layer = None
            ge_pat = re.compile(r"^transformer\.h\.(\d+)\.mlp\.moe\.global_experts\.\d+\.(w1|w2)\.weight$")
            for k in ck:
                m = ge_pat.match(k)
                if m:
                    ge_src_layer = int(m.group(1))
                    break

            if ge_src_layer is not None:
                ge_src_prefix = f"transformer.h.{ge_src_layer}.mlp.moe.global_experts."
                ge_src_keys = [k for k in ck if k.startswith(ge_src_prefix)]
                h_pat = re.compile(r"^transformer\.h\.(\d+)\.")
                num_layers = 0
                for mk_ in mk:
                    m = h_pat.match(mk_)
                    if m:
                        num_layers = max(num_layers, int(m.group(1)) + 1)

                copied = 0
                for tgt in range(num_layers):
                    if tgt == ge_src_layer:
                        continue
                    for src_k in ge_src_keys:
                        tgt_k = src_k.replace(f".h.{ge_src_layer}.", f".h.{tgt}.")
                        if tgt_k in mk:
                            try:
                                sd[tgt_k].copy_(f.get_tensor(src_k))
                                if tgt_k not in loaded_set:
                                    loaded.append(tgt_k)
                                    loaded_set.add(tgt_k)
                                copied += 1
                            except Exception:
                                pass
                if _is_rank0():
                    print(f"ours_com/refine: global_experts params replicated to all layers (copied={copied})")

        if "transformer.wte.weight" in mk and "transformer.wte.weight" not in loaded_set and "lm_head.weight" in ck:
            if _is_rank0():
                print("🔗 Tying: lm_head.weight -> transformer.wte.weight")
            try:
                sd["transformer.wte.weight"].copy_(f.get_tensor("lm_head.weight"))
                loaded.append("transformer.wte.weight")
                loaded_set.add("transformer.wte.weight")
            except Exception:
                pass

    missing = [k for k in mk if k not in loaded_set]
    extra   = [k for k in ck if k not in mk]

    if _is_rank0():
        total_model = len(mk)
        total_ckpt  = len(ck)
        unique_loaded = len(loaded_set)
        copy_ops = len(loaded)
        print(f"🔹 Tensors: model={total_model}, checkpoint={total_ckpt}")
        print(f"📦 Loaded {unique_loaded}/{total_model} tensors from checkpoint "
              f"({copy_ops} copy ops incl. shared copies): {os.path.basename(path)}")
        if missing and not strict:
            print(f"… Missing {len(missing)} keys (kept random init)")
        if extra and strict:
            print(f"… Unexpected {len(extra)} keys in checkpoint (strict=True)")

CURRENT_INPUT_IDS = None
def set_current_input_ids(x):
    global CURRENT_INPUT_IDS
    CURRENT_INPUT_IDS = x
def get_current_input_ids():
    return CURRENT_INPUT_IDS

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
def get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    if _is_rank0():
        print(f"🗓️ Scheduler: cosine (warmup_ratio={warmup_ratio:.3f}, warmup_steps={warmup_steps}, total_steps={total_steps})")
    return scheduler

def enable_sdp_backends():
    import torch, os
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    os.environ.setdefault("PYTORCH_CUDA_SDPA_ALLOW_FLASH_ATTENTION", "1")

def prefer_flash_attention(model):
    import warnings
    try:
        model.config.attn_implementation = "flash_attention_2"
        return "flash_attention_2"
    except Exception:
        try:
            model.config._attn_implementation = "flash_attention_2"
            return "flash_attention_2"
        except Exception:
            pass

    try:
        model.config.attn_implementation = "sdpa"
        return "sdpa"
    except Exception:
        try:
            model.config._attn_implementation = "sdpa"
            return "sdpa"
        except Exception:
            warnings.warn("Could not set attn_implementation; model may fall back to eager attention.")
            return "unknown"

def print_attn_stack_status(model, tag=""):
    import torch
    p = torch.cuda.get_device_properties(0)
    try:
        from transformers.utils import is_flash_attn_2_available
        fa2_avail = is_flash_attn_2_available()
    except Exception:
        fa2_avail = "unknown"
    print(f"[ATTN {tag}] device={p.name} sm={p.major}{p.minor}  torch={torch.__version__}  cuda={torch.version.cuda}")
    print(f"[ATTN {tag}] flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_efficient_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math_sdp={torch.backends.cuda.math_sdp_enabled()}  fa2_avail={fa2_avail}")
    impl = getattr(getattr(model, 'config', object()), 'attn_implementation',
                   getattr(getattr(model, 'config', object()), '_attn_implementation', 'unset'))
    print(f"[ATTN {tag}] model.config.attn_implementation={impl}")

@torch.no_grad()
def save_checkpoint(model, optimizer, scheduler, step, best_loss, total_steps, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, filename)

    state_dict = model.state_dict()

    cloned_sd = {}
    for k, v in state_dict.items():
        if (
            "global_experts" in k
            or "shared_router" in k
        ):
            cloned_sd[k] = v.clone().detach()
        else:
            cloned_sd[k] = v

    if "lm_head.weight" in cloned_sd and "transformer.wte.weight" in cloned_sd:
        cloned_sd["lm_head.weight"] = cloned_sd["lm_head.weight"].clone().detach()

    ckpt = {
        "model": cloned_sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "total_steps": total_steps,
    }

    from safetensors.torch import save_file
    save_file(ckpt["model"], model_path)

    if _is_rank0():
        print(f"[save_checkpoint] Saved to: {model_path}")

def print_model_info(model, config, mode, num_experts,
                     batch_size=None, grad_accum_steps=None, effective_batch=None):
    print("===== Model / MoE Configuration =====")
    print(f"Backbone        : GPT-2 ({config.n_layer} layers)")
    print(f"Mode            : {mode}")
    if mode == "dense":
        print(f"Num Experts     : N/A (Not an MoE model)")
    else:
        print(f"Num Experts     : {num_experts}")
    print(f"Hidden Dim      : {config.n_embd}")
    print(f"FFN Dim        : {config.n_embd * 4}")
    print(f"Attention Heads: {config.n_head}")
    print(f"Vocab Size     : {config.vocab_size}")
    print("-------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params   : {total_params/1e6:.2f}M")
    print(f"Trainable      : {trainable_params/1e6:.2f}M")
    from modeling import GPT2LayerMoE
    moe_layers = [m for m in model.modules() if isinstance(m, GPT2LayerMoE)]
    print(f"MoE Layers     : {len(moe_layers)} (replaced FFN layers)")
    for i, layer in enumerate(moe_layers[:4]):
        moe = layer.moe
        if moe.mode == "ours_com":
            total_experts = 1 + len(moe.global_experts)
            memory_info = " (shared GRU across layers, depth-wise state)"
            print(f"  - MoELayer {i}: mode={moe.mode}, experts={total_experts} "
                  f"(local=1, global={len(moe.global_experts)}){memory_info}")
        else:
            print(f"  - MoELayer {i}: mode={moe.mode}, experts={moe.num_experts}, "
                  f"shared_expert={'Yes' if moe.shared_expert is not None else 'No'}, "
                  f"global_experts={'Yes' if moe.global_experts is not None else 'No'}")
    if batch_size is not None and grad_accum_steps is not None and effective_batch is not None:
        print(f"Batch per step : {batch_size}, Accum steps = {grad_accum_steps}, "
              f"Effective batch = {effective_batch}")
    print("=====================================")

def ensure_flash_attn():
    import importlib.util, subprocess, sys, platform, torch

    spec = importlib.util.find_spec("flash_attn")
    if spec is not None:
        print("✅ FlashAttention already installed.")
        return

    is_windows = platform.system().lower() == "windows"

    if not is_windows:
        print("🔹 Linux detected — installing FlashAttention via pip (upstream).")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "flash-attn", "--no-build-isolation"
            ])
            print("✅ FlashAttention installed (Linux).")
        except Exception as e:
            print("⚠️ FlashAttention install failed (Linux):", e)
        return

    print("🔹 Windows detected — using unofficial flash-attn wheels")

    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda.replace(".", "")
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    wheel_filename = f"flash_attn-2.8.3+cu{cuda_version}torch{torch_version}cxx11abiFALSE-{py_ver}-{py_ver}-win_amd64.whl"

    wheel_url = (
        "https://github.com/kingbri1/flash-attention/releases/download/"
        f"v2.8.3/{wheel_filename}"
    )

    print("📦 Attempting to install:")
    print("   Wheel :", wheel_filename)
    print("   URL   :", wheel_url)

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", wheel_url
        ])
        print("✅ FlashAttention installed (Windows unofficial wheel).")
    except Exception as e:
        print("⚠️ FlashAttention install failed on Windows:", e)
        print("➡ Falling back to SDPA / Eager attention.")

def get_default_optimizer(model):
    params = (p for p in model.parameters() if p.requires_grad)
    try:
        opt = optim.AdamW(
            params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        if _is_rank0():
            print("🧪 Optimizer: AdamW (fused=True)")
        return opt
    except TypeError:
        opt = optim.AdamW(
            params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
        )
        if _is_rank0():
            print("🧪 Optimizer: AdamW (fused unsupported → fallback)")
        return opt
    
def chunked_cross_entropy(logits, labels, ignore_index=-100, chunk_tokens=8192):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, Tm1, V = shift_logits.shape
    bt = B * Tm1

    flat_logits = shift_logits.view(bt, V)
    flat_labels = shift_labels.view(bt)

    total = 0.0
    valid = 0
    for start in range(0, bt, chunk_tokens):
        end = min(start + chunk_tokens, bt)
        loss = F.cross_entropy(
            flat_logits[start:end],
            flat_labels[start:end],
            ignore_index=ignore_index,
            reduction="sum",
        )
        total += loss
        if ignore_index is not None:
            valid += (flat_labels[start:end] != ignore_index).sum().item()
        else:
            valid += (end - start)

    denom = max(valid, 1)
    return total / denom

def build_model_for_mode(mode: str, num_experts: int = 16, config=None):
    from transformers import GPT2Config, GPT2LMHeadModel
    from modeling import convert_gpt2_to_moe
    from patches import (
        patch_model_basic,
        patch_model_for_hash_moe,
        patch_model_for_ours_refine,
        patch_model_for_ours_com,
    )
    
    assert mode in {"dense", "switch", "gshard", "hash", "ours_refine", "ours_com"}, \
        f"Unsupported mode: {mode}"
    
    if config is None:
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1024,
            n_layer=8,
            n_head=8
        )
    
    model = GPT2LMHeadModel(config)
    
    if mode == "dense":
        patch_model_basic(model)
    else:
        eff_num_experts = num_experts if mode != "ours_refine" else (num_experts + 1)
        
        freq_dict = None
        if mode == "hash":
            v_size = getattr(config, "vocab_size", 50257)
            hash_path = get_hash_table_path(v_size)
            freq_dict = {'__load_from_file__': hash_path}

        model = convert_gpt2_to_moe(
            model, 
            config, 
            mode=mode, 
            num_experts=eff_num_experts, 
            alpha=0.01, 
            freq_dict=freq_dict
        )
        
        if mode == "hash":
            patch_model_for_hash_moe(model)
        elif mode in ("ours_refine", "ours_com"):
            if mode == "ours_com":
                patch_model_for_ours_com(model)
            else:
                patch_model_for_ours_refine(model)
        else:
            patch_model_basic(model)
    
    return model

def find_checkpoint_path(mode: str, checkpoints_dir: str) -> str:
    d = os.path.join(checkpoints_dir, f"{mode}_exp1")
    if not os.path.isdir(d):
        return None
    
    for name in ["best_checkpoint.safetensors", "last_checkpoint.safetensors",
                 "best_checkpoint.safetensors.safetensors", "last_checkpoint.safetensors.safetensors"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def load_checkpoint_if_exists(model, mode: str, checkpoints_dir: str, strict: bool = False):
    ckpt_path = find_checkpoint_path(mode, checkpoints_dir)
    if ckpt_path:
        load_safetensors(model, ckpt_path, mode=mode, strict=strict)
        if _is_rank0():
            print(f"✅ Loaded checkpoint: {ckpt_path}")
        return True
    else:
        if _is_rank0():
            print(f"⚠️ No checkpoint found for mode={mode}; using random initialization.")
        return False