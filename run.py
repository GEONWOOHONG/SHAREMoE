#run.py
import os, subprocess, shutil, contextlib
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "warning")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# workspace 경로 설정 (config.py에서 가져옴)
from config import HF_HOME, HF_DATASETS_CACHE
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import argparse
from train import train_moe
from utils import set_seed
from analysis_layers import run_analysis_A
from analysis_specialization_confidence import run_specialization_confidence

def _setenv_if_missing(k, v):
    if os.environ.get(k) in (None, ""):
        os.environ[k] = str(v)

def _has_cmd(cmd):
    return shutil.which(cmd) is not None

def _read_cmd(cmd):
    try:
        out = subprocess.run(cmd, shell=True, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True).stdout
        return out
    except Exception:
        return ""

def _detect_ib():
    if os.path.isdir("/sys/class/infiniband"):
        return True
    if _has_cmd("ibv_devinfo"):
        out = _read_cmd("ibv_devinfo -l")
        return any(line.strip() for line in out.splitlines() if not line.strip().startswith("RDMA device list"))
    return False

def _detect_gdr():
    return os.path.exists("/sys/module/nvidia_peermem") or os.path.exists("/dev/nvidia-peer-mem")

def _detect_nvlink():
    if not _has_cmd("nvidia-smi"):
        return False, ""
    topo = _read_cmd("nvidia-smi topo -m")
    has_nv = any((" NV" in line or "NVL" in line) for line in topo.splitlines())
    return has_nv, topo

def setup_nccl_env_safely(print_topology_rank0=True):
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    
    _setenv_if_missing("NCCL_DEBUG", "WARN")

    has_nvlink, topo_txt = _detect_nvlink()
    has_ib = _detect_ib()
    has_gdr = _detect_gdr()

    _setenv_if_missing("NCCL_IB_DISABLE", "0" if has_ib else "1")

    if has_ib and has_gdr:
        _setenv_if_missing("NCCL_NET_GDR_LEVEL", "2")

    rank = int(os.environ.get("RANK", "0"))
    if print_topology_rank0 and rank == 0 and topo_txt:
        print("==== NVIDIA Topology (nvidia-smi topo -m) ====")
        print(topo_txt.strip())
        print("================================================")
        print(f"[NCCL] has_nvlink={has_nvlink}, has_ib={has_ib}, has_gdr={has_gdr}")

def _maybe_set_expandable_segments():
    try:
        gnames = _read_cmd("nvidia-smi --query-gpu=name --format=csv,noheader").strip().splitlines()
        gname0 = (gnames[0] if gnames else "").upper()
        if ("A100" in gname0) or ("H100" in gname0):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            return
    except Exception:
        pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_maybe_set_expandable_segments()

setup_nccl_env_safely()

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--mode", default="switch")
    tr.add_argument("--num_experts", type=int, default=16)
    tr.add_argument("--batch_size", type=int, default=44)
    tr.add_argument("--seq_len", type=int, default=1024)
    tr.add_argument("--grad_accum", type=int, default=1)
    tr.add_argument("--continue_training", action="store_true")
    tr.add_argument("--mt", action="store_true")
    tr.add_argument("--ablate_local", action="store_true")
    tr.add_argument("--ablate_global", action="store_true")
    tr.add_argument("--ablate_logit_prog", action="store_true", help="Remove logit propagation in ours_refine")
    tr.add_argument("--ablate_global_router", action="store_true", help="Replace GRU router with standard router in ours_refine")
    tr.add_argument("--ffn_dim", type=int, default=None, help="Custom FFN dimension size")
    
    ev = sub.add_parser("eval")
    ev.add_argument("--mode", default="switch")
    ev.add_argument("--num_experts", type=int, default=16)
    ev.add_argument("--batch_size", type=int, default=44)
    ev.add_argument("--seq_len", type=int, default=1024)
    ev.add_argument("--ablate_local", action="store_true")
    ev.add_argument("--ablate_global", action="store_true")
    ev.add_argument("--ablate_logit_prog", action="store_true", help="Remove logit propagation in ours_refine")
    ev.add_argument("--ablate_global_router", action="store_true", help="Replace GRU router with standard router in ours_refine")
           
    an = sub.add_parser("analysis")
    an.add_argument("--modes", type=str, default="dense,switch,gshard,hash,ours_refine", help="Comma-separated mode list")
    an.add_argument("--num_experts", type=int, default=16)
    an.add_argument("--batch_size", type=int, default=44)
    an.add_argument("--seq_len", type=int, default=1024)
    an.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to analyze (None for 10% of total)")
    an.add_argument("--sample_fraction", type=float, default=0.10, help="Fraction of validation batches when max_batches is None")
    an.add_argument("--debug", action="store_true", help="Debug mode: use only 0.1% of validation data for quick testing")
    an.add_argument("--skip_layers", action="store_true", help="Skip layer-level analysis (LEI, CKA, etc.)")
    an.add_argument("--skip_specialization", action="store_true", help="Skip specialization & confidence analysis")
    an.add_argument("--no_flash", action="store_true", help="Disable Flash Attention")
    an.add_argument("--ablate_logit_prog", action="store_true", help="Remove logit propagation in ours_refine")
    an.add_argument("--ablate_global_router", action="store_true", help="Replace GRU router with standard router in ours_refine")

    args = ap.parse_args()

    if args.cmd == "analysis":
        set_seed(42)
        modes_list = [m.strip() for m in args.modes.split(",") if m.strip()]
        
        # Adjust sample_fraction for debug mode
        sample_frac = 0.001 if args.debug else args.sample_fraction
        max_batches = args.max_batches
        if args.debug and max_batches is None:
            max_batches = 2  # minimal batches for debug
        
        print("\n" + "="*80)
        print("🔍 ANALYSIS ORCHESTRATOR")
        print("="*80)
        print(f"Modes: {modes_list}")
        print(f"Num Experts: {args.num_experts}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Max Batches: {max_batches if max_batches else f'Auto (~{sample_frac*100:.1f}%)'}")
        print(f"Debug Mode: {args.debug}")
        print("="*80 + "\n")
        
        # 1) Layer-level analysis (LEI, CKA, Intra-redundancy)
        if not args.skip_layers:
            print("\n[1/2] Running layer-level analysis (LEI, CKA, etc.)...")
            for mode in modes_list:
                print(f"\n  → Analyzing mode: {mode}")
                try:
                    run_analysis_A(
                        mode=mode,
                        num_experts=args.num_experts,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        max_batches=max_batches,
                        use_flash_attn=not args.no_flash,
                    )
                except Exception as e:
                    print(f"⚠️ Layer analysis failed for {mode}: {e}")
        else:
            print("\n[1/2] Skipping layer-level analysis (--skip_layers)")
        
        # 2) Specialization & Confidence analysis
        if not args.skip_specialization:
            print("\n[2/2] Running specialization & confidence analysis...")
            try:
                run_specialization_confidence(
                    modes=modes_list,
                    num_experts=args.num_experts,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    max_batches=max_batches,
                    use_flash_attn=not args.no_flash,
                    sample_fraction=sample_frac,
                )
            except Exception as e:
                print(f"⚠️ Specialization & confidence analysis failed: {e}")
        else:
            print("\n[2/2] Skipping specialization & confidence analysis (--skip_specialization)")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
    elif args.cmd == "train":
        set_seed(42)
        train_moe(
            mode=args.mode,
            num_experts=args.num_experts,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            grad_accum=args.grad_accum,
            continue_training=args.continue_training,
            mt=args.mt,
            ablate_local=args.ablate_local,
            ablate_global=args.ablate_global,
            ablate_logit_prog=args.ablate_logit_prog,
            ablate_global_router=args.ablate_global_router,
            ffn_dim=args.ffn_dim,
        )
    elif args.cmd == "eval":
        from test import run_all_tests
        run_all_tests(
            batch_size=args.batch_size,
            base_num_experts=args.num_experts,
            ablate_local=args.ablate_local,
            ablate_global=args.ablate_global,
            ablate_logit_prog=args.ablate_logit_prog,
            ablate_global_router=args.ablate_global_router,
        )

if __name__ == "__main__":
    main()

#python run.py train --mode ours_refine --num_experts 4 --batch_size 4 --seq_len 1024 --grad_accum 2
#torchrun --nproc_per_node=8 --master_port=29600 run.py train --mode switch --num_experts 16 --batch_size 64 --seq_len 1024 --grad_accum 1 --ffn_dim 731
#python run.py eval --batch_size 44 --num_experts 16

# Full analysis (all modes, all metrics):
#python run.py analysis --modes switch,gshard,hash,ours_refine --batch_size 44 --num_experts 16

# Quick debug run (0.1% data, 2 batches):
#python run.py analysis --modes ours_refine --debug

# Custom sample fraction (5% of validation):
#python run.py analysis --modes switch,ours_refine --sample_fraction 0.05

# Skip specific analyses:
#python run.py analysis --skip_layers --modes ours_refine   # only specialization
#python run.py analysis --skip_specialization --modes ours_refine  # only layer

#apt update && apt install -y nano zip unzip && pip install transformers datasets tensorboard pandas tqdm scipy tiktoken safetensors huggingface_hub hf_transfer calflops
#tensorboard --logdir=.//runs --host=0.0.0.0 --port=6006

#wget https://github.com/schollz/croc/releases/download/v10.2.5/croc_v10.2.5_Linux-64bit.tar.gz
#tar xzf croc_v10.2.5_Linux-64bit.tar.gz
#mv croc /usr/local/bin/
#croc --version