#run.py
import os, subprocess, shutil, contextlib
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "warning")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/hf_cache/datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import argparse
from train import train_moe
from utils import set_seed
from analysis_expert_mapping import run_mapping_analysis

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

    ev = sub.add_parser("eval")
    ev.add_argument("--mode", default="switch")
    ev.add_argument("--num_experts", type=int, default=16)
    ev.add_argument("--batch_size", type=int, default=44)
    ev.add_argument("--seq_len", type=int, default=1024)
    
    an = sub.add_parser("analysis")
    an.add_argument("--num_experts", type=int, default=16)
    an.add_argument("--batch_size", type=int, default=44)
    an.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to analyze (None for 10% of total)")
    an.add_argument("--debug", action="store_true", help="Debug mode: use only 0.1% of validation data for quick testing")

    args = ap.parse_args()

    if args.cmd == "analysis":
        set_seed(42)
        debug_max_batches = args.max_batches
        if args.debug and args.max_batches is None:
            debug_max_batches = "debug"
        
        run_mapping_analysis(
            batch_size=args.batch_size,
            base_num_experts=args.num_experts,
            max_batches=debug_max_batches,
            run_specialization=True,
            run_confidence=True,
            run_routes=True,
        )
    elif args.cmd == "train":
        set_seed(42)
        train_moe(
            mode=args.mode,
            num_experts=args.num_experts,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            grad_accum=args.grad_accum,
            continue_training=args.continue_training,
        )
    elif args.cmd == "eval":
        from test import run_all_tests
        run_all_tests(batch_size=args.batch_size, base_num_experts=args.num_experts)

if __name__ == "__main__":
    main()

#python run.py train --mode stablemoe --num_experts 16 --batch_size 44 --seq_len 1024 --grad_accum 1
#torchrun --nproc_per_node=4 --master_port=29600 run.py train --mode ours_refine --num_experts 16 --batch_size 44 --seq_len 1024 --grad_accum 1
#python run.py eval --batch_size 44 --num_experts 16

#apt update && apt install -y nano zip unzip && pip install transformers datasets tensorboard pandas tqdm scipy tiktoken safetensors huggingface_hub hf_transfer calflops
#tensorboard --logdir=/workspace/runs --host=0.0.0.0 --port=6006

#wget https://github.com/schollz/croc/releases/download/v10.2.5/croc_v10.2.5_Linux-64bit.tar.gz
#tar xzf croc_v10.2.5_Linux-64bit.tar.gz
#mv croc /usr/local/bin/
#croc --version