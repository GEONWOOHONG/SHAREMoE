# run.py — CLI 엔트리포인트
import os, subprocess, shutil, contextlib
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "warning")

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
    # 가장 안전한 판별: 커널 디바이스 경로 or ibv_devinfo
    if os.path.isdir("/sys/class/infiniband"):
        return True
    if _has_cmd("ibv_devinfo"):
        out = _read_cmd("ibv_devinfo -l")
        # 장치 목록에 아무 라인이나 있으면 있음으로 간주
        return any(line.strip() for line in out.splitlines() if not line.strip().startswith("RDMA device list"))
    return False

def _detect_gdr():
    # GPUDirect RDMA 모듈 존재 여부 (이름이 nvidia_peermem 인 경우가 일반적)
    return os.path.exists("/sys/module/nvidia_peermem") or os.path.exists("/dev/nvidia-peer-mem")

def _detect_nvlink():
    if not _has_cmd("nvidia-smi"):
        return False, ""
    topo = _read_cmd("nvidia-smi topo -m")
    # topo 표에 'NV' (NVLink/NV#) 표기가 있으면 NVLink로 간주
    has_nv = any((" NV" in line or "NVL" in line) for line in topo.splitlines())
    return has_nv, topo

def setup_nccl_env_safely(print_topology_rank0=True):
    # PyTorch 권장값으로 교체 (NCCL_ASYNC_ERROR_HANDLING은 deprecated)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # 기존 deprecated 환경변수는 제거
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    
    # OMP 경고 제거 (프로세스당 쓰레드 1 고정; torchrun가 기본 1로 고치지만 명시 권장)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    
    _setenv_if_missing("NCCL_DEBUG", "WARN")

    # 토폴로지 감지
    has_nvlink, topo_txt = _detect_nvlink()
    has_ib = _detect_ib()
    has_gdr = _detect_gdr()

    # IB 없으면 꺼두기, 있으면 켜기
    _setenv_if_missing("NCCL_IB_DISABLE", "0" if has_ib else "1")

    # GDR은 있을 때만 레벨 지정 (없는데 켜면 실패/회피 로직 타는 클러스터가 있음)
    if has_ib and has_gdr:
        _setenv_if_missing("NCCL_NET_GDR_LEVEL", "2")  # 보수적으로 2
    # 없으면 설정 안 함(기존 값 유지)

    # P2P는 NVLink가 없더라도 일반적으로 자동; 굳이 끌 필요 없음.
    # 필요 시: os.environ.setdefault("NCCL_P2P_DISABLE", "0")

    # rank0 에서만 토폴로지 프린트 (torch.distributed 초기화 전이므로 RANK env로 판별)
    rank = int(os.environ.get("RANK", "0"))
    if print_topology_rank0 and rank == 0 and topo_txt:
        print("==== NVIDIA Topology (nvidia-smi topo -m) ====")
        print(topo_txt.strip())
        print("================================================")
        print(f"[NCCL] has_nvlink={has_nvlink}, has_ib={has_ib}, has_gdr={has_gdr}")

# 기타 캐시/메모리 관련 기본값
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/hf_cache/datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

setup_nccl_env_safely()

import argparse
from config import HASH_TABLE_PATH
from utils import set_current_input_ids, get_current_input_ids
from train import train_moe, set_seed
from tools_hash import create_global_hash_table
from analysis_expert_mapping import run_mapping_analysis

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
    
    bh = sub.add_parser("build-hash")
    bh.add_argument("--num_experts", type=int, default=16)

    an = sub.add_parser("analysis")
    an.add_argument("--num_experts", type=int, default=16)
    an.add_argument("--batch_size", type=int, default=44)
    an.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to analyze (None for 10% of total)")
    an.add_argument("--debug", action="store_true", help="Debug mode: use only 0.1% of validation data for quick testing")

    args = ap.parse_args()

    if args.cmd == "build-hash":
        create_global_hash_table(num_experts=args.num_experts)
    elif args.cmd == "analysis":
        set_seed(42)
        # Debug 모드일 때 max_batches를 매우 작게 설정
        debug_max_batches = args.max_batches
        if args.debug and args.max_batches is None:
            # Debug 모드: 전체의 0.1%만 사용하도록 특별 플래그 전달
            debug_max_batches = "debug"
        
        run_mapping_analysis(
            batch_size=args.batch_size,
            base_num_experts=args.num_experts,
            max_batches=debug_max_batches,
            run_specialization=True,
            run_confidence=True,
            run_routes=True,
        )
    elif args.cmd in {"train", "eval"}:
        set_seed(42)
        train_moe(
            mode=args.mode,
            num_experts=args.num_experts,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            grad_accum=args.grad_accum,
            continue_training=(args.cmd == "train" and args.continue_training),
        )

if __name__ == "__main__":
    main()

#python run.py train --mode stablemoe --num_experts 16 --batch_size 44 --seq_len 1024 --grad_accum 1
#torchrun --nproc_per_node=2 --master_port=29600 run.py train --mode switch --num_experts 16 --batch_size 44 --seq_len 1024 --grad_accum 1

#cd "/c/IMML Lab/runpod_a100/repo"
#git status
#git add .
#git commit -m "context"
#git push origin main

#mv /workspace/checkpoints/exp1 /workspace/checkpoints/gshard_exp1

#rm -rf /workspace/checkpoints/exp1
#rm -rf /workspace/runs

#tensorboard --logdir=/workspace/runs --host=0.0.0.0 --port=6006
#watch -n 5 nvidia-smi

#apt-get update && apt-get install -y zip unzip
#apt update && apt install -y nano
#pip install transformers datasets tensorboard pandas tqdm scipy tiktoken safetensors huggingface_hub hf_transfer

#wget https://github.com/schollz/croc/releases/download/v10.2.5/croc_v10.2.5_Linux-64bit.tar.gz
#tar xzf croc_v10.2.5_Linux-64bit.tar.gz
#mv croc /usr/local/bin/
#croc --version

#croc send --transfers 8 /workspace/checkpoints
#croc send /workspace/checkpoints/masking_experiment_8experts.csv
#cd "C:\IMML Lab"
#croc <코드값>

#bias=False + kaiming_uniform_ 초기화를 통해서 dead expert 방지해야함
#top-2 이상일 때 선택 확률을 normalize 해야함

#Fused Kernel / Flash Attention
#Compile 모드

#export HF_HOME=/workspace/hf_cache
#export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
