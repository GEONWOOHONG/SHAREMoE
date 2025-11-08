#config.py
import os

HASH_TABLE_PATH = "/workspace/checkpoints/hash_exp1/global_hash_router_table.pt"

def get_hash_table_path(vocab_size: int):
    base = "/workspace/checkpoints/hash_exp1"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"global_hash_router_table.v{vocab_size}.pt")

HF_HOME = "/workspace/hf_cache"
HF_DATASETS_CACHE = "/workspace/hf_cache/datasets"
RUNS_DIR = "/workspace/runs"

CHECKPOINTS_DIR = "/workspace/checkpoints"