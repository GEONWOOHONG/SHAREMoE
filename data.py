#data.py
import os, random, numpy as np, torch
from datasets import load_dataset
from datasets.utils import logging as ds_logging

def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def worker_init_fn(worker_id):
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    
    worker_seed = 42 + worker_id + rank
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_generator(rank=0):
    g = torch.Generator()
    g.manual_seed(42 + rank)
    return g

def load_or_prepare_pile(cache_path=None, raw_cache=None, verbose=True):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    if (not _is_rank0()) or (not verbose):
        try:
            ds_logging.set_verbosity_error()
            ds_logging.disable_progress_bar()
        except Exception:
            pass

    if verbose and _is_rank0():
        print(f"🔹 Loading Geonwoohong/pile-uncopyrighted-train-tokenized-gpt2 (cache_dir={cache_dir})")

    ds = load_dataset(
        "Geonwoohong/pile-uncopyrighted-train-tokenized-gpt2",
        cache_dir=cache_dir
    )
    return ds["train"], ds["validation"]

def load_or_prepare_mt(cache_path=None, raw_cache=None, verbose=True):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    if (not _is_rank0()) or (not verbose):
        try:
            ds_logging.set_verbosity_error()
            ds_logging.disable_progress_bar()
        except Exception:
            pass

    if verbose and _is_rank0():
        print(f"🔹 Loading Geonwoohong/wmt21-train-tokenized-sentencepiece (cache_dir={cache_dir})")

    ds = load_dataset(
        "Geonwoohong/wmt21-train-tokenized-sentencepiece",
        cache_dir=cache_dir,
    )

    return ds["train"], ds["validation"]