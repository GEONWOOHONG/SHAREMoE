#data.py
import os, random, numpy as np, torch
from datasets import load_dataset
from datasets.utils import logging as ds_logging

from huggingface_hub import snapshot_download
from glob import glob 
from datasets import Features, Sequence, Value

MAX_TRAIN_SHARDS = 10
PILE_REPO_ID = "Geonwoohong/pile-uncopyrighted-train-tokenized-gpt2"

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
    cache_dir_ds = os.environ.get("HF_DATASETS_CACHE", None)
    cache_dir_hub = os.environ.get("HF_HOME", None) or cache_dir_ds

    if MAX_TRAIN_SHARDS is None:
        if verbose and _is_rank0():
            print(f"🔹 Loading {PILE_REPO_ID} (cache_dir={cache_dir_ds}) [ALL shards]")
        ds = load_dataset(PILE_REPO_ID, cache_dir=cache_dir_ds)
        return ds["train"], ds["validation"]

    if verbose and _is_rank0():
        print(f"🔹 Downloading first {MAX_TRAIN_SHARDS} train shards "
              f"from {PILE_REPO_ID} (cache_dir={cache_dir_hub})")

    allow_patterns = [
        *(f"train/train.{i:05d}.parquet" for i in range(MAX_TRAIN_SHARDS)),
        "validation/*.parquet",
    ]

    repo_path = snapshot_download(
        repo_id=PILE_REPO_ID,
        repo_type="dataset",
        cache_dir=cache_dir_hub,
        allow_patterns=allow_patterns,
    )

    train_files = [
        os.path.join(repo_path, f"train/train.{i:05d}.parquet")
        for i in range(MAX_TRAIN_SHARDS)
    ]
    valid_files = sorted(glob(os.path.join(repo_path, "validation", "*.parquet")))

    if verbose and _is_rank0():
        print(f"  ├─ train shards: {len(train_files)} -> {train_files[0]} ...")
        print(f"  └─ validation files: {len(valid_files)}")

    train_ds = load_dataset(
        "parquet",
        data_files={"train": train_files},
    )["train"]

    valid_ds = load_dataset(
        "parquet",
        data_files={"validation": valid_files},
    )["validation"]

    return train_ds, valid_ds

def load_pile_test(verbose=True):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    if verbose:
        print(f"🔹 Loading Geonwoohong/pile-uncopyrighted-test-tokenized-gpt2 (cache_dir={cache_dir})")
    ds = load_dataset("Geonwoohong/pile-uncopyrighted-test-tokenized-gpt2", cache_dir=cache_dir)
    return ds["test"]

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

def make_validation_dataloader(
    batch_size: int = 32,
    seq_len: int = 1024,
    max_batches: int = None,
    sample_fraction: float = 0.10,
    dataset_name: str = "pile",
    num_workers: int = None,
    verbose: bool = True
):
    from torch.utils.data import DataLoader
    
    if dataset_name == "pile":
        _, valid = load_or_prepare_pile(verbose=verbose)
        valid.set_format(type="torch", columns=["input_ids", "attention_mask"])
    elif dataset_name == "mt":
        _, valid = load_or_prepare_mt(verbose=verbose)
        valid.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    if num_workers is None:
        num_workers = min(8, max(2, (os.cpu_count() or 8) // 2))
    
    loader = DataLoader(
        valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
        generator=get_dataloader_generator(0),
    )
    
    total_batches = len(loader)
    
    if max_batches is None:
        effective_batches = max(1, int(total_batches * sample_fraction))
        if verbose and _is_rank0():
            print(f"ℹ️ Using ~{sample_fraction*100:.0f}% of validation: {effective_batches}/{total_batches} batches")
    else:
        effective_batches = min(max_batches, total_batches)
        if verbose and _is_rank0():
            print(f"ℹ️ Using {effective_batches}/{total_batches} batches")
    
    return loader, total_batches, effective_batches