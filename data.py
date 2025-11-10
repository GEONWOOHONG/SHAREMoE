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
        prefetch_factor=2,
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