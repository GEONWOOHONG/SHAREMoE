#tools_hash.py
import os, torch
from tqdm import tqdm
from collections import Counter
from modeling import HashRouter
from data import load_or_prepare_pile, load_or_prepare_mt
from config import HASH_TABLE_PATH, get_hash_table_path

def create_global_hash_table(num_experts, vocab_size=50257, save_path=None, mt=False):
    if save_path is None:
        save_path = get_hash_table_path(vocab_size)

    if os.path.exists(save_path):
        print(f"✅ Global hash table already exists at {save_path}. Skipping creation.")
        return save_path

    print("--- 🌍 Starting Optimized Hash Table Creation (Full Dataset) ---")
    train_dataset, _, _ = (load_or_prepare_mt() if mt else load_or_prepare_pile())
    sample_dataset = train_dataset
    print(f"📊 Using the FULL train dataset of {len(sample_dataset):,} documents.")

    def count_tokens_in_batch(batch):
        tokens_list = []
        counts_list = []
        for ids in batch["input_ids"]:
            c = Counter(ids)
            tokens_list.append(list(c.keys()))
            counts_list.append(list(c.values()))
        return {"tokens": tokens_list, "counts": counts_list}

    num_procs = os.cpu_count() // 2
    print(f"🔄 Processing batches in parallel using {num_procs} processes...")
    result_counters = sample_dataset.map(
        count_tokens_in_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_procs,
        remove_columns=sample_dataset.column_names
    )

    print("🔄 Merging results from all processes...")
    total_counter = Counter()
    iterable_results = result_counters.to_iterable_dataset()

    for item in tqdm(iterable_results, desc="Merging Counters", total=len(result_counters)):
        if isinstance(item['tokens'][0], list):
            tokens = item['tokens'][0]
            counts = item['counts'][0]
        else:
            tokens = item['tokens']
            counts = item['counts']
        if tokens and isinstance(tokens, list):
            batch_dict = dict(zip(tokens, counts))
            total_counter.update(batch_dict)

    freq_dict_data = total_counter
    print("🛠️ Creating balanced assignment table...")
    temp_router = HashRouter(
        vocab_size=vocab_size,
        num_experts=num_experts, method="balanced",
        freq_dict=freq_dict_data, device='cpu'
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(temp_router.table_tensor.cpu(), save_path)
    print(f"✅ Global hash table saved successfully to {save_path}")
    return save_path