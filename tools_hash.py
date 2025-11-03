import os, torch
from tqdm import tqdm
from collections import Counter
from data import load_or_prepare_pile
from modeling import HashRouter
from config import HASH_TABLE_PATH
import torch.nn.functional as F
from contextlib import nullcontext
import math

@torch.no_grad()
def evaluate(model, dataloader, device, show_bar=False, desc="eval", max_batches=None):
    """
    Returns: (avg_loss, perplexity)
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    iterator = dataloader
    if show_bar:
        iterator = tqdm(dataloader, desc=desc, leave=False)
    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        # shift for next-token prediction
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B,T,V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous().float()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.float)

        # flatten
        vocab = shift_logits.size(-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none"
        ).view_as(shift_labels)

        # mask paddings
        loss = (loss * shift_mask)
        n_tokens = torch.clamp_min(shift_mask.sum(), 1).item()
        nll_sum = loss.sum().item()

        total_nll += nll_sum
        total_tokens += int(n_tokens)

    if total_tokens == 0:
        return float("nan"), float("nan")
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return avg_nll, ppl

def create_global_hash_table(num_experts):
    if os.path.exists(HASH_TABLE_PATH):
        print(f"✅ Global hash table already exists. Skipping creation.")
        return

    print("--- 🌍 Starting Optimized Hash Table Creation (Full Dataset) ---")
    train_dataset, _ = load_or_prepare_pile()
    sample_dataset = train_dataset
    print(f"📊 Using the FULL train dataset of {len(sample_dataset):,} documents...")

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
        vocab_size=50257, num_experts=num_experts, method="balanced",
        freq_dict=freq_dict_data, device='cpu'
    )
    os.makedirs(os.path.dirname(HASH_TABLE_PATH), exist_ok=True)
    torch.save(temp_router.table_tensor.cpu(), HASH_TABLE_PATH)
    print(f"✅ Global hash table saved successfully to {HASH_TABLE_PATH}")
