# data.py — 데이터셋 로드 유틸
import os
from datasets import load_dataset

def load_or_prepare_pile(
        cache_path=None,
        raw_cache=None):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    print(f"🔹 Loading Geonwoohong/pile-uncopyrighted-6b-tokenized-gpt2 (cache_dir={cache_dir})")
    ds = load_dataset(
        "Geonwoohong/pile-uncopyrighted-6b-tokenized-gpt2",
        cache_dir=cache_dir
    )
    return ds["train"], ds["validation"]