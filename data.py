# data.py — 데이터셋 로드 유틸
import os, random, numpy as np, torch
from datasets import load_dataset

def worker_init_fn(worker_id):
    """워커 시드 고정을 위한 초기화 함수"""
    # 분산 환경에서 rank 정보 가져오기
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    
    # 각 워커마다 고유한 시드 설정
    worker_seed = 42 + worker_id + rank
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_generator(rank=0):
    """DataLoader용 Generator 생성"""
    g = torch.Generator()
    g.manual_seed(42 + rank)
    return g

def load_or_prepare_pile(
        cache_path=None,
        raw_cache=None):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"🔹 Loading Geonwoohong/pile-uncopyrighted-6b-tokenized-gpt2 (cache_dir={cache_dir})")
    ds = load_dataset(
        "Geonwoohong/pile-uncopyrighted-6b-tokenized-gpt2",
        cache_dir=cache_dir
    )
    return ds["train"], ds["validation"]
