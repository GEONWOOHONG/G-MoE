# data.py — 데이터셋 로드 유틸
import os, random, numpy as np, torch
from datasets import load_dataset
from datasets.utils import logging as ds_logging

def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def worker_init_fn(worker_id):
    """워커 시드 고정을 위한 초기화 함수"""
    # 분산 환경에서 rank 정보 가져오기
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    
    # 각 워커마다 고유한 시드 설정
    worker_seed = 42 + worker_id + rank
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_generator(rank=0):
    """DataLoader용 Generator 생성"""
    g = torch.Generator()
    g.manual_seed(42 + rank)
    return g

def load_or_prepare_pile(cache_path=None, raw_cache=None, verbose=True):
    """
    HF datasets 로드 (분산 안전)
    - verbose=False 면 어떤 랭크든 메시지 최대한 억제
    - verbose=True 면 rank0만 최소 메시지 출력
    """
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    # ☆ 비-rank0 또는 verbose=False 면 프로그레스바/로그 억제
    if (not _is_rank0()) or (not verbose):
        try:
            ds_logging.set_verbosity_error()
            ds_logging.disable_progress_bar()
        except Exception:
            pass  # 구버전 호환

    if verbose and _is_rank0():
        print(f"🔹 Loading Geonwoohong/pile-uncopyrighted-6b-tokenized-gpt2 (cache_dir={cache_dir})")

    ds = load_dataset(
        "Geonwoohong/pile-uncopyrighted-6b-tokenized-gpt2",
        cache_dir=cache_dir
    )
    return ds["train"], ds["validation"]