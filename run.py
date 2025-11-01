# run.py — CLI 엔트리포인트
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/datasets"

os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import argparse
from config import HASH_TABLE_PATH
from utils import set_current_input_ids, get_current_input_ids
from train import train_moe, set_seed
from tools_hash import create_global_hash_table

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--mode", default="switch")
    tr.add_argument("--num_experts", type=int, default=16)
    tr.add_argument("--batch_size", type=int, default=44)
    tr.add_argument("--seq_len", type=int, default=1024)
    tr.add_argument("--continue_training", action="store_true")

    ev = sub.add_parser("eval")
    ev.add_argument("--mode", default="switch")
    ev.add_argument("--num_experts", type=int, default=16)
    ev.add_argument("--batch_size", type=int, default=44)
    ev.add_argument("--seq_len", type=int, default=1024)
    
    bh = sub.add_parser("build-hash")
    bh.add_argument("--num_experts", type=int, default=16)

    args = ap.parse_args()

    if args.cmd == "build-hash":
        create_global_hash_table(num_experts=args.num_experts)
    elif args.cmd in {"train", "eval"}:
        set_seed(42)
        train_moe(
            mode=args.mode,
            num_experts=args.num_experts,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            continue_training=(args.cmd == "train" and args.continue_training),
        )

if __name__ == "__main__":
    main()

#python run.py train --mode stablemoe --num_experts 16 --batch_size 44

#cd "/c/IMML Lab/runpod_a100/repo"
#git status
#git add .
#git commit -m "context"
#git push origin main