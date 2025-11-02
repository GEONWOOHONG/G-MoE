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
from analysis_expert_mapping import run_mapping_analysis

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

    an = sub.add_parser("analysis")
    an.add_argument("--num_experts", type=int, default=16)
    an.add_argument("--batch_size", type=int, default=44)
    an.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to analyze (None for 10% of total)")
    an.add_argument("--debug", action="store_true", help="Debug mode: use only 0.1% of validation data for quick testing")

    args = ap.parse_args()

    if args.cmd == "build-hash":
        create_global_hash_table(num_experts=args.num_experts)
    elif args.cmd == "analysis":
        set_seed(42)
        # Debug 모드일 때 max_batches를 매우 작게 설정
        debug_max_batches = args.max_batches
        if args.debug and args.max_batches is None:
            # Debug 모드: 전체의 0.1%만 사용하도록 특별 플래그 전달
            debug_max_batches = "debug"
        
        run_mapping_analysis(
            batch_size=args.batch_size,
            base_num_experts=args.num_experts,
            max_batches=debug_max_batches,
            run_specialization=True,
            run_confidence=True,
            run_routes=True,
        )
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

#mv /workspace/checkpoints/exp1 /workspace/checkpoints/gshard_exp1

#rm -rf /workspace/checkpoints/exp1
#rm -rf /workspace/runs

#tensorboard --logdir=/workspace/runs --host=0.0.0.0 --port=6006
#watch -n 5 nvidia-smi

#apt-get update && apt-get install -y zip unzip
#apt update && apt install -y nano
#pip install transformers datasets tensorboard pandas tqdm scipy tiktoken safetensors huggingface_hub

#wget https://github.com/schollz/croc/releases/download/v10.2.5/croc_v10.2.5_Linux-64bit.tar.gz
#tar xzf croc_v10.2.5_Linux-64bit.tar.gz
#mv croc /usr/local/bin/
#croc --version

#croc send --transfers 8 /workspace/checkpoints
#croc send /workspace/checkpoints/masking_experiment_8experts.csv
#cd "C:\IMML Lab"
#croc <코드값>

#bias=False + kaiming_uniform_ 초기화를 통해서 dead expert 방지해야함
#top-2 이상일 때 선택 확률을 normalize 해야함

#Fused Kernel / Flash Attention
#Compile 모드

#export HF_HOME=/workspace/hf_cache
#export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True