#config.py
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.join(os.path.dirname(CURRENT_DIR), "workspace")

HASH_TABLE_PATH = os.path.join(WORKSPACE_ROOT, "checkpoints", "hash_exp1", "global_hash_router_table.pt")

def get_hash_table_path(vocab_size: int):
    base = os.path.join(WORKSPACE_ROOT, "checkpoints", "hash_exp1")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"global_hash_router_table.v{vocab_size}.pt")

HF_HOME = os.path.join(WORKSPACE_ROOT, "hf_cache")
HF_DATASETS_CACHE = os.path.join(WORKSPACE_ROOT, "hf_cache", "datasets")

CHECKPOINTS_DIR = os.path.join(WORKSPACE_ROOT, "checkpoints")

os.makedirs(WORKSPACE_ROOT, exist_ok=True)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)