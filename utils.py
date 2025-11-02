# utils.py — 공용 유틸(시드/옵티마이저/스케줄러/체크포인트/런타임 캐시)
import os, math, random, numpy as np, torch
from safetensors.torch import save_model
from transformers import get_cosine_schedule_with_warmup

# 전역 캐시 for input_ids (hash 게이팅 지원) — 원본 인터페이스 유지
CURRENT_INPUT_IDS = None
def set_current_input_ids(x):
    global CURRENT_INPUT_IDS
    CURRENT_INPUT_IDS = x
def get_current_input_ids():
    return CURRENT_INPUT_IDS

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

def get_default_optimizer(model, lr=2.5e-4, weight_decay=0.1):
    from torch.optim import AdamW
    return AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

def get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler

def save_checkpoint(model, optimizer, scheduler, step, best_loss, total_train_steps, path, name="checkpoint.safetensors"):
    # DDP 래핑 대응: 실제 모듈로 저장
    if hasattr(model, "module"):
        model = model.module
    model_path = os.path.join(path, f"{name}")
    save_model(model, model_path)
    trainer_state = {
        "step": int(step),
        "best_loss": float(best_loss),
        "total_train_steps": int(total_train_steps),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    trainer_path = os.path.join(path, f"{name}_trainer.pt")
    torch.save(trainer_state, trainer_path)
    print(f"🔹 Checkpoint saved: {model_path}, {trainer_path}")

def print_model_info(model, config, mode, num_experts,
                     batch_size=None, grad_accum_steps=None, effective_batch=None):
    print("===== Model / MoE Configuration =====")
    print(f"Backbone        : GPT-2 ({config.n_layer} layers)")
    print(f"Mode            : {mode}")
    if mode == "dense":
        print(f"Num Experts     : N/A (Not an MoE model)")
    else:
        print(f"Num Experts     : {num_experts}")
    print(f"Hidden Dim      : {config.n_embd}")
    print(f"FFN Dim        : {config.n_embd * 4}")
    print(f"Attention Heads: {config.n_head}")
    print(f"Vocab Size     : {config.vocab_size}")
    print("-------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params   : {total_params/1e6:.2f}M")
    print(f"Trainable      : {trainable_params/1e6:.2f}M")
    from modeling import GPT2LayerMoE
    moe_layers = [m for m in model.modules() if isinstance(m, GPT2LayerMoE)]
    print(f"MoE Layers     : {len(moe_layers)} (replaced FFN layers)")
    for i, layer in enumerate(moe_layers[:4]):
        moe = layer.moe
        if moe.mode == "ours_com":
            total_experts = 1 + len(moe.global_experts)
            memory_info = " (shared GRU across layers, depth-wise state)"
            print(f"  - MoELayer {i}: mode={moe.mode}, experts={total_experts} "
                  f"(local=1, global={len(moe.global_experts)}){memory_info}")
        else:
            print(f"  - MoELayer {i}: mode={moe.mode}, experts={moe.num_experts}, "
                  f"shared_expert={'Yes' if moe.shared_expert is not None else 'No'}, "
                  f"global_experts={'Yes' if moe.global_experts is not None else 'No'}")
    if batch_size is not None and grad_accum_steps is not None and effective_batch is not None:
        print(f"Batch per step : {batch_size}, Accum steps = {grad_accum_steps}, "
              f"Effective batch = {effective_batch}")
    print("=====================================")

def ensure_flash_attn():
    import importlib.util, subprocess, sys
    try:
        spec = importlib.util.find_spec("flash_attn")
        if spec is None:
            print("🔹 Installing FlashAttention...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "flash-attn==2.7.4.post1", "--no-build-isolation"
            ])
    except Exception as e:
        print("⚠️ FlashAttention install failed:", e)
