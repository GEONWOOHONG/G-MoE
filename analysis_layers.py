# analysis_layers.py
import os, json, time, collections, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHECKPOINTS_DIR
from utils import (
    set_seed,
    ensure_flash_attn,
    build_model_for_mode,
    load_checkpoint_if_exists,
)
from modeling import MoELayer
from transformers import GPT2Config

@torch.no_grad()
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float(); b = b.float()
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (a * b).sum(dim=-1)

@torch.no_grad()
def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)

@torch.no_grad()
def _cka_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xc = _center(X); Yc = _center(Y)
    XT_Y = Xc.T @ Yc
    num = (XT_Y ** 2).sum()
    den = torch.sqrt(((Xc.T @ Xc) ** 2).sum() + 1e-12) * torch.sqrt(((Yc.T @ Yc) ** 2).sum() + 1e-12)
    return num / (den + 1e-12)

@torch.no_grad()
def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1)

class _Recorder:
    def __init__(self, max_tokens:int=16384, save_routes:bool=True):
        self.pre_mlp: Dict[int, List[torch.Tensor]]  = collections.defaultdict(list)
        self.post_mlp: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.route_dist: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)
        self.token_counts: Dict[int, int] = collections.defaultdict(int)  # [Optimization] 토큰 수 추적
        self.layers: List[int] = []
        # CKA 분석에 필요한 충분한 양으로 설정 (기존 8192 -> 16384)
        self.max_tokens = int(max_tokens)
        self.save_routes = save_routes

    def is_full(self) -> bool:
        """모든 레이어에 대해 max_tokens 이상 수집되었는지 확인"""
        if not self.layers: return False
        return all(self.token_counts[l] >= self.max_tokens for l in self.layers)

    def _trimcat(self, lst: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not lst: return None
        X = torch.cat(lst, dim=0)
        if X.size(0) > self.max_tokens:
            X = X[: self.max_tokens]
        return X

    def get_pair(self, l:int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._trimcat(self.pre_mlp[l]), self._trimcat(self.post_mlp[l])

    def get_routes(self, l:int) -> Optional[torch.Tensor]:
        if not self.save_routes: return None
        return self._trimcat(self.route_dist[l])

def _register_hooks(model: nn.Module, rec: _Recorder):
    handles = []
    pat = re.compile(r"^transformer\.h\.(\d+)\.mlp$")

    def pre_hook(module, inputs):
        # [Optimization] 이미 충분한 데이터를 모았으면 복사하지 않음
        lidx = getattr(module, "_layer_idx", None)
        if lidx is not None and rec.token_counts[lidx] >= rec.max_tokens:
            return

        x = inputs[0]
        if not isinstance(x, torch.Tensor): return
        B, T, H = x.shape
        # [Optimization] 필요한 만큼만 자르고 CPU로 이동
        needed = rec.max_tokens - rec.token_counts[lidx]
        flat = x.reshape(B*T, H)
        if flat.size(0) > needed:
            flat = flat[:needed]
        
        flat_cpu = flat.detach().to(torch.float32).cpu()
        if lidx is not None:
            rec.pre_mlp[lidx].append(flat_cpu)
            # post_hook에서 카운트 증가를 위해 여기서는 카운트 체크만 함

    def fwd_hook(module, inputs, outputs):
        lidx = getattr(module, "_layer_idx", None)
        # [Optimization] 이미 충분한 데이터를 모았으면 스킵
        if lidx is not None and rec.token_counts[lidx] >= rec.max_tokens:
            return

        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        B, T, H = out.shape
        
        needed = rec.max_tokens - rec.token_counts[lidx]
        flat = out.reshape(B*T, H)
        if flat.size(0) > needed:
            flat = flat[:needed]
        
        flat_cpu = flat.detach().to(torch.float32).cpu()
        
        if lidx is not None:
            rec.post_mlp[lidx].append(flat_cpu)
            rec.token_counts[lidx] += flat.size(0) # 여기서 카운트 증가

        # 라우팅 정보 수집
        if rec.save_routes:
            for m in module.modules():
                if isinstance(m, MoELayer):
                    scores = None
                    if hasattr(m, "router") and getattr(m.router, "last_scores", None) is not None:
                        scores = m.router.last_scores
                    elif hasattr(m, "xmoe_router") and getattr(m.xmoe_router, "last_scores", None) is not None:
                        scores = m.xmoe_router.last_scores
                    elif getattr(m, "last_scores", None) is not None:
                        scores = m.last_scores
                    
                    if scores is not None:
                        # Scores shape: [B, T, E] -> [B*T, E]
                        flat_scores = scores.view(-1, scores.size(-1))
                        if flat_scores.size(0) > needed:
                            flat_scores = flat_scores[:needed]
                            
                        probs = torch.softmax(flat_scores, dim=-1)
                        rec.route_dist[lidx].append(probs.detach().to(torch.float32).cpu())

    for name, module in model.named_modules():
        m = pat.match(name)
        if m:
            lidx = int(m.group(1))
            module._layer_idx = lidx
            rec.layers.append(lidx)
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(fwd_hook))
    return handles

@torch.no_grad()
def compute_A_metrics(rec: _Recorder) -> Dict:
    layers = sorted(set(rec.layers))
    out = {"LEI": {}, "InterCKA": {}, "InterCKA_by_delta": {}, "IntraRedundancyProxy": {}, "RoutingUsageEntropy": {}, "ActiveExperts": {}, "EmptyExpertsRatio": {}, "IntraExpertCKA": {}}
    
    print("Computing metrics based on collected activations...")
    for l in layers:
        X, Y = rec.get_pair(l)
        if X is None or Y is None or X.size(0) != Y.size(0):
            continue
        cos = _cosine_similarity(X, Y)
        out["LEI"][l] = float((1.0 - cos.mean()).item())
        
    reps = {l: rec.get_pair(l)[1] for l in layers}
    valid_layers = [l for l in layers if reps.get(l) is not None]
    
    # Inter-layer CKA
    inter_pairs = {}
    for i, l1 in enumerate(valid_layers):
        for l2 in valid_layers[i+1:]:
            X, Y = reps[l1], reps[l2]
            n = min(X.size(0), Y.size(0))
            # CKA는 O(N^2) 메모리를 사용하므로 너무 크면 다운샘플링
            if n > 10000:
                indices = torch.randperm(n)[:10000]
                X_sub = X[indices]
                Y_sub = Y[indices]
                cka = _cka_linear(X_sub, Y_sub)
            else:
                cka = _cka_linear(X[:n], Y[:n])
            key = f"{l1}-{l2}"
            inter_pairs[key] = float(cka.item())
            
    out["InterCKA"] = inter_pairs
    delta_acc = collections.defaultdict(list)
    for k, v in inter_pairs.items():
        a, b = k.split("-")
        d = abs(int(a) - int(b))
        delta_acc[d].append(v)
    out["InterCKA_by_delta"] = {int(d): float(np.mean(v)) for d, v in delta_acc.items()}
    
    for l in layers:
        P = rec.get_routes(l)
        if P is None:
            continue
        E = P.shape[1]
        if E < 2:
            continue
        counts = torch.argmax(P, dim=-1).bincount(minlength=E).to(torch.float32)
        freq = counts / counts.sum().clamp_min(1e-9)
        usage_entropy = float(_entropy(freq).item())
        active = int((freq > 0.01).sum().item())
        empty_ratio = float((freq == 0).float().mean().item())
        
        # Intra-redundancy (Sampled for speed)
        if P.size(0) > 5000:
            Q_sub = P[torch.randperm(P.size(0))[:5000]]
        else:
            Q_sub = P
        Q = Q_sub / (Q_sub.norm(dim=0, keepdim=True) + 1e-9)
        S = torch.einsum("ne,nk->ek", Q, Q)
        mask = (~torch.eye(E, dtype=torch.bool))
        intra_proxy = float(S[mask].mean().item())
        
        out["RoutingUsageEntropy"][l] = usage_entropy
        out["ActiveExperts"][l] = active
        out["EmptyExpertsRatio"][l] = empty_ratio
        
        # IntraExpertCKA
        top1 = torch.argmax(P, dim=-1)
        min_tokens = 32
        max_tokens = 512 # CKA sample limit per expert
        feats = rec.get_pair(l)[1]
        if feats is not None:
            per_expert_indices = []
            for e in range(E):
                idx = torch.nonzero(top1 == e, as_tuple=False).view(-1)
                if idx.numel() >= min_tokens:
                    if idx.numel() > max_tokens:
                        idx = idx[torch.linspace(0, idx.numel() - 1, steps=max_tokens).long()]
                    per_expert_indices.append((e, idx))
            cka_vals = []
            for i in range(len(per_expert_indices)):
                ei, ii = per_expert_indices[i]
                Xi = feats[ii]
                for j in range(i + 1, len(per_expert_indices)):
                    ej, jj = per_expert_indices[j]
                    Xj = feats[jj]
                    n = min(Xi.size(0), Xj.size(0))
                    if n >= min_tokens:
                        c = _cka_linear(Xi[:n], Xj[:n]).item()
                        cka_vals.append(c)
            if len(cka_vals) > 0:
                out["IntraExpertCKA"][l] = float(np.mean(cka_vals))
            else:
                out["IntraExpertCKA"][l] = float("nan")
        else:
            out["IntraExpertCKA"][l] = float("nan")
        out["IntraRedundancyProxy"][l] = intra_proxy
    return out

@torch.no_grad()
def run_analysis_A(mode: str = "ours_refine",
                   num_experts: int = 16,
                   batch_size: int = 64,  # Increased default batch size
                   seq_len: int = 1024,
                   max_batches: Optional[int] = None,
                   save_json: Optional[str] = None,
                   use_flash_attn: bool = True,
                   verbose: bool = True):
    assert mode in {"dense","switch","gshard","hash","ours_refine"}, f"Unsupported mode: {mode}"
    set_seed(42)
    if use_flash_attn:
        ensure_flash_attn()
    if verbose:
        print(f"\n{'='*70}")
        print(f"Layer-Level Analysis (Part A) - Mode: {mode}")
        print(f"{'='*70}")
    
    # Test 데이터셋 로드
    from data import load_pile_test, worker_init_fn, get_dataloader_generator
    from torch.utils.data import DataLoader
    
    pile_test = load_pile_test(verbose=verbose)
    pile_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_ds = pile_test
    if verbose:
        print(f"Using test set: {len(pile_test)} samples")
    
    num_workers = min(16, max(4, (os.cpu_count() or 8)//2))
    loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True, # Optimized loader
        worker_init_fn=worker_init_fn, generator=get_dataloader_generator(0)
    )
    total_batches = len(loader)
    
    # Layer Analysis는 CKA/LEI 계산용이므로 전체 데이터셋을 다 돌 필요가 없음.
    # 충분한 통계량(약 16k 토큰)이 모이면 조기 종료함.
    if max_batches is None:
        effective_batches = total_batches
    else:
        effective_batches = min(max_batches, total_batches)

    if verbose:
        print(f"Using up to {effective_batches}/{total_batches} eval batches (will exit early if sufficient data collected)")

    from utils import find_checkpoint_path
    ckpt_path = find_checkpoint_path(mode, CHECKPOINTS_DIR)
    
    if ckpt_path and os.path.exists(os.path.join(os.path.dirname(ckpt_path), "config.json")):
        config = GPT2Config.from_pretrained(os.path.dirname(ckpt_path))
        if verbose: print(f"🔹 Loaded config from {os.path.dirname(ckpt_path)}")
    else:
        print("⚠️ Config not found, using default parameters (n_embd=768, n_layer=12)")
        config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)

    model = build_model_for_mode(mode, num_experts=num_experts, config=config)
    load_checkpoint_if_exists(model, mode, CHECKPOINTS_DIR, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    # Max tokens for CKA metrics (16384 is enough for stable stats)
    rec = _Recorder(max_tokens=16384, save_routes=True)
    handles = _register_hooks(model, rec)
    
    processed = 0
    t0 = time.time()
    
    # Main Loop
    for i, batch in enumerate(loader):
        if i >= effective_batches: break
        
        # [Optimization] Check if we have enough data
        if rec.is_full():
            if verbose: print("✅ Sufficient data collected for layer metrics. Stopping early.")
            break
            
        input_ids = batch["input_ids"][:, :seq_len].to(device, non_blocking=True)
        attn      = batch["attention_mask"][:, :seq_len].to(device, non_blocking=True)
        
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            else:
                _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        processed += 1
        
        if verbose and processed % 10 == 0:
            print(f"Processed {processed} batches...")
            
    resA = compute_A_metrics(rec)
    result = {
        "mode": mode,
        "num_experts": num_experts,
        "batches": processed,
        "batch_size": batch_size,
        "runtime_sec": float(time.time() - t0),
        "A": resA,
    }
    if save_json is None:
        save_json = os.path.join(CHECKPOINTS_DIR, f"analysis_A_{mode}.json")
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"Saved Part-A metrics to: {save_json}")
        print(f"Runtime: {result['runtime_sec']:.2f}s")
    for h in handles:
        try: h.remove()
        except Exception: pass
    del model
    torch.cuda.empty_cache()
    return result

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense","switch","gshard","hash","ours_refine"], default="ours_refine")
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--no_flash", action="store_true")
    args = ap.parse_args()
    run_analysis_A(
        mode=args.mode,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        save_json=args.save_json,
        use_flash_attn=not args.no_flash,
    )