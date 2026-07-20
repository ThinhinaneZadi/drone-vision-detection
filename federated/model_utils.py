"""
model_utils.py — checkpoint/state-dict handling and FedAvg core.

Aggregation rules:
- float tensors (weights, biases, BN running stats): sample-weighted average
- integer tensors (BN num_batches_tracked): copied from the largest-weight
  client, never averaged
- optimizer state, epoch counters, EMA: never aggregated; stale EMA is
  stripped from the saved checkpoint so aggregated weights are actually used
"""

from pathlib import Path
import argparse
import sys
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
BEST = REPO / "models/best_yolo11s_visdrone.pt"

def get_state(path: Path) -> dict:
    """Effective model weights as float32 CPU tensors."""
    model = YOLO(str(path))
    return {k: v.detach().cpu().float().clone()
            for k, v in model.model.state_dict().items()}

def check_compatible(states: list[dict]) -> None:
    keys = set(states[0])
    for i, s in enumerate(states[1:], 1):
        if set(s) != keys:
            sys.exit(f"ERROR: state {i} has mismatched keys")
        for k in keys:
            if s[k].shape != states[0][k].shape:
                sys.exit(f"ERROR: shape mismatch at {k}: "
                         f"{s[k].shape} vs {states[0][k].shape}")

def fedavg(states: list[dict], weights: list[float]) -> dict:
    """Sample-count-weighted FedAvg over compatible state dicts."""
    if len(states) != len(weights) or not states:
        sys.exit("ERROR: states/weights length mismatch or empty")
    check_compatible(states)
    total = float(sum(weights))
    ref = weights.index(max(weights))          # donor for integer tensors
    out = {}
    for k, v0 in states[0].items():
        if v0.is_floating_point():
            out[k] = sum(s[k] * (w / total) for s, w in zip(states, weights))
        else:
            out[k] = states[ref][k].clone()
    return out

def save_aggregated(base_path: Path, state: dict, out_path: Path) -> None:
    """Inject aggregated weights into a copy of the base checkpoint."""
    if out_path.resolve() == base_path.resolve():
        sys.exit("ERROR: refusing to overwrite the base checkpoint")
    ckpt = torch.load(str(base_path), map_location="cpu", weights_only=False)
    ckpt["model"].float().load_state_dict(state, strict=True)
    ckpt["ema"] = None                          # stale EMA must not shadow us
    ckpt["updates"] = None
    ckpt["optimizer"] = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out_path))
    print(f"saved aggregated checkpoint: {out_path}")

def selftest() -> None:
    """FedAvg(X, X) must equal X exactly; save/reload must round-trip."""
    print(f"self-test using {BEST.name}")
    s = get_state(BEST)
    agg = fedavg([s, {k: v.clone() for k, v in s.items()}], [1.0, 1.0])
    max_diff = max((agg[k] - s[k]).abs().max().item() for k in s)
    print(f"max |FedAvg(X,X) - X| = {max_diff:.2e}")
    out = REPO / "federated/experiments/selftest_identity.pt"
    save_aggregated(BEST, agg, out)
    s2 = get_state(out)
    max_rt = max((s2[k] - s[k]).abs().max().item() for k in s)
    print(f"max round-trip diff (fp16 storage) = {max_rt:.2e}")
    out.unlink()                                # clean up the temp file
    ok = max_diff == 0.0 and max_rt < 1e-2
    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        print("import module or run with --selftest")
