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

# ----------------------------------------------------------------------
# Unit tests — synthetic tensors, no GPU or real checkpoint required.
# Each raises AssertionError on failure with a clear message.
# ----------------------------------------------------------------------

def test_weighted_average_known_values() -> None:
    """Weighted FedAvg on hand-computed tensors must match exactly."""
    s1 = {"w": torch.tensor([1.0, 10.0])}
    s2 = {"w": torch.tensor([3.0, 20.0])}
    # weights 1:3 -> expected = (1*1 + 3*3)/4, (1*10 + 3*20)/4
    out = fedavg([s1, s2], [1.0, 3.0])
    expected = torch.tensor([2.5, 17.5])
    assert torch.allclose(out["w"], expected, atol=0.0), \
        f"expected {expected}, got {out['w']}"
    print("  [pass] weighted average matches hand-computed values")

def test_mismatched_architecture_rejected() -> None:
    """fedavg must refuse states with different keys or shapes."""
    s1 = {"w": torch.tensor([1.0, 2.0])}
    s2 = {"w": torch.tensor([1.0, 2.0, 3.0])}   # different shape
    try:
        fedavg([s1, s2], [1.0, 1.0])
    except SystemExit:
        pass
    else:
        raise AssertionError("fedavg accepted mismatched shapes without error")
    s3 = {"different_key": torch.tensor([1.0, 2.0])}
    try:
        fedavg([s1, s3], [1.0, 1.0])
    except SystemExit:
        pass
    else:
        raise AssertionError("fedavg accepted mismatched keys without error")
    print("  [pass] mismatched architectures correctly rejected")

def test_bn_counter_integrity() -> None:
    """Integer tensors (e.g. BN num_batches_tracked) must be COPIED from
    the largest-weight client, never averaged — averaging would silently
    corrupt BatchNorm behavior."""
    s1 = {"num_batches_tracked": torch.tensor(10, dtype=torch.long)}
    s2 = {"num_batches_tracked": torch.tensor(50, dtype=torch.long)}
    out = fedavg([s1, s2], [1.0, 3.0])          # s2 has the larger weight
    assert out["num_batches_tracked"].item() == 50, \
        (f"expected copy of largest-weight client's value (50), "
         f"got {out['num_batches_tracked'].item()} "
         f"(looks averaged: {(10 + 50) / 2})")
    assert out["num_batches_tracked"].dtype == torch.long, \
        "integer tensor dtype was not preserved"
    print("  [pass] integer BN counter copied from largest-weight client, "
          "not averaged")

def test_ema_stripped_on_save() -> None:
    """save_aggregated must strip any stale EMA/optimizer/updates so the
    aggregated weights are what actually gets used on reload."""
    ckpt = torch.load(str(BEST), map_location="cpu", weights_only=False)
    has_ema_field = "ema" in ckpt
    assert has_ema_field, "checkpoint has no 'ema' field to test against"
    s = get_state(BEST)
    agg = fedavg([s, {k: v.clone() for k, v in s.items()}], [1.0, 1.0])
    out = REPO / "federated/experiments/selftest_ema_check.pt"
    save_aggregated(BEST, agg, out)
    raw = torch.load(str(out), map_location="cpu", weights_only=False)
    assert raw["ema"] is None, "stale EMA was not stripped from saved checkpoint"
    assert raw["optimizer"] is None, "optimizer state was not stripped"
    # confirm reloading through the real loader also yields the aggregated
    # weights, not something else
    s2 = get_state(out)
    max_diff = max((s2[k] - agg[k]).abs().max().item() for k in agg)
    assert max_diff < 1e-2, \
        f"reloaded model does not match aggregated weights (diff={max_diff:.2e})"
    out.unlink()
    print("  [pass] stale EMA/optimizer stripped; reload uses aggregated weights")

def selftest() -> None:
    """Full correctness suite: identity property, save/reload round-trip,
    and the four unit tests above."""
    print(f"self-test using {BEST.name}\n")

    print("unit tests (synthetic tensors):")
    test_weighted_average_known_values()
    test_mismatched_architecture_rejected()
    test_bn_counter_integrity()
    test_ema_stripped_on_save()

    print("\nreal-checkpoint tests:")
    s = get_state(BEST)
    agg = fedavg([s, {k: v.clone() for k, v in s.items()}], [1.0, 1.0])
    max_diff = max((agg[k] - s[k]).abs().max().item() for k in s)
    print(f"  max |FedAvg(X,X) - X| = {max_diff:.2e}")
    out = REPO / "federated/experiments/selftest_identity.pt"
    save_aggregated(BEST, agg, out)
    s2 = get_state(out)
    max_rt = max((s2[k] - s[k]).abs().max().item() for k in s)
    print(f"  max round-trip diff (fp16 storage) = {max_rt:.2e}")
    out.unlink()

    ok = max_diff == 0.0 and max_rt < 1e-2
    print("\nSELF-TEST PASS" if ok else "\nSELF-TEST FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        print("import module or run with --selftest")
