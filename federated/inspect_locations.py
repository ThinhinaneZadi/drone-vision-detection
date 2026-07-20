"""
inspect_locations.py — VisDrone-DET capture-group audit.

Reproduces the dataset audit for the federated-learning project:
counts capture groups (location proxies) from filename prefixes,
checks train/val group overlap, applies client-eligibility rules,
and writes the client registry CSV.

Project-defined conventions (NOT official VisDrone metadata):
- "capture group" = first underscore-delimited filename token
- origin inferred from prefix: 9999* = still capture, else video-derived
"""

from pathlib import Path
from collections import Counter
import csv
import sys

# ---------------------------------------------------------------- config
REPO = Path(__file__).resolve().parents[1]
TRAIN_IMG = REPO / "data/VisDrone-DET/VisDrone2019-DET-train/images"
VAL_IMG   = REPO / "data/VisDrone-DET/VisDrone2019-DET-val/images"
OUT_CSV   = REPO / "federated/experiments/registry/clients.csv"
THRESHOLDS = (20, 50, 100)          # candidate minimum-images-per-client tiers
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------- section 1: scan
def count_groups(img_dir: Path) -> Counter:
    """Return Counter{group_prefix: n_images} for one split directory."""
    if not img_dir.is_dir():
        sys.exit(f"ERROR: image directory not found: {img_dir}")
    counts = Counter()
    n_files = 0
    for f in img_dir.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        n_files += 1
        prefix = f.name.split("_", 1)[0]
        if not prefix.isdigit():
            sys.exit(f"ERROR: unexpected filename pattern: {f.name}")
        counts[prefix] += 1
    if n_files == 0:
        sys.exit(f"ERROR: no images found in {img_dir}")
    print(f"scanned {img_dir.name}: {n_files} images, {len(counts)} groups")
    return counts

train_counts = count_groups(TRAIN_IMG)
val_counts   = count_groups(VAL_IMG)

# ---------------------------------------------------------------- section 2: overlap
overlap = sorted(set(train_counts) & set(val_counts))
print(f"\ntrain/val overlapping groups: {len(overlap)}")
overlap_train_imgs = sum(train_counts[g] for g in overlap)
overlap_val_imgs   = sum(val_counts[g] for g in overlap)
print(f"  overlap images: train={overlap_train_imgs}, val={overlap_val_imgs}")
clean_val_imgs = sum(v for g, v in val_counts.items() if g not in overlap)
print(f"  clean global-val size: {clean_val_imgs} images, "
      f"{len(val_counts) - len(overlap)} groups")

# ---------------------------------------------------------------- section 3+4: registry rows
rows = []
for g in sorted(train_counts):
    n = train_counts[g]
    origin = "still_capture" if g.startswith("9999") else "video_derived"
    in_overlap = g in overlap
    if in_overlap:
        excluded_reason = "train_val_group_overlap"
    elif n < min(THRESHOLDS):
        excluded_reason = f"below_min_size_{min(THRESHOLDS)}"
    else:
        excluded_reason = ""
    rows.append({
        "group_id": g,
        "n_train_images": n,
        "n_val_images": val_counts.get(g, 0),
        "origin_inferred": origin,
        "train_val_overlap": int(in_overlap),
        "eligible_20":  int(n >= 20  and not in_overlap),
        "eligible_50":  int(n >= 50  and not in_overlap),
        "eligible_100": int(n >= 100 and not in_overlap),
        "excluded_reason": excluded_reason,
    })

# ---------------------------------------------------------------- section 5: write + summary
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote registry: {OUT_CSV} ({len(rows)} groups)")

print("\neligibility summary (excluding overlap groups):")
for t in THRESHOLDS:
    elig = [r for r in rows if r[f"eligible_{t}"]]
    imgs = sum(r["n_train_images"] for r in elig)
    print(f"  >= {t:>3} images: {len(elig):>3} clients, "
          f"{imgs} images ({imgs / sum(train_counts.values()):.0%} of train)")
