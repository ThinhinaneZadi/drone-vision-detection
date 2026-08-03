"""
inspect_partitions.py — non-IID analysis of client partitions.

Reads each client's train list, tallies per-class object counts from YOLO
label files, and compares each client's class distribution to the clean
global validation distribution (KL divergence). Writes class_stats.csv.
"""

from pathlib import Path
from collections import Counter
import argparse
import csv
import math
import sys

REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]
EPS = 1e-9  # avoids log(0) when a class is absent

def label_for(img_path: Path) -> Path:
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")

def tally(list_file: Path) -> tuple[Counter, int]:
    """Return (per-class object Counter, n_images) for one image list."""
    counts, n_imgs = Counter(), 0
    for line in list_file.read_text().splitlines():
        if not line.strip():
            continue
        n_imgs += 1
        lbl = label_for(Path(line))
        if not lbl.is_file():
            sys.exit(f"ERROR: missing label {lbl}")
        for row in lbl.read_text().splitlines():
            if row.strip():
                cls = int(row.split()[0])
                if not 0 <= cls < len(CLASS_NAMES):
                    sys.exit(f"ERROR: class id {cls} out of range in {lbl}")
                counts[cls] += 1
    return counts, n_imgs

def to_dist(counts: Counter) -> list[float]:
    total = sum(counts.values())
    return [(counts.get(i, 0) + EPS) / (total + EPS * len(CLASS_NAMES))
            for i in range(len(CLASS_NAMES))]

def kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[20, 50, 100], required=True)
    args = ap.parse_args()

    part_root = REPO / f"federated/experiments/partitions/tier{args.tier}"
    if not part_root.is_dir():
        sys.exit(f"ERROR: partition not found: {part_root} "
                 "(run partition_by_location.py first)")

    val_counts, val_imgs = tally(part_root / "clean_val.txt")
    val_dist = to_dist(val_counts)
    print(f"clean val: {val_imgs} images, {sum(val_counts.values())} objects")

    rows = []
    for client_dir in sorted(part_root.glob("client_*")):
        counts, n_imgs = tally(client_dir / "train.txt")
        n_obj = sum(counts.values())
        dist = to_dist(counts)
        row = {"client_id": client_dir.name,
               "n_images": n_imgs,
               "n_objects": n_obj,
               "objects_per_image": round(n_obj / n_imgs, 1),
               "kl_vs_global": round(kl(dist, val_dist), 3)}
        for i, name in enumerate(CLASS_NAMES):
            row[f"pct_{name}"] = round(100 * counts.get(i, 0) / n_obj, 1)
        rows.append(row)
        print(f"{client_dir.name}: {n_imgs} imgs, {n_obj} objs, "
              f"{row['objects_per_image']}/img, KL={row['kl_vs_global']}")

    out = part_root / "class_stats.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
