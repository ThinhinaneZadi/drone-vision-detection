"""
prepare_option_b.py — build the Option B initialization checkpoint.

Creates a 10-class YOLO11s whose backbone/neck weights come from the
COCO-pretrained yolo11s.pt and whose detection head is freshly initialized
ONCE (seeded). All Option B clients and the aggregation base start from
this exact file, satisfying FedAvg's identical-initialization requirement.
The model has never seen any VisDrone data.
"""

from pathlib import Path
import sys
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "models/init_yolo11s_coco10.pt"
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]

def main() -> None:
    if OUT.exists():
        sys.exit(f"ERROR: {OUT} already exists — delete it first if you "
                 "really want to regenerate (clients must share ONE init)")

    torch.manual_seed(0)
    coco = YOLO("yolo11s.pt")            # auto-downloads if absent
    coco_sd = coco.model.state_dict()

    new = DetectionModel(cfg="yolo11s.yaml", nc=len(CLASS_NAMES), verbose=False)
    new_sd = new.state_dict()
    transfer = {k: v for k, v in coco_sd.items()
                if k in new_sd and v.shape == new_sd[k].shape}
    missing = [k for k in new_sd if k not in transfer]
    new.load_state_dict(transfer, strict=False)
    new.names = {i: n for i, n in enumerate(CLASS_NAMES)}

    print(f"transferred {len(transfer)}/{len(new_sd)} tensors from COCO")
    print(f"freshly initialized (head): {len(missing)} tensors")
    for k in missing[:6]:
        print(f"  new: {k}")
    if not all(".23." in k or "dfl" in k for k in missing):
        print("WARNING: some non-head tensors were NOT transferred — inspect!")

    torch.save({"model": new, "train_args": {}}, OUT)

    # verify the checkpoint round-trips through the same loader the
    # pipeline uses, and that shapes match the aggregation path
    check = YOLO(str(OUT))
    n = sum(p.numel() for p in check.model.parameters())
    print(f"reload OK: {n:,} parameters, nc={check.model.yaml.get('nc')}")
    print(f"saved: {OUT}")

if __name__ == "__main__":
    main()
