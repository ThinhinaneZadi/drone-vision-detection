"""
pseudo_label.py — generate pseudo-labels for the partial-label-space
experiments, as the standard-literature comparison point to this
project's loss-reweighting fix (see docs/partial_label_profiles.md).

WHAT THIS DOES DIFFERENTLY FROM THE LOSS-REWEIGHTING FIX:
The loss-reweighting fix (client_train.py --class-weights) tells the
model to simply ignore classes a client has no real labels for -- neither
rewarding nor punishing predictions there.

Pseudo-labeling instead asks the CURRENT GLOBAL MODEL what it thinks is
in a client's images, for exactly the classes that client has no real
labels for, and treats sufficiently confident predictions as if they
were real ground-truth labels for this round's local training. This is
the standard approach in the missing/partial-label literature, and is
the comparison point our contribution needs to be evaluated against.

Because the global model changes every round, pseudo-labels are
regenerated FRESH every round for every client, using that round's
current global checkpoint -- not generated once and reused.
"""
from pathlib import Path
import shutil

from ultralytics import YOLO


def generate_pseudo_labeled_partition(
    model_ckpt: Path,
    source_client_dir: Path,
    allowed_classes: set[int],
    out_dir: Path,
    conf_threshold: float = 0.5,
    imgsz: int = 960,
    device: str = "0",
) -> Path:
    """For ONE client, for ONE round: run the current global model on
    this client's own training images, and for any predicted object
    whose class is OUTSIDE allowed_classes (i.e. a class this client has
    no real ground truth for), add it as a pseudo-label line -- merged
    alongside the client's existing real labels for allowed_classes,
    which are copied through unchanged.

    Returns the path to a data.yaml pointing at the merged result, in
    the same images/+labels/ structure build_partial_label_partition.py
    uses, so client_train.py needs no changes to consume it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(exist_ok=True)
    out_labels.mkdir(exist_ok=True)

    src_images_dir = source_client_dir / "images"
    src_labels_dir = source_client_dir / "labels"
    image_paths = sorted(src_images_dir.glob("*.jpg"))

    model = YOLO(str(model_ckpt))
    # batch inference across all this client's images at once, quiet mode
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save=False,
    )

    n_pseudo_added = 0
    for img_path, result in zip(image_paths, results):
        link_path = out_images / img_path.name
        if not link_path.exists():
            link_path.symlink_to(img_path.resolve())

        # start from this client's REAL, already profile-filtered labels
        src_label_path = src_labels_dir / (img_path.stem + ".txt")
        real_lines = []
        if src_label_path.is_file():
            real_lines = [l for l in src_label_path.read_text().splitlines() if l.strip()]

        # add pseudo-labels ONLY for classes outside this client's profile
        pseudo_lines = []
        boxes = result.boxes
        if boxes is not None:
            for cls_id, xywhn in zip(boxes.cls.tolist(), boxes.xywhn.tolist()):
                cls_id = int(cls_id)
                if cls_id not in allowed_classes:
                    cx, cy, w, h = xywhn
                    pseudo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    n_pseudo_added += 1

        merged = real_lines + pseudo_lines
        out_label_path = out_labels / (img_path.stem + ".txt")
        out_label_path.write_text("\n".join(merged) + ("\n" if merged else ""))

    train_txt = out_dir / "train.txt"
    train_txt.write_text("\n".join(str((out_images / p.name).resolve()) for p in image_paths) + "\n")

    return train_txt, n_pseudo_added
