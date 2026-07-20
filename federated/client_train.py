"""
client_train.py — train ONE federated client locally, then exit.

Designed to be launched as a subprocess by server.py so GPU memory is
fully released between sequential clients (critical on 4 GB VRAM).
"""

from pathlib import Path
import argparse
import shutil
import sys
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="client data.yaml")
    ap.add_argument("--init", required=True, help="starting checkpoint")
    ap.add_argument("--out", required=True, help="where to copy trained weights")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default="federated/experiments/runs_local")
    args = ap.parse_args()

    data, init, out = Path(args.data), Path(args.init), Path(args.out)
    if not data.is_file():
        sys.exit(f"ERROR: data yaml not found: {data}")
    if not init.is_file():
        sys.exit(f"ERROR: init checkpoint not found: {init}")

    client_name = data.parent.name          # e.g. client_9999981

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO / run_dir).resolve()

    model = YOLO(str(init))
    model.train(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        val=False,                # evaluation happens separately on clean val
        plots=False,
        cache=False,
        workers=2,
        project=str(run_dir),
        name=client_name,
        exist_ok=True,
        verbose=True,
    )

    trained = run_dir / client_name / "weights/last.pt"
    if not trained.is_file():
        sys.exit(f"ERROR: expected trained weights not found: {trained}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained, out)
    print(f"CLIENT_DONE name={client_name} weights={out}")

if __name__ == "__main__":
    main()
