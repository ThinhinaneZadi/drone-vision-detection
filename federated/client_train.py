"""
client_train.py — train ONE federated client locally, then exit.

Designed to be launched as a subprocess by server.py so GPU memory is
fully released between sequential clients (critical on 4 GB VRAM).
Supports layer freezing (freeze the first N layers during local training).

--continuous-lr disables per-round warmup and uses a small fixed learning
rate, appropriate for multi-round (50-100+) federated schedules where
restarting warmup every round would prevent the model from ever training
at a stable rate.
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
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N model layers (0 = train all)")
    ap.add_argument("--continuous-lr", action="store_true",
                    help="disable per-round warmup; use a small fixed lr0 "
                         "(needed for many-round schedules, e.g. 50-100+)")
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="fixed learning rate when --continuous-lr is set")
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

    train_kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        freeze=args.freeze if args.freeze > 0 else None,
        val=False,                # evaluation happens separately on clean val
        plots=False,
        cache=False,
        workers=2,
        project=str(run_dir),
        name=client_name,
        exist_ok=True,
        verbose=True,
    )

    if args.continuous_lr:
        # fixed lr, no per-round warmup restart, no auto-optimizer reselection
        train_kwargs.update(
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=1.0,               # no decay within a round; lr stays ~lr0
            warmup_epochs=0.0,
            cos_lr=False,
        )

    model = YOLO(str(init))
    model.train(**train_kwargs)

    trained = run_dir / client_name / "weights/last.pt"
    if not trained.is_file():
        sys.exit(f"ERROR: expected trained weights not found: {trained}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained, out)
    print(f"CLIENT_DONE name={client_name} weights={out}")

if __name__ == "__main__":
    main()
