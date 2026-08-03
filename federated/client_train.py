"""
client_train.py — train ONE federated client locally, then exit.
Designed to be launched as a subprocess by server.py so GPU memory is
fully released between sequential clients (critical on 4 GB VRAM).
Supports layer freezing (freeze the first N layers during local training).
--continuous-lr disables per-round warmup and uses a small fixed learning
rate, appropriate for multi-round (50-100+) federated schedules where
restarting warmup every round would prevent the model from ever training
at a stable rate.
--prox-mu > 0 enables FedProx: a proximal term that discourages this
client's local weights from drifting too far from the global model it
started the round with (see federated/fedprox.py).
--class-weights enables FIXED per-class loss weighting for partial-label-
space experiments: a client whose profile excludes certain classes can
set those classes' weight to 0.0, so the loss neither rewards nor
punishes predictions there (see docs/partial_label_profiles.md).

NOTE: setting `model.class_weights` on the YOLO wrapper's .model BEFORE
calling .train() does NOT work — Ultralytics constructs its own internal
training model/trainer objects with different identities, verified via
id() debug instrumentation during development (see git history). The
correct hook is overriding DetectionTrainer.set_class_weights(), which
Ultralytics itself calls at the correct point in setup, normally to
auto-compute class weights from --cls_pw; we override it to apply our
FIXED weights instead when --class-weights is given.
"""
from pathlib import Path
import argparse
import csv
import shutil
import sys
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
REPO = Path(__file__).resolve().parents[1]


def make_custom_trainer(prox_mu, global_state_dict, fixed_class_weights):
    """Build a DetectionTrainer subclass with FedProx's optimizer and/or
    fixed class weights swapped in, as needed. Captures everything via
    closure rather than class attributes, since each subprocess run is
    a fresh process anyway but this avoids any risk of state leaking."""
    from fedprox import ProxAdamW

    class CustomTrainer(DetectionTrainer):
        def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9,
                             decay=1e-5, iterations=1e5):
            if prox_mu <= 0:
                return super().build_optimizer(model, name, lr, momentum, decay, iterations)
            trainable = [p for p in model.parameters() if p.requires_grad]
            trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
            global_params = [global_state_dict[n].to(trainable[0].device)
                              for n in trainable_names]
            return ProxAdamW(trainable, global_params, prox_mu, lr=lr,
                              betas=(momentum, 0.999), weight_decay=decay)

        def set_class_weights(self):
            if fixed_class_weights is None:
                return super().set_class_weights()
            self.model.class_weights = torch.tensor(
                fixed_class_weights, device=self.device)
            print(f"CLASS_WEIGHTS applied to trainer.model: {fixed_class_weights}")

    return CustomTrainer


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
    ap.add_argument("--prox-mu", type=float, default=0.0,
                    help="FedProx proximal term strength (0 = plain FedAvg, "
                         "the default; try 0.01-0.1 to start)")
    ap.add_argument("--class-weights", default=None,
                    help="comma-separated FIXED per-class loss weights "
                         "(e.g. '1,1,1,0,0,0,0,0,0,0' zeroes out classes "
                         "3-9). Overrides Ultralytics' automatic --cls_pw "
                         "weighting. Default None = normal training.")
    ap.add_argument("--run-dir", default="federated/experiments/runs_local")
    args = ap.parse_args()

    if args.freeze >= 23:
        sys.exit(f"ERROR: --freeze {args.freeze} would freeze the entire "
                  f"model (layer 23 is the detection head) — no parameters "
                  f"would be trainable. This is almost certainly a mistake; "
                  f"pass a smaller --freeze value or --freeze 0.")

    data, init, out = Path(args.data), Path(args.init), Path(args.out)
    if not data.is_file():
        sys.exit(f"ERROR: data yaml not found: {data}")
    if not init.is_file():
        sys.exit(f"ERROR: init checkpoint not found: {init}")
    client_name = data.parent.name
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
        val=False,
        plots=False,
        cache=False,
        workers=2,
        project=str(run_dir),
        name=client_name,
        exist_ok=True,
        verbose=True,
    )
    if args.continuous_lr:
        train_kwargs.update(
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=1.0,
            warmup_epochs=0.0,
            cos_lr=False,
        )

    fixed_class_weights = None
    if args.class_weights:
        fixed_class_weights = [float(w) for w in args.class_weights.split(",")]
        if len(fixed_class_weights) != 10:
            sys.exit(f"ERROR: --class-weights must have exactly 10 values "
                     f"(got {len(fixed_class_weights)}): {args.class_weights}")

    model = YOLO(str(init))

    n_total = sum(p.numel() for p in model.model.parameters())
    n_trainable = sum(
        p.numel() for name, p in model.model.named_parameters()
        if not (args.freeze > 0
                and any(name.startswith(f"model.{j}.") for j in range(args.freeze)))
    )
    print(f"CLIENT_INFO client={client_name} freeze={args.freeze} "
          f"trainable={n_trainable:,}/{n_total:,} "
          f"({100 * n_trainable / n_total:.1f}%) prox_mu={args.prox_mu} "
          f"class_weights={fixed_class_weights}")
    if n_trainable == 0:
        sys.exit(f"ERROR: 0 trainable parameters for client={client_name} "
                  f"with freeze={args.freeze} — refusing to train nothing.")

    if args.prox_mu > 0 or fixed_class_weights is not None:
        global_state = None
        if args.prox_mu > 0:
            # capture the global model's weights BEFORE any local training
            # happens — this is the fixed reference point ProxAdamW pulls
            # each step back toward
            global_state = {k: v.detach().clone()
                             for k, v in model.model.state_dict().items()}
        model.train(trainer=make_custom_trainer(args.prox_mu, global_state, fixed_class_weights),
                    **train_kwargs)
    else:
        model.train(**train_kwargs)

    trained = run_dir / client_name / "weights/last.pt"
    if not trained.is_file():
        sys.exit(f"ERROR: expected trained weights not found: {trained}")

    results_csv = run_dir / client_name / "results.csv"
    box_loss = cls_loss = dfl_loss = "None"
    if results_csv.is_file():
        with results_csv.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            box_loss = last.get("train/box_loss", "None").strip()
            cls_loss = last.get("train/cls_loss", "None").strip()
            dfl_loss = last.get("train/dfl_loss", "None").strip()
    print(f"LOSS box_loss={box_loss} cls_loss={cls_loss} dfl_loss={dfl_loss}")

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained, out)
    print(f"CLIENT_DONE name={client_name} weights={out}")
if __name__ == "__main__":
    main()
