"""
server.py — federated round loop: sequential client training (subprocesses),
sample-weighted FedAvg aggregation, and clean-global-val evaluation.
Supports layer freezing (--freeze N) and continuous-LR mode for many-round
schedules (--continuous-lr). Logs trainable/communicated parameter counts.
Client subprocess output streams live (not buffered) to avoid pipe deadlock.
"""

from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time

import torch
from ultralytics import YOLO

from model_utils import get_state, fedavg, save_aggregated, count_trainable_params

REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]

def evaluate(ckpt: Path, eval_yaml: Path, tag: str) -> dict:
    model = YOLO(str(ckpt))
    r = model.val(data=str(eval_yaml), batch=2, imgsz=640, device="0",
                  plots=False, verbose=False)
    del model
    torch.cuda.empty_cache()
    m = {"model": tag,
         "precision": round(float(r.box.mp), 4),
         "recall": round(float(r.box.mr), 4),
         "mAP50": round(float(r.box.map50), 4),
         "mAP50_95": round(float(r.box.map), 4)}
    print(f"EVAL {tag}: P={m['precision']} R={m['recall']} "
          f"mAP50={m['mAP50']} mAP50-95={m['mAP50_95']}")
    return m

def run_client_streaming(cmd: list[str]) -> tuple[int, str]:
    """Run a client subprocess, streaming its output live to stdout while
    also collecting it (for error reporting on failure). Avoids the pipe
    deadlock that subprocess.run(capture_output=True) can hit."""
    lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[20, 50, 100], required=True)
    ap.add_argument("--partition", default=None,
                    help="partition directory name under experiments/partitions/ "
                         "(default: tier{tier})")
    ap.add_argument("--clients", required=True,
                    help="comma-separated group ids, or 'all' for every "
                         "client in the tier")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N model layers in local training")
    ap.add_argument("--continuous-lr", action="store_true",
                    help="disable per-round warmup; use fixed small lr0 "
                         "(recommended for --rounds >= ~10)")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--init", default="models/best_yolo11s_visdrone.pt")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--keep-client-ckpts", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate global model every N rounds (default: "
                         "every round; use e.g. 5 for long 50-100 round runs "
                         "to save time)")
    args = ap.parse_args()

    if args.rounds >= 10 and not args.continuous_lr:
        print(f"WARNING: {args.rounds} rounds requested without "
              f"--continuous-lr — every round will restart LR warmup, "
              f"which is very likely to distort results at this length.")

    part_name = args.partition or f"tier{args.tier}"
    part_root = REPO / f"federated/experiments/partitions/{part_name}"
    if not part_root.is_dir():
        sys.exit(f"ERROR: partition not found: {part_root}")
    init_ckpt = (REPO / args.init).resolve()
    if not init_ckpt.is_file():
        sys.exit(f"ERROR: init checkpoint not found: {init_ckpt}")

    counts = {}
    with (part_root / "partition_summary.csv").open() as fh:
        for row in csv.DictReader(fh):
            counts[row["group_id"]] = int(row["n_train_images"])

    if args.clients.strip().lower() == "all":
        gids = sorted(counts)
    else:
        gids = [g.strip() for g in args.clients.split(",") if g.strip()]
    for g in gids:
        if g not in counts:
            sys.exit(f"ERROR: group {g} not in partition {part_name}")
        if not (part_root / f"client_{g}" / "data.yaml").is_file():
            sys.exit(f"ERROR: missing data.yaml for client_{g}")

    exp_name = args.exp_name or f"{part_name}_c{len(gids)}_r{args.rounds}_e{args.epochs}"
    exp_dir = REPO / "federated/experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    eval_yaml = exp_dir / "eval.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    clean_val = part_root / "clean_val.txt"
    eval_yaml.write_text(f"train: {clean_val}\nval: {clean_val}\n"
                         f"names:\n{names_block}\n")

    param_info = count_trainable_params(init_ckpt, args.freeze)
    print(f"parameters: {param_info['trainable']:,} trainable / "
          f"{param_info['total']:,} total "
          f"({param_info['trainable_pct']:.1f}%), "
          f"~{param_info['trainable_bytes'] / 1e6:.1f} MB communicated/round "
          f"(fp32)")

    config = vars(args) | {"clients": gids,
                           "sample_counts": {g: counts[g] for g in gids},
                           "param_info": param_info}
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"experiment: {exp_name}\nclients: "
          f"{', '.join(f'{g}({counts[g]} imgs)' for g in gids)}")

    metrics = [evaluate(init_ckpt, eval_yaml, "initial_global")]
    global_ckpt = init_ckpt
    comm_log = []

    for rnd in range(1, args.rounds + 1):
        rdir = exp_dir / f"round{rnd}"
        rdir.mkdir(exist_ok=True)
        client_ckpts = []
        t_round0 = time.time()

        for g in gids:
            out = rdir / f"client_{g}.pt"
            print(f"\n--- round {rnd}: training client_{g} "
                  f"({counts[g]} images) ---")
            t0 = time.time()
            cmd = [sys.executable, str(REPO / "federated/client_train.py"),
                   "--data", str(part_root / f"client_{g}" / "data.yaml"),
                   "--init", str(global_ckpt),
                   "--out", str(out),
                   "--epochs", str(args.epochs),
                   "--batch", str(args.batch),
                   "--freeze", str(args.freeze)]
            if args.continuous_lr:
                cmd += ["--continuous-lr", "--lr0", str(args.lr0)]
            rc, output = run_client_streaming(cmd)
            if rc != 0 or not out.is_file():
                sys.exit(f"ERROR: client_{g} training failed (round {rnd}, "
                         f"exit code {rc})")
            print(f"client_{g} done in {time.time() - t0:.0f}s")
            client_ckpts.append(out)

        comm_bytes_round = 2 * len(gids) * param_info["trainable_bytes"]
        comm_log.append({"round": rnd, "comm_bytes": comm_bytes_round})

        do_eval = (rnd % args.eval_every == 0) or (rnd == args.rounds)
        if do_eval:
            for g, ck in zip(gids, client_ckpts):
                metrics.append(evaluate(ck, eval_yaml, f"round{rnd}_client_{g}"))

        states = [get_state(ck) for ck in client_ckpts]
        weights = [float(counts[g]) for g in gids]
        agg = fedavg(states, weights)
        new_global = rdir / f"global_round{rnd}.pt"
        save_aggregated(init_ckpt, agg, new_global)

        if do_eval:
            metrics.append(evaluate(new_global, eval_yaml, f"round{rnd}_global"))
        global_ckpt = new_global

        if not args.keep_client_ckpts:
            for ck in client_ckpts:
                ck.unlink()

        elapsed = time.time() - t_round0
        print(f"round {rnd}/{args.rounds} done in {elapsed:.0f}s"
              + (" (evaluated)" if do_eval else " (eval skipped)"))

        (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (exp_dir / "communication_log.json").write_text(
            json.dumps(comm_log, indent=2))

    total_comm = sum(r["comm_bytes"] for r in comm_log)
    print(f"\ntotal communicated (fp32, upload+download): "
          f"{total_comm / 1e9:.2f} GB over {args.rounds} rounds")

    print("\n===== summary =====")
    for m in metrics:
        print(f"{m['model']:<28} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f}")
    print(f"\nresults: {exp_dir}")

if __name__ == "__main__":
    main()
