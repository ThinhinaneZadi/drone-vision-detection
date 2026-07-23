"""
server.py — federated round loop: sequential client training (subprocesses),
sample-weighted FedAvg aggregation, and clean-global-val evaluation.

Features:
- --freeze N: layer freezing during local training
- --continuous-lr: fixed small lr0, no per-round warmup restart (required
  for many-round schedules)
- --epoch-schedule: vary local epochs across rounds, e.g.
  "1-35:1,36-50:2" means rounds 1-35 use 1 epoch, 36-50 use 2 epochs
- --resume: continue an existing experiment from its last completed round
  (reads config.json/metrics.json, finds the last saved global checkpoint)
- Every round: saves metrics.json (all eval results), communication_log.json,
  round{N}/client_losses.json, and round_summary.csv (one row per round with
  loss + accuracy + params + communication in a single table)
"""

from pathlib import Path
import argparse
import csv
import json
import re
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
    lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)

def parse_loss_line(output: str) -> dict:
    m = re.search(r"LOSS box_loss=(\S+) cls_loss=(\S+) dfl_loss=(\S+)", output)
    if not m:
        return {"box_loss": None, "cls_loss": None, "dfl_loss": None}
    def parse(v):
        return None if v == "None" else float(v)
    return {"box_loss": parse(m.group(1)), "cls_loss": parse(m.group(2)),
            "dfl_loss": parse(m.group(3))}

def parse_epoch_schedule(spec: str, total_rounds: int) -> dict:
    schedule = {r: 1 for r in range(1, total_rounds + 1)}
    if not spec:
        return schedule
    for segment in spec.split(","):
        rng, ep = segment.split(":")
        ep = int(ep)
        if "-" in rng:
            start, end = (int(x) for x in rng.split("-"))
        else:
            start = end = int(rng)
        for r in range(start, end + 1):
            if r <= total_rounds:
                schedule[r] = ep
    return schedule

def find_last_completed_round(exp_dir: Path) -> int:
    last = 0
    for d in exp_dir.glob("round*"):
        if not d.is_dir():
            continue
        try:
            n = int(d.name.replace("round", ""))
        except ValueError:
            continue
        if (d / f"global_round{n}.pt").is_file():
            last = max(last, n)
    return last

def write_round_summary_row(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[20, 50, 100], required=True)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--clients", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--epoch-schedule", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--continuous-lr", action="store_true")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--init", default="models/best_yolo11s_visdrone.pt")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--keep-client-ckpts", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.rounds >= 10 and not args.continuous_lr:
        print(f"WARNING: {args.rounds} rounds requested without "
              f"--continuous-lr — every round will restart LR warmup.")

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
    epoch_sched = parse_epoch_schedule(args.epoch_schedule, args.rounds)
    if args.epoch_schedule:
        for r, ep in sorted(epoch_sched.items()):
            if ep != epoch_sched.get(r - 1, ep):
                print(f"epoch schedule: from round {r} onward, epochs={ep}")

    eval_yaml = exp_dir / "eval.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    clean_val = part_root / "clean_val.txt"

    param_info = count_trainable_params(init_ckpt, args.freeze)
    round_summary_csv = exp_dir / "round_summary.csv"

    start_round = 1
    metrics = []
    comm_log = []
    global_ckpt = init_ckpt

    if args.resume:
        if not exp_dir.is_dir():
            sys.exit(f"ERROR: --resume given but experiment dir not found: {exp_dir}")
        last_round = find_last_completed_round(exp_dir)
        if last_round == 0:
            sys.exit(f"ERROR: --resume given but no completed rounds found in {exp_dir}")
        with (exp_dir / "metrics.json").open() as f:
            metrics = json.load(f)
        comm_log_path = exp_dir / "communication_log.json"
        if comm_log_path.is_file():
            with comm_log_path.open() as f:
                comm_log = json.load(f)
        global_ckpt = exp_dir / f"round{last_round}" / f"global_round{last_round}.pt"
        start_round = last_round + 1
        print(f"RESUMING from round {start_round} "
              f"(last completed: round {last_round}, ckpt: {global_ckpt})")
        if start_round > args.rounds:
            sys.exit(f"Nothing to do: last completed round {last_round} "
                     f">= requested rounds {args.rounds}")
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)
        eval_yaml.write_text(f"train: {clean_val}\nval: {clean_val}\n"
                             f"names:\n{names_block}\n")
        print(f"parameters: {param_info['trainable']:,} trainable / "
              f"{param_info['total']:,} total "
              f"({param_info['trainable_pct']:.1f}%), "
              f"~{param_info['trainable_bytes'] / 1e6:.1f} MB "
              f"communicated/round (fp32)")
        config = vars(args) | {"clients": gids,
                               "sample_counts": {g: counts[g] for g in gids},
                               "param_info": param_info,
                               "epoch_schedule_resolved": epoch_sched}
        (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"experiment: {exp_name}\nclients: "
              f"{', '.join(f'{g}({counts[g]} imgs)' for g in gids)}")
        metrics = [evaluate(init_ckpt, eval_yaml, "initial_global")]

    if not eval_yaml.is_file():
        eval_yaml.write_text(f"train: {clean_val}\nval: {clean_val}\n"
                             f"names:\n{names_block}\n")

    for rnd in range(start_round, args.rounds + 1):
        rdir = exp_dir / f"round{rnd}"
        rdir.mkdir(exist_ok=True)
        client_ckpts = []
        client_losses = {}
        t_round0 = time.time()
        rnd_epochs = epoch_sched.get(rnd, args.epochs)

        for g in gids:
            out = rdir / f"client_{g}.pt"
            print(f"\n--- round {rnd}/{args.rounds} (epochs={rnd_epochs}): "
                  f"training client_{g} ({counts[g]} images) ---")
            t0 = time.time()
            cmd = [sys.executable, str(REPO / "federated/client_train.py"),
                   "--data", str(part_root / f"client_{g}" / "data.yaml"),
                   "--init", str(global_ckpt),
                   "--out", str(out),
                   "--epochs", str(rnd_epochs),
                   "--batch", str(args.batch),
                   "--freeze", str(args.freeze)]
            if args.continuous_lr:
                cmd += ["--continuous-lr", "--lr0", str(args.lr0)]
            rc, output = run_client_streaming(cmd)
            if rc != 0 or not out.is_file():
                sys.exit(f"ERROR: client_{g} training failed (round {rnd}, "
                         f"exit code {rc})")
            client_losses[g] = parse_loss_line(output)
            print(f"client_{g} done in {time.time() - t0:.0f}s "
                  f"(loss: box={client_losses[g]['box_loss']} "
                  f"cls={client_losses[g]['cls_loss']} "
                  f"dfl={client_losses[g]['dfl_loss']})")
            client_ckpts.append(out)

        comm_bytes_round = 2 * len(gids) * param_info["trainable_bytes"]
        comm_log.append({"round": rnd, "epochs": rnd_epochs,
                         "comm_bytes": comm_bytes_round})

        do_eval = (rnd % args.eval_every == 0) or (rnd == args.rounds)
        if do_eval:
            for g, ck in zip(gids, client_ckpts):
                metrics.append(evaluate(ck, eval_yaml, f"round{rnd}_client_{g}"))

        states = [get_state(ck) for ck in client_ckpts]
        weights = [float(counts[g]) for g in gids]
        agg = fedavg(states, weights)
        new_global = rdir / f"global_round{rnd}.pt"
        save_aggregated(init_ckpt, agg, new_global)

        global_metric = None
        if do_eval:
            global_metric = evaluate(new_global, eval_yaml, f"round{rnd}_global")
            metrics.append(global_metric)
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
        (rdir / "client_losses.json").write_text(
            json.dumps(client_losses, indent=2))

        box_vals = [v["box_loss"] for v in client_losses.values() if v["box_loss"] is not None]
        cls_vals = [v["cls_loss"] for v in client_losses.values() if v["cls_loss"] is not None]
        dfl_vals = [v["dfl_loss"] for v in client_losses.values() if v["dfl_loss"] is not None]
        write_round_summary_row(round_summary_csv, {
            "round": rnd,
            "epochs": rnd_epochs,
            "n_clients": len(gids),
            "trainable_params": param_info["trainable"],
            "total_params": param_info["total"],
            "trainable_pct": round(param_info["trainable_pct"], 2),
            "comm_MB_this_round": round(comm_bytes_round / 1e6, 2),
            "avg_box_loss": round(sum(box_vals) / len(box_vals), 5) if box_vals else None,
            "avg_cls_loss": round(sum(cls_vals) / len(cls_vals), 5) if cls_vals else None,
            "avg_dfl_loss": round(sum(dfl_vals) / len(dfl_vals), 5) if dfl_vals else None,
            "evaluated": do_eval,
            "global_precision": global_metric["precision"] if global_metric else None,
            "global_recall": global_metric["recall"] if global_metric else None,
            "global_mAP50": global_metric["mAP50"] if global_metric else None,
            "global_mAP50_95": global_metric["mAP50_95"] if global_metric else None,
            "elapsed_seconds": round(elapsed),
        })
        print(f"round_summary.csv updated ({rnd} rows)")

    total_comm = sum(r["comm_bytes"] for r in comm_log)
    print(f"\ntotal communicated (fp32, upload+download): "
          f"{total_comm / 1e9:.2f} GB over {len(comm_log)} rounds")

    print("\n===== summary (global only) =====")
    for m in metrics:
        if "global" in m["model"] or m["model"] == "initial_global":
            print(f"{m['model']:<28} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f}")
    print(f"\nresults: {exp_dir}")
    print(f"full per-round table: {round_summary_csv}")

if __name__ == "__main__":
    main()
