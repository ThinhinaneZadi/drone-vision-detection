# Federated Learning Experiments — Results Summary

Project: location-based federated object detection (VisDrone-DET, YOLO11s)
Branch: federated-learning
All metrics measured on the clean global validation set (520 images, 52
capture groups, zero group overlap with any client training data).

## Dataset audit (script-verified: inspect_locations.py)

- 208 distinct capture groups in DET-train (filename-prefix location proxies;
  city labels are not recoverable from the public dataset)
- 24 groups overlap the official train/val split (297 train / 28 val images,
  all video-derived, all below client-size thresholds) — excluded
- Client eligibility tiers: 45 clients (>=20 img, 83% of data),
  30 (>=50, 77%), 19 (>=100, 64%)
- Experiments below use the 19-client tier-100 partition (all still-capture
  groups, 101–434 images each)

## Main comparison table (clean val, mAP50)

| Model | mAP50 | Notes |
|---|---|---|
| COCO-10 init (no VisDrone exposure) | 0.027 | Option B starting line |
| Best single client, Option B round 5 | 0.147 | single-location ceiling |
| Federated, 19 clients, 5 rounds x 2 epochs (Option B) | 0.171 | still climbing at round 5 |
| Centralized control, union of 19 clients, 10 epochs | 0.250 | same init, matched epochs |
| Best fine-tuned baseline (full VisDrone-train) | 0.387 | practical upper reference |

Federation cost at matched epochs: 0.079 mAP50 (~68% of centralized
performance retained) under location-based non-IID.

## Option A — init from best fine-tuned model (0.387 on clean val)

Measures forgetting/recovery, not learning (init already saw all client data).

- 3 clients, 5 rounds x 1 epoch: global 0.367 -> 0.362 -> 0.360 -> 0.355 ->
  0.351 (client drift toward the 3-location subset optimum)
- All 19 clients, 3 rounds x 1 epoch: global 0.383 / 0.381 / 0.378 —
  full participation nearly preserves the initial model
- Aggregate beat every individual client model in every round (57/57)

## Option B — init from COCO-pretrained YOLO11s, fresh 10-class head

Measures genuine federated learning (init never saw VisDrone).

- Global trajectory: 0.027 -> 0.100 -> 0.132 -> 0.146 -> 0.161 -> 0.171
- Rounds 1–2: largest clients (9999999, 9999998) match/beat the aggregate;
  from round 3 the aggregate pulls decisively ahead (round 5: 0.171 vs
  best client 0.147) — cross-location knowledge accumulates in the average

## Non-IID analysis links to outcomes

- Pre-training KL divergence vs global class distribution predicted
  forgetting: client_9999960 (KL=0.671, highest) was the worst client in
  every Option A round (0.323/0.322/0.321)
- client_9999940 (101 images + KL=0.436) was the weakest Option B learner
- Per-class results mirror class imbalance: car mAP50 0.709 vs bicycle 0.043
  (centralized control)

## Caveats and known limitations

1. Option A initialization saw all client data centrally; its runs measure
   preservation, not learning (documented by design).
2. Every federated round restarts Ultralytics LR warmup (warmup_epochs=3 >
   local epochs), so federated training never reached full LR; the
   centralized control ran one continuous schedule. The 0.079 gap is
   therefore an upper-bound estimate at this budget.
3. Both federated and centralized curves were still rising when stopped.
4. Capture groups are location *proxies* (capture sessions), not verified
   distinct geographic locations. Scene categories, where used, are
   project-defined annotations, not official VisDrone metadata.
5. Clean val (0.387 for the baseline) is not comparable to full official
   val (0.464) — the official split leaks 24 capture groups across
   train/val; both numbers are reported deliberately.
