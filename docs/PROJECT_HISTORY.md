# Federated Object Detection for Drones — Complete Project History & Handoff Document

**Purpose of this document:** this is a full reconstruction guide. If every past conversation and memory of this project were lost, reading this document alone should let you (or anyone else) understand exactly what was built, why every decision was made, what has been verified to work, what is still broken or unfinished, and exactly what to do next. Nothing here is invented — everything is drawn from real, verified work.

---

## 1. The Big Picture: What This Project Is

**Researcher:** Thinhinane Zadi ("Tina"), AFRL Summer 2026 Intern, mentored by Dr. Simon Khan.

**The core idea:** simulate a swarm of drones, each with its own camera and its own local images, that collaboratively train one shared object-detection model **without ever sending raw images to a central server**. This is called **federated learning**. Instead of centralizing data, each "drone" (client) trains a copy of the model on its own local images, and only the *trained model itself* (not the images) gets sent back and averaged together with every other drone's trained model. This repeats over many rounds.

**Why this matters for real drones:** real drone swarms often can't or shouldn't send raw video back to a base station — bandwidth is limited, connections are unreliable, and there may be privacy/security reasons to keep raw footage on-device. Federated learning is a way to still get a smart, shared model without that centralization.

**The dataset:** VisDrone2019-DET, a public dataset of drone-captured images labeled with 10 object classes: `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`.

**The model:** YOLO11s (a real-time object detector), 9,431,662 parameters total.

**The repository:** `github.com/ThinhinaneZadi/drone-vision-detection`, branch `federated-learning`.

---

## 2. Phase 0 — Before Federated Learning: The Centralized Baseline

Before any federated work began, a strong centralized (normal, non-federated) YOLO11s model was fine-tuned on VisDrone through a separate series of experiments (frozen-head training, resolution adaptation, etc.). The best result was saved as:

**`models/best_yolo11s_visdrone.pt`**
- DET validation: Precision 0.579, Recall 0.469, **mAP50 0.464**, mAP50-95 0.280
- Video evaluation: P 0.516, R 0.551, mAP50 0.518, mAP50-95 0.272
- This model was trained/validated at **960×960 resolution** — a fact that turned out to be extremely important later (see Section 8).

This checkpoint became the standard starting point ("Option A" — see Section 4) for most federated experiments.

---

## 3. Building the Federated Pipeline — The Core Scripts

Four scripts form the backbone of every federated experiment. All live in `federated/`.

### 3.1 `partition_by_location.py` — turning VisDrone into simulated "drones"

VisDrone doesn't come with GPS/location labels. Instead, each image's filename has a numeric **capture-group prefix** (e.g., `9999966_00000_d_0000001.jpg`) — images sharing a prefix came from the same capture session. This prefix was used as a **proxy for "which drone/location this came from."**

- Only capture groups with **≥100 images** were used as clients (the "tier-100" eligibility threshold) — this gave **19 real clients**.
- Client sample counts range from 86 to 434 images each.
- **Leakage prevention:** any capture group whose images appeared in *both* the official VisDrone train and validation splits was excluded entirely from the shared validation set, so no client's training images could leak into the evaluation set. This excluded 24 overlap groups, leaving a clean **520-image shared validation set**.
- A `--local-val-frac` option (added later) can additionally hold out a slice of each client's own images (e.g., every 7th image, ~15%) into a private `local_val.txt`, for testing whether a client's own training actually helps *that client specifically* — this produces the `tier100_lv15` partition variant.
- Output: one folder per client (`client_<id>/`), each with a `train.txt` (list of that client's image paths) and a `data.yaml` (Ultralytics config pointing at those paths).

### 3.2 `model_utils.py` — the FedAvg aggregation math

This is where the actual "federated averaging" happens.

- **`get_state(path)`**: loads a checkpoint's raw weights as float32 tensors.
- **`fedavg(states, weights)`**: for floating-point tensors (all real learnable weights, plus BatchNorm running statistics), computes a **sample-count-weighted average** across clients — a client with more training images gets proportionally more influence. For integer tensors (BatchNorm's `num_batches_tracked` counter), the value is **copied** from whichever client had the largest weight, never averaged (averaging an integer counter would corrupt BatchNorm behavior).
- **`save_aggregated(base_path, state, out_path)`**: injects the averaged weights into a copy of the checkpoint, and explicitly strips any stale `ema`/`optimizer`/`updates` fields so the aggregated weights are actually what gets used on the next load.
- **`count_trainable_params(ckpt, freeze_n)`**: counts how many of the model's 9,431,662 parameters are actually trainable at a given freeze depth (see Section 4).
- This file includes its own unit tests (`--selftest`), including a test that `FedAvg(X,X) == X` (averaging a model with itself should return the same model, unchanged) and a test confirming BatchNorm counters are copied, not averaged.

### 3.3 `client_train.py` — trains ONE client, then exits

Launched as a fresh subprocess for every single client, every single round (not one long-running process) — this matters because it fully releases GPU memory between clients, critical on memory-constrained GPUs.

Core behavior: load the current global checkpoint, fine-tune it on this client's local images only, save the result, print a summary line, exit.

**Key flags (as of the current version):**
- `--freeze N`: freezes the first N layers of the network (they don't receive gradient updates) — see Section 4.
- `--continuous-lr`: uses a small, fixed learning rate with no per-round warmup restart — necessary for many-round (50+) schedules, since restarting LR warmup every single round would prevent the model from ever settling into stable training.
- `--prox-mu`: FedProx proximal term strength — see Section 9.
- `--class-weights`: fixed per-class loss weights for the partial-label-space experiment — see Section 10.
- `--imgsz`: image size for training (see Section 8 for why this mattered).

**Two safety mechanisms added after a real bug was found (see Section 6):**
- Refuses to run at all if `--freeze` would leave **zero** trainable parameters (this exact situation caused a 36-round-long silent failure once — see Section 6).
- Prints the exact trainable parameter count and percentage before every round's training begins, so a misconfiguration is visible immediately in the logs instead of hiding for many rounds.

### 3.4 `server.py` — the round loop that ties everything together

For each round: sends the current global model to every client (sequentially, via subprocess), collects their trained results, evaluates each client individually plus the aggregate on the shared clean validation set, runs FedAvg to produce the new global model, saves a checkpoint, and logs everything to disk.

**Logged every round**, to `round_summary.csv`: round number, local epochs used, image size, trainable parameter count, communication cost (MB), exact mini-batch step count, average training loss (box/cls/dfl), global precision/recall/mAP50/mAP50-95, wall-clock time.

**Key flags added over the project's lifetime:**
- `--epoch-schedule "1-35:1,36-50:2"`: lets local epochs vary across rounds (e.g., 1 epoch for most of the run, then increase to 2 partway through, to test stability under a training-intensity change).
- `--resume`: continue an interrupted experiment from its last completed round, by reading the saved config/metrics and finding the last saved checkpoint.
- `--imgsz`: passed through to training AND evaluation (originally hardcoded to 640 — see Section 8).
- `--use-profile-class-weights`: for partial-label-space experiments, automatically looks up each client's assigned mission profile and passes the correct class-weight string — see Section 10.

---

## 4. The Two Experimental Paths

**Option A — preserve an already-good model.** Every Option A experiment initializes from `best_yolo11s_visdrone.pt` (0.464 mAP50). The question: can federated training preserve this quality as multiple clients train on it locally, or does it degrade ("drift")?

**Option B — learn from scratch.** Initializes from a COCO-pretrained YOLO11s with a **freshly initialized detection head** — a model that has never seen a single VisDrone image, starting at near-zero VisDrone accuracy. Built by `federated/prepare_option_b.py`: transfers 493 of 499 tensors from COCO, freshly (seed=0) initializes the remaining 6 head tensors (since VisDrone's 10 classes don't match COCO's 80), saves as `models/init_yolo11s_coco10.pt`. The question: can federated learning actually *build* a good detector, not just preserve one?

**Freezing, as a drift-control technique (used throughout Option A):** freezing the first N layers means those layers' weights never update during local training — only the unfrozen later layers (closer to the detection head) adapt. This is a way to limit how far a client's local training can pull the model away from the shared starting point. Three depths were tested:
- `freeze=0`: full fine-tuning, 9,431,662 / 9,431,662 trainable (100%)
- `freeze=11`: 3,989,678 / 9,431,662 trainable (42.3%)
- `freeze=22`: 2,334,702 / 9,431,662 trainable (24.75%) — only the last two structural blocks (a C3k2 block and the detection head) remain trainable

---

## 5. Complete Experiment History (Part A — Short/Local Experiments)

All seven ran on Generation-1 scripts (before the freeze-safety guard and trainable-param logging existed), all on a local laptop GPU (GTX 1650, 4GB VRAM), all at **imgsz=640** (the significance of this is explained in Section 8).

| # | Experiment | Setup | Result |
|---|---|---|---|
| 1 | `tier100_c3_r1_e1` | 3 clients, 1 round, freeze=0 | Smoke test. mAP50 0.3872→0.3675. Aggregate beat every individual client — first proof FedAvg was combining information constructively. |
| 2 | `tier100_c3_r5_e1` | Same 3 clients, 5 rounds | mAP50 declined every round to 0.3512 (drop of 0.0360) — real drift with narrow, 3-client participation. |
| 3 | `tier100_c19_r3_e1` | 19 clients, 3 rounds, freeze=0 | mAP50 declined only to 0.3779 (drop of 0.0093) — roughly 4-5x smaller per-round decline than 3 clients. First evidence broader participation reduces drift. |
| 4 | `tier100_c19_r3_e1_freeze11` | Same 19 clients, freeze=11 | mAP50 to 0.3827 (drop of only 0.0045) — about half the unfrozen decline. First evidence freezing further reduces drift. |
| 5 | `optionB_tier100_c19_r5_e2` | 19 clients, 5 rounds, Option B init, 2 epochs | mAP50 ROSE every round: 0.027→0.171. First proof FedAvg can learn from scratch, not just preserve. |
| 6 | `lv15_c19_r1_e1_freeze11` | 19 clients, lv15 partition, 1 round, freeze=11 | Tested "local adaptation": does 1 round help a client on its OWN held-out data? Average delta was -0.0098 (net negative) — inconclusive/negative at 1 epoch. |
| 7 | `lv15_c19_r1_e1_freeze22` | Same, freeze=22 | Average local-adaptation delta +0.0022 — near flat. Neither freeze depth showed clear specialization at 1 epoch — still an open question. |

---

## 6. Complete Experiment History (Part B — Long-Horizon 50-Round Experiments)

Three experiments, all on a rented RunPod H100 GPU, using `--continuous-lr`, batch=4, epoch-schedule `1-35:1, 36-50:2`, at **imgsz=640** (later identified as a mistake — see Section 8). **Critical limitation: the RunPod pod these ran on was later terminated, which deleted its storage volume — only each experiment's `round_summary.csv` survived (recovered from local downloads); the exact server.py version, config.json, and full communication logs for these three specific runs are permanently unrecoverable.**

### 6.1 `real_c19_r50_freeze11_schedule`
mAP50 declined gradually and smoothly across all 50 rounds: 0.3786 (round 1) → 0.3589 (round 50), a drop of only 0.0197. No instability at the round-36 schedule change. Communication: 606.43 MB/round (~29.6 GB total).

### 6.2 `real_c19_r50_freeze0_schedule`
Direct comparison, no freezing. mAP50 reached only 0.3506 at round 50 — **0.0083 worse than freeze=11**, despite an identical number of training steps. Communication: 1,433.61 MB/round — **2.36x more** than freeze=11. **This is the strongest complete finding of the whole project: freeze=11 beats full fine-tuning on BOTH accuracy and communication cost simultaneously, at the same compute cost.**

### 6.3 `optionB_c19_r50_e2_continuous` — THE BROKEN RUN
This was meant to answer the most important open question: does federated learning-from-scratch (Option B) keep improving over a long horizon toward centralized-level accuracy? Instead:

- Rounds 1-14: real learning, mAP50 climbing (0.0287 → 0.0409)
- **Rounds 15-50 (36 rounds): mAP50, precision, and recall became bit-for-bit IDENTICAL every single round** — the model silently stopped learning, but the pipeline kept running and logging normally with no error.

**Root-cause investigation (this is a good example of the debugging discipline used throughout this project):**
1. Two checkpoints (round 10 and round 30) were compared tensor-by-tensor. **All 256 learnable weight/bias tensors were exactly identical.** Only BatchNorm running statistics moved (and only negligibly) — BN stats update automatically from data flow, without needing a real gradient step, so this proved the model had stopped receiving real weight updates.
2. **Hypothesis 1 (tested and DISPROVEN):** maybe the aggregation code was silently loading a stale EMA (exponential moving average) snapshot instead of real trained weights. Tested directly by inspecting a real checkpoint's `ema` field — it was `None`. Ruled out.
3. **Hypothesis 2 (most likely, never confirmed):** a misconfigured `--freeze` value (possibly ≥23, which would freeze the entire model including the head) was accidentally used for this specific run. The exact launch command could not be recovered — it was typed directly into the now-deleted RunPod terminal, never saved anywhere.
4. **Remediation (regardless of root cause):** two safeguards were added to `client_train.py` — a hard refusal to run if `--freeze >= 23`, and mandatory printing of the exact trainable-parameter count before every round. These are now permanent parts of the pipeline (Section 3.3).

**As of this writing, this experiment has never been successfully completed** — every attempted rerun (Colab, a second Windows machine, and finally successfully on RunPod — see Section 7) hit new infrastructure problems before finally succeeding on the current RunPod pods described in Section 8.

---

## 7. Infrastructure Journey — What Broke, Where, and What We Learned

This project moved across many machines over time, and nearly every move surfaced a new class of bug. This history matters because these lessons are now baked into the pipeline as permanent safeguards.

1. **Local laptop (GTX 1650, 4GB VRAM):** the original environment for all Part A experiments. Fine for small-scale/short runs, too weak for 50-round/high-resolution work.
2. **Google Colab (T4 GPU):** used for the first 50-round attempts and later Option B reruns. Repeatedly hit: Drive disconnecting mid-session, `pip install ultralytics` silently downgrading a working CUDA PyTorch install to a CPU-only build (twice), and general session fragility.
3. **A second Windows machine (Quadro T2000, 4GB):** attempted as an alternative. Hit: PyTorch has no wheels for very new Python versions (had to install Python 3.11 alongside the existing 3.14), a disk-space crisis from unrelated large files sharing the drive, the same CUDA-downgrade-via-pip issue, a Windows GPU driver timeout (TDR) crash from running the display and compute on the same GPU, and a `UnicodeDecodeError` crash caused by Ultralytics' emoji output hitting Windows' default non-UTF-8 terminal encoding (fixed by adding `encoding="utf-8", errors="replace"` to the subprocess call in `server.py`).
4. **RunPod (rented cloud GPUs — H100s and one RTX PRO 6000):** the current, working environment. A silent checkpoint-file-vanishing bug was seen once (likely antivirus/endpoint-security interference on a different machine, not RunPod itself) but RunPod itself has been reliable. **`tmux` is used on every RunPod session** to keep training running even if the browser disconnects — sessions are named per-experiment (`fedlearn`, `freeze11`, `freeze22`) so multiple pods can be managed without confusion.

**A key process lesson learned the hard way:** local, uncommitted fixes made directly on one pod's files do NOT automatically appear on a fresh pod — they must be committed and pushed to GitHub first. Early in the RunPod phase, fixes were manually re-pasted onto every new pod; this was fixed by committing everything to the `federated-learning` branch before spinning up new pods, so a plain `git clone` now brings in every fix automatically.

---

## 8. The Resolution Bug — A Significant, Late-Discovered Issue

**The discovery:** `best_yolo11s_visdrone.pt` was trained and validated at **960×960** resolution, achieving 0.464 mAP50 on its own validation set. But **every single federated experiment across this entire project — all 7 Part A experiments and all 3 Part B experiments — evaluated and trained at 640×640**, YOLO's default, because:
1. `client_train.py`'s `--imgsz` flag defaulted to 640 and was never explicitly overridden in any launch command.
2. `server.py`'s `evaluate()` function had **`imgsz=640` hardcoded directly in the code**, not even exposed as a flag — so even the very first `initial_global` baseline evaluation in every experiment (before any federated training happened) was measured at the wrong resolution.

**Why this matters:** every experiment's `initial_global` baseline came out to mAP50 ≈ 0.387, a full 0.077 (about 17% relative) below the model's real 0.464. **Some unknown fraction of every reported "drift" or "degradation" in this entire project's results may actually be an artifact of evaluating at the wrong resolution, not a real property of federated learning.**

**Direct confirmation:** a smoke test of `best_yolo11s_visdrone.pt` at `imgsz=960` (freeze=11, 1 client) scored **mAP50 = 0.458** — almost exactly matching the model's real 0.464 — versus 0.387 at 640px on the same setup otherwise. This confirms the resolution mismatch was a real, substantial factor.

**Fix applied:** `server.py`'s `evaluate()` function now accepts `imgsz` as a parameter instead of hardcoding it, threaded through from a new `--imgsz` command-line flag. Default behavior is unchanged (still 640, so nothing breaks silently) — but every new experiment now explicitly passes `--imgsz 960` to match the model's real training resolution. **ALL new experiments from this point forward must use imgsz=960, no exceptions, including smoke tests, to avoid any ambiguity about which numbers came from which resolution.**

**Consequence:** the three Part B 50-round experiments are currently being **rerun from scratch at imgsz=960** (see Section 8.1) — this is considered necessary for any results to be trustworthy or publication-ready, not optional polish.

### 8.1 The Three Parallel Reruns — Results
--- To avoid waiting ~24+ hours running these sequentially, three separate RunPod 
pods were launched simultaneously (same total GPU-hour cost as sequential, much less 
wall-clock time):
## 9. FedProx — An Additional Drift-Control Technique
| Pod | GPU | Experiment | Settings |
**Why:** freezing (Section 4) controls drift by physically preventing most weights 
from changing — effective, but blunt; it limits the model's ability to adapt at 
all in the frozen portion. **FedProx** (Li et al., 2020, MLSys — a real, 
published, well-established technique, NOT invented in this project) is a more 
surgical alternative: it lets every weight train normally, but adds a mathematical 
penalty term that discourages any single client's weights from drifting too far from 
the global model it started the round with.
|---|---|---|---|
**The math:** during each client's local training, alongside the normal detection 
loss, an extra term is effectively added: Implemented as a gradient modification 
(mathematically equivalent to adding the term to the loss): before every real 
optimizer step, `mu x (current_weight - global_weight)` is added to that weight's 
gradient.
| Pod 1 (`fedlearn` session) | H100 80GB | `optionB_c19_r50_imgsz960_paper` | Option 
| B init, freeze=0, 50 rounds, batch=16, imgsz=960, epoch-schedule 1→2 at round 
| 36 |
**Implementation:** a new file `federated/fedprox.py` defines `ProxAdamW`, a 
subclass of PyTorch's AdamW optimizer with this gradient modification built into its 
`step()` method. `client_train.py` gained a `--prox-mu` flag; when set > 0, it 
captures the model's weights *before* training starts (the fixed reference point) 
and swaps in a custom Ultralytics `DetectionTrainer` subclass whose 
`build_optimizer()` method returns `ProxAdamW` instead of the normal optimizer.
| Pod 2 (`freeze11` session) | H100 80GB | `freeze11_c19_r50_imgsz960_paper` | 
| `best_yolo11s_visdrone.pt` init, freeze=11, otherwise identical settings |
**A real bug found and fixed during development — an important lesson about 
Ultralytics internals:** the very first version appeared to run successfully at 
every tested `--prox-mu` value (0, 0.01, 1.0) but produced **bit-for-bit identical 
results regardless of mu** — clearly wrong, since a strong pull (mu=1.0) should 
visibly change training. Root cause, found via careful debug instrumentation: 
**Ultralytics uses gradient accumulation** — with a small batch size, it only 
calls the optimizer's real `.step()` once every `~64/batch_size` mini-batches, not 
every mini-batch. At `batch=2` with only ~60 total mini-batches in one local epoch, 
a client might get only **one real optimizer step** — and at that first step, 
the model's weights are, by definition, still identical to the global reference, so 
the proximal term is mathematically zero regardless of mu. This isn't a bug in the 
FedProx math — it's a real, previously-unknown-to-us property of how Ultralytics 
trains, which also means "mini-batches processed" and "actual weight updates 
applied" are different numbers throughout this whole project (worth stating 
explicitly in a paper's methods section). **Retested at `batch=16` (the project's 
real setting), the proximal term was confirmed working correctly** — by the 
second real optimizer step, the injected term was comparable in magnitude to the 
normal gradient itself, exactly as intended.
| Pod 3 (`freeze22` session) | RTX PRO 6000 97GB | `freeze22_c19_r50_imgsz960_paper` 
| | `best_yolo11s_visdrone.pt` init, freeze=22, otherwise identical settings |
**Status: implementation complete, debugged, and verified working. A real validation 
run (comparing FedProx against plain FedAvg at freeze=0, 11, and 22) has not yet 
been performed — it is queued behind the three Section 8.1 reruns, waiting for a 
free GPU.** Each pod's setup was smoke-tested before the real run launched. All 
three completed 50 rounds. Results (final, saved permanently to GitHub, including 
the round-50 checkpoint for each): ---

## 10. The Novel Contribution — Federated Detection Under Partial Label Spaces
**freeze=11: SUCCESS.** Final round: P=0.547 R=0.446 
This is intended to be the actual original contribution of a potential paper, not just a re-verification of existing techniques.

### 10.1 The research gap

Standard federated object detection (everything done so far in this project) assumes every client labels every class. Real drone deployments don't work this way: a traffic-monitoring drone has no operational reason to label bicycles; a pedestrian-safety drone has no reason to label buses. When a client's local annotations don't cover every class, objects of the unlabeled classes are still physically present in the images but get treated as unlabeled background during training — actively teaching the model to suppress real objects it was simply never told about. This combination (object *detection*, specifically, under *partial/non-overlapping label spaces*) is a genuinely under-explored combination in the literature.

### 10.2 Designing realistic client "mission profiles"

Rather than restricting classes arbitrarily, real per-client class distributions were measured first, revealing that **8 of 10 classes are present in nearly every client already**, but **tricycle and awning-tricycle are naturally rare or completely absent in several clients** — real, pre-existing non-IID structure, not something artificially introduced.

Five mission profiles were designed and clients assigned based on which profiles they could **actually support** (>=20 real instances of every required class — avoiding confounding the partial-label effect with pre-existing data scarcity):

| Profile | Classes | Real-world justification | Clients assigned |
|---|---|---|---|
| A_traffic | car, van, truck, bus | Traffic-monitoring drone | 9999940, 9999953, 9999962, 9999977 |
| B_pedsafety | pedestrian, people, bicycle, motor | Pedestrian/crosswalk safety drone | 9999945, 9999955, 9999966, 9999981 |
| C_lastmile | tricycle, awning-tricycle, motor, people | Informal transit corridor drone | 9999972, 9999998, 9999999 |
| D_full | all 10 classes | Control group (full coverage) | 9999937, 9999942, 9999982, 9999984 |
| E_vehicle_only | car, van, truck | Highway-only drone (added for an edge case — see below) | 9999951, 9999956, 9999960, 9999994 |

**A genuine edge case, treated as a finding rather than hidden:** client `9999960` is heavily car-dominated (4,744 car instances) with almost nothing else (0 buses, 8 bicycles) — it didn't cleanly fit any of the first four profiles. Rather than force it in with a lowered threshold, a fifth, narrower profile (`E_vehicle_only`) was added specifically to accommodate this kind of real client — the honest conclusion being that not every real client fits a predefined mission profile.

**A real bug found and fixed during this design work:** the first-pass instance counts were computed using `cat file1.txt file2.txt ... | awk ...`, which silently merges the last line of one file with the first line of the next when files don't end in a trailing newline (true of every VisDrone label file). This caused a small, consistent undercount (~1-3%) across all clients. Verified via direct comparison against a correct, per-file Python-based recount — critically, **the undercount never changed which classes were completely absent per client**, so it never actually affected the profile assignment, but the hardcoded counts in the code were corrected for accuracy regardless.

**Files:** `federated/assign_label_profiles.py` (the deterministic assignment logic and corrected per-client counts, plus `profile_to_weight_string()`/`get_client_weight_strings()` helper functions) and `docs/partial_label_profiles.md` (the full methodology writeup, including this bug and why it didn't matter).

### 10.3 The label-filtering pipeline

`federated/build_partial_label_partition.py`: for each client, creates a new partition directory containing `images/` (symlinks to the client's real original images — no copying) and `labels/` (real label files, but with every line whose class isn't in that client's assigned profile removed). Ultralytics automatically finds an image's labels by swapping `/images/` for `/labels/` in the image's own path, so placing filtered labels as a sibling of symlinked images makes YOLO use the filtered version automatically, without touching the original dataset at all. Validation always stays on the full, unfiltered ground truth — the point is measuring restricted training against reality. Also writes `partition_summary.csv` (group_id, n_train_images per client) — this was missing in the first version, which silently blocked `server.py` from being able to read client image counts; caught and fixed before any real run.

**Verified correct:** a `D_full` client (100% classes kept) was confirmed to retain exactly the same label content as the unfiltered source (13,734 = 13,734 lines, via the corrected Python counting method). A `C_lastmile` client's filtered labels were confirmed to contain ONLY its 4 allowed classes (verified via `uniq -c` on the class IDs), zero leakage of excluded classes.

### 10.4 The loss-reweighting fix — the actual novel technical contribution

**The idea:** during a client's local training, don't let the loss function punish the model for predictions in regions that might contain a real, unlabeled object of an excluded class — mask out that class's contribution to the loss entirely for that client.

**Discovery:** Ultralytics' own `v8DetectionLoss` already has a built-in mechanism for this — `self.class_weights`, multiplied directly into the classification loss (`bce_loss *= self.class_weights`) if present. This meant we could reuse an existing, supported mechanism rather than modifying Ultralytics' loss code ourselves.

**A real bug found and fixed — a second important lesson about Ultralytics internals:** the first attempt set `model.model.class_weights = torch.tensor(...)` directly on the loaded YOLO object, *before* calling `.train()`. This ran without error and produced identical loss values regardless of the weights used — again clearly wrong. Root cause, found via `id()` debug instrumentation added temporarily to Ultralytics' own installed `loss.py`: **Ultralytics constructs entirely different internal model objects during training setup** — the object our code set the attribute on was never the same object the loss function later read from (three different Python object IDs were observed across one single training run). The correct fix, discovered by inspecting Ultralytics' own `DetectionTrainer` class: it already has a built-in `set_class_weights()` method (normally used to *auto-compute* weights from class frequency via `--cls_pw`), called at the correct point in training setup, operating on `self.model` (the trainer's own, correct model reference). The fix was to **override this method** in a custom trainer subclass to apply our fixed weights instead of Ultralytics' automatic computation.

**Verified correct:** with 7 of 10 classes zeroed out, `cls_loss` dropped from 1.08037 to 0.59955 (roughly half, consistent with removing most of the classification signal), while `box_loss` and `dfl_loss` were essentially unchanged (1.55->1.56, 0.959->0.961) — exactly the expected pattern, since class weighting should only affect classification loss, not box regression. This internal consistency is strong evidence the mechanism works precisely as intended, not just "does something."

**Files:** `federated/client_train.py` now supports `--class-weights "1,1,1,0,0,0,0,0,0,0"` (comma-separated, one value per class), implemented via the same custom-trainer pattern as FedProx (both live in one `make_custom_trainer()` function, since both need to override different parts of Ultralytics' `DetectionTrainer`).

### 10.5 Wiring it all into server.py for real multi-client runs

`server.py` gained a `--use-profile-class-weights` flag: when set, it calls `get_client_weight_strings()` from `assign_label_profiles.py` once at startup, and automatically passes each client's correct, profile-matched `--class-weights` string when launching that client's training subprocess — no manual per-client specification needed.

**Verified end-to-end at imgsz=960:** a real 2-client test run confirmed client `9999937` (D_full profile) received weights `[1,1,1,1,1,1,1,1,1,1]` while client `9999940` (A_traffic profile) received `[0,0,0,1,1,1,0,0,1,0]` — correctly different, correctly matched to each client's real assigned profile, both applied successfully during real training with no errors.

**Status: the full partial-label-space pipeline (profile design -> label filtering -> loss-reweighting -> automatic server.py wiring) is complete, debugged, and verified working end-to-end. NO actual multi-round experiment has been run yet** — this is the very next piece of work, waiting on a free GPU.

---

## 11. Complete Current Status (as of this document)

| Item | Status |
|---|---|
| Centralized baseline model | Done - 0.464 mAP50 @ 960px |
| Federated pipeline (server/client/model_utils/partition scripts) | Done, debugged, stable |
| Part A short experiments (7 total) | Done - all at imgsz=640 (known limitation) |
| Part B original 50-round experiments (3 total) | Done but flawed - imgsz=640, and Option B run broken (see Section 6.3) |
| Resolution bug (640 vs 960) discovered and fixed | Done |
| Part B reruns at imgsz=960 (3 pods) | IN PROGRESS |
| FedProx implementation | Done, debugged, verified in isolation |
| FedProx real validation run (vs plain FedAvg, all 3 freeze depths) | Not started |
| Partial label-space profile design (19 clients -> 5 profiles) | Done, verified, bug-corrected |
| Label-filtering pipeline (+ partition_summary.csv fix) | Done, verified |
| Loss-reweighting (`--class-weights`) implementation | Done, debugged, verified in isolation |
| Loss-reweighting wired into server.py (`--use-profile-class-weights`) | Done, verified end-to-end |
| Baseline: naive FedAvg under partial labels | Not started |
| Baseline: pseudo-labeling under partial labels | Not started |
| Contribution experiment: loss-reweighting under partial labels | Not started |
| Combined best-drift-method + loss-reweighting experiment | Not started |
| Paper draft | Not started |

---

## 12. Detailed Next Steps

1. **Finish the three imgsz=960 Part B reruns** (Section 8.1) — no action needed, just time; these become the project's real headline Option A/B results once complete.
2. **Run FedProx's real validation** — repeat the freeze=0 / 11 / 22 comparison with `--prox-mu` enabled, to see whether it controls drift as well as or better than freezing, without freezing's adaptation-limiting tradeoff.
3. **Run the partial-label-space baseline (plain FedAvg, no fix)** using the `tier100_partial_labels` partition (`--use-profile-class-weights` OFF) — this is expected to show degraded accuracy on classes most clients don't label, and is the paper's core "motivating problem" result.
4. **Run the partial-label-space baseline (pseudo-labeling)** — the existing literature's typical fix, for comparison. Not yet implemented.
5. **Run the partial-label-space contribution experiment** — same partition, `--use-profile-class-weights` ON — the actual novel result; needs to beat both baselines above to be worth publishing.
6. **Combine the best drift-control method (freeze=11 or FedProx, whichever wins step 2) with loss-reweighting** — ties the whole summer's work into one coherent system rather than a pile of separate experiments.
7. **Write the paper** — motivation -> related work -> federated pipeline & baseline results (steps 1-2) -> partial-label-space method & results (steps 3-6) -> conclusion. Target: a federated-learning workshop (NeurIPS/ICML FL workshop) or a detection-focused venue like WACV, given realistic scope for a summer internship project. FedProx itself is NOT novel (cite Li et al., 2020) — the novelty is in the specific combination: federated *detection*, under *partial label spaces*, on *drone imagery*, with a *loss-reweighting* fix, compared directly against an *architectural* drift-control baseline (freezing).
8. **Every step above should follow the discipline established throughout this project:** smoke-test small before committing real GPU hours, verify with direct evidence rather than assuming code works because it ran without crashing, commit and push to GitHub before moving to a new machine, and document reasoning (not just results) in `docs/` so nothing is ever lost again.

