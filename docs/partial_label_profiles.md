# Partial Label-Space Client Profiles — Design Documentation

## Motivation

Standard federated object detection assumes every client labels every
class. Real drone deployments don't work this way: a traffic-monitoring
drone has no operational reason to label bicycles; a pedestrian-safety
drone has no reason to label buses. This document describes how each of
the 19 real VisDrone federated clients was assigned a "mission profile" —
a restricted set of object classes it is allowed to have labeled — to
simulate this realistic constraint, as the foundation for a novel
contribution: loss-reweighting to handle federated object detection under
partial, non-overlapping label spaces.

## Step 1 — Real per-client class distributions

Before designing any profile, we measured each client's REAL, naturally
occurring instance counts per class, using:

```bash
for c in <client_ids>; do
  cat data/VisDrone-DET/VisDrone2019-DET-train/labels/${c}_*.txt \
    | awk '{print $1}' | sort -n | uniq -c
done
```

Class index -> name mapping (matches visdrone.yaml):
0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van, 5=truck, 6=tricycle,
7=awning-tricycle, 8=bus, 9=motor

**Key finding: 8 of 10 classes are present in nearly every client already**
(pedestrian, people, bicycle, car, van, truck, motor = 19/19 clients;
bus = 18/19). Two classes are naturally rare and client-dependent:

- tricycle: completely ABSENT in 4 clients (9999953, 9999955, 9999956, 9999994)
- awning-tricycle: completely ABSENT in 3 clients (9999940, 9999953, 9999956)

This means client heterogeneity in class distribution is a REAL, pre-existing
property of the dataset — not something we introduce artificially.

## Step 2 — Candidate mission profiles

Four initial profiles were proposed, grounded in plausible real drone
missions (not arbitrary class groupings):

| Profile | Classes | Real-world justification |
|---|---|---|
| A_traffic | car, van, truck, bus | Traffic-monitoring drone patrolling roads |
| B_pedsafety | pedestrian, people, bicycle, motor | Sidewalk/crosswalk safety drone |
| C_lastmile | tricycle, awning-tricycle, motor, people | Informal/market transit corridor drone |
| D_full | all 10 classes | Control group — full-coverage client |

## Step 3 — Checking which real clients can support each profile

A client "supports" a profile only if it has >= 20 real instances of EVERY
class in that profile. This threshold avoids assigning a client to a
profile it has no real signal for, which would confound the partial-label
effect we want to study with pre-existing data scarcity.

Result: A_traffic and B_pedsafety were each supportable by 16/19 clients.
C_lastmile only 7/19. D_full only 6/19 (constrained by the rare classes
identified in Step 1).

## Step 4 — The edge case: client 9999960

Client 9999960 is heavily car-dominated (4,744 car instances) with almost
nothing else: bus=0, bicycle=8, people=13. It does not cleanly support
ANY of the four profiles above (fails A_traffic on bus=0, fails
B_pedsafety on bicycle=8 < 20 threshold).

Rather than force this client into a profile via a lowered threshold, a
fifth, narrower profile was added:

| Profile | Classes | Justification |
|---|---|---|
| E_vehicle_only | car, van, truck | Highway/vehicle-only monitoring drone |

This is treated as a legitimate methodological finding — not every real
client fits a predefined mission profile — rather than an inconvenience
to hide. Worth stating explicitly in the paper's methods section.

## Step 5 — Final balanced assignment

Assignment logic (see federated/assign_label_profiles.py):
1. D_full and C_lastmile (the most constrained profiles) are filled first,
   up to 4 clients each, from their real supporter lists.
2. Remaining clients are assigned to whichever of {A_traffic, B_pedsafety,
   E_vehicle_only} they qualify for AND which currently has the fewest
   assigned clients — this keeps the three "easy" profiles balanced
   instead of greedily filling one first.

Final result (19 clients total):

| Profile | Client count | Clients |
|---|---|---|
| D_full | 4 | 9999937, 9999942, 9999982, 9999984 |
| C_lastmile | 3 | 9999972, 9999998, 9999999 |
| A_traffic | 4 | 9999940, 9999953, 9999962, 9999977 |
| B_pedsafety | 4 | 9999945, 9999955, 9999966, 9999981 |
| E_vehicle_only | 4 | 9999951, 9999956, 9999960, 9999994 |

This assignment is fully deterministic and reproducible — re-running
`python3 federated/assign_label_profiles.py` regenerates it exactly.

## Next steps (not yet implemented as of this writing)

1. Build a label-filtering step (new script or extension to
   partition_by_location.py) that, for each client, DROPS every label
   line whose class is not in that client's assigned profile — images
   stay unchanged, only annotations are filtered.
2. Baseline experiment: plain FedAvg under this partial-label-space split
   — expect degraded accuracy on classes many clients don't label.
3. Baseline experiment: pseudo-labeling under the same split.
4. Contribution experiment: loss-reweighting (mask out loss contribution
   for classes outside a client's known profile) under the same split.
5. Compare all three, and combine the best drift-control method (freeze
   or FedProx, from separate experiments) with loss-reweighting.
