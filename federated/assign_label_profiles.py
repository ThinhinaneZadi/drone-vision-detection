"""
assign_label_profiles.py — assign each federated client a "mission profile"
(a restricted set of object classes it is allowed to have labeled), for the
partial-label-space federated detection experiments.

WHY THIS EXISTS:
Standard federated object detection assumes every client labels every class.
Real drone deployments don't work this way — a traffic-monitoring drone has
no reason to label bicycles; a pedestrian-safety drone has no reason to
label buses. This script simulates that realistic constraint by assigning
each of the 19 real VisDrone clients to a mission profile, based on
class labels it can actually support in its own real data (not an
arbitrary/random assignment).

METHODOLOGY:
1. For each client, count real instances of each of the 10 VisDrone classes
   in its own label files (already computed once, hardcoded below from that
   analysis — see docs/partial_label_profiles.md for the full derivation).
2. A client "supports" a profile if it has >= MIN_INSTANCES real instances
   of every class in that profile. This avoids assigning a client to a
   profile it has no real signal for, which would confound the partial-
   label-space effect we're trying to study with pre-existing data scarcity.
3. Two of the ten classes (tricycle, awning-tricycle) are naturally rare
   and missing entirely in several clients (see docs/partial_label_profiles.md)
   — this constrains which clients can support classes-including-those
   profiles (C_lastmile, D_full), so those are assigned first (most
   constrained), then remaining clients fill A_traffic / B_pedsafety /
   E_vehicle_only.
4. One client (9999960) is heavily car-dominated and does not cleanly
   support ANY of the first four profiles (bus=0 for A_traffic, bicycle=8
   for B_pedsafety, below the 20-instance threshold) — rather than forcing
   it into a profile with a lowered threshold, a fifth narrower profile
   (E_vehicle_only) was added specifically to accommodate this class of
   real client. This is treated as a legitimate finding (not every real
   client fits a predefined mission profile), not an error to hide.

This assignment is deterministic and re-derivable from
docs/partial_label_profiles.md — re-run this script to regenerate it.
"""

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]

PROFILES = {
    "A_traffic":    [3, 4, 5, 8],          # car, van, truck, bus
    "B_pedsafety":  [0, 1, 2, 9],          # pedestrian, people, bicycle, motor
    "C_lastmile":   [6, 7, 9, 1],          # tricycle, awning-tricycle, motor, people
    "D_full":       list(range(10)),       # all classes (control group)
    "E_vehicle_only": [3, 4, 5],           # car, van, truck (added for clients
                                            # that don't fit A-D — see docstring)
}

MIN_INSTANCES = 20  # minimum real instances of a class to call it "supported"

# Real per-client class instance counts, from
# federated/experiments/partitions/tier100/*/train.txt label files,
# counted 2026-08-03 (see docs/partial_label_profiles.md for the exact
# command used to derive this).
CLIENT_CLASS_COUNTS = {
    "9999937": {0:3227, 1:1838, 2:295, 3:2026, 4:562, 5:698, 6:1128, 7:1209, 8:42, 9:2709},
    "9999940": {0:1005, 1:83, 2:36, 3:2429, 4:1019, 5:454, 6:1, 8:285, 9:175},
    "9999942": {0:2846, 1:804, 2:483, 3:3863, 4:1166, 5:373, 6:67, 7:320, 8:28, 9:1236},
    "9999945": {0:2326, 1:742, 2:46, 3:5401, 4:276, 5:74, 6:3, 7:14, 8:54, 9:1109},
    "9999951": {0:3260, 1:588, 2:27, 3:7248, 4:1372, 5:1173, 6:1, 7:4, 8:842, 9:152},
    "9999953": {0:447, 1:37, 2:6, 3:2209, 4:205, 5:246, 8:219, 9:77},
    "9999955": {0:8447, 1:454, 2:34, 3:6313, 4:3837, 5:648, 7:4, 8:1874, 9:984},
    "9999956": {0:368, 1:37, 2:3, 3:2013, 4:297, 5:366, 8:250, 9:59},
    "9999960": {0:244, 1:15, 2:8, 3:4884, 4:301, 5:93, 6:4, 9:28},
    "9999962": {0:1814, 1:372, 2:217, 3:2996, 4:428, 5:183, 6:63, 7:7, 8:61, 9:507},
    "9999966": {0:2763, 1:938, 2:530, 3:3582, 4:318, 5:100, 6:39, 7:3, 8:124, 9:390},
    "9999972": {0:217, 1:219, 2:20, 3:2474, 4:296, 5:180, 6:67, 7:72, 8:14, 9:320},
    "9999977": {0:190, 1:516, 2:72, 3:2068, 4:352, 5:33, 6:24, 7:6, 8:74, 9:686},
    "9999981": {0:1311, 1:892, 2:166, 3:2370, 4:469, 5:156, 6:119, 7:15, 8:15, 9:812},
    "9999982": {0:1818, 1:431, 2:152, 3:4388, 4:650, 5:1125, 6:128, 7:33, 8:161, 9:1397},
    "9999984": {0:1187, 1:440, 2:84, 3:2697, 4:525, 5:1221, 6:93, 7:41, 8:145, 9:415},
    "9999994": {0:899, 1:112, 2:143, 3:6911, 4:346, 5:447, 7:70, 8:111, 9:71},
    "9999998": {0:1989, 1:1121, 2:1612, 3:11231, 4:2004, 5:581, 6:142, 7:56, 8:193, 9:1049},
    "9999999": {0:8660, 1:2860, 2:1094, 3:9634, 4:2670, 5:986, 6:475, 7:142, 8:47, 9:2354},
}


def can_support(client_row: dict, class_ids: list[int]) -> bool:
    return all(client_row.get(c, 0) >= MIN_INSTANCES for c in class_ids)


def assign_profiles() -> dict[str, str]:
    """Deterministic assignment: most-constrained profiles (D, C) filled
    first, then remaining clients distributed across A/B/E based on which
    they can actually support."""
    all_clients = sorted(CLIENT_CLASS_COUNTS.keys())
    supporters = {
        name: [c for c in all_clients if can_support(CLIENT_CLASS_COUNTS[c], classes)]
        for name, classes in PROFILES.items()
    }

    assigned: dict[str, str] = {}
    target_counts = {"D_full": 4, "C_lastmile": 4}
    for profile_name in ["D_full", "C_lastmile"]:
        for c in supporters[profile_name]:
            if c not in assigned and sum(1 for v in assigned.values() if v == profile_name) < target_counts[profile_name]:
                assigned[c] = profile_name

    remaining = [c for c in all_clients if c not in assigned]
    balance_pool = ["A_traffic", "B_pedsafety", "E_vehicle_only"]
    for c in remaining:
        # among profiles this client actually qualifies for, pick whichever
        # currently has the FEWEST assigned clients, to keep A/B/E balanced
        # instead of greedily always filling A_traffic first
        eligible = [p for p in balance_pool if c in supporters[p]]
        if not eligible:
            raise RuntimeError(
                f"client {c} does not support ANY defined profile — "
                f"its real class counts are: {CLIENT_CLASS_COUNTS[c]}. "
                f"Add a new profile or adjust MIN_INSTANCES."
            )
        counts = {p: sum(1 for v in assigned.values() if v == p) for p in eligible}
        chosen = min(eligible, key=lambda p: counts[p])
        assigned[c] = chosen
    return assigned


if __name__ == "__main__":
    assignment = assign_profiles()
    print(f"{'client':<10} {'profile':<16} {'classes labeled'}")
    for c in sorted(assignment):
        profile = assignment[c]
        class_names = [CLASS_NAMES[i] for i in PROFILES[profile]]
        print(f"{c:<10} {profile:<16} {class_names}")

    from collections import Counter
    print(f"\nProfile counts: {dict(Counter(assignment.values()))}")
