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
    "9999937": {0:3175,1:1821,2:290,3:1960,4:548,5:672,6:1118,7:1196,8:40,9:2687},
    "9999940": {0:990,1:81,2:36,3:2387,4:999,5:442,6:1,8:279,9:172},
    "9999942": {0:2811,1:798,2:480,3:3721,4:1122,5:356,6:67,7:318,8:24,9:1220},
    "9999945": {0:2302,1:739,2:46,3:5270,4:269,5:71,6:3,7:14,8:52,9:1101},
    "9999951": {0:3233,1:586,2:27,3:7064,4:1329,5:1127,6:1,7:3,8:815,9:149},
    "9999953": {0:417,1:37,2:6,3:2060,4:194,5:224,8:203,9:73},
    "9999955": {0:8363,1:450,2:34,3:6102,4:3773,5:628,7:4,8:1833,9:975},
    "9999956": {0:355,1:37,2:3,3:1939,4:285,5:358,8:243,9:56},
    "9999960": {0:222,1:13,2:8,3:4744,4:293,5:82,6:4,9:27},
    "9999962": {0:1774,1:369,2:215,3:2912,4:415,5:175,6:63,7:7,8:57,9:496},
    "9999966": {0:2692,1:929,2:525,3:3542,4:311,5:99,6:39,7:3,8:122,9:386},
    "9999972": {0:210,1:216,2:20,3:2363,4:280,5:169,6:64,7:69,8:14,9:313},
    "9999977": {0:189,1:512,2:72,3:1990,4:343,5:30,6:24,7:6,8:70,9:685},
    "9999981": {0:1287,1:891,2:160,3:2286,4:456,5:152,6:116,7:14,8:14,9:793},
    "9999982": {0:1806,1:431,2:152,3:4244,4:623,5:1075,6:125,7:33,8:151,9:1387},
    "9999984": {0:1159,1:433,2:83,3:2648,4:514,5:1154,6:92,7:41,8:137,9:413},
    "9999994": {0:862,1:108,2:143,3:6850,4:341,5:438,7:70,8:109,9:71},
    "9999998": {0:1943,1:1115,2:1597,3:11006,4:1974,5:558,6:139,7:55,8:186,9:1040},
    "9999999": {0:8567,1:2834,2:1085,3:9440,4:2628,5:970,6:472,7:140,8:47,9:2333},
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
