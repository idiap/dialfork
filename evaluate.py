# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Score a prediction file on one of the three DialFork tasks.

    python evaluate.py --task 1 predictions/task1/flowkl.csv
    python evaluate.py --task 2 my_distances.csv --per-task
    python evaluate.py --task 3 my_turn_indices.csv

Prediction files are CSV (plain or gzipped, .csv.gz) with a header row; the column
names are fixed:

Task 1 (set-to-set distance)   task, hallucination_type, fork_position, distance
    One row per (task, hallucination type, fork position k): the distance between the
    task's conformant dialogues and its forks at position k (96 x 4 = 384 rows).
    Expected: d(k=1) > d(k=2) > d(k=3) > d(k=4).

Task 2 (dialog-to-dialog distance)   id, distance
    One row per fork (34,144 rows; ids as in the Hugging Face dataset, e.g.
    2115-WIC-k2): the distance from the fork to its conformant original
    (2115-conformant). Expected, per dialogue and type: d(k=1) > ... > d(k=4).

Task 3 (hallucination turn prediction)   id, predicted_turn_index
    One row per dialogue (36,278 rows; ids as in the Hugging Face dataset, e.g.
    2115-WIC-k2 or 2115-conformant): the 0-based position in `turns`
    of the first turn the method flags as off-track, or empty if it flags none.
    The ground truth is the dialogue's `fork_turn_index`. Task 3 is scored on the
    official Task 3 test split (datasets/dialfork/task3_splits.csv); rows for other dialogues are
    ignored, so a file covering all 36,278 dialogues is fine (--split all scores them all).

In Task 1, `hallucination_type` may be the full name (goal_replacement) or the short one (GR).
No third-party dependencies; the ground truth is read from datasets/dialfork/
(see scripts/download_data.py).
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict

TYPES = {"workflow_identity_corruption": "WIC",
         "step_selection_policy_corruption": "SSPC",
         "goal_replacement": "GR",
         "conflicting_system_constraints": "CSC"}
SHORT = {v: k for k, v in TYPES.items()}
COLUMNS = ("WIC", "SSPC", "GR", "CSC")      # table order in the paper
N_FORKS = 4
FILE_RE = re.compile(r"^dialog_(\d+)\.json$")
FORK_ID_RE = re.compile(r"^(\d+)-(WIC|SSPC|GR|CSC)-k(\d+)$")
ASSISTANT = {"system", "assistant", "agent"}
SPLITS_FILE = "task3_splits.csv"            # the Task 3 train/test split (scripts/make_splits.py)
# Task 3 composite Score weights (paper, Appendix "Composite Score")
W_HALLU_MAE, W_CONF_MAE, W_HALLU_COV, W_CONF_COV = 0.35, 0.35, 0.15, 0.15


# ---------------------------------------------------------------- ground truth
def fork_dir(root, task, htype, k):
    return os.path.join(root, task, htype, "n_hallucinations_%d" % N_FORKS,
                        "fork_at_segment_%d" % k)


def dialog_ids(folder):
    return sorted(int(m.group(1)) for m in map(FILE_RE.match, os.listdir(folder)) if m)


def load_index(root):
    """task -> {"conformant": [ids], htype: {k: [ids]}} (file names only: fast)."""
    if not os.path.isdir(root):
        sys.exit("dataset not found at %s; run: python scripts/download_data.py" % root)
    index = {}
    for task in sorted(os.listdir(root)):
        tdir = os.path.join(root, task)
        if not os.path.isdir(os.path.join(tdir, "originals")):
            continue
        index[task] = {"conformant": dialog_ids(os.path.join(tdir, "originals"))}
        for h in TYPES:
            index[task][h] = {k: dialog_ids(fork_dir(root, task, h, k))
                              for k in range(1, N_FORKS + 1)}
    return index


def load_splits(root):
    """dialog_id (source dialogue) -> "train" | "test"; its forks share it."""
    path = os.path.join(root, SPLITS_FILE)
    if not os.path.isfile(path):
        sys.exit("%s not found; run: python scripts/download_data.py (or scripts/make_splits.py)"
                 % path)
    with open(path, encoding="utf-8", newline="") as fh:
        return {int(r["dialog_id"]): r["split"] for r in csv.DictReader(fh)}


def load_turns(root, index):
    """id -> (task, kind, htype, fork_turn_index, n_turns), reading every file."""
    out = {}
    for task, t in index.items():
        jobs = [(os.path.join(root, task, "originals", "dialog_%d.json" % i), "%d-conformant" % i,
                 None) for i in t["conformant"]]
        for h in TYPES:
            for k in range(1, N_FORKS + 1):
                jobs += [(os.path.join(fork_dir(root, task, h, k), "dialog_%d.json" % i),
                          "%d-%s-k%d" % (i, TYPES[h], k), h) for i in t[h][k]]
        for path, did, h in jobs:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            out[did] = (task, "fork" if h else "conformant", h,
                        d["annotations"].get("fork_turn_index") if h else None, len(d["turns"]))
    return out


# ---------------------------------------------------------------- helpers
def read_csv(path, required):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit("%s: missing column(s) %s; expected %s" % (path, missing, list(required)))
        return list(reader)


def htype_of(value):
    v = value.strip()
    if v in TYPES:
        return v
    if v.upper() in SHORT:
        return SHORT[v.upper()]
    sys.exit("unknown hallucination_type %r (use one of %s)" % (value, sorted(SHORT)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def order_tau(dist):
    """Kendall tau between the distances {k: d} and the expected order
    d(1) > d(2) > ... (later forks are closer to the original). Tied distances count
    as neither concordant nor discordant: the expected tau under random tie-breaking."""
    ks = sorted(dist)
    score = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = dist[ks[i]], dist[ks[j]]     # ks[i] is earlier: should be larger
            score += (a > b) - (a < b)
    n = len(ks)
    return score / (n * (n - 1) / 2.0)


def print_table(title, rows, cols, fmt):
    print("\n" + title)
    w = max([len(r[0]) for r in rows] + [6])
    print("  %-*s %s" % (w, "", " ".join("%8s" % c for c in cols)))
    for name, vals in rows:
        print("  %-*s %s" % (w, name, " ".join(
            "%8s" % ("-" if v is None else fmt % v) for v in vals)))


# ---------------------------------------------------------------- Task 1 and 2
def evaluate_ordering(args, per_unit_key):
    index = load_index(args.data)
    if per_unit_key == "task":
        rows = read_csv(args.predictions, ("task", "hallucination_type", "fork_position", "distance"))
    else:
        rows = read_csv(args.predictions, ("id", "distance"))
    task_of = {i: t for t, v in index.items() for i in v["conformant"]}
    pred = defaultdict(dict)                      # (task, htype, unit) -> {k: distance}
    for r in rows:
        if per_unit_key == "task":
            h, k = htype_of(r["hallucination_type"]), int(r["fork_position"])
            task = r["task"].strip()
            unit = task
        else:
            m = FORK_ID_RE.match(r["id"].strip())
            if not m or int(m.group(1)) not in task_of or not 1 <= int(m.group(3)) <= N_FORKS:
                sys.exit("unknown fork id %r (expected e.g. 2115-WIC-k2)" % r["id"])
            unit, h, k = int(m.group(1)), SHORT[m.group(2)], int(m.group(3))
            task = task_of[unit]
        pred[(task, h, unit)][k] = float(r["distance"])

    taus = defaultdict(list)                      # (task, htype) -> [tau per unit]
    missing = 0
    for task, t in index.items():
        for h in TYPES:
            units = [task] if per_unit_key == "task" else \
                sorted(set().union(*[set(t[h][k]) for k in range(1, N_FORKS + 1)]))
            for u in units:
                d = pred.get((task, h, u), {})
                if len(d) < N_FORKS:
                    missing += 1
                    continue
                taus[(task, h)].append(order_tau(d))
    if missing and not args.allow_missing:
        sys.exit("%d (task/dialogue, type) unit(s) lack one or more of the 4 fork positions; "
                 "pass --allow-missing to score the rest" % missing)

    # mean over the units of each (task, type), then over tasks: the paper's aggregation
    by_type = {h: mean([mean(taus[(task, h)]) for task in index if taus.get((task, h))])
               for h in TYPES}
    row = [by_type[SHORT[c]] for c in COLUMNS]
    row.append(mean(row))
    name = os.path.basename(args.predictions).split(".")[0]
    print_table("Task %d: Kendall tau with the expected fork order (chance = 0)%s"
                % (args.task, "" if not missing else " -- %d unit(s) missing" % missing),
                [(name, row)], list(COLUMNS) + ["Avg."], "%+.3f")
    if args.per_task:
        print_table("Per task (mean over types)",
                    [(task, [mean([mean(taus.get((task, h), [])) for h in TYPES])])
                     for task in index], ["tau"], "%+.3f")
    return {"by_type": {TYPES[h]: v for h, v in by_type.items()}, "avg": row[-1]}


# ---------------------------------------------------------------- Task 3
def evaluate_turns(args):
    index = load_index(args.data)
    all_gold = load_turns(args.data, index)
    if args.split == "all":
        gold = all_gold
    else:
        split_of = load_splits(args.data)
        gold = {d: g for d, g in all_gold.items() if split_of[int(d.split("-")[0])] == args.split}
    try:
        rows = read_csv(args.predictions, ("id", "predicted_turn_index"))
    except SystemExit:
        with (gzip.open if args.predictions.endswith(".gz") else open)(args.predictions, "rt") as fh:
            header = fh.readline()
        if "predicted_turn" in header:
            sys.exit("%s: Task 3 predictions are now 0-based turn indices: rename the column to "
                     "predicted_turn_index and give the 0-based position in `turns`"
                     % args.predictions)
        raise
    pred = {}
    for r in rows:
        did, v = r["id"].strip(), r["predicted_turn_index"].strip()
        if did not in all_gold:
            sys.exit("unknown id %r" % did)
        if did not in gold:                    # another split: not scored
            continue
        if v in ("", "none", "None", "null"):
            pred[did] = None
            continue
        p = int(float(v))
        if not 0 <= p < gold[did][4]:
            sys.exit("%s: predicted_turn_index %d outside 0..%d (its turns)" % (did, p, gold[did][4] - 1))
        pred[did] = p
    missing = [d for d in gold if d not in pred]
    if missing and not args.allow_missing:
        sys.exit("%d dialogue(s) have no row (e.g. %s); a dialogue the method does not flag "
                 "still needs a row with an empty predicted_turn_index" % (len(missing), missing[0]))

    def metrics(get_pred):
        hall = defaultdict(list)                 # (task, htype) -> abs errors
        h_flag = h_tot = c_flag = c_tot = 0
        conf = defaultdict(list)                 # task -> abs errors of the flagged
        for did, (task, kind, h, fork_turn_index, n_turns) in gold.items():
            if did not in pred and args.allow_missing:
                continue
            p = get_pred(did)
            if kind == "fork":
                h_tot += 1
                h_flag += p is not None
                # a fork the method does not flag counts as flagged just past its end
                hall[(task, h)].append(abs((n_turns if p is None else p) - fork_turn_index))
            else:
                c_tot += 1
                if p is not None:              # a false alarm: how early it fires
                    c_flag += 1
                    conf[task].append(abs(p - n_turns))
        by_type = {h: mean([mean(hall[(t, h)]) for t in index if hall.get((t, h))]) for h in TYPES}
        return {"by_type": by_type, "hall_mae": mean(list(by_type.values())),
                "hall_cov": 100.0 * h_flag / h_tot if h_tot else None,
                "conf_mae": mean([mean(v) for v in conf.values()]),
                "conf_cov": 100.0 * c_flag / c_tot if c_tot else None}

    # Score anchors: the last-turn baseline's hallucinated MAE and the first-turn
    # baseline's conformant MAE (the worst credible values), computed from the data
    h_upper = metrics(lambda d: gold[d][4] - 1)["hall_mae"]
    c_upper = metrics(lambda d: 0)["conf_mae"]
    m = metrics(lambda d: pred.get(d))

    def desir(v, upper):
        return None if v is None else 1.0 - min(max(v / upper, 0.0), 1.0)
    w_hm, w_cm, w_hc, w_cc = args.weights
    comps = [(desir(m["hall_mae"], h_upper), w_hm),
             (desir(m["conf_mae"] if m["conf_mae"] is not None else 0.0, c_upper), w_cm),
             (None if m["hall_cov"] is None else m["hall_cov"] / 100.0, w_hc),
             (None if m["conf_cov"] is None else 1.0 - m["conf_cov"] / 100.0, w_cc)]
    comps = [(x, w) for x, w in comps if x is not None]
    score = 100.0 * math.exp(sum(w * math.log(max(x, 1e-12)) for x, w in comps)
                             / sum(w for _, w in comps))
    m["score"] = score

    name = os.path.basename(args.predictions).split(".")[0]
    cols = ["CSC", "GR", "SSPC", "WIC", "MAE", "Cov.", "Conf.MAE", "Conf.Cov.", "Score"]
    vals = [m["by_type"][SHORT[c]] for c in ("CSC", "GR", "SSPC", "WIC")] + \
           [m["hall_mae"], m["hall_cov"], m["conf_mae"], m["conf_cov"], score]
    print_table("Task 3: hallucination turn prediction, %s (turn indices; MAE and conformant "
                "coverage: lower is better)%s"
                % ("all dialogues" if args.split == "all" else "%s split, %d dialogues"
                   % (args.split, len(gold)),
                   "" if not missing else " -- %d dialogue(s) missing" % len(missing)),
                [(name, vals)], cols, "%.2f")
    print("  Score anchors: last-turn hallucinated MAE %.2f, first-turn conformant MAE %.2f"
          % (h_upper, c_upper))
    return {"by_type": {TYPES[h]: v for h, v in m["by_type"].items()}, **{
        k: m[k] for k in ("hall_mae", "hall_cov", "conf_mae", "conf_cov", "score")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="prediction CSV file")
    ap.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--data", default="datasets/dialfork", help="DialFork root folder")
    ap.add_argument("--per-task", action="store_true", help="also print the per-task breakdown")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score only the units/dialogues present in the file")
    ap.add_argument("--weights", default=None, metavar="HM,CM,HC,CC",
                    help="Task 3: Score weights for hallucination MAE, conformant MAE, hallucination "
                         "coverage and conformant coverage (default: the paper's %s,%s,%s,%s; only "
                         "the default is comparable with the leaderboard)"
                         % (W_HALLU_MAE, W_CONF_MAE, W_HALLU_COV, W_CONF_COV))
    ap.add_argument("--split", choices=("test", "train", "all"), default="test",
                    help="Task 3: the split to score (default: the official Task 3 test split)")
    ap.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    args = ap.parse_args()
    if args.weights is None:
        args.weights = (W_HALLU_MAE, W_CONF_MAE, W_HALLU_COV, W_CONF_COV)
    else:
        try:
            args.weights = tuple(float(x) for x in args.weights.split(","))
            assert len(args.weights) == 4 and min(args.weights) >= 0 and sum(args.weights) > 0
        except (ValueError, AssertionError):
            ap.error("--weights takes four non-negative numbers, e.g. 0.25,0.25,0.25,0.25")
    if args.task == 1:
        out = evaluate_ordering(args, "task")
    elif args.task == 2:
        out = evaluate_ordering(args, "dialog")
    else:
        out = evaluate_turns(args)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
