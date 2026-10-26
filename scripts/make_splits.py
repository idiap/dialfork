# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Write the official Task 3 train/test split: <dataset root>/task3_splits.csv.

    python scripts/make_splits.py                 # datasets/dialfork/task3_splits.csv
    python scripts/make_splits.py --check         # verify an existing task3_splits.csv

The split is over source dialogues: a conformant dialogue and its 16 forks always fall
in the same split (a fork replays its source before the fork turn, so splitting files
would leak the test prefixes into training). Within each task, the source dialogues are
shuffled with a fixed seed and half of them (the larger half, if odd) go to `test`.
Deterministic: the same dataset always gives the same file.

The split exists for Task 3 only: Task 3 is scored on `test`, and methods that learn from
data (e.g. the streaming detectors' flow graph, n-gram model and thresholds) may learn only
from the conformant dialogues of `train`, never from its forks or labels (they would exploit
regularities of the construction). Tasks 1 and 2 use every dialogue, regardless of it: there,
conformant dialogues are the reference forks are compared with, never items to score.
"""
import argparse
import csv
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate as E  # noqa: E402

SEED = "dialfork-split-v1"


def make_splits(root):
    """[(task, dialog_id, split)] sorted by task and dialog_id."""
    out = []
    for task, t in E.load_index(root).items():
        ids = list(t["conformant"])
        random.Random("%s:%s" % (SEED, task)).shuffle(ids)
        n_test = (len(ids) + 1) // 2
        split = {i: ("test" if j < n_test else "train") for j, i in enumerate(ids)}
        out += [(task, i, split[i]) for i in sorted(ids)]
    return out


def to_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(("task", "dialog_id", "split"))
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/dialfork", help="DialFork root folder")
    ap.add_argument("--check", action="store_true",
                    help="only check that task3_splits.csv matches the rule")
    args = ap.parse_args()
    path = os.path.join(args.data, E.SPLITS_FILE)
    text = to_csv(make_splits(args.data))
    if args.check:
        with open(path, encoding="utf-8", newline="") as fh:
            ok = fh.read() == text
        print("%s: %s" % (path, "OK" if ok else "DIFFERS from the split rule"))
        sys.exit(0 if ok else 1)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    n_test = sum(r["split"] == "test" for r in rows)
    print("%s: %d source dialogues, %d train / %d test" % (path, len(rows), len(rows) - n_test, n_test))


if __name__ == "__main__":
    main()
