# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Score Task 3 predictions against the onset people saw, not only the fork turn.

    python scripts/task3_human_onset.py                       # the paper's methods
    python scripts/task3_human_onset.py my_method.csv.gz ...  # your own files

The fork turn is where the instruction is injected. An injected behaviour can take a turn
or two to become visible, for instance when the substituted workflow starts with the same
steps as the intended one. In the human study, annotators marked the first assistant turn
where the behaviour starts (the observable onset). This script compares each method's flag
with both references, on the forks of the Task 3 test split that annotators judged and
confirmed (first annotation per fork, as in the study's analysis).

Columns: MAE against the fork turn and against the human onset (a fork the method does
not flag counts as flagged just past its last turn, as in evaluate.py), then where the
flags fall: before the fork turn (on the still-conformant part), between the fork turn and
the human onset (before the deviation is visible), at the human onset, or after it. All in
turn indices (both speakers), as in evaluate.py.

Standard library only; reads the dataset folder, its Task 3 split, and
annotool/data/dialfork_human_judgments.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import evaluate  # noqa: E402  (its loaders and CSV reader)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JUDGMENTS = os.path.join(ROOT, "annotool", "data", "dialfork_human_judgments.csv")
CONFIRMED = ("yes", "partly")

# the rows of the Task 3 results table in docs/task3.md: (label, prediction file)
PAPER_METHODS = [
    ("Random", "random"), ("First turn", "first-turn"), ("Last turn", "last-turn"),
    ("SS", "streaming-ss__labse"), ("SS + NC", "streaming-ss-nc__labse"),
    ("IT", "streaming-it__labse"), ("IT + NC", "streaming-it-nc__labse"),
] + [("%s%s" % (name, suffix), "llm-%s%s" % (model, file_suffix))
     for suffix, file_suffix in (("", ""), (" + workflow", "-workflow"))
     for name, model in (("GPT-OSS-20B", "gpt-oss-20b"), ("Qwen3.6-35B", "qwen3.6-35b"),
                         ("Gemma4-31B", "gemma4-31b"), ("Llama-3.3-70B", "llama3.3-70b"),
                         ("Mistral-Small-3.2-24B", "mistral-small3.2-24b"))]


def load_human_onsets(data, split):
    """dataset id -> (fork_turn_index, human_onset_index, n_turns) for the confirmed forks."""
    split_of = evaluate.load_splits(data)
    first = {}
    with open(JUDGMENTS, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "fork" and r["is_foil"] == "0":
                first.setdefault(r["item_id"], r)     # the first annotation of each fork
    out = {}
    for r in first.values():
        if r["appears"] not in CONFIRMED or not r["onset_sys_turn"]:
            continue
        task, short, k, dialog_id = r["ref"].split("/")
        if split != "all" and split_of[int(dialog_id)] != split:
            continue
        path = os.path.join(evaluate.fork_dir(data, task, evaluate.SHORT[short], int(k[1:])),
                            "dialog_%s.json" % dialog_id)
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        # the study counts assistant turns from 1; the dataset indexes every turn from 0
        system = [i for i, t in enumerate(d["turns"]) if t["speaker"] == "System"]
        onset = int(r["onset_sys_turn"])
        if onset > len(system):
            continue
        out["%s-%s-%s" % (dialog_id, short, k)] = (
            d["annotations"]["fork_turn_index"], system[onset - 1], len(d["turns"]))
    return out


def score(path, forks):
    pred = {}
    for r in evaluate.read_csv(path, ("id", "predicted_turn_index")):
        v = r["predicted_turn_index"].strip()
        pred[r["id"].strip()] = None if v in ("", "none", "None", "null") else int(float(v))
    covered = [d for d in forks if d in pred]
    if not covered:
        return None
    err_fork, err_human = [], []
    where = dict.fromkeys(("before fork", "fork-onset", "at onset", "after", "not flagged"), 0)
    for d in covered:
        fork_turn, onset, n_turns = forks[d]
        p = pred[d]
        q = n_turns if p is None else p
        err_fork.append(abs(q - fork_turn))
        err_human.append(abs(q - onset))
        if p is None:
            where["not flagged"] += 1
        elif p < fork_turn:
            where["before fork"] += 1
        elif p < onset:
            where["fork-onset"] += 1
        elif p == onset:
            where["at onset"] += 1
        else:
            where["after"] += 1
    n = len(covered)
    return n, sum(err_fork) / n, sum(err_human) / n, {b: 100.0 * c / n for b, c in where.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="*",
                    help="Task 3 prediction files (default: the paper's, from predictions/task3/)")
    ap.add_argument("--data", default=os.path.join(ROOT, evaluate.RELEASES["v1"]))
    ap.add_argument("--split", choices=("test", "train", "all"), default="test",
                    help="forks to use (default: test, which every released file covers)")
    args = ap.parse_args()

    forks = load_human_onsets(args.data, args.split)
    delayed = sum(1 for f, o, _ in forks.values() if o > f)
    early = sum(1 for f, o, _ in forks.values() if o < f)
    print("%d confirmed forks with a human onset (%s split): the onset is after the fork turn "
          "in %d, before it in %d, and at it in %d.\n"
          % (len(forks), args.split, delayed, early, len(forks) - delayed - early))

    if args.predictions:
        runs = [(os.path.basename(p).split(".")[0], p) for p in args.predictions]
    else:
        runs = []
        for label, stem in PAPER_METHODS:
            found = glob.glob(os.path.join(ROOT, "predictions", "task3", stem + ".csv*"))
            if found:
                runs.append((label, found[0]))
    print("| method | n | MAE vs fork turn | MAE vs human onset | before fork turn "
          "| fork turn → onset | at onset | after onset | not flagged |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, path in runs:
        res = score(path, forks)
        if res is None:
            print("| %s | 0 | | | | | | | |" % label)
            continue
        n, mf, mh, w = res
        print("| %s | %d | %.2f | %.2f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |"
              % (label, n, mf, mh, w["before fork"], w["fork-onset"], w["at onset"],
                 w["after"], w["not flagged"]))


if __name__ == "__main__":
    main()
