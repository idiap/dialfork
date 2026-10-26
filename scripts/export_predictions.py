# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Export the paper's results (results/task{1,2,3}/) as prediction files for evaluate.py.

    python scripts/export_predictions.py -o predictions

Writes predictions/task1/<method>.csv.gz, predictions/task2/<method>__<encoder>.csv.gz
and predictions/task3/<method>.csv.gz (gzipped CSV; evaluate.py reads them directly).
Task 3 predictions are 0-based positions in `turns`: the streaming detectors store
1-based positions over all turns (shifted here), the LLMs an assistant-turn number
(mapped here to that turn's position).
"""
import argparse
import csv
import glob
import gzip
import io
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate as E  # noqa: E402

TASK1_DIR = "results/task1/dialog2flow-joint-dse-base"
TASK1_METHODS = {"FBD": "fbd", "PRD": "prd", "kde-distance-FlowKL": "flowkl",
                 "MeanLenDiff-turns": "turn-count-diff", "MeanLenDiff-words": "word-count-diff"}
TASK2_METHODS = ("mean-turn-emb", "full-dialog-emb", "flowdtw", "flowscore")
TASK2_CONTROLS = ("turn-count-diff", "word-count-diff", "shared-prefix")
TASK2_ENCODERS = ("LaBSE", "Qwen3-Embedding-8B", "all-mpnet-base-v2", "bge-m3",
                  "dialog2flow-joint-dse-base", "dse-bert-base", "embeddinggemma-300m")
# streaming detectors trained on the Task 3 train split and run on its test split
# (methods/task3/streaming.py --split-file); their prediction files cover the test split only
TASK3_STREAM_DIR = "results/task3_split"
TASK3_STREAM_SETTING = "lm-3-gram"         # the paper's N
TASK3_STREAM = {("raw", "max_jump"): "ss", ("smooth", "max_jump"): "ss-nc",
                ("raw", "first_below"): "it", ("smooth", "first_below"): "it-nc"}
TASK3_STREAM_ENCODERS = {"dialog2flow-joint-dse-base": "d2f", "embedding_BAAI-bge-m3": "bge-m3",
                         "LaBSE": "labse", "all-mpnet-base-v2": "sbert",
                         "Qwen3-Embedding-8B": "qwen3-emb", "embeddinggemma-300m": "embgemma",
                         "dse-bert-base": "dse"}


def write(path, header, rows):
    """Gzipped CSV, with a fixed gzip timestamp so identical predictions give
    byte-identical files (no spurious git changes on re-export)."""
    path += ".gz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                 filename="") as fh:
        fh.write(buf.getvalue().encode("utf-8"))
    print("  %-55s %6d rows" % (path, len(rows)))


def task1(out):
    rows = {m: [] for m in TASK1_METHODS}
    for p in sorted(glob.glob(os.path.join(TASK1_DIR, "*", "results_all.json"))):
        task = os.path.basename(os.path.dirname(p))
        res = json.load(open(p))
        for h, per_k in res.items():
            for key, vals in per_k.items():
                k = int(key.rsplit("_", 1)[1])
                for m in TASK1_METHODS:
                    rows[m].append((task, h, k, vals[m]))
    for m, name in TASK1_METHODS.items():
        write(os.path.join(out, "task1", name + ".csv"),
              ("task", "hallucination_type", "fork_position", "distance"), sorted(rows[m]))


def task2(out):
    combos = [(m, e) for m in TASK2_METHODS for e in TASK2_ENCODERS] + \
             [(m, "none") for m in TASK2_CONTROLS]
    for m, enc in combos:
        rows = []
        for p in sorted(glob.glob(os.path.join("results/task2", m, enc, "*",
                                               "results_d2d_rankings.json"))):
            for h, per_dialog in json.load(open(p)).items():
                for did, rec in per_dialog.items():
                    for key, d in rec["distances"].items():
                        rows.append((int(did), E.TYPES[h], int(key.rsplit("_", 1)[1]), d))
        name = m if enc == "none" else "%s__%s" % (m, enc)
        write(os.path.join(out, "task2", name + ".csv"), ("id", "distance"),
              [("%d-%s-k%d" % (did, h, k), d) for did, h, k, d in sorted(rows)])


def assistant_turn_index(turn, speakers):
    """1-based assistant turn (what the LLMs output) -> 0-based position in turns, or None
    for a turn number outside the dialogue (it flags nothing in it)."""
    positions = [i for i, s in enumerate(speakers) if s.lower() in E.ASSISTANT]
    return positions[turn - 1] if 1 <= turn <= len(positions) else None


def task3(out, data):
    index = E.load_index(data)
    speakers = {}

    def spk(task, h, k, did):
        key = (task, h, k, did)
        if key not in speakers:
            p = (os.path.join(E.fork_dir(data, task, h, k), "dialog_%d.json" % did) if h
                 else os.path.join(data, task, "originals", "dialog_%d.json" % did))
            d = json.load(open(p))
            speakers[key] = ([u["speaker"] for u in d["turns"]], d["annotations"].get("fork_assistant_turn"))
        return speakers[key]

    # streaming: per_dialog entries carry dialog_id, the fork turn and the conversation length,
    # which identify the fork (the four forks of a dialogue have distinct fork turns)
    for exp, enc in TASK3_STREAM_ENCODERS.items():
        exp_dir = os.path.join(TASK3_STREAM_DIR, exp, TASK3_STREAM_SETTING)
        if not os.path.isdir(exp_dir):
            continue
        n_done = len([f for f in glob.glob(os.path.join(exp_dir, "*.json"))
                      if not f.endswith("AVERAGE.json")])
        if n_done < len(index):
            print("  SKIPPED streaming %s: %d of %d tasks finished" % (enc, n_done, len(index)))
            continue
        preds = {name: {} for name in TASK3_STREAM.values()}
        for p in sorted(glob.glob(os.path.join(exp_dir, "*.json"))):
            task = os.path.basename(p)[:-5]
            if task == "AVERAGE":
                continue
            det = json.load(open(p))["detection"]
            for (variant, method), name in TASK3_STREAM.items():
                for h in E.TYPES:
                    for r in det["hallucination_types"][h]["variants"][variant]["per_dialog"]:
                        did = r["dialog_id"]
                        k = [k for k in range(1, 5) if spk(task, h, k, did)[1] == r["ground_truth"]
                             and len(spk(task, h, k, did)[0]) == r["last_turn_index"]][0]
                        pos = r["pred_online"].get(method)
                        preds[name]["%d-%s-k%d" % (did, E.TYPES[h], k)] = \
                            None if pos is None else pos - 1
                for r in det["happy"]["variants"][variant]["per_dialog"]:
                    pos = r["pred_online"].get(method)
                    preds[name]["%d-conformant" % r["dialog_id"]] = \
                        None if pos is None else pos - 1
        for name, pr in preds.items():
            write(os.path.join(out, "task3", "streaming-%s__%s.csv" % (name, enc)),
                  ("id", "predicted_turn_index"), sorted((i, "" if v is None else v) for i, v in pr.items()))

    # trivial baselines, defined directly as turn indices
    import random
    gold = E.load_turns(data, index)
    rng = random.Random(0)
    for name, f in (("first-turn", lambda n: 0), ("last-turn", lambda n: n - 1),
                    ("random", lambda n: rng.randint(0, n - 1))):
        write(os.path.join(out, "task3", name + ".csv"), ("id", "predicted_turn_index"),
              [(i, f(gold[i][4])) for i in sorted(gold)])

    # non-streaming LLMs: the per-dialogue caches, in assistant turns (converted below)
    # results/task3_rerun/offline/ holds complete re-runs of the runs whose caches were
    # overwritten (GPT-OSS, Qwen); a re-run, when present, replaces the original run
    runs = {os.path.basename(d): d for d in sorted(glob.glob("results/task3/offline/llm-*"))}
    runs.update({os.path.basename(d): d for d in sorted(glob.glob("results/task3_rerun/offline/llm-*"))})
    for exp_dir in [runs[k] for k in sorted(runs)]:
        model = os.path.basename(exp_dir)[4:]
        name = "llm-" + (model[:-len("-no-workflow")] if model.endswith("-no-workflow")
                         else model + "-workflow").replace(":", "-")
        pr, dup, n_outside = {}, Counter(), 0
        for p in sorted(glob.glob(os.path.join(exp_dir, "*_cache.jsonl"))):
            task = os.path.basename(p)[:-len("_cache.jsonl")]
            for line in open(p):
                r = json.loads(line)
                did = r["dialog_id"]
                if r["type"] == "originals":
                    key = "%d-conformant" % did
                    turn_speakers = spk(task, None, None, did)[0]
                else:
                    h = r["type"]
                    ks = [k for k in range(1, 5) if spk(task, h, k, did)[1] == r["ground_truth"]]
                    key = "%d-%s-k%d" % (did, E.TYPES[h], ks[0])
                    turn_speakers = spk(task, h, ks[0], did)[0]
                dup[key] += 1
                pr[key] = None if r["prediction"] is None else assistant_turn_index(r["prediction"], turn_speakers)
                n_outside += r["prediction"] is not None and pr[key] is None
        if set(pr) != set(gold):
            print("  SKIPPED %s: its cache covers %d of %d dialogues (re-run "
                  "methods/task3/llm.py to regenerate it)" % (name, len(pr), len(gold)))
            continue
        write(os.path.join(out, "task3", name + ".csv"), ("id", "predicted_turn_index"),
              sorted((i, "" if v is None else v) for i, v in pr.items()))
        if n_outside:
            print("    (%d predictions pointed past the dialogue's last assistant turn: kept as no flag)" % n_outside)
        n_dup = sum(1 for v in dup.values() if v > 1)
        if n_dup:
            print("    (%d ids appeared more than once in the cache; the last entry was kept)" % n_dup)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="predictions")
    ap.add_argument("--data", default="datasets/dialfork")
    ap.add_argument("--tasks", default="1,2,3")
    args = ap.parse_args()
    todo = set(args.tasks.split(","))
    if "1" in todo:
        print("Task 1"); task1(args.output)
    if "2" in todo:
        print("Task 2"); task2(args.output)
    if "3" in todo:
        print("Task 3"); task3(args.output, args.data)


if __name__ == "__main__":
    main()
