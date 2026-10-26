# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Export the DialFork folder tree to Parquet shards for the Hugging Face Hub.

    python scripts/export_hf.py -i datasets/dialfork -o hf_export

Writes `hf_export/data/test-XXXXX-of-NNNNN.parquet` (one row per dialogue, with its Task 3
split from `<input>/task3_splits.csv`, see scripts/make_splits.py),
`hf_export/extra/done_markers.json` (the per-task DONE.txt files),
`hf_export/extra/task3_splits.csv`, `hf_export/extra/dialfork_llm_judge.csv.gz` (the LLM
judge's answers, from judge/answers/) and the dataset
card `hf_export/README.md` with its images (`hf_export/assets/`; `--card-only` rewrites
just those). Every row keeps the dialogue file exactly as it is on
disk (`dialog_json`), so `scripts/download_data.py` rebuilds the original tree
byte for byte. Upload with:

    huggingface-cli upload DialFork/dialfork hf_export . --repo-type dataset
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

TYPES = {"workflow_identity_corruption": "WIC",
         "step_selection_policy_corruption": "SSPC",
         "goal_replacement": "GR",
         "conflicting_system_constraints": "CSC"}
N_FORKS = 4
FORK_DIR = "n_hallucinations_%d" % N_FORKS
FILE_RE = re.compile(r"^dialog_(\d+)\.json$")

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("task3_split", pa.string()),
    ("task", pa.string()),
    ("domain", pa.string()),
    ("dialog_id", pa.int64()),
    ("kind", pa.string()),
    ("hallucination_type", pa.string()),
    ("hallucination_type_short", pa.string()),
    ("variant_id", pa.string()),
    ("fork_position", pa.int64()),
    ("fork_assistant_turn", pa.int64()),
    ("fork_turn_index", pa.int64()),
    ("num_turns", pa.int64()),
    ("turns", pa.list_(pa.struct([("speaker", pa.string()), ("text", pa.string())]))),
    ("dialog_json", pa.string()),
])


def row(path, task, dialog_id, kind, split, htype=None, k=None):
    with open(path, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    d = json.loads(raw)
    ann = d.get("annotations") or {}
    if kind == "fork":
        assert ann.get("hallucination_type") == htype, path
        assert ann.get("fork_segment") == "%d/%d" % (k, N_FORKS), path
    return {
        "id": "%d-%s" % (dialog_id, "conformant" if kind == "conformant"
                         else "%s-k%d" % (TYPES[htype], k)),
        "task3_split": split,
        "task": task,
        "domain": (ann.get("task") or {}).get("domain"),
        "dialog_id": dialog_id,
        "kind": kind,
        "hallucination_type": htype,
        "hallucination_type_short": TYPES.get(htype),
        "variant_id": ann.get("variant_id"),
        "fork_position": k,
        "fork_assistant_turn": ann.get("fork_assistant_turn"),
        "fork_turn_index": ann.get("fork_turn_index"),
        "num_turns": len(d.get("turns") or []),
        "turns": [{"speaker": t["speaker"], "text": t["text"]} for t in d.get("turns") or []],
        "dialog_json": raw,
    }


def load_splits(root):
    path = os.path.join(root, "task3_splits.csv")
    if not os.path.isfile(path):
        sys.exit("%s not found; run: python scripts/make_splits.py" % path)
    with open(path, encoding="utf-8", newline="") as fh:
        return {int(r["dialog_id"]): r["split"] for r in csv.DictReader(fh)}


def collect(root):
    rows, done = [], {}
    split_of = load_splits(root)
    for task in sorted(os.listdir(root)):
        tdir = os.path.join(root, task)
        if not os.path.isdir(tdir):
            continue
        marker = os.path.join(tdir, "DONE.txt")
        if os.path.exists(marker):
            with open(marker, encoding="utf-8", newline="") as fh:
                done[task] = fh.read()
        odir = os.path.join(tdir, "originals")
        for f in sorted(os.listdir(odir)):
            m = FILE_RE.match(f)
            assert m, os.path.join(odir, f)
            rows.append(row(os.path.join(odir, f), task, int(m.group(1)), "conformant",
                            split_of[int(m.group(1))]))
        for htype in TYPES:
            for k in range(1, N_FORKS + 1):
                fdir = os.path.join(tdir, htype, FORK_DIR, "fork_at_segment_%d" % k)
                for f in sorted(os.listdir(fdir)):
                    m = FILE_RE.match(f)
                    assert m, os.path.join(fdir, f)
                    rows.append(row(os.path.join(fdir, f), task, int(m.group(1)),
                                    "fork", split_of[int(m.group(1))], htype, k))
    rows.sort(key=lambda r: (r["task"], r["dialog_id"], r["kind"] != "conformant",
                             r["hallucination_type"] or "", r["fork_position"] or 0))
    assert len({r["id"] for r in rows}) == len(rows), "ids are not unique"
    return rows, done


# images the dataset card shows, copied next to it (assets/ on the Hub)
CARD_ASSETS = ("banner_light.png", "banner_dark.png", "forking.png", "dataset_stats.png",
               "human_study_full.png")


def write_card_assets(out):
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)
    for name in CARD_ASSETS:
        shutil.copyfile(os.path.join(src, name), os.path.join(out, "assets", name))


def write_card(args, n_rows, n_conf):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_card.md"),
              encoding="utf-8") as fh:
        card = fh.read()
    card = card.replace("{{N_ROWS}}", "{:,}".format(n_rows)) \
               .replace("{{N_CONFORMANT}}", "{:,}".format(n_conf)) \
               .replace("{{N_FORKED}}", "{:,}".format(n_rows - n_conf)) \
               .replace("{{REPO_ID}}", args.repo_id)
    with open(os.path.join(args.output, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default="datasets/dialfork")
    ap.add_argument("-o", "--output", default="hf_export")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--repo-id", default="DialFork/dialfork")
    ap.add_argument("--card-only", action="store_true",
                    help="only rewrite the dataset card and its images (no Parquet export)")
    args = ap.parse_args()

    write_card_assets(args.output)
    if args.card_only:
        import glob    # counting the files is enough for the card
        n_conf = len(glob.glob(os.path.join(args.input, "*", "originals", "dialog_*.json")))
        n_rows = n_conf + len(glob.glob(os.path.join(
            args.input, "*", "*", "n_hallucinations_*", "fork_at_segment_*", "dialog_*.json")))
        if n_rows == 0:     # no dataset folder: count the rows of the exported Parquet files
            shards = glob.glob(os.path.join(args.output, "data", "*.parquet"))
            if not shards:
                sys.exit("no dialogues found in %s nor Parquet files in %s/data"
                         % (args.input, args.output))
            kinds = [k for f in shards for k in pq.read_table(f, columns=["kind"])["kind"].to_pylist()]
            n_rows, n_conf = len(kinds), kinds.count("conformant")
        write_card(args, n_rows, n_conf)
        print("card and images -> %s (%d dialogues, %d conformant)" % (args.output, n_rows, n_conf))
        return

    rows, done = collect(args.input)
    shutil.rmtree(os.path.join(args.output, "data"), ignore_errors=True)
    os.makedirs(os.path.join(args.output, "data"))
    os.makedirs(os.path.join(args.output, "extra"), exist_ok=True)
    per = -(-len(rows) // args.shards)
    for s in range(args.shards):
        table = pa.Table.from_pylist(rows[s * per:(s + 1) * per], schema=SCHEMA)
        pq.write_table(table, os.path.join(
            args.output, "data", "test-%05d-of-%05d.parquet" % (s, args.shards)),
            compression="zstd")
    shutil.copyfile(os.path.join(args.input, "task3_splits.csv"),
                    os.path.join(args.output, "extra", "task3_splits.csv"))
    judge = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "judge", "answers", "dialfork_llm_judge.csv.gz")
    if os.path.isfile(judge):
        shutil.copyfile(judge, os.path.join(args.output, "extra", "dialfork_llm_judge.csv.gz"))
    with open(os.path.join(args.output, "extra", "done_markers.json"), "w",
              encoding="utf-8") as fh:
        json.dump(done, fh, indent=1, sort_keys=True)
    n_conf = sum(r["kind"] == "conformant" for r in rows)
    write_card(args, len(rows), n_conf)
    print("%d dialogues (%d conformant, %d forked) over %d tasks -> %s"
          % (len(rows), n_conf, len(rows) - n_conf, len(done), args.output))


if __name__ == "__main__":
    main()
