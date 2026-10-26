# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Pack judge runs into the released layout (judge/answers/), and the selected judge's
answers on the whole corpus into one flat CSV, with the dataset's dialogue ids.

    python -m judge.export_answers --study results/judge/outputs \\
        --full results/judge/outputs/full/qwen3.6_35b/P0

--study copies every run under <dir>/<model>/<prompt>/seed<S>/ to
judge/answers/study/, shards merged into one judgments.jsonl.gz, one line per
(item_id, stage), sorted. --full does the same for the full-corpus run (to
judge/answers/full/<model>/<prompt>/), and writes judge/answers/dialfork_llm_judge.csv.gz,
one row per dialogue.
"""
import argparse
import csv
import glob
import gzip
import io
import json
import os

from judge import common
from annotool.dialogs import TYPE_ABBREV

CSV_NAME = "dialfork_llm_judge.csv.gz"
COLUMNS = ("id", "task", "dialog_id", "kind", "hallucination_type", "variant_id", "k",
           "fork_assistant_turn", "cue_type", "cue_variant", "is_foil", "in_study",
           "slider", "appears", "onset_sys_turn", "prefix_ok")


def merged(files):
    """One record per (item_id, stage) over a run's files: the last one, as run.py
    resumes it, except that a call with no reply never replaces one with a reply. Calls
    with no reply are kept (e.g. reasoning that ran out of context); scoring skips them."""
    done = {}
    for f in files:
        for r in common.read_jsonl(f):
            k = (r["item_id"], r["stage"])
            if r["raw"] or r["answer"] is not None or k not in done:
                done[k] = r
    return [done[k] for k in sorted(done)]


def write_run(records, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "judgments.jsonl.gz")
    # mtime=0: the same answers give the same bytes
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for r in records:
            raw.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def dataset_id(meta):
    if meta["kind"] == "conformant":
        return "%s-conformant" % meta["dialog_id"]
    return "%s-%s-k%d" % (meta["dialog_id"], TYPE_ABBREV[meta["hallucination_type"]],
                          meta["segment"])


def write_csv(records, items, path):
    by = {}
    for r in records:
        by.setdefault(r["item_id"], {})[r["stage"]] = r["answer"] or {}
    rows = []
    for i, st in by.items():
        m = items[i]["meta"]
        a1, a2, a3 = st.get(1, {}), st.get(2, {}), st.get(3, {})
        rows.append({
            "id": dataset_id(m), "task": m["task"], "dialog_id": m["dialog_id"],
            "kind": m["kind"], "hallucination_type": TYPE_ABBREV.get(m["hallucination_type"], ""),
            "variant_id": m["variant_id"] or "", "k": m["segment"] or "",
            "fork_assistant_turn": m["fork_assistant_turn"] or "",
            "cue_type": TYPE_ABBREV.get(m["cue_type"], ""), "cue_variant": m["cue_variant"],
            "is_foil": int(bool(m["is_foil"])), "in_study": int(bool(items[i]["in_study"])),
            "slider": a1.get("slider", ""), "appears": a2.get("appears") or "",
            "onset_sys_turn": a2.get("onset_sys_turn") or "",
            "prefix_ok": a3.get("prefix_ok") or ("" if items[i]["has_prefix"] else "n/a")})
    rows.sort(key=lambda r: (r["task"], int(r["dialog_id"]), r["id"]))
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study", help="folder of study runs: <model>/<prompt>/seed<S>/")
    ap.add_argument("--full", help="folder of the selected judge's full-corpus run")
    args = ap.parse_args(argv)
    if args.study:
        for seed_dir in sorted(glob.glob(os.path.join(args.study, "*", "*", "seed*"))):
            files = common.answer_files(seed_dir)
            if not files:
                continue
            rel = os.path.relpath(seed_dir, args.study)
            recs = merged(files)
            print("  %-36s %5d stage answers -> %s" % (rel, len(recs), write_run(
                recs, os.path.join(common.STUDY_ANSWERS, rel))))
    if args.full:
        fz = common.load_frozen()
        recs = merged(common.answer_files(args.full))
        out = os.path.join(common.FULL_ANSWERS, common.model_dir(fz["model"]), fz["prompt"])
        print("  full corpus: %d stage answers -> %s" % (len(recs), write_run(recs, out)))
        from judge.full_score import load_full
        n = write_csv(recs, load_full(), os.path.join(common.ANSWERS, CSV_NAME))
        print("  %d dialogues -> %s" % (n, os.path.join(common.ANSWERS, CSV_NAME)))


if __name__ == "__main__":
    main()
