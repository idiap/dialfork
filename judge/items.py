# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Build the judge inputs: one record per human-study conversation (428).

Each record holds what the annotators saw at each step (the task's steps, the user's
brief, the transcript with assistant-turn numbers, the step-2 cue, the step-3 reveal)
and their answers. Read from the released, anonymized files only.

    python -m judge.items            # writes results/judge/items.jsonl and checks the counts
"""
import csv
import json
import os
import sqlite3
from collections import Counter

from judge import common
from annotool.dialogs import (cue_sentence, load_dialog, load_variant_pools,
                              substitute_workflow, system_turn_indices)
from annotool.workflow import TASK_DESCRIPTIONS, build_flow, render_text


def load_study_items():
    # immutable: not even the -shm index of the WAL database is touched
    con = sqlite3.connect("file:%s?mode=ro&immutable=1" % common.HUMAN_DB, uri=True)
    rows = [json.loads(r[0]) for r in con.execute("SELECT item_json FROM items")]
    con.close()
    return {r["item_id"]: r for r in rows}


def load_human_answers():
    def num(v):
        return int(float(v)) if v not in ("", None) else None
    out = {}
    with open(common.HUMAN_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["item_id"], []).append({
                "annotator": r["annotator"],
                "slider": num(r["slider"]),
                "appears": r["appears"] or None,
                "onset_sys_turn": num(r["onset_sys_turn"]),
                "prefix_ok": r["prefix_ok"] or "n/a",
            })
    return out


def workflow_text(task, dialog):
    meta = dialog["annotations"]["task"]
    return render_text(build_flow(task, meta["graph"], meta["replies"]))


def turns_of(dialog):
    """Same numbering the annotators clicked: assistant turns counted from 1."""
    out, n = [], 0
    for t in dialog.get("turns") or []:
        sys_turn = t.get("speaker") == "System"
        n += sys_turn
        out.append({"speaker": "assistant" if sys_turn else "user",
                    "sys_n": n if sys_turn else None, "text": t.get("text") or ""})
    return out


def build_record(meta, answers, pools, root=common.DATASET):
    d = load_dialog(os.path.join(root, meta["path"]))
    task = meta["task"]
    fork = meta["kind"] == "fork"
    reveal = {"fork_assistant_turn": meta["fork_assistant_turn"] if fork else None,
              "true_cue": None, "substitute": None}
    if fork:
        reveal["true_cue"] = cue_sentence(meta["hallucination_type"], meta["variant_id"], pools)
        if meta["hallucination_type"] == "workflow_identity_corruption":
            sub = substitute_workflow(d)
            if sub:
                name, graph, replies = sub
                reveal["substitute"] = {
                    "name": name, "description": TASK_DESCRIPTIONS.get(name, ""),
                    "steps": render_text(build_flow(name, graph, replies))}
    return {
        "item_id": meta["item_id"],
        "split": common.split_of(task),
        "meta": meta,
        "task_description": TASK_DESCRIPTIONS.get(task, ""),
        "steps": workflow_text(task, d),
        "user_goal": ((d.get("personas") or {}).get("User") or {}).get("circumstances", ""),
        "turns": turns_of(d),
        "n_system_turns": len(system_turn_indices(d)),
        "cue": cue_sentence(meta["cue_type"], meta["cue_variant"], pools),
        # as noPrefix in annotool/static/app.js: no step-3 question without a pre-fork part
        "has_prefix": fork and meta["fork_assistant_turn"] != 1,
        "reveal": reveal,
        "humans": answers,
    }


def check(records):
    kinds = Counter(r["meta"]["kind"] for r in records)
    judgments = [h for r in records for h in r["humans"]]
    multi = Counter(len(r["humans"]) for r in records if len(r["humans"]) > 1)
    got = {
        "items": len(records),
        "fork/conformant": (kinds["fork"], kinds["conformant"]),
        "foils": sum(bool(r["meta"]["is_foil"]) for r in records),
        "conformant foils": sum(r["meta"]["kind"] == "conformant" for r in records
                                if r["meta"]["is_foil"]),
        "judgments": len(judgments),
        "appears yes/partly/no": tuple(Counter(h["appears"] for h in judgments)[k]
                                       for k in ("yes", "partly", "no")),
        "prefix yes/mostly/no/n-a": tuple(Counter(h["prefix_ok"] for h in judgments)[k]
                                          for k in ("yes", "mostly", "no", "n/a")),
        "multi-judged 2/3": (multi[2], multi[3]),
        "select/report": (sum(r["split"] == "select" for r in records),
                          sum(r["split"] == "report" for r in records)),
    }
    want = {"items": 428, "fork/conformant": (342, 86), "foils": 164, "conformant foils": 86,
            "judgments": 526, "appears yes/partly/no": (287, 58, 181),
            "prefix yes/mostly/no/n-a": (276, 26, 14, 210), "multi-judged 2/3": (42, 28)}
    for k, v in got.items():
        flag = "" if k not in want else ("ok" if want[k] == v else "MISMATCH, expected %s" % (want[k],))
        print("  %-26s %-18s %s" % (k, v, flag))
    missing = [r["item_id"] for r in records if r["cue"] is None]
    if missing:
        print("  WARNING: no cue text for", missing)
    # an annotator never sees a prefix question where the item has none, and vice versa
    bad = [r["item_id"] for r in records for h in r["humans"]
           if (h["prefix_ok"] == "n/a") == r["has_prefix"]]
    print("  has_prefix consistent with the human answers:", "ok" if not bad else bad[:5])
    return all(got[k] == v for k, v in want.items()) and not missing and not bad


def main():
    metas, answers = load_study_items(), load_human_answers()
    pools = load_variant_pools(common.GENERATE_PY)
    ids = sorted(i for i in metas if i in answers)      # items in a batch and judged
    records = [build_record(metas[i], answers[i], pools) for i in ids]
    ok = check(records)
    os.makedirs(os.path.dirname(common.ITEMS), exist_ok=True)
    with open(common.ITEMS, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d items to %s%s" % (len(records), common.ITEMS, "" if ok else "  (CHECK FAILED)"))


if __name__ == "__main__":
    main()
