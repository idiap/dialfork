# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""The frozen judge over all of DialFork, against the construction labels.

A fork's labels say what was injected (type, variant) and from which assistant turn
(t_k); a conformant dialogue has nothing injected. With the study's cues, a correct
judge confirms a true cue, starts it at t_k, rejects a foil on a conformant
dialogue, and rates later forks higher. Disagreements are written out for
inspection; they are candidates, not errors: the label or the judge may be wrong.

    python -m judge.full_score      # the released answers (judge/answers/full/)
    python -m judge.full_score --answers results/judge/outputs/full/<model>/<prompt> [--items ...]

Writes results/judge/full_measures.csv (one row per group: all, study tasks, other
tasks, each type, each task) and results/judge/full_flags.csv.
"""
import bisect
import csv
import argparse
import json
import os
from collections import defaultdict

from judge import common
from judge import score
from annotool.dialogs import HALLUCINATION_TYPES, TYPE_ABBREV
from annotool.analyze import WORKFLOW_BREAKING
from annotool.stats import kendall_tau_b

def load_full(path=common.FULL_ITEMS):
    if not os.path.isfile(path):
        raise SystemExit("%s is missing: run python -m judge.full_items first" % path)
    with open(path, encoding="utf-8") as fh:
        return {r["item_id"]: r for r in map(json.loads, fh)}


def ans(done, i):
    a = done[i][0]
    return a[1]["slider"], a[2]["appears"], a[2]["onset_sys_turn"], a[3]["prefix_ok"]


def rate(xs):
    return (sum(xs) / float(len(xs)) if xs else float("nan")), len(xs)


def measures(items, done, ids):
    """The construction-label measures over one group of conversations."""
    m = {}
    true = [i for i in ids if items[i]["meta"]["kind"] == "fork" and not items[i]["meta"]["is_foil"]]
    conf = [i for i in ids if items[i]["meta"]["kind"] == "conformant"]
    ffoil = [i for i in ids if items[i]["meta"]["kind"] == "fork" and items[i]["meta"]["is_foil"]]
    yes = lambda i: ans(done, i)[1] in score.CONFIRMED                   # noqa: E731
    m["confirm_true"], m["n_true"] = rate([yes(i) for i in true])
    m["false_confirm_conformant"], m["n_conformant"] = rate([yes(i) for i in conf])
    m["confirm_fork_foil"], m["n_fork_foil"] = rate([yes(i) for i in ffoil])
    on = [(ans(done, i)[2], items[i]["meta"]["fork_assistant_turn"]) for i in true
          if yes(i) and ans(done, i)[2]]
    m["onset_exact"], _ = rate([a == b for a, b in on])
    m["onset_pm1"], _ = rate([abs(a - b) <= 1 for a, b in on])
    m["onset_early"], _ = rate([a < b - 1 for a, b in on])
    m["onset_late"], _ = rate([a > b + 1 for a, b in on])
    forks = [i for i in ids if items[i]["meta"]["kind"] == "fork"]
    # rating by fork position, over WIC/SSPC/GR as in the human study (CSC keeps every
    # step of the workflow), and over all four types as *_all
    for scope, suf in ((WORKFLOW_BREAKING, ""), (None, "_all")):
        fs = [i for i in forks if scope is None or items[i]["meta"]["hallucination_type"] in scope]
        for k in (1, 2, 3, 4):
            v = [ans(done, i)[0] for i in fs if items[i]["meta"]["segment"] == k]
            m["slider_k%d%s" % (k, suf)] = sum(v) / float(len(v)) if v else float("nan")
        # tau-b over each complete quartet (one source, one type, k = 1..4), averaged
        q = defaultdict(dict)
        for i in fs:
            me = items[i]["meta"]
            q[me["quartet_id"]][me["segment"]] = ans(done, i)[0]
        taus = [kendall_tau_b(sorted(d), [d[k] for k in sorted(d)])
                for d in q.values() if len(d) == 4]
        taus = [t for t in taus if t == t]
        m["quartet_tau" + suf] = sum(taus) / len(taus) if taus else float("nan")
        m["n_quartets" + suf] = len(taus)
    v = [ans(done, i)[0] for i in conf]
    m["slider_conf"] = sum(v) / float(len(v)) if v else float("nan")
    # the rating as a cue-free detector: flag a dialogue when rated below a threshold
    for t in (50, 70, 90):
        m["rating_flags_conformant_lt%d" % t], _ = rate([ans(done, i)[0] < t for i in conf])
        m["rating_flags_forks_lt%d" % t], _ = rate([ans(done, i)[0] < t for i in forks])
    cs, fs = [ans(done, i)[0] for i in conf], [ans(done, i)[0] for i in forks]
    m["rating_auc"] = auc(cs, fs)
    pre = [ans(done, i)[3] for i in ids if items[i]["has_prefix"]]
    m["prefix_ok"], _ = rate([p in ("yes", "mostly") for p in pre])
    m["n"] = len(ids)
    return m


def auc(pos, neg):
    """P(a conformant dialogue is rated above a fork), ties count 1/2."""
    if not pos or not neg:
        return float("nan")
    neg = sorted(neg)
    win = sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x)
              - bisect.bisect_left(neg, x)) for x in pos)
    return win / float(len(pos) * len(neg))


def flags(items, done):
    """Conversations where the judge and the construction labels disagree."""
    out = []
    for i in sorted(done):
        r, (sl, ap, on, pre) = items[i], ans(done, i)
        me = r["meta"]
        why = []
        if me["kind"] == "conformant":
            if ap in score.CONFIRMED:
                why.append("foil confirmed on a conformant dialogue")
            if sl is not None and sl < 50:
                why.append("conformant dialogue rated below 50")
        elif not me["is_foil"]:
            if ap == "no":
                why.append("true instruction not confirmed")
            elif on and abs(on - me["fork_assistant_turn"]) > 1:
                why.append("onset off by more than one turn")
            if pre == "no":
                why.append("turns before the fork judged not conformant")
        for w in why:
            out.append({"reason": w, "ref": me["ref"], "task": me["task"],
                        "type": TYPE_ABBREV.get(me["hallucination_type"] or "", "conformant"),
                        "variant": me["variant_id"] or "", "k": me["segment"] or "",
                        "t_k": me["fork_assistant_turn"] or "", "slider": sl, "appears": ap,
                        "onset": on or "", "prefix_ok": pre, "in_study": r["in_study"],
                        "path": me["path"]})
    return out


def write_csv(path, rows):
    cols = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def study_check(items, done, fz):
    """The 428 study conversations were judged twice (study run, full run): greedy
    decoding should give the same answers."""
    study = score.load_items()
    path = common.answer_files(os.path.join(common.STUDY_ANSWERS, common.model_dir(fz["model"]),
                                            fz["prompt"], "seed0"))
    if not path:
        return
    prev, _ = score.judge_done(path, study)
    by_ref = {r["meta"]["ref"]: i for i, r in study.items()}
    same = tot = 0
    for i, r in items.items():
        if r["in_study"] and i in done and by_ref[i] in prev:
            tot += 1
            same += ans(done, i) == ans(prev, by_ref[i])
    print("study conversations judged again: %d of %d identical to the study run" % (same, tot))


def main(argv=None):
    fz = common.load_frozen()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--answers", default=os.path.join(common.FULL_ANSWERS, common.model_dir(fz["model"]),
                                                      fz["prompt"]),
                    help="folder with the run's judgments*.jsonl[.gz]")
    ap.add_argument("--items", default=common.FULL_ITEMS, help="from python -m judge.full_items")
    args = ap.parse_args(argv)
    items = load_full(args.items)
    done, fail = score.judge_done(common.answer_files(args.answers), items)
    done = {i: r for i, r in done.items() if score.complete(r[0])}
    print("%s %s: %d of %d conversations judged (unparsed stages %.4f)"
          % (fz["model"], fz["prompt"], len(done), len(items), fail))

    def group(pred):
        return [i for i in done if pred(items[i])]

    rows = []
    for lab, pred in (("all", lambda r: True),
                      ("in the human study's 5 tasks", lambda r: r["split"] is not None),
                      ("the other 19 tasks", lambda r: r["split"] is None)):
        rows.append(dict(group=lab, **measures(items, done, group(pred))))
    for h in HALLUCINATION_TYPES:
        rows.append(dict(group="type " + TYPE_ABBREV[h], **measures(
            items, done, group(lambda r, h=h: r["meta"]["kind"] == "conformant"
                               or r["meta"]["hallucination_type"] == h))))
    tasks = sorted({r["meta"]["task"] for r in items.values()})
    for t in tasks:
        rows.append(dict(group="task " + t, **measures(
            items, done, group(lambda r, t=t: r["meta"]["task"] == t))))
    os.makedirs(common.WORK, exist_ok=True)
    write_csv(os.path.join(common.WORK, "full_measures.csv"), rows)
    fl = flags(items, done)
    if fl:
        write_csv(os.path.join(common.WORK, "full_flags.csv"), fl)

    print("\n%-34s %6s %6s %6s %6s %6s | %s | %5s %5s | %s" % (
        "group", "n", "conf", "FCconf", "FCfoil", "on=", "slider k1-k4 / conformant",
        "tauQ", "pre", "rating<70: conformant / forks, AUC"))
    for r in rows:
        print("%-34s %6d %6.3f %6.3f %6.3f %6.3f | %3.0f %3.0f %3.0f %3.0f / %3.0f | %5.2f %5.2f"
              " | %.3f / %.3f, %.3f" % (
            r["group"], r["n"], r["confirm_true"], r["false_confirm_conformant"],
            r["confirm_fork_foil"], r["onset_exact"], r["slider_k1"], r["slider_k2"],
            r["slider_k3"], r["slider_k4"], r["slider_conf"], r["quartet_tau"], r["prefix_ok"],
            r["rating_flags_conformant_lt70"], r["rating_flags_forks_lt70"], r["rating_auc"]))
    reasons = defaultdict(int)
    for f in fl:
        reasons[f["reason"]] += 1
    print("\nflagged for inspection (results/judge/full_flags.csv):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("  %6d  %s" % (v, k))
    study_check(items, done, fz)


if __name__ == "__main__":
    main()
