# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Score the judges against the human study.

Two kinds of numbers, for any rater (a judge, or the humans themselves):
  * the study's own measures (RQ1-RQ4): slider by fork position, confirmation of the
    true cue, false confirmation of foils, onset vs t_k, prefix conformance;
  * agreement with people: the judge paired with every human judgment of the same
    conversation, against the human-human pairs of the 70 multi-judged ones (the
    ceiling), with the same statistics (Gwet AC1, Krippendorff alpha interval).

The RQ1 helpers are the paper's (annotool/analyze.py), fed with judge answers in the
same {item_id: [answer]} shape.

    python -m judge.score --validate      # humans only: must reproduce the paper
    python -m judge.score                 # the released runs (judge/answers/study/)
    python -m judge.score --runs results/judge/outputs     # your own runs

Writes results/judge/summary.csv (one row per run and split) and, if the selected
judge has sampled runs, results/judge/self_consistency.json.
"""
import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict

from judge import common
from annotool.analyze import (ordering_units, quartet_taus, slider_by_segment,
                              superiority, PAIRWISE, WORKFLOW_BREAKING)
from annotool.stats import gwet_ac1, percent_agreement, spearman
from annotool.stats import krippendorff_alpha_interval as _alpha_reference

CONFIRMED = ("yes", "partly")


def krippendorff_alpha_interval(units):
    """annotool.stats.krippendorff_alpha_interval in O(n) rather than O(n^2), for the
    bootstrap: sum_{i!=j} (x_i - x_j)^2 = 2 m sum x^2 - 2 (sum x)^2. Same value
    (checked in --validate)."""
    def sq(u):
        return 2.0 * len(u) * sum(x * x for x in u) - 2.0 * sum(u) ** 2
    usable = [u for u in units if len(u) >= 2]
    vals = [x for u in usable for x in u]
    n = len(vals)
    if n < 2:
        return float("nan")
    do = sum(sq(u) / (len(u) - 1) for u in usable) / n
    de = sq(vals) / (n * (n - 1))
    return 1 - do / de if de else float("nan")


SPLITS = ("select", "report", "all")


# --- loading -------------------------------------------------------------------

def load_items(path=common.ITEMS):
    if not os.path.isfile(path):
        raise SystemExit("%s is missing: run python -m judge.items first" % path)
    with open(path, encoding="utf-8") as fh:
        return {r["item_id"]: r for r in map(json.loads, fh)}


def as_rec(annotator, slider, appears, onset, prefix_ok):
    return {"annotator": annotator, 1: {"slider": slider},
            2: {"appears": appears, "onset_sys_turn": onset}, 3: {"prefix_ok": prefix_ok}}


def human_done(items):
    return {i: [as_rec(h["annotator"], h["slider"], h["appears"], h["onset_sys_turn"],
                       h["prefix_ok"]) for h in r["humans"]] for i, r in items.items()}


def judge_done(path, items):
    """{item_id: [rec]} from a run's JSONL (or a list of them, e.g. shards); a stage
    without a parsed answer is None."""
    by = defaultdict(dict)
    for p in ([path] if isinstance(path, str) else path):
        for r in common.read_jsonl(p):
            if r["raw"] or r["answer"] is not None:         # as run.py: no reply = not done
                by[r["item_id"]][r["stage"]] = r
    out, n_fail, n_stage = {}, 0, 0
    for i, st in by.items():
        if i not in items:
            continue
        need = (1, 2, 3) if items[i]["has_prefix"] else (1, 2)
        if any(s not in st for s in need):
            continue                                    # still running
        a = {s: (st[s]["answer"] or {}) for s in need}
        n_stage += len(need)
        n_fail += sum(st[s]["answer"] is None for s in need)
        out[i] = [as_rec("judge", a[1].get("slider"), a[2].get("appears"),
                         a[2].get("onset_sys_turn"),
                         a[3].get("prefix_ok") if 3 in a else "n/a")]
    return out, (n_fail / n_stage if n_stage else float("nan"))


# --- the study's measures, for one rater ---------------------------------------

def rate(k, n):
    return k / float(n) if n else float("nan")


def study_measures(items, done):
    """As annotool/analyze.py reports them: RQ1 over the types that break the workflow
    (WIC, SSPC, GR; all four as *_all), RQ2-RQ4 per conversation on its first judgment
    (a judge gives one anyway)."""
    meta = {i: items[i]["meta"] for i in done}
    m = {}
    for types, suf in ((WORKFLOW_BREAKING, ""), (None, "_all")):
        for seg, mean, _, _, n in slider_by_segment(meta, done, types):
            m[("slider_k%d" % seg if seg else "slider_conf") + suf] = mean
        taus = quartet_taus(meta, done, types)
        m["quartet_tau_mean" + suf] = sum(taus) / len(taus) if taus else float("nan")
        units = ordering_units(meta, done, min_forks=8, types=types)
        m["order_tau_mean" + suf] = (sum(u["tau"] for u in units.values()) / len(units)
                                     if units else float("nan"))
        for (lab, keep), key in zip(PAIRWISE, ("sup_later", "sup_k4_k1", "sup_conf")):
            s = superiority(meta, done, keep, types, n_boot=200)
            m[key + suf] = s[0] / 100.0 if s else float("nan")

    first = {i: recs[0] for i, recs in done.items()}
    conf = lambda r: r[2]["appears"] in CONFIRMED                       # noqa: E731
    true_cue = [i for i in first if meta[i]["kind"] == "fork" and not meta[i]["is_foil"]]
    conf_foil = [i for i in first if meta[i]["kind"] == "conformant"]
    fork_foil = [i for i in first if meta[i]["kind"] == "fork" and meta[i]["is_foil"]]
    for key, ids in (("confirm_true", true_cue), ("false_confirm_conformant_foil", conf_foil),
                     ("confirm_fork_foil", fork_foil)):
        m[key] = rate(sum(conf(first[i]) for i in ids), len(ids))
    on = [(first[i][2]["onset_sys_turn"], meta[i]["fork_assistant_turn"]) for i in true_cue
          if conf(first[i]) and first[i][2]["onset_sys_turn"]]
    m["onset_exact"] = rate(sum(a == b for a, b in on), len(on))
    m["onset_pm1"] = rate(sum(abs(a - b) <= 1 for a, b in on), len(on))
    pre = [r[3]["prefix_ok"] for r in first.values() if r[3]["prefix_ok"] not in (None, "n/a")]
    m["prefix_ok"] = rate(sum(p in ("yes", "mostly") for p in pre), len(pre))
    m["n_items"] = len(done)
    return m


# --- agreement -----------------------------------------------------------------

def unit_stats(units):
    """units: one list per conversation of (slider, appears, onset, prefix) tuples."""
    s = {}
    conf = [Counter(a in CONFIRMED for _, a, _, _ in u) for u in units]
    three = [Counter(a for _, a, _, _ in u) for u in units]
    pre = [Counter(p for _, _, _, p in u) for u in units
           if all(p not in (None, "n/a") for _, _, _, p in u)]
    s["confirm_raw"] = percent_agreement(conf)
    s["confirm_ac1"] = gwet_ac1(conf)
    s["appears3_ac1"] = gwet_ac1(three)
    s["prefix_ac1"] = gwet_ac1(pre) if pre else float("nan")
    s["slider_alpha"] = krippendorff_alpha_interval([[x for x, _, _, _ in u] for u in units])
    on = [[o for _, _, o, _ in u if o] for u in units]
    s["onset_alpha"] = krippendorff_alpha_interval([u for u in on if len(u) >= 2])
    return s


def tup(r):
    return (r[1]["slider"], r[2]["appears"], r[2]["onset_sys_turn"], r[3]["prefix_ok"])


def complete(r):
    return r[1]["slider"] is not None and r[2]["appears"] is not None


def judge_pairs(done, humans):
    """One unit per (conversation, human judgment): [judge, that human]."""
    out = []
    for i, (j,) in done.items():
        if complete(j):
            out += [((i, h["annotator"]), [tup(j), tup(h)]) for h in humans[i]]
    return out


def human_pairs(humans, ids):
    out = []
    for i in ids:
        hs = humans[i]
        out += [((i, a["annotator"], b["annotator"]), [tup(a), tup(b)])
                for x, a in enumerate(hs) for b in hs[x + 1:]]
    return out


def bootstrap(pairs, n_boot=1000, seed=0):
    """95% CI of each statistic, resampling conversations."""
    by = defaultdict(list)
    for key, u in pairs:
        by[key[0]].append(u)
    ids, rng, boots = sorted(by), random.Random(seed), defaultdict(list)
    for _ in range(n_boot):
        sample = [u for i in (rng.choice(ids) for _ in ids) for u in by[i]]
        for k, v in unit_stats(sample).items():
            if v == v:
                boots[k].append(v)
    ci = {}
    for k, v in boots.items():
        v.sort()
        ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1])
    return ci


def agreement(done, humans, items, ids, n_boot=1000):
    ids = [i for i in ids if i in done]
    sub = {i: done[i] for i in ids}
    pairs = judge_pairs(sub, humans)
    s = unit_stats([u for _, u in pairs])
    js = [(done[i][0][1]["slider"], sum(h[1]["slider"] for h in humans[i]) / len(humans[i]))
          for i in ids if done[i][0][1]["slider"] is not None]
    s["slider_spearman"] = spearman([a for a, _ in js], [b for _, b in js]) if js else float("nan")
    multi = [i for i in ids if len(humans[i]) > 1]
    # like for like with the ceiling: the same multi-judged conversations only
    mp = judge_pairs({i: done[i] for i in multi}, humans)
    for k, v in unit_stats([u for _, u in mp]).items():
        s["multi_" + k] = v
    if n_boot:
        for k, (lo, hi) in bootstrap(pairs, n_boot).items():
            s[k + "_lo"], s[k + "_hi"] = lo, hi
    s["n_pairs"] = len(pairs)
    return s


def ceiling(humans, ids, n_boot=1000):
    multi = [i for i in ids if len(humans[i]) > 1]
    pairs = human_pairs(humans, multi)
    s = unit_stats([u for _, u in pairs])
    if n_boot:
        for k, (lo, hi) in bootstrap(pairs, n_boot).items():
            s[k + "_lo"], s[k + "_hi"] = lo, hi
    s["n_pairs"], s["n_multi"] = len(pairs), len(multi)
    return s


# --- validation: the humans must reproduce the paper ----------------------------

def validate(items):
    humans = human_done(items)
    multi = [i for i in items if len(humans[i]) > 1]
    # the paper's way: one unit per conversation with all its raters
    units = [[tup(r) for r in humans[i]] for i in multi]
    paper = unit_stats(units)
    sl = [[x for x, _, _, _ in u] for u in units]
    assert abs(krippendorff_alpha_interval(sl) - _alpha_reference(sl)) < 1e-9
    print("Agreement on the %d multi-judged conversations (paper: AC1 0.69 confirmation, "
          "alpha 0.84 onset, 0.66 slider)" % len(multi))
    for k in ("confirm_raw", "confirm_ac1", "appears3_ac1", "prefix_ac1", "slider_alpha",
              "onset_alpha"):
        print("  %-14s %.3f" % (k, paper[k]))
    print("Same, as rater pairs (the form the judge is compared in):")
    c = ceiling(humans, list(items), n_boot=0)
    for k in ("confirm_ac1", "appears3_ac1", "slider_alpha", "onset_alpha"):
        print("  %-14s %.3f" % (k, c[k]))
    print("Study measures (paper: slider 14/19/41/58, conformant 94; confirmed 91%, "
          "conformant foils 8%, fork foils 45%; onset 67% exact / 77% +-1; prefix 95%)")
    for k, v in study_measures(items, humans).items():
        print("  %-30s %s" % (k, "%.3f" % v if isinstance(v, float) else v))
    # a human in the judge's seat: the first judgment of each conversation, paired with
    # the others -> must land on the ceiling (up to which pairs are used)
    first = {i: [humans[i][0]] for i in multi}
    rest = {i: humans[i][1:] for i in multi}
    a = agreement(first, rest, items, multi, n_boot=0)
    print("First annotator as 'judge' vs the others:",
          "AC1 %.3f, slider alpha %.3f, onset alpha %.3f"
          % (a["confirm_ac1"], a["slider_alpha"], a["onset_alpha"]))


# --- all runs --------------------------------------------------------------------

def runs(root):
    """(model, prompt, seed, [jsonl files]) per run under root/<model>/<prompt>/seed<S>/;
    a run may be split in shards."""
    for seed_dir in sorted(glob.glob(os.path.join(root, "*", "*", "seed*"))):
        path = common.answer_files(seed_dir)
        if not path:
            continue
        prompt_dir = os.path.dirname(seed_dir)
        yield (os.path.basename(os.path.dirname(prompt_dir)), os.path.basename(prompt_dir),
               os.path.basename(seed_dir), path)


def ids_of(items, split=None, tasks=None):
    return [i for i, r in items.items()
            if (split in (None, "all") or r["split"] == split)
            and (tasks is None or r["meta"]["task"] in tasks)]


def select_score(a):
    """The selection criterion: mean agreement with people on the three questions that
    carry signal (confirmation AC1, slider alpha, onset alpha)."""
    v = [a["confirm_ac1"], a["slider_alpha"], a["onset_alpha"]]
    return sum(v) / len(v)


def loto(rows, tasks):
    """Leave one task out: pick the best judge on the other four tasks, score it on the
    one left out, next to the frozen judge and the humans of that task."""
    for pool, lab in ((False, "any judge"), (True, "generator excluded")):
        judges = [r for r in rows if r["model"] != "humans" and r["seed"] == "seed0"
                  and "_t" not in r["prompt"] and not (pool and r.get("generator"))]
        _loto(rows, tasks, judges, lab)


def _loto(rows, tasks, judges, lab):
    print("\nleave one task out, %s (selected on the other four -> scored on this one):" % lab)
    print("  %-18s %-26s %6s %6s %6s %6s | %6s %6s %6s" % ("task", "selected", "score", "AC1",
          "sl.a", "on.a", "h.AC1", "h.sl.a", "h.on.a"))
    for t in tasks:
        rest = [r for r in judges if r["split"] == "rest:" + t]
        if not rest:
            continue
        best = max(rest, key=lambda r: r["select_score"])
        got = next(r for r in judges if r["split"] == "task:" + t
                   and (r["model"], r["prompt"]) == (best["model"], best["prompt"]))
        h = next(r for r in rows if r["model"] == "humans" and r["split"] == "task:" + t)
        print("  %-18s %-26s %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f" % (
            t, "%s %s" % (best["model"], best["prompt"]), got["select_score"],
            got["confirm_ac1"], got["slider_alpha"], got["onset_alpha"],
            h["confirm_ac1"], h["slider_alpha"], h["onset_alpha"]))


def self_consistency(items, root, model, prompt):
    """Agreement of the judge with itself across the seeds of a sampled run (at
    temperature 0 the seed changes nothing): one unit per conversation holding each
    seed's answer."""
    seeds = sorted(glob.glob(os.path.join(root, model, prompt, "seed*")))
    runs_ = [judge_done(common.answer_files(d), items)[0] for d in seeds]
    if len(runs_) < 2:
        return None
    ids = [i for i in items if all(i in r and complete(r[i][0]) for r in runs_)]
    s = unit_stats([[tup(r[i][0]) for r in runs_] for i in ids])
    same = sum(len({r[i][0][2]["appears"] for r in runs_}) == 1 for i in ids)
    s.update(n_seeds=len(runs_), n_items=len(ids), appears_identical=same / float(len(ids)))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--runs", default=common.STUDY_ANSWERS,
                    help="folder of runs, as <model>/<prompt>/seed<S>/judgments*.jsonl[.gz]")
    args = ap.parse_args(argv)
    items = load_items()
    if args.validate:
        return validate(items)

    humans = human_done(items)
    rows = []
    tasks = common.SELECT_TASKS + common.REPORT_TASKS
    for sp in SPLITS + tuple("task:" + t for t in tasks):
        ids = ids_of(items, sp) if sp in SPLITS else ids_of(items, tasks=[sp[5:]])
        c = ceiling(humans, ids, args.n_boot if sp in SPLITS else 0)
        rows.append(dict(model="humans", prompt="-", seed="-", split=sp, fail_rate=0,
                         select_score=select_score(c), **c,
                         **study_measures(items, {i: humans[i] for i in ids})))
    for model, prompt, seed, path in runs(args.runs):
        done, fail = judge_done(path, items)
        # task:<t> = that task alone; rest:<t> = the other four (the LOTO selection set)
        for sp in SPLITS + tuple("task:" + t for t in tasks) + tuple("rest:" + t for t in tasks):
            if sp in SPLITS:
                ids = ids_of(items, sp)
            elif sp.startswith("task:"):
                ids = ids_of(items, tasks=[sp[5:]])
            else:
                ids = ids_of(items, tasks=[x for x in tasks if x != sp[5:]])
            ids = [i for i in ids if i in done]
            if not ids:
                continue
            a = agreement(done, humans, items, ids, args.n_boot if ":" not in sp else 0)
            sm = study_measures(items, {i: done[i] for i in ids})
            rows.append(dict(model=model, prompt=prompt, seed=seed, split=sp,
                             fail_rate=fail, generator=model == common.model_dir(common.GENERATOR),
                             select_score=select_score(a), **a, **sm))
    os.makedirs(common.WORK, exist_ok=True)
    out = os.path.join(common.WORK, "summary.csv")
    cols = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("model", "prompt", "seed", "split"), k))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), out))
    if os.path.isfile(common.FROZEN):
        fz = common.load_frozen()
        # sampled runs (temperature 0.7): greedy ones are identical whatever the seed
        sc = self_consistency(items, args.runs, common.model_dir(fz["model"]), fz["prompt"] + "_t0.7")
        if sc:
            with open(os.path.join(common.WORK, "self_consistency.json"), "w") as fh:
                json.dump(sc, fh, indent=1)
            print("\nself-consistency of %s %s at temperature 0.7, %d seeds, %d conversations: "
                  "appears identical %.3f, AC1 %.3f, slider alpha %.3f, onset alpha %.3f"
                  % (fz["model"], fz["prompt"], sc["n_seeds"], sc["n_items"],
                     sc["appears_identical"], sc["confirm_ac1"], sc["slider_alpha"],
                     sc["onset_alpha"]))
    loto(rows, tasks)
    # quick view: the selection split, best first
    sel = sorted((r for r in rows if r["split"] == "select"),
                 key=lambda r: -r["select_score"] if r["select_score"] == r["select_score"] else 9)
    print("\nselection split (%s), mean agreement with people:" % ", ".join(common.SELECT_TASKS))
    print("  %-14s %-4s %-6s %5s | %6s %6s %6s | %6s %5s" % ("model", "", "seed", "score",
          "AC1", "sl.a", "on.a", "FC.foil", "fail"))
    for r in sel:
        print("  %-14s %-4s %-6s %5.3f | %6.3f %6.3f %6.3f | %6.3f %5.3f" % (
            r["model"], r["prompt"], r["seed"], r["select_score"], r["confirm_ac1"],
            r["slider_alpha"], r["onset_alpha"], r["false_confirm_conformant_foil"],
            r["fail_rate"]))


if __name__ == "__main__":
    main()
