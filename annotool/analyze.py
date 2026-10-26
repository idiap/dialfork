# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Turn the collected annotations into the result tables.

  python3 -m annotool.analyze --db annotool/data/annot.sqlite

Primary outputs validate the three ground-truth labels the benchmark relies on
(hallucination type/variant, fork turn, fork order); the correlation with the
automatic metrics is secondary.

Aggregation rule, stated once and applied throughout:
  * categorical answers -> the FIRST submitted annotation per item (so n = items)
  * the 0-100 slider    -> the MEAN across annotators (more stable, and what the
                           ordering and correlation analyses want)
  * agreement           -> computed only on the double-annotated overlap
  * fork order          -> within each (annotator, task): one person's own scale,
                           one workflow, all four fork positions mixed
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from . import db
from .sampling import has_prefix
from .dialogs import HALLUCINATION_TYPES, SEGMENTS, TYPE_ABBREV
from .stats import (bootstrap_ci, fleiss_kappa, fmt_rate, gwet_ac1, kendall_tau_b,
                    krippendorff_alpha_interval, percent_agreement, spearman, wilson)

CONFIRMED = ("yes", "partly")


def load_answers(db_path, exclude=None):
    """item_id -> [ {annotator, order, 1:payload, 2:payload, 3:payload} ]

    `exclude` drops annotators by name. Excluding people after seeing the results
    is a researcher-degrees-of-freedom hazard, so decide the RULE in advance
    (section 7 gives the objective basis: the foil false-confirm rate) and record
    who was dropped and why.
    """
    exclude = {e.strip() for e in (exclude or []) if e.strip()}
    con = db.connect(db_path)
    rows = con.execute(
        "SELECT r.item_id, r.annotator_id, r.step, r.payload_json, r.ms, r.id, "
        "n.name AS annotator_name FROM responses r "
        "JOIN annotators n ON n.id=r.annotator_id ORDER BY r.id").fetchall()
    con.close()
    by = defaultdict(dict)
    order = {}
    for r in rows:
        key = (r["item_id"], r["annotator_id"])
        if key not in order:
            order[key] = len(order)
        rec = by[key]
        rec["annotator"] = r["annotator_name"]
        rec["annotator_id"] = r["annotator_id"]
        rec["order"] = order[key]
        rec[r["step"]] = json.loads(r["payload_json"])
        rec.setdefault("ms", []).append(r["ms"] or 0)
    # each person's own sequence position (0, 1, 2, ...), for the learning curve
    seq = defaultdict(int)
    for rec in sorted(by.values(), key=lambda r: r["order"]):
        rec["seq"] = seq[rec["annotator_id"]]
        seq[rec["annotator_id"]] += 1
    out = defaultdict(list)
    for (item_id, _), rec in sorted(by.items(), key=lambda kv: kv[1]["order"]):
        if 3 in rec and rec["annotator"] not in exclude:
            out[item_id].append(rec)
    return out


def first(recs):
    return recs[0] if recs else None


def mean_slider(recs):
    vals = [r[1]["slider"] for r in recs if 1 in r and r[1].get("slider") is not None]
    return sum(vals) / float(len(vals)) if vals else None


def section(title):
    return "\n## %s\n" % title


def load_items(db_path):
    con = db.connect(db_path)
    items = db.study_items(con)      # only conversations that are in a batch
    con.close()
    return items


# CSC keeps every component of the workflow (states, actions, transitions, goal) and
# adds a competing objective on top, so the step-1 question ("how well does the
# assistant follow the steps?") is answered over the three types that break a
# component of it; RQ1 is reported for those and, for completeness, for all four.
CSC = "conflicting_system_constraints"
WORKFLOW_BREAKING = tuple(h for h in HALLUCINATION_TYPES if h != CSC)
RQ1_SCOPES = (("WIC, SSPC, GR", WORKFLOW_BREAKING), ("all four types", HALLUCINATION_TYPES))


def in_scope(item, types):
    """Conformant items always count (they are the anchor); forks only of `types`."""
    return item["kind"] != "fork" or types is None or item["hallucination_type"] in types


def ordering_units(items, done, min_forks=8, types=None):
    """(annotator, task) -> tau-b between that person's own sliders and k.

    The primary fork-order statistic. Each unit is one person judging one
    workflow, so neither rater scale nor task difficulty enters the tau; the
    sources differ from item to item, which adds noise but no bias. A unit
    needs all four segments and at least `min_forks` forks (one block = 16).
    `types` restricts the forks to those hallucination types.
    """
    pts = defaultdict(list)
    for i, recs in done.items():
        it = items[i]
        if it["kind"] != "fork" or not in_scope(it, types):
            continue
        for r in recs:
            if 1 in r and r[1].get("slider") is not None:
                pts[(r["annotator"], it["task"])].append(
                    (it["segment"], r[1]["slider"], it["hallucination_type"]))
    out = {}
    for unit, xs in pts.items():
        if len(xs) < min_forks or len({k for k, _, _ in xs}) < 4:
            continue
        t = kendall_tau_b([k for k, _, _ in xs], [v for _, v, _ in xs])
        if t == t:
            by_type = {}
            for h in HALLUCINATION_TYPES:
                sub = [(k, v) for k, v, hh in xs if hh == h]
                if len({k for k, _ in sub}) == 4:
                    th = kendall_tau_b([k for k, _ in sub], [v for _, v in sub])
                    if th == th:
                        by_type[h] = th
            out[unit] = {"tau": t, "n": len(xs), "by_type": by_type}
    return out


PAIRWISE = (("later fork rated above an earlier one", lambda a, b: b <= 4),
            ("k=4 above k=1", lambda a, b: a == 1 and b == 4),
            ("conformant above a fork", lambda a, b: b == 5))


def superiority(items, done, keep, types=None, n_boot=2000, seed=0):
    """Probability of superiority: among pairs of conversations judged by the same
    annotator within the same task, at different positions, the share in which the
    later one is rated higher (ties count 1/2; chance is 50%). Conformant = position
    5. `keep(k_lo, k_hi)` picks the position pairs. 95% CI by bootstrap over the
    (annotator, task) units. Returns (pct, lo, hi, pairs) or None."""
    import random as _r
    units = defaultdict(list)
    for i, recs in done.items():
        it = items[i]
        if not in_scope(it, types):
            continue
        for r in recs:
            if 1 in r and r[1].get("slider") is not None:
                units[(r["annotator"], it["task"])].append(
                    (it["segment"] if it["kind"] == "fork" else 5, r[1]["slider"]))
    per = []
    for xs in units.values():
        w = n = 0
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                (k1, v1), (k2, v2) = xs[a], xs[b]
                if k1 == k2:
                    continue
                if k1 > k2:
                    k1, v1, k2, v2 = k2, v2, k1, v1
                if keep(k1, k2):
                    n += 1
                    w += 1.0 if v2 > v1 else (0.5 if v2 == v1 else 0.0)
        if n:
            per.append((w, n))
    if not per:
        return None
    rng = _r.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [per[rng.randrange(len(per))] for _ in per]
        boots.append(sum(w for w, _ in s) / sum(n for _, n in s))
    boots.sort()
    return (100.0 * sum(w for w, _ in per) / sum(n for _, n in per),
            100.0 * boots[int(0.025 * n_boot)], 100.0 * boots[int(0.975 * n_boot) - 1],
            int(sum(n for _, n in per)))


def quartet_taus(items, done, types=None):
    """tau-b over each complete quartet (one source's four forks under one type),
    on the mean slider; the four forks are judged by four different annotators."""
    by_q = defaultdict(dict)
    for i, recs in done.items():
        it = items[i]
        if it["kind"] != "fork" or not in_scope(it, types):
            continue
        m = mean_slider(recs)
        if m is not None:
            by_q[it["quartet_id"]][it["segment"]] = m
    out = []
    for d in by_q.values():
        if len(d) == 4:
            ks = sorted(d)
            t = kendall_tau_b(ks, [d[k] for k in ks])
            if t == t:
                out.append(t)
    return out


def slider_by_segment(items, done, types=None):
    """[(segment, mean, lo, hi, n)] for conformant (0) and k=1..4, slider mean per item."""
    by = defaultdict(list)
    for i, recs in done.items():
        it = items[i]
        if not in_scope(it, types):
            continue
        m = mean_slider(recs)
        if m is not None:
            by[it["segment"] or 0].append(m)
    out = []
    for seg in (0,) + SEGMENTS:
        v = by.get(seg, [])
        lo, hi = bootstrap_ci(v, n_boot=800) if len(v) > 2 else (float("nan"),) * 2
        out.append((seg, sum(v) / len(v) if v else None, lo, hi, len(v)))
    return out


def anchor_check(items, done, types=None):
    """(ok, total): (annotator, task) pairs, with conformant and all four positions
    judged, in which conformant conversations get the highest mean slider."""
    groups = defaultdict(lambda: defaultdict(list))
    for i, recs in done.items():
        it = items[i]
        if not in_scope(it, types):
            continue
        for r in recs:
            if 1 in r and r[1].get("slider") is not None:
                groups[(r["annotator"], it["task"])][it["segment"] or 0].append(r[1]["slider"])
    ok = tot = 0
    for g in groups.values():
        if 0 not in g or len(g) < 5:
            continue
        tot += 1
        means = {k: sum(v) / float(len(v)) for k, v in g.items()}
        ok += int(means[0] >= max(means.values()))
    return ok, tot


def conformant_foil_check(items, done, min_foils=2):
    """{annotator: (confirmed, shown, excluded)} over the foils shown on CONFORMANT
    dialogues only, the pre-defined exclusion criterion: nothing was injected into
    them, so 'no' is unambiguously correct (a foil on a fork can partly match the
    deviation it does contain). Excluded = confirmed more than half, assessed only
    with at least `min_foils` of them (one mini-block shows a single one)."""
    per = defaultdict(lambda: [0, 0])
    for i, recs in done.items():
        if items[i]["kind"] != "conformant":
            continue
        for r in recs:
            per[r["annotator"]][1] += 1
            per[r["annotator"]][0] += int(r[2]["appears"] in CONFIRMED)
    return {a: (k, n, n >= min_foils and k > n / 2.0) for a, (k, n) in per.items()}


def binary_slider_users(done, share=0.9):
    """Annotators who put at least `share` of their sliders at 0 or 100, i.e. used
    it as follows / does not follow rather than how closely."""
    v = defaultdict(list)
    for recs in done.values():
        for r in recs:
            if 1 in r and r[1].get("slider") is not None:
                v[r["annotator"]].append(r[1]["slider"] in (0, 100))
    return sorted(a for a, x in v.items() if len(x) >= 5 and sum(x) >= share * len(x))


def notes_of(items, done):
    """Every free-text note, with what is needed to open its conversation."""
    out = []
    for i, recs in done.items():
        it = items[i]
        for r in recs:
            t = (r.get(3, {}).get("note") or "").strip()
            if t:
                out.append({"item_id": i, "ref": it["ref"], "task": it["task"],
                            "type": TYPE_ABBREV.get(it["hallucination_type"] or "", "conformant"),
                            "segment": it["segment"], "is_foil": bool(it["is_foil"]),
                            "annotator": r["annotator"], "note": t})
    out.sort(key=lambda n: (n["task"], n["type"], n["segment"] or 0, n["ref"]))
    return out


def analyze(items, answers, task2_root, method, encoder, corr=None):
    """The markdown report. task2_root=None skips the metric correlation (it
    is slow); `corr` passes in rows already computed by metric_rows()."""
    done = {k: v for k, v in answers.items() if k in items}
    L = []
    L.append("# DialFork human annotation results")
    L.append("")
    L.append("Aggregation: categorical = first annotation per item; slider = mean "
             "across annotators; agreement computed on the double-annotated overlap only.")

    # ---------- coverage
    L.append(section("0. Coverage"))
    n_ann = len({r["annotator_id"] for recs in done.values() for r in recs})
    dbl = sum(1 for recs in done.values() if len(recs) >= 2)
    L.append("- annotators: **%d**" % n_ann)
    L.append("- conversations completed: **%d** of %d drawn into batches "
             "(%d double-annotated), over %d task(s)"
             % (len(done), len(items), dbl, len({items[i]["task"] for i in done})))
    secs = [sum(r["ms"]) / 1000.0 for recs in done.values() for r in recs if r.get("ms")]
    if secs:
        secs.sort()
        L.append("- median time per item: **%.1f s** (%.1f min for all three steps)"
                 % (secs[len(secs) // 2], secs[len(secs) // 2] / 60.0))

    forks = [(i, done[i]) for i in done if items[i]["kind"] == "fork"]
    true_forks = [(i, r) for i, r in forks if not items[i]["is_foil"]]
    foil_forks = [(i, r) for i, r in forks if items[i]["is_foil"]]
    confs = [(i, done[i]) for i in done if items[i]["kind"] == "conformant"]

    # ---------- 1. instruction-following rate
    L.append(section("1. RQ2 -- is the injected behaviour carried out? (per hallucination type)"))
    L.append("Forked items shown their TRUE injected instruction. "
             "'Confirmed' = answered yes or partly.")
    L.append("")
    L.append("| type | confirmed (yes+partly) | yes only |")
    L.append("|---|---|---|")
    for h in HALLUCINATION_TYPES:
        sub = [r for i, r in true_forks if items[i]["hallucination_type"] == h]
        n = len(sub)
        k = sum(1 for r in sub if first(r)[2]["appears"] in CONFIRMED)
        ky = sum(1 for r in sub if first(r)[2]["appears"] == "yes")
        L.append("| %s | %s | %s |" % (TYPE_ABBREV[h], fmt_rate(k, n), fmt_rate(ky, n)))
    n = len(true_forks)
    k = sum(1 for _, r in true_forks if first(r)[2]["appears"] in CONFIRMED)
    L.append("| **all** | **%s** | %s |"
             % (fmt_rate(k, n),
                fmt_rate(sum(1 for _, r in true_forks if first(r)[2]["appears"] == "yes"), n)))
    L.append("")
    L.append("Per fork segment. Segment-4 forks leave the fewest turns for the behaviour "
             "to show, so a weak rate is most likely there:")
    L.append("")
    L.append("| segment | confirmed |")
    L.append("|---|---|")
    for s in SEGMENTS:
        sub = [r for i, r in true_forks if items[i]["segment"] == s]
        L.append("| k=%d | %s | " % (s, fmt_rate(
            sum(1 for r in sub if first(r)[2]["appears"] in CONFIRMED), len(sub))))

    # ---------- 2. foil control
    L.append(section("2. RQ2 foil control -- are the cued judgements rubber-stamping?"))
    L.append("A foil shows a variant drawn from a DIFFERENT hallucination type. "
             "Conformant dialogues are always foils: nothing was injected into them.")
    L.append("")
    L.append("| shown | confirmed |")
    L.append("|---|---|")
    rows = [("true instruction (forked)", true_forks),
            ("foil instruction (forked)", foil_forks),
            ("foil instruction (conformant)", confs)]
    for lab, sub in rows:
        n = len(sub)
        k = sum(1 for _, r in sub if first(r)[2]["appears"] in CONFIRMED)
        L.append("| %s | %s |" % (lab, fmt_rate(k, n)))
    L.append("")
    L.append("A large gap between the first row and the other two is what makes a cued "
             "design defensible; a small gap invalidates section 1.")

    # ---------- 3. onset vs fork_assistant_turn
    L.append(section("3. RQ3 -- does the instructed behaviour appear at the intended turn?"))
    deltas_by = defaultdict(list)
    for i, recs in true_forks:
        a = first(recs)[2]
        if a["appears"] in CONFIRMED and a.get("onset_sys_turn"):
            d = a["onset_sys_turn"] - items[i]["fork_assistant_turn"]
            deltas_by["all"].append(d)
            deltas_by[items[i]["hallucination_type"]].append(d)
            deltas_by["k%d" % items[i]["segment"]].append(d)
    L.append("| group | n | mean abs error | median abs | within +/-1 | exact |")
    L.append("|---|---|---|---|---|---|")
    keys = ["all"] + list(HALLUCINATION_TYPES) + ["k%d" % s for s in SEGMENTS]
    for key in keys:
        ds = deltas_by.get(key, [])
        if not ds:
            continue
        ab = sorted(abs(d) for d in ds)
        lab = TYPE_ABBREV.get(key, key)
        L.append("| %s | %d | %.2f | %d | %.1f%% | %.1f%% |"
                 % (lab, len(ds), sum(ab) / float(len(ab)), ab[len(ab) // 2],
                    100.0 * sum(1 for d in ds if abs(d) <= 1) / len(ds),
                    100.0 * sum(1 for d in ds if d == 0) / len(ds)))
    if deltas_by["all"]:
        early = sum(1 for d in deltas_by["all"] if d < 0)
        L.append("")
        L.append("Signed: %d of %d onsets were placed *before* the annotated fork turn."
                 % (early, len(deltas_by["all"])))

    # ---------- 4. prefix conformance
    L.append(section("4. RQ4 -- prefix conformance (pre-fork turns)"))
    L.append("Forks injected before assistant turn 1 (every k=1 fork) have no pre-fork "
             "turns, so they are not asked and not counted here.")
    L.append("")
    # only forks with at least one assistant turn before the injection
    # (segment-1 forks start at turn 1: nothing to judge, recorded as n/a)
    c = Counter(first(r)[3].get("prefix_ok") for i, r in forks if has_prefix(items[i]))
    n = sum(v for k, v in c.items() if k in ("yes", "mostly", "no"))
    L.append("| answer | n | share |")
    L.append("|---|---|---|")
    for k in ("yes", "mostly", "no"):
        L.append("| %s | %d | %.1f%% |" % (k, c.get(k, 0),
                                           100.0 * c.get(k, 0) / n if n else 0))
    L.append("")
    L.append("yes or mostly: %s" % fmt_rate(c.get("yes", 0) + c.get("mostly", 0), n))
    L.append("")
    L.append("| fork position | yes | mostly | no |")
    L.append("|---|---|---|---|")
    for seg in SEGMENTS:
        ck = Counter(first(r)[3].get("prefix_ok") for i, r in forks
                     if has_prefix(items[i]) and items[i]["segment"] == seg)
        nk = sum(v for k, v in ck.items() if k in ("yes", "mostly", "no"))
        if nk:
            L.append("| k=%d | %s | %s | %s |" % ((seg,) + tuple(
                fmt_rate(ck.get(k, 0), nk) for k in ("yes", "mostly", "no"))))

    # ---------- 5. RQ1: fork order
    L.append(section("5. RQ1 -- fork order: does conformance fall as the fork moves earlier?"))
    L.append("Step-1 slider (0-100), blind to how the conversation was built. Higher k = "
             "later fork = more of the dialogue produced under the intended policy, so "
             "ratings should rise with k, with conformant conversations highest. "
             "**CSC keeps every component of the workflow** and adds a competing objective "
             "on top, so the step-1 question is answered over the types that break a "
             "component of it (WIC, SSPC, GR); all four types are shown for completeness.")
    L.append("")
    L.append("**5a. Mean slider per fork position** (mean over annotators per conversation, "
             "95% bootstrap CI):")
    L.append("")
    segs = [slider_by_segment(items, done, t) for _, t in RQ1_SCOPES]
    csc = slider_by_segment(items, done, (CSC,))
    L.append("| position | %s | %s | CSC only |" % tuple(n for n, _ in RQ1_SCOPES))
    L.append("|---|---|---|---|")
    for j, seg in enumerate((0,) + SEGMENTS):
        cells = []
        for rows in segs + [csc]:
            _, m, lo, hi, n = rows[j]
            cells.append("-" if m is None else ("%.1f [%.1f, %.1f] (n=%d)" % (m, lo, hi, n)
                                                 if lo == lo else "%.1f (n=%d)" % (m, n)))
        if seg == 0:
            cells[2] = "(same anchor)"
        L.append("| %s | %s | %s | %s |" % (("conformant" if seg == 0 else "k=%d" % seg,)
                                            + tuple(cells)))
    L.append("")
    L.append("**5b. Probability of superiority**, within each (annotator, task): the share "
             "of pairs at different positions in which the later one is rated higher "
             "(ties = 1/2; chance 50%; conformant = latest position):")
    L.append("")
    L.append("| comparison | %s | %s |" % tuple(n for n, _ in RQ1_SCOPES))
    L.append("|---|---|---|")
    for lab, keep in PAIRWISE:
        cells = []
        for _, t in RQ1_SCOPES:
            r = superiority(items, done, keep, t)
            cells.append("-" if r is None else "%.1f%% [%.1f, %.1f] (%d pairs)" % r)
        L.append("| %s | %s | %s |" % ((lab,) + tuple(cells)))
    r = superiority(items, done, PAIRWISE[2][1], (CSC,))
    if r:
        L.append("")
        L.append("CSC forks rated below a conformant dialogue by the same annotator: "
                 "**%.1f%%** [%.1f, %.1f] of %d pairs." % r)
    L.append("")
    L.append("**5c. Rank correlation (Kendall tau-b) with k.** Within annotator: one tau-b "
             "per (annotator, task), averaged. Quartets: the four forks of one source under "
             "one type, each judged by a different annotator (the Task-2 setting).")
    L.append("")
    L.append("| scope | within annotator | units | quartets | complete quartets |")
    L.append("|---|---|---|---|---|")
    for name, t in RQ1_SCOPES:
        units = ordering_units(items, done, types=t)
        taus = [u["tau"] for u in units.values()]
        qt = quartet_taus(items, done, t)
        def fmt(v):
            if not v:
                return "-"
            lo, hi = bootstrap_ci(v)
            return "**%+.3f** [%+.3f, %+.3f]" % (sum(v) / len(v), lo, hi) if lo == lo \
                else "**%+.3f**" % (sum(v) / len(v))
        L.append("| %s | %s | %d | %s | %d |" % (name, fmt(taus), len(taus), fmt(qt), len(qt)))
    L.append("")
    L.append("Scale: with 16 forks, four at each k, the ties in k cap the within-annotator "
             "tau-b at about +0.89 even for a perfectly ordered annotator.")
    units = ordering_units(items, done, types=WORKFLOW_BREAKING)
    if units:
        L.append("")
        L.append("Per (annotator, task), WIC/SSPC/GR forks:")
        L.append("")
        L.append("| annotator | task | forks | tau-b |")
        L.append("|---|---|---|---|")
        for (a, t), u in sorted(units.items()):
            L.append("| %s | %s | %d | %+.3f |" % (a, t, u["n"], u["tau"]))

    # ---------- 6. anchor check
    L.append(section("6. RQ1 anchor -- do conformant conversations score highest?"))
    L.append("If they do not, the slider is not measuring conformance and section 5 is "
             "not interpretable. Per (annotator, task) with conformant and all four "
             "positions judged: conformant mean vs the mean of each fork position.")
    L.append("")
    for name, t in RQ1_SCOPES:
        ok, tot = anchor_check(items, done, t)
        L.append("- %s: conformant highest in %s" % (name, fmt_rate(ok, tot)))

    # ---------- 7. agreement
    L.append(section("7. Inter-annotator agreement (double-judged items only)"))
    dbl_items = {i: r for i, r in done.items() if len(r) >= 2}
    if not dbl_items:
        L.append("_No conversation has been judged by two people yet, so there is "
                 "nothing to compute agreement from. Give a batch someone already "
                 "holds to a second person who works the same task._")
    else:
        app = [Counter(x[2]["appears"] for x in r) for r in dbl_items.values()]
        pre = [Counter(x[3].get("prefix_ok") for x in r) for i, r in dbl_items.items()
               if has_prefix(items[i])]
        sl = [[x[1]["slider"] for x in r] for r in dbl_items.values()]
        on = [[x[2]["onset_sys_turn"] for x in r if x[2].get("onset_sys_turn")]
              for r in dbl_items.values()]
        on = [u for u in on if len(u) >= 2]

        L.append("| quantity | units | raw agreement | Gwet AC1 | Fleiss kappa |")
        L.append("|---|---|---|---|---|")
        app2 = [Counter(x[2]["appears"] in CONFIRMED for x in r) for r in dbl_items.values()]
        for lab, d in (("behaviour carried out (yes/partly vs no; the RQ2 measure)", app2),
                       ("does the behaviour appear (yes / partly / no)", app),
                       ("prefix conformance", pre)):
            if d:
                L.append("| %s | %d | %.3f | %+.3f | %+.3f |"
                         % (lab, len(d), percent_agreement(d), gwet_ac1(d), fleiss_kappa(d)))
        L.append("")
        yes = sum(c.get("yes", 0) + c.get("partly", 0) for c in app)
        tot = sum(sum(c.values()) for c in app)
        L.append("**Read AC1, not kappa, for these two.** %.0f%% of all confirmation "
                 "answers fall on one side, and kappa collapses under skewed "
                 "marginals: simulated raters who agree 90%% of the time on a 90%%-"
                 "skewed question score kappa 0.58 (95%% CI [0.24, 0.85] at n=60) but "
                 "AC1 0.87 [0.75, 0.96]. Kappa is reported above only so nobody has "
                 "to ask for it." % (100.0 * yes / tot if tot else 0))
        L.append("")
        L.append("| quantity | units | Krippendorff alpha (interval) |")
        L.append("|---|---|---|")
        L.append("| conformance slider | %d | %+.3f |"
                 % (len(sl), krippendorff_alpha_interval(sl)))
        if on:
            L.append("| onset turn | %d | %+.3f |" % (len(on), krippendorff_alpha_interval(on)))
            gaps = sorted(max(u) - min(u) for u in on)
            L.append("")
            L.append("Median between-annotator gap on the onset turn: **%d** turns; "
                     "%.0f%% of pairs are within 1 turn of each other."
                     % (gaps[len(gaps) // 2],
                        100.0 * sum(1 for g in gaps if g <= 1) / len(gaps)))

        # per-pair, so agreement is not one lucky pairing
        pairs = defaultdict(lambda: {"n": 0, "same": 0, "sl": []})
        for i, recs in dbl_items.items():
            for a in range(len(recs)):
                for b in range(a + 1, len(recs)):
                    key = tuple(sorted([recs[a]["annotator"], recs[b]["annotator"]]))
                    p = pairs[key]
                    p["n"] += 1
                    p["same"] += int(recs[a][2]["appears"] == recs[b][2]["appears"])
                    p["sl"].append([recs[a][1]["slider"], recs[b][1]["slider"]])
        if len(pairs) > 1:
            L.append("")
            L.append("Per pair, so the figure is not one lucky pairing:")
            L.append("")
            L.append("| pair | items | raw agreement | slider alpha |")
            L.append("|---|---|---|---|")
            for (x, y), p in sorted(pairs.items()):
                L.append("| %s / %s | %d | %.3f | %+.3f |"
                         % (x, y, p["n"], p["same"] / float(p["n"]),
                            krippendorff_alpha_interval(p["sl"])))
        L.append("")
        L.append("These were produced independently: annotators never see each "
                 "other's answers and each holds its own randomised order.")

    # ---------- 7b. per-annotator reliability
    L.append(section("7b. Per-annotator reliability (the basis for any exclusion)"))
    L.append("A foil shows an instruction that was never injected, so the correct answer "
             "is 'no'. **Exclusion criterion (stated before the analysis):** only the "
             "foils shown on *conformant* dialogues count, since nothing was injected "
             "into them, while a foil on a fork can partly match the deviation the fork "
             "does contain. An annotator is excluded if they confirm more than half of "
             "them, assessed only with at least two (one mini-block shows a single one). "
             "This needs no comparison with other annotators.")
    L.append("")
    per = defaultdict(lambda: {"foil": 0, "foil_bad": 0, "true": 0, "true_ok": 0,
                               "n": 0, "secs": []})
    for i, recs in done.items():
        it = items[i]
        for r in recs:
            p = per[r["annotator"]]
            p["n"] += 1
            p["secs"].append(sum(r.get("ms") or []) / 1000.0)
            conf = r[2]["appears"] in CONFIRMED
            if it["is_foil"]:
                p["foil"] += 1
                p["foil_bad"] += int(conf)
            else:
                p["true"] += 1
                p["true_ok"] += int(conf)
    crit = conformant_foil_check(items, done)
    binary = set(binary_slider_users(done))
    L.append("| annotator | items | foils on conformant confirmed (the criterion) | "
             "all foils confirmed | true instruction confirmed | median time | flags |")
    L.append("|---|---|---|---|---|---|---|")
    for name in sorted(per):
        p = per[name]
        secs = sorted(p["secs"])
        k, n, out = crit.get(name, (0, 0, False))
        flags = (["**excluded**"] if out else []) + \
                (["binary slider"] if name in binary else [])
        L.append("| %s | %d | %s | %s | %s | %.0f s | %s |"
                 % (name, p["n"], fmt_rate(k, n) if n >= 2 else "%s (too few)" % fmt_rate(k, n),
                    fmt_rate(p["foil_bad"], p["foil"]), fmt_rate(p["true_ok"], p["true"]),
                    secs[len(secs) // 2] if secs else 0, ", ".join(flags)))
    L.append("")
    excl = sorted(a for a, (_, _, o) in crit.items() if o)
    L.append("Meeting the exclusion criterion: **%s**." % (", ".join(excl) if excl else "nobody"))
    if binary:
        L.append("")
        L.append("Binary slider use (>= 90%% of sliders at 0 or 100, i.e. judging *whether* "
                 "the steps were followed rather than *how closely*): %s. Not an exclusion "
                 "criterion; check that excluding them leaves RQ1 unchanged."
                 % ", ".join(sorted(binary)))

    # ---------- 7c. learning over time
    L.append(section("7c. Does accuracy change as annotators go?"))
    L.append("After each conversation annotators see what really happened next to "
             "their own answer (neutral, never scored). That can teach the "
             "construction, so it is measured rather than assumed: each person's "
             "conversations are split into thirds by the order they did them in.")
    L.append("")
    counts = defaultdict(int)
    for recs in done.values():
        for r in recs:
            counts[r["annotator"]] += 1
    thirds = defaultdict(lambda: {"on": [], "foil": [], "true": []})
    for i, recs in done.items():
        it = items[i]
        for r in recs:
            n = counts[r["annotator"]]
            if n < 6:
                continue
            part = ("early", "middle", "late")[min(2, 3 * r["seq"] // n)]
            conf = r[2]["appears"] in CONFIRMED
            if it["is_foil"]:
                thirds[part]["foil"].append(conf)
            elif it["kind"] == "fork":
                thirds[part]["true"].append(conf)
                if conf and r[2].get("onset_sys_turn"):
                    thirds[part]["on"].append(abs(r[2]["onset_sys_turn"] - it["fork_assistant_turn"]) <= 1)
    if thirds:
        L.append("| part of each person's sequence | onset within +/-1 | confirmed the true "
                 "instruction | confirmed a foil |")
        L.append("|---|---|---|---|")
        for part in ("early", "middle", "late"):
            t = thirds.get(part)
            if not t:
                continue
            L.append("| %s | %s | %s | %s |" % (
                part, fmt_rate(sum(t["on"]), len(t["on"])),
                fmt_rate(sum(t["true"]), len(t["true"])),
                fmt_rate(sum(t["foil"]), len(t["foil"]))))
        L.append("")
        L.append("Rising onset accuracy with a flat foil rate reads as learning the task; "
                 "a rising foil rate would mean learning to say yes. Report either way.")
    else:
        L.append("_Not enough answers per person yet._")

    # ---------- 8. metric correlation
    L.append(section("8. Secondary -- do the metrics track human judgement?"))
    if task2_root is None:
        L.append("_Not computed. Tick \"include metric correlation\" in the admin page, "
                 "or pass --task2 on the command line._")
        corr = None
    elif corr is None:
        corr = metric_correlation(items, done, task2_root, method, encoder)
    if task2_root is not None and not corr:
        L.append("_No Task 2 per-dialogue results under %s for the annotated tasks._"
                 % task2_root)
    elif corr:
        L.append("Spearman rho between human *deviation* (100 - mean slider) and each "
                 "metric's dialogue-to-dialogue distance from the conformant parent, "
                 "over the annotated forks that have Task 2 results. Positive = agrees "
                 "with the annotators.")
        L.append("")
        L.append("| method | encoder | n | Spearman rho | 95% CI |")
        L.append("|---|---|---|---|---|")
        for row in corr:
            L.append("| %s | %s | %d | %+.3f | [%+.3f, %+.3f] |" % row)

    # ---------- 9. failure cases
    L.append(section("9. Failure cases (free text)"))
    notes = []
    for i, recs in done.items():
        for r in recs:
            t = (r[3].get("note") or "").strip()
            if t:
                it = items[i]
                notes.append("- **%s / %s%s** `%s` (%s): %s"
                             % (it["task"], TYPE_ABBREV.get(it["hallucination_type"] or "", "conformant"),
                                "" if it["segment"] is None else " k=%d" % it["segment"],
                                it["ref"], r["annotator"], t))
    L.extend(notes if notes else ["_None recorded._"])
    return "\n".join(L)


def metric_rows(items, answers, task2_root, method=None,
                encoder="dialog2flow-joint-dse-base"):
    """metric_correlation() over the items that have answers."""
    done = {k: v for k, v in answers.items() if k in items}
    return metric_correlation(items, done, task2_root, method, encoder)


DEFAULT_TASK2 = "predictions/task2"     # the paper's Task 2 methods, shipped with the repo


def _prediction_files(task2_root):
    """{(method, encoder): path} when task2_root is a Task 2 prediction file
    (evaluate.py format: id,distance) or a folder of them, named
    <method>__<encoder>.csv[.gz] or <method>.csv[.gz] (encoder 'none'); else None."""
    def parse(path):
        stem = os.path.basename(path)
        for ext in (".csv.gz", ".csv"):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                method, _, enc = stem.partition("__")
                return (method, enc or "none")
        return None
    if os.path.isfile(task2_root):
        key = parse(task2_root)
        return {key: task2_root} if key else None
    if not os.path.isdir(task2_root):
        return None
    files = {}
    for name in sorted(os.listdir(task2_root)):
        key = parse(name)
        if key:
            files[key] = os.path.join(task2_root, name)
    return files or None


def _read_predictions(path):
    """{fork id: distance} from an id,distance CSV (plain or gzipped)."""
    import csv
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return {r["id"].strip(): float(r["distance"]) for r in csv.DictReader(fh)}


def task2_methods(task2_root, encoder, method_filter=None):
    """[(method, encoder)] available in task2_root (a results/task2 folder, or Task 2
    prediction files): the requested encoder where the method has one, else its only
    one (the count baselines use 'none')."""
    files = _prediction_files(task2_root)
    if files is not None:
        by_method = defaultdict(list)
        for method, enc in sorted(files):
            by_method[method].append(enc)
    elif os.path.isdir(task2_root):
        by_method = {m: sorted(os.listdir(os.path.join(task2_root, m)))
                     for m in sorted(os.listdir(task2_root))
                     if os.path.isdir(os.path.join(task2_root, m))}
    else:
        return []
    out = []
    for method, encs in by_method.items():
        if method_filter and method != method_filter:
            continue
        pick = [e for e in encs if e == encoder] or encs[:1]
        out += [(method, e) for e in pick]
    return out


def metric_distances(items, item_ids, task2_root, encoder, method_filter=None):
    """{item_id: {method: distance from the conformant parent}} for forks with
    Task 2 results. Each results file is read once."""
    cache, out = {}, {}
    files = _prediction_files(task2_root)
    for method, enc in task2_methods(task2_root, encoder, method_filter):
        if files is not None:
            pred = _read_predictions(files[(method, enc)])
            for i in item_ids:
                it = items.get(i)
                if it and it["kind"] == "fork":
                    fid = "%d-%s-k%d" % (int(it["dialog_id"]), TYPE_ABBREV[it["hallucination_type"]],
                                         int(it["segment"]))
                    if fid in pred:
                        out.setdefault(i, {})[method] = pred[fid]
            continue
        for i in item_ids:
            it = items.get(i)
            if not it or it["kind"] != "fork":
                continue
            p = os.path.join(task2_root, method, enc, it["task"], "results_d2d_rankings.json")
            if p not in cache:
                try:
                    with open(p, encoding="utf-8") as fh:
                        cache[p] = json.load(fh)
                except (OSError, ValueError):
                    cache[p] = None
            try:
                dist = cache[p][it["hallucination_type"]][str(it["dialog_id"])][
                    "distances"]["fork_at_segment_%d" % it["segment"]]
            except (KeyError, TypeError):
                continue
            out.setdefault(i, {})[method] = dist
    return out


def metric_correlation(items, done, task2_root, method_filter, encoder):
    methods = task2_methods(task2_root, encoder, method_filter)
    if not methods:
        return None
    human = {}
    for i, recs in done.items():
        if items[i]["kind"] != "fork":
            continue
        m = mean_slider(recs)
        if m is not None:
            human[i] = 100.0 - m
    if not human:
        return None
    dist = metric_distances(items, human, task2_root, encoder, method_filter)
    rows = []
    for method, enc in methods:
        pairs = [(hv, dist[i][method]) for i, hv in human.items()
                 if method in dist.get(i, {})]
        if len(pairs) >= 5:
            xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
            rho = spearman(xs, ys)
            # percentile CI over bootstrap replicates of rho itself -- NOT
            # bootstrap_ci(), which would resample the mean of those
            # replicates and collapse the interval.
            reps = sorted(_boot_rho(xs, ys))
            if len(reps) >= 40:
                lo = reps[int(0.025 * len(reps))]
                hi = reps[int(0.975 * len(reps))]
            else:
                lo = hi = float("nan")
            rows.append((method, enc, len(xs), rho, lo, hi))
    return rows or None


def _boot_rho(xs, ys, n_boot=1000, seed=0):
    import random as _r
    rng = _r.Random(seed)
    n = len(xs)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(spearman([xs[i] for i in idx], [ys[i] for i in idx]))
    return [v for v in out if v == v]


def summary_data(items, answers):
    """Structured numbers for the admin dashboard's charts. Partial data is fine."""
    done = {k: v for k, v in answers.items() if k in items}
    out = {"n_items": len(items), "n_done": len(done),
           "n_double": sum(1 for r in done.values() if len(r) >= 2)}

    # RQ1: mean slider by fork segment (0 = conformant), bars over the types that
    # break the workflow, `mean_all` (markers) over all four types
    nan = lambda v: None if v is None or v != v else v
    wb = slider_by_segment(items, done, WORKFLOW_BREAKING)
    al = slider_by_segment(items, done, HALLUCINATION_TYPES)
    out["slider_by_segment"] = [
        {"segment": seg, "n": n, "mean": m, "lo": nan(lo), "hi": nan(hi),
         "mean_all": al[j][1], "n_all": al[j][4]}
        for j, (seg, m, lo, hi, n) in enumerate(wb)]
    rq1 = {}
    for key, t in (("breaking", WORKFLOW_BREAKING), ("all", HALLUCINATION_TYPES)):
        sup = superiority(items, done, PAIRWISE[0][1], t)
        qt = quartet_taus(items, done, t)
        qlo, qhi = bootstrap_ci(qt, n_boot=800) if len(qt) > 1 else (None, None)
        rq1[key] = {"later_higher": None if sup is None else
                    {"pct": sup[0], "lo": sup[1], "hi": sup[2], "pairs": sup[3]},
                    "quartet": {"n": len(qt), "mean": sum(qt) / len(qt) if qt else None,
                                "lo": nan(qlo), "hi": nan(qhi)}}
    out["rq1"] = rq1
    out["notes"] = notes_of(items, done)

    # confirmation rates: true instruction vs foil
    groups = {"true (forked)": [], "foil (forked)": [], "foil (conformant)": []}
    for i, recs in done.items():
        it = items[i]
        key = ("foil (conformant)" if it["kind"] == "conformant"
               else ("foil (forked)" if it["is_foil"] else "true (forked)"))
        groups[key].append(first(recs)[2]["appears"] in CONFIRMED)
    out["confirmation"] = [
        {"group": g, "n": len(v), "rate": (100.0 * sum(v) / len(v)) if v else None,
         "lo": 100 * wilson(sum(v), len(v))[0] if v else None,
         "hi": 100 * wilson(sum(v), len(v))[1] if v else None}
        for g, v in groups.items()]

    # per-type confirmation (true instruction only)
    per_type = {}
    for i, recs in done.items():
        it = items[i]
        if it["kind"] != "fork" or it["is_foil"]:
            continue
        per_type.setdefault(TYPE_ABBREV[it["hallucination_type"]], []).append(
            first(recs)[2]["appears"] in CONFIRMED)
    out["confirmation_by_type"] = [
        {"group": k, "n": len(v), "rate": 100.0 * sum(v) / len(v)}
        for k, v in sorted(per_type.items())]

    # |onset - fork_assistant_turn|
    hist = Counter()
    for i, recs in done.items():
        it = items[i]
        a = first(recs)[2]
        if it["kind"] == "fork" and not it["is_foil"] and a["appears"] in CONFIRMED \
                and a.get("onset_sys_turn"):
            hist[min(5, abs(a["onset_sys_turn"] - it["fork_assistant_turn"]))] += 1
    out["onset_error"] = [{"err": k if k < 5 else "5+", "n": hist.get(k, 0)}
                          for k in range(6)]

    # ordering tau-b, within annotator (see ordering_units)
    # tau-b is NaN for a unit whose sliders all tie, and the CI is NaN below two
    # units: both are "not yet", and NaN is not JSON (the browser refuses it)
    taus = [u["tau"] for u in ordering_units(items, done, types=WORKFLOW_BREAKING).values()
            if u["tau"] == u["tau"]]
    if taus:
        lo, hi = bootstrap_ci(taus, n_boot=800)
        out["ordering"] = {"n": len(taus), "mean": sum(taus) / len(taus),
                           "lo": None if lo != lo else lo, "hi": None if hi != hi else hi}
    else:
        out["ordering"] = {"n": 0}
    return out


def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


ANSWER_COLS = ["item_id", "ref", "annotator", "task", "source_id", "kind",
               "hallucination_type", "variant_id", "segment", "fork_assistant_turn", "cue_type",
               "cue_variant", "is_foil", "slider", "appears", "onset_sys_turn", "prefix_ok",
               "note", "seconds"]


def write_answers_csv(items, answers, fh, distances=None, metrics=()):
    """One row per (conversation, annotator): the releasable raw data. With
    `distances`, one extra column per Task-2 metric (distance from the conformant
    parent), so the metric correlation can be recomputed from this file alone."""
    import csv
    w = csv.writer(fh)
    w.writerow(ANSWER_COLS + ["metric_" + m for m in metrics])
    for item_id in sorted(answers):
        it = items.get(item_id)
        if it is None:
            continue
        d = (distances or {}).get(item_id, {})
        for r in answers[item_id]:
            w.writerow([
                item_id, it["ref"], r["annotator"], it["task"], it["source_id"], it["kind"],
                it["hallucination_type"], it["variant_id"], it["segment"],
                it["fork_assistant_turn"], it["cue_type"], it["cue_variant"], int(it["is_foil"]),
                r.get(1, {}).get("slider"), r.get(2, {}).get("appears"),
                r.get(2, {}).get("onset_sys_turn"), r.get(3, {}).get("prefix_ok"),
                (r.get(3, {}).get("note") or "").replace("\n", " "),
                round(sum(r.get("ms") or []) / 1000.0, 1)] + [d.get(m, "") for m in metrics])


def dump_csv(items, answers, path, distances=None, metrics=()):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        write_answers_csv(items, answers, fh, distances, metrics)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="annotool/data/annot.sqlite")
    ap.add_argument("--task2", default=DEFAULT_TASK2,
                    help="Task 2 distances for the metric correlation: a results/task2 "
                         "folder, a folder of prediction files (e.g. predictions/task2) or "
                         "a single prediction file (default: predictions/task2)")
    ap.add_argument("--method", default=None, help="restrict to one Task-2 method")
    ap.add_argument("--encoder", default="dialog2flow-joint-dse-base")
    ap.add_argument("--exclude", default="",
                    help="comma-separated annotator names to drop from the report")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--dump", default=None,
                    help="also write every raw response as flat CSV -- the live "
                         "SQLite is gitignored, so this is the releasable artifact")
    args = ap.parse_args(argv)

    items = load_items(args.db)
    answers = load_answers(args.db, args.exclude.split(","))
    if args.dump:
        dump_csv(items, answers, args.dump)
        print("wrote %s" % args.dump)
    text = analyze(items, answers, args.task2, args.method, args.encoder)
    if args.out:
        _ensure_dir(args.out)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("wrote %s" % args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
