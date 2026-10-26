# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Human-study figure and numbers for the DialFork appendix, from the annotation export.

One panel per research question:
(a) RQ1: mean step-1 conformance slider per fork position (bars: WIC, SSPC, GR;
    markers: all four types), with the conformant anchor.
(b) RQ2: share of cued instructions confirmed, true instruction per type vs. foils.
(c) RQ3: onset error |annotated onset - t_k| over confirmed true instructions.
(d) RQ4: prefix conformance (yes / mostly / no) per fork position.
The Spearman rho with the Task-2 distances (secondary analysis) is printed only.

Same aggregation as the admin panel (annotool/analyze.py): categorical answers use the
first judgment per conversation, the slider the mean over annotators. Also prints the
numbers quoted in the paper, with all annotators and without the binary slider users.

Usage (from the repo root):
    python3 scripts/make_human_figure.py -a annotool/data/dialfork_human_judgments.csv \
        [-o human_study.pdf] [--main-out human_study_main.pdf]
"""
import argparse
import csv
import glob
import itertools
import os
import sys
from collections import Counter, OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from annotool.stats import (bootstrap_ci, gwet_ac1, kendall_tau_b,  # noqa: E402
                            percent_agreement, spearman, wilson)

GRAY, GRAY_DARK = "#8C8C8C", "#5E5E5C"     # conformant / foils / length-only controls
BLUE, BLUE_DARK = "#2A78D6", "#1C5BA8"     # forked / true instruction
BLUE_LIGHT = "#8DB8EA"
INK, INK_MUTED, GRID = "#1A1A19", "#5E5E5C", "#E4E3DF"
TEXT_WIDTH_IN = 6.3                        # ACL two-column text width (16 cm)
COLUMN_WIDTH_IN = 3.03                     # ACL column width (7.7 cm)

TYPES = OrderedDict([("workflow_identity_corruption", "WIC"),
                     ("step_selection_policy_corruption", "SSPC"),
                     ("goal_replacement", "GR"),
                     ("conflicting_system_constraints", "CSC")])
CONFIRMED = ("yes", "partly")
# CSC keeps the workflow (states, actions, transitions) and its goal, adding a competing
# objective on top; the step-1 question asks how well the steps are followed, so panel (a)
# shows the three types that break a component of the workflow, with all four as markers
STEP_PRESERVING = "conflicting_system_constraints"
METHODS = OrderedDict([("mean-turn-emb", "Mean-turn emb."), ("full-dialog-emb", "Full-dialog emb."),
                       ("flowdtw", "FlowDTW"), ("flowscore", "FlowScore"),
                       ("turn-count-diff", "Turn-count diff."),
                       ("word-count-diff", "Word-count diff.")])


def load(path, exclude=()):
    """item_id -> list of rows, in submission order (the export keeps that order)."""
    items = defaultdict(list)
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["annotator"] in exclude:
                continue
            r["segment"] = int(r["segment"]) if r["segment"] not in ("", "None") else 0
            r["slider"] = float(r["slider"]) if r["slider"] not in ("", "None") else None
            r["is_foil"] = r["is_foil"] == "1"
            items[r["item_id"]].append(r)
    return items


def binary_users(items, share):
    v = defaultdict(list)
    for rows in items.values():
        for r in rows:
            if r["slider"] is not None:
                v[r["annotator"]].append(r["slider"] in (0, 100))
    return sorted(a for a, x in v.items() if np.mean(x) >= share)


def mean_slider(rows):
    v = [r["slider"] for r in rows if r["slider"] is not None]
    return sum(v) / len(v) if v else None


def slider_by_segment(items):
    by = defaultdict(list)
    for rows in items.values():
        m = mean_slider(rows)
        if m is not None:
            by[rows[0]["segment"]].append(m)
    out = []
    for k in range(5):
        lo, hi = bootstrap_ci(by[k])
        out.append((k, np.mean(by[k]), lo, hi, len(by[k])))
    return out


def ordering(items, min_forks=8):
    """Kendall tau-b between slider and k within each (annotator, task); overall and per type."""
    pts = defaultdict(list)
    for rows in items.values():
        for r in rows:
            if r["kind"] == "fork" and r["slider"] is not None:
                pts[(r["annotator"], r["task"])].append(
                    (r["segment"], r["slider"], r["hallucination_type"]))
    taus, by_type = [], defaultdict(list)
    for xs in pts.values():
        if len(xs) < min_forks or len({k for k, _, _ in xs}) < 4:
            continue
        t = kendall_tau_b([k for k, _, _ in xs], [v for _, v, _ in xs])
        if t != t:
            continue
        taus.append(t)
        for h in TYPES:
            sub = [(k, v) for k, v, hh in xs if hh == h]
            if len({k for k, _ in sub}) == 4:
                th = kendall_tau_b([k for k, _ in sub], [v for _, v in sub])
                if th == th:
                    by_type[h].append(th)
    return taus, by_type


def pairwise(items, keep):
    """Share of within-(annotator, task) pairs with the later position rated higher
    (ties count 1/2), i.e. the probability of superiority; conformant = position 5.
    `keep(k_lo, k_hi)` selects the position pairs. 95% CI by bootstrap over units."""
    units = defaultdict(list)
    for rows in items.values():
        for r in rows:
            if r["slider"] is not None:
                units[(r["annotator"], r["task"])].append(
                    (r["segment"] if r["kind"] == "fork" else 5, r["slider"]))
    per = []
    for xs in units.values():
        w = n = 0
        for (k1, v1), (k2, v2) in itertools.combinations(xs, 2):
            if k1 == k2:
                continue
            if k1 > k2:
                k1, v1, k2, v2 = k2, v2, k1, v1
            if keep(k1, k2):
                n += 1
                w += 1.0 if v2 > v1 else (0.5 if v2 == v1 else 0.0)
        if n:
            per.append((w, n))
    rng = np.random.default_rng(0)
    w, n = np.array(per).T
    boots = [w[i].sum() / n[i].sum() for i in
             (rng.integers(0, len(per), len(per)) for _ in range(5000))]
    return 100 * w.sum() / n.sum(), 100 * np.percentile(boots, 2.5), \
        100 * np.percentile(boots, 97.5), int(n.sum())


def quartets(items):
    """tau-b over the four forks of one source under one type, each judged by a
    different annotator (the Task-2 setting); mean slider per fork."""
    by = defaultdict(dict)
    for rows in items.values():
        r, m = rows[0], mean_slider(rows)
        if r["kind"] == "fork" and m is not None:
            by[(r["source_id"], r["hallucination_type"])][r["segment"]] = m
    taus = [kendall_tau_b(sorted(d), [d[k] for k in sorted(d)])
            for d in by.values() if len(d) == 4]
    return [t for t in taus if t == t]


def rate(flags):
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100 * lo, 100 * hi, n


def confirmation(items):
    true_by_type, true_by_k = defaultdict(list), defaultdict(list)
    foil_fork, foil_conf = [], []
    for rows in items.values():
        r = rows[0]
        ok = r["appears"] in CONFIRMED
        if r["kind"] == "conformant":
            foil_conf.append(ok)
        elif r["is_foil"]:
            foil_fork.append(ok)
        else:
            true_by_type[r["hallucination_type"]].append(ok)
            true_by_k[r["segment"]].append(ok)
    true_all = sum(true_by_type.values(), [])
    return true_all, true_by_type, true_by_k, foil_fork, foil_conf


def onset(items):
    errs, by_type, by_k = [], defaultdict(list), defaultdict(list)
    for rows in items.values():
        r = rows[0]
        if r["kind"] == "fork" and not r["is_foil"] and r["appears"] in CONFIRMED \
                and r["onset_sys_turn"] not in ("", "None"):
            e = int(r["onset_sys_turn"]) - int(r["fork_assistant_turn"])
            errs.append(e)
            by_type[r["hallucination_type"]].append(e)
            by_k[r["segment"]].append(e)
    return errs, by_type, by_k


def prefix(items):
    ans = [rows[0]["prefix_ok"] for rows in items.values()
           if rows[0]["prefix_ok"] in ("yes", "mostly", "no")]
    return ans


def prefix_by_k(items):
    out = defaultdict(list)
    for rows in items.values():
        if rows[0]["prefix_ok"] in ("yes", "mostly", "no"):
            out[rows[0]["segment"]].append(rows[0]["prefix_ok"])
    return out


def correlation(items):
    xs, ys = defaultdict(list), defaultdict(list)
    for rows in items.values():
        r = rows[0]
        m = mean_slider(rows)
        if r["kind"] != "fork" or m is None:
            continue
        for meth in METHODS:
            d = r.get("metric_" + meth, "")
            if d not in ("", "None"):
                xs[meth].append(100 - m)
                ys[meth].append(float(d))
    out = OrderedDict()
    rng = np.random.default_rng(0)
    for meth in METHODS:
        x, y = np.array(xs[meth]), np.array(ys[meth])
        if not len(x):
            continue
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(x), len(x))
            boots.append(spearman(list(x[i]), list(y[i])))
        out[meth] = (spearman(list(x), list(y)), np.percentile(boots, 2.5),
                     np.percentile(boots, 97.5), len(x))
    # fork position itself as a predictor of perceived deviation (later fork = less deviant)
    fx, fy = [], []
    for rows in items.values():
        m = mean_slider(rows)
        if rows[0]["kind"] == "fork" and m is not None:
            fx.append(100 - m)
            fy.append(-rows[0]["segment"])
    return out, spearman(fx, fy)


def report(items, label):
    print("\n==== %s ====" % label)
    ann = {r["annotator"] for rows in items.values() for r in rows}
    print("annotators %d  conversations %d  judgments %d  multi-judged %d"
          % (len(ann), len(items), sum(len(v) for v in items.values()),
             sum(len(v) > 1 for v in items.values())))
    for k, m, lo, hi, n in slider_by_segment(items):
        print("slider %-11s %5.1f [%5.1f, %5.1f]  n=%d" % ("conformant" if k == 0 else "k=%d" % k,
                                                          m, lo, hi, n))
    tk = defaultdict(list)
    for rows in items.values():
        m = mean_slider(rows)
        if rows[0]["kind"] == "fork" and m is not None:
            tk[(rows[0]["hallucination_type"], rows[0]["segment"])].append(m)
    for h, a in TYPES.items():
        print("  slider %-4s k=1..4: %s" % (a, " ".join("%5.1f" % np.mean(tk[(h, k)])
                                                        for k in range(1, 5))))
    ft = defaultdict(list)
    for rows in items.values():
        if rows[0]["kind"] == "fork":
            ft[rows[0]["segment"]].append(int(rows[0]["fork_assistant_turn"]))
    print("  mean t_k k=1..4: %s" % " ".join("%.2f" % np.mean(ft[k]) for k in range(1, 5)))
    for lab, keep in (("later fork > earlier fork", lambda a, b: b <= 4),
                      ("k=4 > k=1", lambda a, b: a == 1 and b == 4),
                      ("conformant > fork", lambda a, b: b == 5)):
        print("pairwise %-26s %.1f%% [%.1f, %.1f]  pairs=%d" % ((lab,) + pairwise(items, keep)))
    q = quartets(items)
    lo, hi = bootstrap_ci(q)
    print("quartet tau_b %+.3f [%+.3f, %+.3f]  quartets=%d" % (np.mean(q), lo, hi, len(q)))
    taus, by_type = ordering(items)
    lo, hi = bootstrap_ci(taus)
    print("tau_b within (annotator, task): %+.3f [%+.3f, %+.3f]  units=%d"
          % (np.mean(taus), lo, hi, len(taus)))
    for h, a in TYPES.items():
        if by_type[h]:
            lo, hi = bootstrap_ci(by_type[h])
            print("  %-4s %+.3f [%+.3f, %+.3f]  units=%d" % (a, np.mean(by_type[h]), lo, hi,
                                                            len(by_type[h])))
    true_all, by_t, by_k, ff, fc = confirmation(items)
    print("confirmed true %.1f%% [%.1f, %.1f] n=%d" % rate(true_all))
    yes = [rows[0]["appears"] == "yes" for rows in items.values()
           if rows[0]["kind"] == "fork" and not rows[0]["is_foil"]]
    print("  answered yes %.1f%%" % (100 * np.mean(yes)))
    for h, a in TYPES.items():
        print("  %-4s %.1f%% [%.1f, %.1f] n=%d" % ((a,) + rate(by_t[h])))
    for k in range(1, 5):
        print("  k=%d  %.1f%% [%.1f, %.1f] n=%d" % ((k,) + rate(by_k[k])))
    print("confirmed foil (forked) %.1f%% [%.1f, %.1f] n=%d" % rate(ff))
    print("confirmed foil (conformant) %.1f%% [%.1f, %.1f] n=%d" % rate(fc))
    print("confirmed foil (all) %.1f%% [%.1f, %.1f] n=%d" % rate(ff + fc))
    errs, e_t, e_k = onset(items)
    a = np.abs(errs)
    print("onset median abs error %.0f" % np.median(a))
    print("onset exact %.1f%%  within1 %.1f%%  before t_k %.1f%%  n=%d"
          % (100 * np.mean(a == 0), 100 * np.mean(a <= 1), 100 * np.mean(np.array(errs) < 0),
             len(a)))
    for h, ab in TYPES.items():
        x = np.abs(e_t[h])
        print("  %-4s exact %.1f%% within1 %.1f%%" % (ab, 100 * np.mean(x == 0),
                                                       100 * np.mean(x <= 1)))
    for k in range(1, 5):
        x = np.abs(e_k[k])
        print("  k=%d  exact %.1f%% within1 %.1f%%" % (k, 100 * np.mean(x == 0),
                                                        100 * np.mean(x <= 1)))
    p = prefix(items)
    for k, a in sorted(prefix_by_k(items).items()):
        print("  prefix k=%d yes %.1f%% mostly %.1f%% no %.1f%% n=%d" % (
            k, 100 * a.count("yes") / len(a), 100 * a.count("mostly") / len(a),
            100 * a.count("no") / len(a), len(a)))
    print("prefix yes %.1f%%  mostly %.1f%%  no %.1f%%  yes+mostly %.1f%% [%.1f, %.1f] n=%d"
          % ((100 * p.count("yes") / len(p), 100 * p.count("mostly") / len(p),
              100 * p.count("no") / len(p)) + rate([x in ("yes", "mostly") for x in p])))
    dbl = [v for v in items.values() if len(v) > 1]
    for lab, f in (("3-way", lambda a: a), ("confirmed", lambda a: a in CONFIRMED)):
        app = [Counter(f(r["appears"]) for r in v) for v in dbl]
        print("agreement (%s) raw %.3f AC1 %.3f" % (lab, percent_agreement(app),
                                                     gwet_ac1(app)))
    corr, rho_k = correlation(items)
    for meth, (r, lo, hi, n) in corr.items():
        print("rho %-16s %+.3f [%+.3f, %+.3f] n=%d" % (meth, r, lo, hi, n))
    print("rho fork position  %+.3f" % rho_k)


def report_rq1(items, label):
    print("\n==== %s ====" % label)
    for k, m, lo, hi, n in slider_by_segment(items):
        print("slider %-11s %5.1f [%5.1f, %5.1f]  n=%d" % ("conformant" if k == 0 else "k=%d" % k,
                                                          m, lo, hi, n))
    for lab, keep in (("later fork > earlier fork", lambda a, b: b <= 4),
                      ("k=4 > k=1", lambda a, b: a == 1 and b == 4),
                      ("conformant > fork", lambda a, b: b == 5)):
        print("pairwise %-26s %.1f%% [%.1f, %.1f]  pairs=%d" % ((lab,) + pairwise(items, keep)))
    q = quartets(items)
    lo, hi = bootstrap_ci(q)
    print("quartet tau_b %+.3f [%+.3f, %+.3f]  quartets=%d" % (np.mean(q), lo, hi, len(q)))
    taus, _ = ordering(items)
    lo, hi = bootstrap_ci(taus)
    print("tau_b within (annotator, task): %+.3f [%+.3f, %+.3f]  units=%d"
          % (np.mean(taus), lo, hi, len(taus)))
    g = defaultdict(lambda: defaultdict(list))
    for rows in items.values():
        for r in rows:
            g[(r["annotator"], r["task"])][r["segment"] if r["kind"] == "fork" else 0].append(
                r["slider"])
    ok = tot = 0
    for d in g.values():
        if 0 in d and len(d) == 5:
            means = {k: np.mean(v) for k, v in d.items()}
            tot += 1
            ok += means[0] >= max(means.values())
    print("anchor: conformant highest in %d/%d (annotator, task) pairs" % (ok, tot))


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=2)
    (ax.yaxis if grid == "y" else ax.xaxis).grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, fontsize=7.5, color=INK, loc="left", pad=4)


def bars(ax, labels, vals, los=None, his=None, colors=BLUE, fmt="%.0f", top=100):
    x = np.arange(len(vals))
    yerr = None if los is None else [np.array(vals) - los, np.array(his) - vals]
    ax.bar(x, vals, width=0.66, color=colors, yerr=yerr,
           error_kw=dict(ecolor=INK_MUTED, lw=0.7, capsize=1.8))
    ends = his if his is not None else vals
    for xi, v, e in zip(x, vals, ends):
        ax.text(xi, e + 0.02 * top, fmt % v, ha="center", va="bottom", fontsize=6.3,
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0, top * 1.12)


def pct_axis(ax, ticks=(0, 50, 100)):
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%d%%" % t for t in ticks])


def without_csc(items):
    return {i: rows for i, rows in items.items()
            if rows[0]["hallucination_type"] != STEP_PRESERVING}


def figure(items, out):
    for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman*.ttf"):
        font_manager.fontManager.addfont(f)
    have_times = any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist)
    plt.rcParams.update({  # STIX (bundled with matplotlib) is a Times-like fallback
        "font.family": "Times New Roman" if have_times else "STIXGeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 6.5})
    fig, axs = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 3.5),
                            gridspec_kw=dict(width_ratios=[1, 1.1]))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.93, wspace=0.28,
                        hspace=0.55)

    # (a) RQ1: conformance by fork position, conformant dialogues as anchor
    ax = axs[0, 0]
    g, a = slider_by_segment(without_csc(items)), slider_by_segment(items)
    order = [1, 2, 3, 4, 0]
    bars(ax, ["$k$=1", "$k$=2", "$k$=3", "$k$=4", "conformant"],
         np.array([g[k][1] for k in order]), np.array([g[k][2] for k in order]),
         np.array([g[k][3] for k in order]), [BLUE] * 4 + [GRAY])
    ax.plot(np.arange(4) + 0.42, [a[k][1] for k in order[:4]], "D", ms=2.8, color=INK,
            mec="none", label="incl. CSC")
    ax.legend(loc="upper left", fontsize=6.3, frameon=False, handletextpad=0.2,
              borderaxespad=0.1)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("mean rating (0\u2013100)", color=INK_MUTED)
    title(ax, "(a) RQ1: perceived conformance by fork position")
    style(ax)

    # (b) RQ2: cued confirmation, true instruction per type vs. foils
    ax = axs[0, 1]
    _, by_t, _, ff, fc = confirmation(items)
    rs = [rate(by_t[h]) for h in TYPES] + [rate(ff), rate(fc)]
    bars(ax, list(TYPES.values()) + ["foil\n(forked)", "foil\n(conformant)"],
         np.array([r[0] for r in rs]), np.array([r[1] for r in rs]),
         np.array([r[2] for r in rs]), [BLUE] * 4 + [GRAY] * 2, fmt="%.0f%%")
    pct_axis(ax)
    ax.set_ylabel("confirmed (yes or partly)", color=INK_MUTED)
    title(ax, "(b) RQ2: instruction confirmed in the conversation")
    style(ax)

    # (c) RQ3: onset error distribution, as in the admin panel
    ax = axs[1, 0]
    errs = np.minimum(np.abs(onset(items)[0]), 5)
    bars(ax, ["0", "1", "2", "3", "4", "5+"], [100 * np.mean(errs == e) for e in range(6)],
         fmt="%.0f%%", top=80)
    pct_axis(ax, (0, 25, 50, 75))
    w1 = 100 * np.mean(errs <= 1)
    ax.annotate("", xy=(-0.33, 79), xytext=(1.33, 79),
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7,
                                shrinkA=0, shrinkB=0))
    for xe in (-0.33, 1.33):
        ax.plot([xe, xe], [77, 79], color=INK_MUTED, lw=0.7)
    ax.text(0.5, 80.5, "within one turn: %.0f%%" % w1, ha="center", va="bottom",
            fontsize=6.5, color=INK)
    ax.set_xlabel("$|$annotated onset $-$ $t_k|$ (assistant turns)", color=INK_MUTED,
                  labelpad=1)
    ax.set_ylabel("share of confirmed forks", color=INK_MUTED)
    title(ax, "(c) RQ3: onset localization error")
    style(ax)

    # (d) RQ4: prefix conformance per fork position (k=1 has no pre-fork part)
    ax = axs[1, 1]
    ans = prefix_by_k(items)
    ks = [4, 3, 2]
    cols = [("yes", BLUE_DARK, "yes"), ("mostly", BLUE_LIGHT, "mostly"), ("no", GRAY, "no")]
    left = np.zeros(len(ks))
    for key, c, lab in cols:
        v = np.array([100 * ans[k].count(key) / len(ans[k]) for k in ks])
        ax.barh(np.arange(len(ks)), v, left=left, height=0.6, color=c, label=lab,
                edgecolor="white", lw=0.8)
        for yi, (l, vi) in enumerate(zip(left, v)):
            if vi >= 6:
                ax.text(l + vi / 2, yi, "%.0f%%" % vi, ha="center", va="center",
                        fontsize=6.3, color="white" if key != "mostly" else INK)
        left += v
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels(["$k$=%d" % k for k in ks], color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("does the assistant follow the steps before the fork?",
                  color=INK_MUTED, labelpad=1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=6.5,
              frameon=False, handlelength=1, columnspacing=0.8, borderaxespad=0.1)
    title(ax, "(d) RQ4: prefix conformance")
    style(ax, grid="x")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("\nwrote %s" % out)


def figure_main(items, out):
    """Single-column version for the main body: panels (a) RQ1 and (c) RQ3 only, the
    checks of the labels Tasks 1-2 (fork order) and Task 3 (onset) rely on."""
    fig, axs = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 1.45),
                            gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.2, top=0.86, wspace=0.42)

    ax = axs[0]
    g, a = slider_by_segment(without_csc(items)), slider_by_segment(items)
    order = [1, 2, 3, 4, 0]
    bars(ax, ["1", "2", "3", "4", "conf."],
         np.array([g[k][1] for k in order]), np.array([g[k][2] for k in order]),
         np.array([g[k][3] for k in order]), [BLUE] * 4 + [GRAY])
    ax.plot(np.arange(4) + 0.42, [a[k][1] for k in order[:4]], "D", ms=2.4, color=INK,
            mec="none", label="incl. CSC")
    ax.legend(loc="upper left", fontsize=6, frameon=False, handletextpad=0.2,
              borderaxespad=0.1)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("fork position $k$", color=INK_MUTED, labelpad=1)
    ax.set_ylabel("mean rating (0–100)", color=INK_MUTED, labelpad=1)
    title(ax, "(a) Perceived conformance")
    style(ax)

    ax = axs[1]
    errs = np.minimum(np.abs(onset(items)[0]), 5)
    bars(ax, ["0", "1", "2", "3", "4", "5+"], [100 * np.mean(errs == e) for e in range(6)],
         fmt="%.0f%%", top=80)
    pct_axis(ax, (0, 25, 50, 75))
    w1 = 100 * np.mean(errs <= 1)   # same bracket as panel (c) of the appendix figure
    ax.set_ylim(0, 98)              # headroom above the bar labels for the bracket
    ax.plot([-0.33, 1.33], [86, 86], color=INK_MUTED, lw=0.7)
    for xe in (-0.33, 1.33):
        ax.plot([xe, xe], [84, 86], color=INK_MUTED, lw=0.7)
    ax.text(-0.33, 87.5, "within one turn: %.0f%%" % w1, ha="left", va="bottom", fontsize=6,
            color=INK)
    ax.set_xlabel("$|$annotated onset $-$ $t_k|$", color=INK_MUTED, labelpad=1)
    ax.set_ylabel("share of forks", color=INK_MUTED, labelpad=1)
    title(ax, "(b) Onset localization error")
    style(ax)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("wrote %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--answers", required=True, help="answers.csv from the export")
    ap.add_argument("--binary-share", type=float, default=0.9,
                    help="an annotator whose slider answers are exactly 0 or 100 at least "
                         "this often uses it as a binary scale; the numbers are also "
                         "printed without them, as a robustness check")
    ap.add_argument("-o", "--out", default="human_study.pdf")
    ap.add_argument("--main-out", default="human_study_main.pdf",
                    help="single-column version (panels a and c) for the main body")
    args = ap.parse_args()
    full = load(args.answers)
    binary = binary_users(full, args.binary_share)
    graded = load(args.answers, binary)
    report(full, "all annotators")
    report_rq1(without_csc(full), "RQ1 / panel (a) bars: without CSC")
    report_rq1({i: r for i, r in full.items() if r[0]["kind"] == "conformant"
                or r[0]["hallucination_type"] == STEP_PRESERVING}, "RQ1: CSC only")
    report(graded, "without binary slider users: " + ", ".join(binary))
    figure(full, args.out)
    figure_main(full, args.main_out)


if __name__ == "__main__":
    main()
