# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Small statistics helpers. Kept separate so analyze.py stays readable."""
import math
import random


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval -- the right one for rates near 1 at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = float(k) / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fmt_rate(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return "%.1f%% [%.1f, %.1f] (%d/%d)" % (100.0 * k / n, 100 * lo, 100 * hi, k, n)


def bootstrap_ci(values, n_boot=5000, seed=0):
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / float(n))
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def kendall_tau_b(x, y):
    """tau-b: handles the ties a 0-100 slider will certainly produce."""
    n = len(x)
    if n < 2:
        return float("nan")
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if a == 0 and b == 0:
                tx += 1
                ty += 1
            elif a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    n0 = conc + disc + tx
    n1 = conc + disc + ty
    if n0 == 0 or n1 == 0:
        return float("nan")
    return (conc - disc) / math.sqrt(float(n0) * n1)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def fleiss_kappa(items):
    """items: list of dicts {category: count}. Only units with >=2 raters count."""
    units = [d for d in items if sum(d.values()) >= 2]
    if not units:
        return float("nan")
    cats = sorted({c for d in units for c in d})
    n = len(units)
    # requires a constant number of raters; use the modal count and drop others
    sizes = {}
    for d in units:
        sizes[sum(d.values())] = sizes.get(sum(d.values()), 0) + 1
    m = max(sizes, key=lambda k: sizes[k])
    units = [d for d in units if sum(d.values()) == m]
    n = len(units)
    if n == 0 or m < 2:
        return float("nan")
    p_j = {}
    for c in cats:
        p_j[c] = sum(d.get(c, 0) for d in units) / float(n * m)
    P_i = []
    for d in units:
        s = sum(d.get(c, 0) ** 2 for c in cats)
        P_i.append((s - m) / float(m * (m - 1)))
    P_bar = sum(P_i) / n
    P_e = sum(v * v for v in p_j.values())
    if P_e >= 1:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha_interval(units):
    """units: list of lists of numeric ratings (>=2 ratings to be pairable)."""
    usable = [u for u in units if len(u) >= 2]
    if not usable:
        return float("nan")
    Do_num = 0.0
    n = 0
    allvals = []
    for u in usable:
        m = len(u)
        n += m
        allvals.extend(u)
        s = 0.0
        for i in range(m):
            for j in range(m):
                if i != j:
                    s += (u[i] - u[j]) ** 2
        Do_num += s / (m - 1)
    if n < 2:
        return float("nan")
    Do = Do_num / n
    De = 0.0
    for i in range(len(allvals)):
        for j in range(len(allvals)):
            if i != j:
                De += (allvals[i] - allvals[j]) ** 2
    De /= (len(allvals) * (len(allvals) - 1))
    if De == 0:
        return float("nan")
    return 1 - Do / De


def percent_agreement(items):
    """items: list of dicts {category: count}. Share of rater PAIRS that agree."""
    agree = total = 0
    for d in items:
        counts = list(d.values())
        m = sum(counts)
        if m < 2:
            continue
        total += m * (m - 1) / 2.0
        agree += sum(c * (c - 1) / 2.0 for c in counts)
    return agree / total if total else float("nan")


def gwet_ac1(items):
    """Gwet's AC1 -- the right statistic when one answer dominates.

    Cohen/Fleiss kappa collapses under skewed marginals (the "kappa paradox"):
    with 90% of answers on one category, raters agreeing 90% of the time score
    kappa ~= 0.58, and at n=60 the 95% CI runs [0.24, 0.85]. AC1 on the same data
    gives 0.87 [0.75, 0.96]. Our confirmation question is exactly that skewed, so
    report raw agreement and AC1 first and kappa as a footnote.
    """
    units = [d for d in items if sum(d.values()) >= 2]
    if not units:
        return float("nan")
    cats = sorted({c for d in units for c in d})
    k = len(cats)
    if k < 2:
        # every rater gave the same single answer everywhere: no disagreement
        # possible, and no chance-correction to make. Report perfect agreement
        # rather than nan, but the raw rate beside it is what carries the meaning.
        return 1.0
    po = percent_agreement(units)
    pe = 0.0
    for c in cats:
        pis = [d.get(c, 0) / float(sum(d.values())) for d in units]
        pi = sum(pis) / len(pis)
        pe += pi * (1 - pi)
    pe /= (k - 1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
