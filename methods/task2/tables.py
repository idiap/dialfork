# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""
Dialog-to-dialog table generator for manuscript reporting.

This script aggregates dialog-to-dialog ranking results and renders paper-ready
main and appendix tables (markdown or LaTeX).

================================================================================
TASK DEFINITION: FLOW-AWARE DIALOG DISTANCE RANKING
================================================================================

Unlike classical NLP similarity tasks that emphasize semantic overlap, this
task evaluates whether a dialog-distance metric captures conversation flow
structure (implicit workflow steps).

For each original dialog, we split the conversation into 4 segments and create
exactly 4 hallucinated forks: one at the beginning of each segment. This
provides a natural 4-item ground-truth ranking of distances to the original
dialog:

    - fork at segment 1 start -> most different (largest flow deviation)
    - fork at segment 2 start -> less different
    - fork at segment 3 start -> more similar
    - fork at segment 4 start -> most similar (longest shared workflow prefix)

The evaluation target is therefore ranking quality, not absolute similarity.

================================================================================
DISTANCE METHODS EVALUATED
================================================================================

For each embedding model, four dialog2dialog distance methods are evaluated:

1. mean-turn-emb
     - Encode each turn, average (centroid) turn embeddings per dialog,
         and compute cosine distance between dialog centroids.

2. full-dialog-emb
     - Serialize the full dialog as text, encode once, and compute cosine
         distance between full-dialog embeddings.

3. flowdtw
     - Encode each turn, align turn-embedding sequences with DTW, and use the
         DTW-aligned cosine-cost distance as dialog distance.

             Implemented equations (matching methods/task2/run.py):
                 Given L2-normalized turn embeddings E = [e_1, ..., e_n] and
                 E' = [e'_1, ..., e'_m], local costs are

                     c(i,j) = 1 - cos(e_i, e'_j) = 1 - e_i^T e'_j.

                 DTW dynamic program:

                     D(i,j) = c(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1)),

                 with D(0,0)=0 and +inf borders as usual.

                 Final FlowDTW distance is path-length normalized:

                     d_FlowDTW = D(n,m) / L,

                 where L is the number of alignment steps in the selected DTW path.

4. flowscore
     - Encode each turn and apply a BERTScore-style matching equation over turn
         embeddings (instead of token embeddings), yielding a flow-aware distance.

             Implemented equations (matching methods/task2/run.py):
                 Build similarity matrix S with

                     S(i,j) = e_i^T e'_j,

                 because embeddings are L2-normalized.

                 Precision and recall over turn sets:

                     P = (1/n) * sum_i max_j S(i,j)
                     R = (1/m) * sum_j max_i S(i,j)

                 F1-style flow matching score:

                     F_flow = 2PR / (P + R).

                 FlowScore is reported as a distance:

                     d_FlowScore = 1 - F_flow.

Methods (3) and (4) are flow-aware sequence methods and part of the novelty
of the paper, not previously published, as far as we know (since there were actually no previous dataset to measure this,
which is what we introduce in our paper, DialFork).

================================================================================
EMBEDDING FAMILIES
================================================================================

The aggregated tables include both classic general-purpose embeddings
(e.g., all-mpnet-base-v2 and LaBSE) and newer general-purpose embeddings
(e.g., Qwen3-Embedding-8B, embeddinggemma variants, and bge-m3), alongside
dialog-specific sentence embedding families (DSE and Dialog2Flow).

================================================================================
METRIC AND REPORTING
================================================================================

Primary metric: Kendall tau correlation between predicted fork ranking and
ground-truth ranking.

This script reports:
    - main table: Kendall tau by hallucination type (CSC, GR, SSPC, WIC) + AVG
    - appendix table: Kendall tau by task + AVG
    - a per-encoder significance table (on by default, disable with
        --no-significance): paired Wilcoxon signed-rank test and paired t-test
        (both one-sided, FlowScore > baseline) on per-dialog Kendall tau against
        each baseline method, with paired-bootstrap 95% CIs on the mean gap.
        P-values for each test are Holm-Bonferroni corrected within their own
        family across encoders; Wilcoxon is the primary test (robust to the
        bounded/discrete shape of Kendall tau), the t-test is reported for
        comparison. Requires the per-dialog results_d2d_rankings.json files on
        disk (not just the aggregated *_overall*.json summaries).

Higher is better. Best values are highlighted per model and globally in LaTeX
output.

Observed high-level outcome used in the paper narrative:
    - FlowScore is consistently strongest across embedding families, typically
        reaching >0.80 Kendall tau against the ground-truth ranking.

================================================================================
IMPLEMENTATION NOTES
================================================================================

- This script is a reporting/aggregation layer only. It reads JSON summaries
    generated by methods/task2/run.py and formats final tables.
- Table values are rendered at 3 decimal places.
- The temporary correction block for one model/method combination is kept as
    in code and should be removed once the upstream issue is fully resolved.

"""

import os
import glob
import pandas as pd
import numpy as np
import json
import argparse
from itertools import permutations
from scipy.stats import wilcoxon, ttest_rel

parser = argparse.ArgumentParser(description="Generate dialog-to-dialog result tables.")
parser.add_argument(
    "--root",
    default="results/task2",
    help="Path containing the dialog-to-dialog results (default: results/task2).",
)
parser.add_argument(
    "--latex",
    action="store_true",
    help="Print LaTeX tables instead of markdown output.",
)
parser.add_argument(
    "--significance",
    dest="significance",
    action="store_true",
    default=True,
    help=("Report paired Wilcoxon signed-rank and paired t-tests (one-sided, FlowScore > baseline) "
          "with bootstrap confidence intervals on the mean Kendall tau gap, comparing FlowScore "
          "against each baseline method per encoder. P-values are Holm-Bonferroni corrected "
          "within each baseline's family of tests across encoders. Enabled by default."),
)
parser.add_argument(
    "--no-significance",
    dest="significance",
    action="store_false",
    help="Skip the significance-testing section.",
)
parser.add_argument(
    "--n-bootstrap",
    type=int,
    default=10000,
    help="Number of paired bootstrap resamples used for the significance CIs (default: 10000).",
)
parser.add_argument(
    "--significance-seed",
    type=int,
    default=42,
    help="Random seed for the bootstrap resampling (default: 42).",
)
args = parser.parse_args()

ROOT = args.root

# Abbreviations for hallucination types
HT_ABBREV = {
    "conflicting_system_constraints": "CSC",
    "goal_replacement": "GR",
    "step_selection_policy_corruption": "SSPC",
    "workflow_identity_corruption": "WIC",
}

# Task name abbreviations (short for fitting in table)
TASK_ABBREV = {
    "apartment_schedule": "ase",
    "apartment_search": "as",
    "bank_balance": "bb",
    "bank_fraud_report": "bfr",
    "doctor_followup": "df",
    "doctor_schedule": "ds",
    "hotel_book": "hb",
    "hotel_search": "hs",
    "hotel_service_request": "hsr",
    "meeting_schedule": "ms",
    "party_plan": "pp",
    "party_rsvp": "pr",
    "plane_book": "pb",
    "plane_search": "ps",
    "restaurant_book": "rb",
    "restaurant_search": "rs",
    "ride_book": "rib",
    "ride_change": "ric",
    "ride_status": "ris",
    "spaceship_access_codes": "sac",
    "spaceship_life_support": "spl",
    "trip_directions": "td",
    "trivia": "trv",
    "weather": "wth",
}

# Method ordering
METHOD_ORDER = ["mean-turn-emb", "full-dialog-emb", "flowdtw", "flowscore"]
# METHOD_ORDER = ["mean-turn-emb", "flowdtw", "flowscore"]

# Naive length-only controls: computed once (no embedding model involved), stored under a
# fixed placeholder "none" model folder by methods/task2/run.py, and rendered as a separate leading
# block rather than as extra rows in the model x method grid.
BASELINE_METHODS = ["random", "turn-count-diff", "word-count-diff", "shared-prefix"]
BASELINE_MODEL_PLACEHOLDER = "none"

# Number of forks ranked per dialog in Task 2 (generation/generate.py's --hallucinations-per-dialog
# default / DialFork's fixed 4-segment split, see README). Needed to compute the exact
# (not simulated) chance-level Kendall tau for the "random" baseline below.
TASK2_N_FORKS = 4


def _exact_random_kendall_tau(n=TASK2_N_FORKS):
    """Expected Kendall tau of a uniformly random n-way ranking against a fixed ground
    truth, via exact enumeration over all n! permutations (n is small, so this is exact,
    not a Monte Carlo estimate). Mirrors _kendall_tau_from_orders()'s tau-a formula from
    methods/task2/run.py. Mathematically this is always 0 by symmetry; enumerating it explicitly
    keeps the value tied to the actual scoring formula rather than an assertion.
    """
    ground_truth = list(range(n))
    taus = []
    for perm in permutations(ground_truth):
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] < perm[j]:
                    concordant += 1
                else:
                    discordant += 1
        taus.append((concordant - discordant) / (concordant + discordant))
    return float(np.mean(taus))

MODELS_TO_IGNORE = ["dialog2flow-joint-bert-base", "dialog2flow-single-bert-base", "dialog2flow-single-dse-base"]
# MODELS_TO_IGNORE = []


def _latex_escape(value):
    s = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


def _pretty_method_label(method):
    pretty = {
        "mean-turn-emb": "Mean-turn emb.",
        "full-dialog-emb": "Full-dialog emb.",
        "flowdtw": "FlowDTW",
        "flowscore": "FlowScore",
        "turn-count-diff": "Turn-count diff.",
        "word-count-diff": "Word-count diff.",
        "shared-prefix": "Shared prefix",
        "random": "Random (chance)",
    }
    return pretty.get(str(method), str(method))


def _safe_float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def _best_index_set(series):
    numeric = series.apply(_safe_float_or_none).dropna()
    if numeric.empty:
        return set()
    best = numeric.max()
    tol = 1e-12
    return set(numeric.index[(numeric - best).abs() <= tol])


def _style_latex_value(text, is_group_best=False, is_global_best=False):
    if text == "?":
        return text
    if is_global_best:
        return r"\underline{\textbf{" + text + "}}"
    if is_group_best:
        return r"\textbf{" + text + "}"
    return text


def _tie_aware_kendall_tau(record):
    """Kendall's tau of one dialog's predicted ranking, recomputed from the stored
    per-fork distances so that tied distances count as neither concordant nor
    discordant (i.e. the expected tau under random tie-breaking).

    methods/task2/run.py ranks forks with a stable sort, so ties used to keep the ground-truth
    order and were scored as correct. That inflated integer-valued distances
    (turn-count-diff: ~87% of dialogs have ties) while leaving continuous
    embedding distances unchanged. Falls back to the stored value if the record
    has no distances.
    """
    distances = record.get("distances")
    order = record.get("ground_truth_rank")
    if not distances or not order:
        tau = record.get("kendall_tau")
        return None if tau is None else float(tau)
    order = [k for k in order if k in distances]
    n = len(order)
    if n < 2:
        return 0.0
    # order is most-similar first, so ground truth says distance increases along it
    score = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = distances[order[i]], distances[order[j]]
            score += (a < b) - (a > b)
    return score / (n * (n - 1) / 2)


def _load_dialog_kendall_taus(root, method, model):
    """Load per-dialog Kendall tau values for one method/model from disk.

    Returns a dict keyed by (task, hallucination_type, dialog_id) -> kendall_tau,
    pooled across every task and hallucination type, matching the unit of
    analysis used to compute the reported mean (one value per original dialog).
    """
    values = {}
    pattern = os.path.join(root, method, model, "*", "results_d2d_rankings.json")
    for f in glob.glob(pattern):
        task = os.path.basename(os.path.dirname(f))
        try:
            with open(f) as fin:
                payload = json.load(fin)
        except Exception:
            continue
        for ht, per_dialog in payload.items():
            for dialog_id, record in per_dialog.items():
                tau = _tie_aware_kendall_tau(record)
                if tau is not None:
                    values[(task, ht, dialog_id)] = tau
    return values


def _holm_bonferroni(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in input order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min((n - rank) * pvals[i], 1.0)
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted


def _paired_bootstrap_ci_mean(diff, n_boot=10000, alpha=0.05, rng=None, batch_size=2000):
    """Percentile bootstrap CI for the mean of paired differences."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(diff)
    boot_means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        this_batch = min(batch_size, n_boot - done)
        idx = rng.integers(0, n, size=(this_batch, n))
        boot_means[done:done + this_batch] = diff[idx].mean(axis=1)
        done += this_batch
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _compute_significance(root, models, baseline_methods, target_method="flowscore",
                           n_boot=10000, seed=42, alpha=0.05):
    """Paired Wilcoxon signed-rank test and paired t-test (both one-sided,
    FlowScore > baseline) per encoder, with paired bootstrap CIs on the mean
    Kendall tau gap. P-values for each test are Holm-Bonferroni corrected
    within their own family across encoders (the two tests are corrected
    independently of each other). The Wilcoxon test is treated as primary
    (non-parametric, robust to the bounded/discrete shape of Kendall tau);
    the t-test is reported alongside it for comparison.
    """
    rng = np.random.default_rng(seed)
    target_data = {model: _load_dialog_kendall_taus(root, target_method, model) for model in models}

    rows = []
    for baseline in baseline_methods:
        family_rows = []
        for model in models:
            baseline_data = _load_dialog_kendall_taus(root, baseline, model)
            if not baseline_data and baseline in BASELINE_METHODS:
                # Naive length baselines don't depend on the embedding model; they're
                # stored once under the placeholder model folder.
                baseline_data = _load_dialog_kendall_taus(root, baseline, BASELINE_MODEL_PLACEHOLDER)
            tgt = target_data.get(model, {})
            keys = sorted(set(tgt) & set(baseline_data))
            if len(keys) < 2:
                continue

            x = np.array([tgt[k] for k in keys], dtype=np.float64)
            y = np.array([baseline_data[k] for k in keys], dtype=np.float64)
            # Kendall taus over 4 forks are multiples of 1/6: round the differences so that
            # equal gaps are exact ties for the Wilcoxon ranking (float noise otherwise splits
            # them and makes W depend on rounding).
            diff = np.round(x - y, 10)

            if np.all(diff == 0):
                w_stat, w_p_raw = np.nan, 1.0
            else:
                w_stat, w_p_raw = wilcoxon(diff, alternative="greater")

            t_stat, t_p_raw = ttest_rel(x, y, alternative="greater")

            ci_low, ci_high = _paired_bootstrap_ci_mean(diff, n_boot=n_boot, alpha=alpha, rng=rng)

            family_rows.append({
                "baseline": baseline,
                "model": model,
                "n_pairs": len(keys),
                "mean_diff": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_stat": float(w_stat) if w_stat is not None and not np.isnan(w_stat) else float("nan"),
                "wilcoxon_p_raw": float(w_p_raw),
                "ttest_stat": float(t_stat) if not np.isnan(t_stat) else float("nan"),
                "ttest_p_raw": float(t_p_raw),
            })

        if family_rows:
            wilcoxon_corrected = _holm_bonferroni([r["wilcoxon_p_raw"] for r in family_rows])
            ttest_corrected = _holm_bonferroni([r["ttest_p_raw"] for r in family_rows])
            for r, w_p_corr, t_p_corr in zip(family_rows, wilcoxon_corrected, ttest_corrected):
                r["wilcoxon_p_corrected"] = w_p_corr
                r["wilcoxon_significant"] = w_p_corr < alpha
                r["ttest_p_corrected"] = t_p_corr
                r["ttest_significant"] = t_p_corr < alpha
            rows.extend(family_rows)

    return pd.DataFrame(rows)


def _format_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _print_significance_table_markdown(df_sig, target_method="flowscore"):
    if df_sig.empty:
        print("No significance results available (missing per-dialog rankings files).")
        return

    print(f"### Significance: {_pretty_method_label(target_method)} vs. each baseline\n")
    print("*Paired Wilcoxon signed-rank test and paired t-test (both one-sided, FlowScore > baseline) "
          "on per-dialog Kendall τ, with 95% paired-bootstrap CIs on the mean gap. P-values for each "
          "test are Holm-Bonferroni corrected within their own family across encoders. Wilcoxon is the "
          "primary test (robust to the bounded/discrete shape of Kendall tau); the t-test is shown for "
          "comparison and drives no conclusion on its own.*\n")

    display_rows = []
    for _, row in df_sig.iterrows():
        display_rows.append({
            "Baseline": _pretty_method_label(row["baseline"]),
            "Model": row["model"],
            "N": int(row["n_pairs"]),
            "Mean Δ": f"{row['mean_diff']:.3f}",
            "95% CI": f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            "W": f"{row['wilcoxon_stat']:,.0f}" if not np.isnan(row["wilcoxon_stat"]) else "?",
            "p_Wilcoxon (Holm)": _format_p(row["wilcoxon_p_corrected"]),
            "t": f"{row['ttest_stat']:.2f}" if not np.isnan(row["ttest_stat"]) else "?",
            "p_ttest (Holm)": _format_p(row["ttest_p_corrected"]),
            "Sig.": "yes" if row["wilcoxon_significant"] else "no",
        })
    print(pd.DataFrame(display_rows).to_markdown(index=False))


def _print_significance_table_latex(df_sig, target_method="flowscore"):
    if df_sig.empty:
        print("% No significance results available (missing per-dialog rankings files).")
        return

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(
        r"\caption{Paired Wilcoxon signed-rank tests and paired $t$-tests (both one-sided, "
        + _latex_escape(_pretty_method_label(target_method))
        + r" $>$ baseline) on per-dialog Kendall $\tau$, with 95\% paired-bootstrap CIs on the mean gap. "
        r"P-values for each test are Holm-Bonferroni corrected within their own family across encoders. "
        r"Wilcoxon is the primary test; the $t$-test is shown for comparison. Bold marks significance "
        r"(Wilcoxon, $\alpha=0.05$).}"
    )
    print(r"\label{tab:d2d_significance}")
    print(r"\begin{tabular}{llrrlrlrl}")
    print(r"\toprule")
    print(r"Baseline & Model & $N$ & Mean $\Delta$ & 95\% CI & $W$ & $p_W$ (Holm) & $t$ & $p_t$ (Holm) \\")
    print(r"\midrule")
    for baseline, group in df_sig.groupby("baseline", sort=False):
        for _, row in group.iterrows():
            w_txt = f"{row['wilcoxon_stat']:,.0f}" if not np.isnan(row["wilcoxon_stat"]) else "?"
            t_txt = f"{row['ttest_stat']:.2f}" if not np.isnan(row["ttest_stat"]) else "?"
            pw_txt = _format_p(row["wilcoxon_p_corrected"])
            pt_txt = _format_p(row["ttest_p_corrected"])
            if row["wilcoxon_significant"]:
                pw_txt = r"\textbf{" + pw_txt + "}"
            if row["ttest_significant"]:
                pt_txt = r"\textbf{" + pt_txt + "}"
            print(
                _latex_escape(_pretty_method_label(row["baseline"])) + " & "
                + _latex_escape(row["model"]) + " & "
                + str(int(row["n_pairs"])) + " & "
                + f"{row['mean_diff']:.3f}" + " & "
                + f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]" + " & "
                + w_txt + " & "
                + pw_txt + " & "
                + t_txt + " & "
                + pt_txt + r" \\"
            )
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _print_main_table_latex(df, df_baselines=None):
    print("% Requires: \\usepackage{booktabs,multirow}")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\caption{Dialog2Dialog results measured with Kendall $\tau$ by hallucination type. "
          r"Leading rows (above the rule) are construction-artifact controls, excluded from the "
          r"best-value highlighting below: Random is the exact chance-level $\tau$ for a uniformly "
          r"random ranking (0 by symmetry); Turn-count diff. / Word-count diff. are naive "
          r"length-only heuristics with no embedding model, computed once rather than per encoder "
          r"--- see \S4.1. Best values within each model are bold; best overall values are "
          r"bold+underlined. Higher is better.}")
    print(r"\label{tab:d2d_main}")
    print(r"\begin{tabular}{llccccc}")
    print(r"\toprule")
    print(r"Model & Method & WIC & SSPC & GR & CSC & Avg. $\uparrow$ \\")
    print(r"\midrule")

    metric_cols = ["WIC", "SSPC", "GR", "CSC", "AVG"]

    if df_baselines is not None and not df_baselines.empty:
        for _, row in df_baselines.iterrows():
            vals = [str(row[col]) for col in metric_cols]
            method = _latex_escape(_pretty_method_label(row["method"]))
            print(r"-- & " + method + " & " + " & ".join(vals) + r" \\")
        print(r"\midrule")

    global_best = {col: _best_index_set(df[col]) for col in metric_cols}
    model_best = {}
    for model, group in df.groupby("model", sort=False):
        for col in metric_cols:
            model_best[(model, col)] = _best_index_set(group[col])

    for model, group in df.groupby("model", sort=False):
        row_count = len(group)
        first = True
        for idx, row in group.iterrows():
            model_cell = (r"\multirow{" + str(row_count) + r"}{*}{" + _latex_escape(row["model"]) + "}") if first else ""
            first = False
            vals = []
            for col in metric_cols:
                txt = str(row[col])
                vals.append(
                    _style_latex_value(
                        txt,
                        is_group_best=(idx in model_best[(model, col)]),
                        is_global_best=(idx in global_best[col]),
                    )
                )
            method = _latex_escape(_pretty_method_label(row["method"]))
            print(model_cell + " & " + method + " & " + " & ".join(vals) + r" \\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _print_appendix_table_latex(df, task_cols, df_baselines=None):
    print("% Requires: \\usepackage{booktabs,multirow,graphicx,rotating}")
    print(r"\begin{sidewaystable*}[p]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\caption{Dialog2Dialog appendix results measured with Kendall $\tau$ by task. "
          r"Leading rows (above the rule) are construction-artifact controls, excluded from the "
          r"best-value highlighting below: Random is the exact chance-level $\tau$ for a uniformly "
          r"random ranking; Turn-count diff. / Word-count diff. are naive length-only heuristics "
          r"with no embedding model --- see \S4.1. Best values within each model are bold; best "
          r"overall values are bold+underlined. Higher is better.}")
    print(r"\label{tab:d2d_appendix}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{ll" + "c" * (len(task_cols) + 1) + "}")
    print(r"\toprule")
    print("Model & Method & " + " & ".join(_latex_escape(col) for col in task_cols) + r" & Avg. $\uparrow$ \\")
    print(r"\midrule")

    metric_cols = list(task_cols) + ["AVG"]

    if df_baselines is not None and not df_baselines.empty:
        for _, row in df_baselines.iterrows():
            vals = [str(row[col]) if col in row else "?" for col in metric_cols]
            method = _latex_escape(_pretty_method_label(row["method"]))
            print("-- & " + method + " & " + " & ".join(vals) + r" \\")
        print(r"\midrule")

    global_best = {col: _best_index_set(df[col]) for col in metric_cols}
    model_best = {}
    for model, group in df.groupby("model", sort=False):
        for col in metric_cols:
            model_best[(model, col)] = _best_index_set(group[col])

    for model, group in df.groupby("model", sort=False):
        row_count = len(group)
        first = True
        for idx, row in group.iterrows():
            model_cell = (r"\multirow{" + str(row_count) + r"}{*}{" + _latex_escape(row["model"]) + "}") if first else ""
            first = False
            vals = []
            for col in metric_cols:
                txt = str(row[col])
                vals.append(
                    _style_latex_value(
                        txt,
                        is_group_best=(idx in model_best[(model, col)]),
                        is_global_best=(idx in global_best[col]),
                    )
                )
            method = _latex_escape(_pretty_method_label(row["method"]))
            print(model_cell + " & " + method + " & " + " & ".join(vals) + r" \\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{sidewaystable*}")


# ================================================================================
# TEMPORARY: recompute results_d2d_overall.json / results_d2d_overall_by_ht.json
# from the per-task results_d2d_summary.json files on disk.
#
# Each Slurm job (methods/task2/run_all.sh) only processes a single task, but all jobs for
# a given model/method share the same output directory, so whichever job
# finished last overwrote the shared "overall" files with a version reflecting
# only its own task. The per-task summaries themselves are complete and
# unaffected. This block rebuilds the overall files in place so the tables
# below are correct without re-launching any jobs.
#
# methods/task2/run.py now aggregates from disk the same way (see
# load_all_task_summaries_from_disk), so once all results directories have
# been regenerated by that fix this block can be removed.
# ================================================================================
def _recompute_overall_files(root):
    metrics_cols = ["exact_match_rate", "position_accuracy_mean", "kendall_tau_mean"]

    for experiment_dir in glob.glob(os.path.join(root, "*", "*")):
        if not os.path.isdir(experiment_dir):
            continue

        all_task_summaries = {}
        for entry in sorted(os.listdir(experiment_dir)):
            task_dir = os.path.join(experiment_dir, entry)
            summary_path = os.path.join(task_dir, "results_d2d_summary.json")
            if os.path.isdir(task_dir) and os.path.isfile(summary_path):
                try:
                    with open(summary_path) as fin:
                        all_task_summaries[entry] = json.load(fin)
                except Exception:
                    continue
                # Replace the stored per-type tau means with tie-aware ones
                # recomputed from the per-dialog distances (see _tie_aware_kendall_tau).
                rankings_path = os.path.join(task_dir, "results_d2d_rankings.json")
                if os.path.isfile(rankings_path):
                    with open(rankings_path) as fin:
                        rankings = json.load(fin)
                    for ht, per_dialog in rankings.items():
                        taus = [t for t in (_tie_aware_kendall_tau(r) for r in per_dialog.values()) if t is not None]
                        if taus and ht in all_task_summaries[entry]:
                            all_task_summaries[entry][ht]["kendall_tau_mean"] = float(np.mean(taus))

        if not all_task_summaries:
            continue

        overall_rows = {}
        for task, summaries in all_task_summaries.items():
            task_df = pd.DataFrame(summaries).T
            task_df = task_df[[c for c in metrics_cols if c in task_df.columns]]
            if not task_df.empty:
                overall_rows[task] = task_df.mean(axis=0)

        if overall_rows:
            df_overall = pd.DataFrame(overall_rows).T.sort_index()
            df_overall.loc["AVG"] = df_overall.mean(axis=0)
            with open(os.path.join(experiment_dir, "results_d2d_overall.json"), "w") as fout:
                json.dump(df_overall.to_dict(), fout, indent=2)

        ht_records = []
        for task, summaries in all_task_summaries.items():
            for ht, summary in summaries.items():
                record = {"task": task, "hallucination_type": ht}
                for c in metrics_cols:
                    record[c] = summary.get(c)
                ht_records.append(record)

        if ht_records:
            df_ht = pd.DataFrame(ht_records).groupby("hallucination_type", as_index=True)[metrics_cols].mean().sort_index()
            if not df_ht.empty:
                df_ht.loc["AVG"] = df_ht.mean(axis=0)
                with open(os.path.join(experiment_dir, "results_d2d_overall_by_ht.json"), "w") as fout:
                    json.dump(df_ht.to_dict(), fout, indent=2)


_recompute_overall_files(ROOT)


# MAIN TABLE: Load per-hallucination-type Kendall τ
pattern_ht = os.path.join(ROOT, "*", "*", "results_d2d_overall_by_ht.json")
files_ht = glob.glob(pattern_ht)

rows_main = []
task_names = set()  # Collect all task names for appendix
for f in files_ht:
    parts = f.split(os.sep)
    if len(parts) < 3:
        continue
    method = parts[-3]
    model = parts[-2]

    if method.startswith("_") or model in MODELS_TO_IGNORE:
        continue

    try:
        with open(f) as fin:
            payload = json.load(fin)
    except Exception:
        continue

    tau_data = payload.get("kendall_tau_mean", {})
    if not tau_data:
        continue

    row = {"model": model, "method": method}
    for ht_full, ht_abbr in HT_ABBREV.items():
        val = tau_data.get(ht_full)
        row[ht_abbr] = f"{val:.3f}" if val is not None else "?"
    avg = tau_data.get("AVG")
    row["AVG"] = f"{avg:.3f}" if avg is not None else "?"
    rows_main.append(row)

rows_main_baseline = [r for r in rows_main if r["method"] in BASELINE_METHODS]
rows_main = [r for r in rows_main if r["method"] in METHOD_ORDER]

# "random" isn't loaded from disk (no dialog-distance metric to run): it's the exact
# chance-level Kendall tau, the same constant for every hallucination type/task/model.
_random_tau_str = f"{_exact_random_kendall_tau():.3f}"
rows_main_baseline.append({
    "model": BASELINE_MODEL_PLACEHOLDER,
    "method": "random",
    "WIC": _random_tau_str, "SSPC": _random_tau_str, "GR": _random_tau_str, "CSC": _random_tau_str,
    "AVG": _random_tau_str,
})

if not rows_main and not rows_main_baseline:
    print("No results found.")
    exit(0)

# APPENDIX TABLE: Load per-task Kendall τ
pattern_overall = os.path.join(ROOT, "*", "*", "results_d2d_overall.json")
files_overall = glob.glob(pattern_overall)

rows_appendix = []
for f in files_overall:
    parts = f.split(os.sep)
    if len(parts) < 3:
        continue
    method = parts[-3]
    model = parts[-2]

    if method.startswith("_"):
        continue

    try:
        with open(f) as fin:
            payload = json.load(fin)
    except Exception:
        continue

    tau_data = payload.get("kendall_tau_mean", {})
    if not tau_data:
        continue

    row = {"model": model, "method": method}
    for task, val in tau_data.items():
        if task != "AVG":
            task_names.add(task)
            task_abbr = TASK_ABBREV.get(task, task)
            row[task_abbr] = f"{val:.3f}" if val is not None else "?"
    avg = tau_data.get("AVG")
    row["AVG"] = f"{avg:.3f}" if avg is not None else "?"
    rows_appendix.append(row)

rows_appendix_baseline = [r for r in rows_appendix if r["method"] in BASELINE_METHODS]
rows_appendix = [r for r in rows_appendix if r["method"] in METHOD_ORDER]

# Sort task names and get their abbreviations
sorted_tasks = sorted(task_names)
sorted_task_abbrevs = [TASK_ABBREV.get(t, t) for t in sorted_tasks]

# Same constant "random" row as the main table, one column per task.
rows_appendix_baseline.append({
    "model": BASELINE_MODEL_PLACEHOLDER,
    "method": "random",
    **{abbrev: _random_tau_str for abbrev in sorted_task_abbrevs},
    "AVG": _random_tau_str,
})

# Main table: single table grouped by model (sorted alphabetically)
df_main = pd.DataFrame(rows_main)

# Fill in missing model-method combinations with "?"
all_models = sorted(df_main["model"].unique())
model_method_pairs = []
for model in all_models:
    for method in METHOD_ORDER:
        model_method_pairs.append({"model": model, "method": method})

df_complete = pd.DataFrame(model_method_pairs)
df_main = df_complete.merge(df_main, on=["model", "method"], how="left")

# Fill NaN with "?"
for col in ["WIC", "SSPC", "GR", "CSC", "AVG"]:
    df_main[col] = df_main[col].fillna("?")

# --- Compute AVG for main table from visible columns ---
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


main_cols_for_avg = ["CSC", "GR", "SSPC", "WIC"]

def avg_high_precision(row):
    vals = [safe_float(x) for x in row if x != '?']
    return f"{pd.Series(vals).mean():.6f}" if vals else "?"
df_main["AVG"] = df_main[main_cols_for_avg].apply(avg_high_precision, axis=1)
# For display, round to 3 decimals
if not df_main.empty:
    df_main["AVG"] = df_main["AVG"].apply(lambda x: f"{float(x):.3f}" if x != "?" else "?")

cols_main = ["model", "method", "WIC", "SSPC", "GR", "CSC", "AVG"]
df_main_out = df_main[cols_main].copy()

# Naive length-only baselines: no model grid to fill in, just recompute AVG the same way.
df_main_baseline = pd.DataFrame(rows_main_baseline)
if not df_main_baseline.empty:
    for col in ["WIC", "SSPC", "GR", "CSC", "AVG"]:
        if col not in df_main_baseline.columns:
            df_main_baseline[col] = None
        df_main_baseline[col] = df_main_baseline[col].fillna("?")
    df_main_baseline["AVG"] = df_main_baseline[main_cols_for_avg].apply(avg_high_precision, axis=1)
    df_main_baseline["AVG"] = df_main_baseline["AVG"].apply(lambda x: f"{float(x):.3f}" if x != "?" else "?")
    df_main_baseline["_order"] = df_main_baseline["method"].apply(
        lambda m: BASELINE_METHODS.index(m) if m in BASELINE_METHODS else len(BASELINE_METHODS)
    )
    df_main_baseline = df_main_baseline.sort_values(["_order", "model"]).drop(columns="_order")
df_main_baseline_out = df_main_baseline[cols_main].copy() if not df_main_baseline.empty else df_main_baseline

df_appendix = pd.DataFrame(rows_appendix)

# Fill in missing model-method combinations for appendix too
df_complete_appendix = pd.DataFrame(model_method_pairs)
df_appendix = df_complete_appendix.merge(df_appendix, on=["model", "method"], how="left")

# Fill NaN with "?"
for col in sorted_task_abbrevs + ["AVG"]:
    if col in df_appendix.columns:
        df_appendix[col] = df_appendix[col].fillna("?")


# --- Compute AVG for appendix table from visible columns ---
appendix_cols_for_avg = sorted_task_abbrevs
def row_avg(row):
    vals = [safe_float(row[c]) for c in appendix_cols_for_avg if row[c] != '?']
    return f"{pd.Series(vals).mean():.6f}" if vals else "?"
df_appendix["AVG"] = df_appendix.apply(row_avg, axis=1)
if not df_appendix.empty:
    df_appendix["AVG"] = df_appendix["AVG"].apply(lambda x: f"{float(x):.3f}" if x != "?" else "?")

cols_appendix = ["model", "method"] + sorted_task_abbrevs + ["AVG"]
df_appendix_out = df_appendix[cols_appendix].copy()

df_appendix_baseline = pd.DataFrame(rows_appendix_baseline)
if not df_appendix_baseline.empty:
    for col in sorted_task_abbrevs + ["AVG"]:
        if col not in df_appendix_baseline.columns:
            df_appendix_baseline[col] = None
        df_appendix_baseline[col] = df_appendix_baseline[col].fillna("?")
    df_appendix_baseline["AVG"] = df_appendix_baseline.apply(row_avg, axis=1)
    df_appendix_baseline["AVG"] = df_appendix_baseline["AVG"].apply(lambda x: f"{float(x):.3f}" if x != "?" else "?")
    df_appendix_baseline["_order"] = df_appendix_baseline["method"].apply(
        lambda m: BASELINE_METHODS.index(m) if m in BASELINE_METHODS else len(BASELINE_METHODS)
    )
    df_appendix_baseline = df_appendix_baseline.sort_values(["_order", "model"]).drop(columns="_order")
df_appendix_baseline_out = df_appendix_baseline[cols_appendix].copy() if not df_appendix_baseline.empty \
    else df_appendix_baseline

if args.latex:
    _print_main_table_latex(df_main_out, df_baselines=df_main_baseline_out)
    print()
    _print_appendix_table_latex(df_appendix_out, sorted_task_abbrevs, df_baselines=df_appendix_baseline_out)
else:
    print("### Dialog2Dialog: Kendall τ by Hallucination Type\n")
    print("*Main paper: Kendall τ per hallucination type (CSC, GR, SSPC, WIC) and overall average.*\n")
    print(df_main_out.to_markdown(index=False))

    if not df_main_baseline_out.empty:
        print("\n### Construction-artifact baseline controls\n")
        print("*'random' is the exact chance-level Kendall tau for a uniformly random ranking "
              "(0 by symmetry, computed via exact permutation enumeration, not simulated). "
              "'turn-count-diff'/'word-count-diff' use distance = |turn-count| / |word-count| "
              "difference between the original dialog and each fork, no embedding model involved "
              "(computed once, not per encoder). See §4.1: together these check whether a trivial "
              "heuristic already tracks the fork-position ranking used as ground truth.*\n")
        print(df_main_baseline_out.drop(columns=["model"]).to_markdown(index=False))

    print("\n\n<details><summary>Appendix: Kendall τ by Task</summary>\n")
    print("*Detailed per-task Kendall τ across all models and methods.*\n")
    print(df_appendix_out.to_markdown(index=False))
    if not df_appendix_baseline_out.empty:
        print("\n*Construction-artifact baseline controls (per task):*\n")
        print(df_appendix_baseline_out.drop(columns=["model"]).to_markdown(index=False))
    print("\n</details>")

if args.significance:
    baseline_methods = [m for m in METHOD_ORDER if m != "flowscore"] + BASELINE_METHODS
    df_sig = _compute_significance(
        ROOT,
        models=all_models,
        baseline_methods=baseline_methods,
        target_method="flowscore",
        n_boot=args.n_bootstrap,
        seed=args.significance_seed,
    )
    print()
    if args.latex:
        _print_significance_table_latex(df_sig, target_method="flowscore")
    else:
        _print_significance_table_markdown(df_sig, target_method="flowscore")
