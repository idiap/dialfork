# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""
Set-to-set dialog distance table generator for manuscript reporting.

This script aggregates set-to-set dialog distance ranking results and renders
paper-ready tables (markdown or LaTeX) measuring how well different distance
metrics can separate sets of dialogs according to their degree of workflow
deviation.

================================================================================
TASK DEFINITION: SET-TO-SET DIALOG DISTANCE RANKING AS A DATASET SANITY CHECK
================================================================================

Unlike the dialog2dialog task (which measures pairwise dialog distances), this
task operates at the *set level*: it measures the distance between the full
distribution of original dialogs and each distribution of hallucinated dialogs.

The primary goal is NOT to find the best distance metric; it is to serve as a
*sanity check* that the DialFork dataset is correctly constructed — i.e. that
the forking / hallucination injection mechanism produces a well-ordered spectrum
of workflow deviations.

If existing set-to-set distance metrics can rank the four hallucinated sets in
the expected order far above chance (raw Kendall τ, chance = 0), that confirms
the dataset's construction is sound and the forking concept works as intended.

FORK-BASED SET CONSTRUCTION
----------------------------
For each task, the original dialogs all share the same underlying workflow.
Hallucinated dialogs are forked at one of four segment boundaries:

    - fork at segment 1 start  ->  most deviant  (smallest shared prefix)
    - fork at segment 2 start  ->  more deviant
    - fork at segment 3 start  ->  less deviant
    - fork at segment 4 start  ->  least deviant (longest shared workflow prefix)

All dialogs forked at the same segment boundary are grouped into one set.
This yields four sets with a natural, unambiguous ground-truth distance ranking
relative to the original set:

    d(original, fork@1) > d(original, fork@2) > d(original, fork@3) > d(original, fork@4)

One such ranking problem is defined per task (i.e. per underlying workflow).
Because every task carries a single workflow, all dialogs within the original set
are strictly comparable, which is the key property that makes set-level distance
meaningful here.

EVALUATION TARGET
-----------------
The evaluation asks: can the distance metric recover the above ground-truth
ranking of the four hallucinated sets?  This is measured with Kendall τ
correlation (raw, in [-1, 1]; chance = 0) between the metric-induced ranking
and the ground-truth ranking.

================================================================================
DISTANCE METHODS EVALUATED
================================================================================

We evaluate three families of set-to-set dialog distance metrics:

1. FBD — Frechet BERT Distance
     Compute the Frechet distance between the multivariate Gaussian distributions
    fitted to sentence-pair embeddings (CLS channels) of the reference set and
    the target set. Because each embedding is computed from consecutive turns,
    FBD captures local transition information implicitly in latent space via
    shifts in the per-channel embedding distribution.
     Reference: Mehri & Eskenazi, ACL Findings 2021
                https://arxiv.org/pdf/2105.02573

2. PRD — Precision-Recall Distance
    Build cluster histograms over sentence-pair embeddings (k-means) for both
    the reference and target sets, sweep a family of linear trade-off
    constraints, and report 1 minus the maximum F1 on the resulting PRD curve.
    As with FBD, sequential cues are represented implicitly through latent
    embedding geometry of consecutive-turn encodings, rather than explicit
    graph transition probabilities.
     Reference: Mehri & Eskenazi, ACL Findings 2021
                https://arxiv.org/pdf/2105.02573

3. FlowKL — Dialog Flow Graph KL Divergence  [proposed]
     Arguably (since graph can be very large in practice) a more interpretable
     alternative grounded in an explicit, task-specific dialog flow representation.
     Full procedure:

       Step 1 — Graph construction:
           Build a dialog flow graph G from the original set using the Dialog2Flow
           embeddings (Burdisso et al., EMNLP 2024;
           https://aclanthology.org/2024.emnlp-main.310.pdf).
           Each graph node n_i represents a cluster of semantically similar
           utterances (a dialog act / workflow step); directed edges encode
           observed action/step transitions across all original dialogs, with
           edge weights proportional to transition frequencies.

       Step 2 — Per-dialog flow likelihood:
           For each dialog d = (u_1, ..., u_T) in a target set, map every
           utterance u_t to the node n_t with the nearest centroid (hard
           assignment, no node-confidence weighting), and treat G as a bigram
           language model over the node sequence.  The per-dialog flow likelihood
           is the geometric mean of the Laplace-smoothed transition probabilities
           over the transitions of d observed in G (set K_d):

               L(d) = ( prod_{t in K_d} P_G(n_t | n_{t-1}) )^(1 / |K_d|)

           (sdialog's DialogFlowScore with use_softmax=False and
           use_only_known_edges=True; see methods/task1/run.py.)

       Step 3 — Distribution distance:
           Apply kernel density estimation (KDE) to the per-dialog L(d) values
           in each set, yielding a smooth distribution per set.  Compute the KL
           divergence between the reference (original) distribution p_ref and
           the target (hallucinated) distribution p_tgt:

               d_FlowKL(target) = KL( p_ref(L) || p_tgt(L) )

     Interpretability advantage:
           Unlike FBD and PRD — which operate on high-dimensional per-channel
           embedding distributions — FlowKL uses a single scalar (L(d)) derived
           from an explicit, human-readable workflow graph G.  The graph can be
           inspected directly, making it straightforward to understand *why* a
           set is close to or distant from the reference.

================================================================================
METRIC AND RESULTS SUMMARY
================================================================================

Primary evaluation metric: Kendall τ correlation between the metric-induced
ranking of the four hallucinated sets and the ground-truth ranking
(fork@4 < fork@3 < fork@2 < fork@1 in terms of distance from original).

The script also supports accuracy (exact rank match) and NDCG as alternatives.

Results are reported per task and averaged across tasks.

Observed outcome (raw Kendall τ, averaged over tasks; rows = metrics, columns = hallucination types):

     Hallucination type abbreviations used below:
        WIC  = workflow_identity_corruption
        SSPC = step_selection_policy_corruption
        GR   = goal_replacement
        CSC  = conflicting_system_constraints

                                  WIC      SSPC     GR       CSC      AVG
     Random (chance)              0.000    0.000    0.000    0.000    0.000
     MeanLenDiff-turns           -0.111   -0.347    0.278   -0.222   -0.101
     MeanLenDiff-words            0.222    0.208    0.583   -0.208    0.201
     FBD                          0.972    0.792    0.986    0.903    0.913
     PRD                          0.931    0.764    0.986    0.833    0.878
     FlowKL  [proposed]           0.847    0.903    0.944    0.833    0.882

All three distance metrics recover the ground-truth ordering far above chance
(Fisher-combined p < 0.001; exact 4-way recovery in 68-77% of the 96
(task, type) units vs. 4.2% by chance), while the length-only controls stay
near chance.  This provides set-level evidence that DialFork's fork-based
construction is valid and is not explained by dialog length.

Key observations for the paper:

  1. FlowKL is statistically indistinguishable from FBD and PRD (paired
      Wilcoxon, Holm-corrected, p = 0.71), while being more interpretable
      (an explicit, human-readable flow graph; see the Appendix figure).

  2. FlowKL is the best metric on SSPC (step_selection_policy_corruption):
      0.903 vs FBD 0.792 and PRD 0.764.  This is the hallucination type where the
      forked dialog follows a *different policy* for selecting the next workflow
      step — precisely what an explicit transition model captures, whereas FBD and
      PRD capture step transitions only implicitly through consecutive-turn
      embedding distributions.  Conversely, FlowKL is weaker on WIC (0.847 vs 0.972).

  3. Easiest hallucination type for FBD/PRD: GR and WIC (>= 0.93); hardest: SSPC.

================================================================================
IMPLEMENTATION NOTES
================================================================================

- This script is a reporting/aggregation layer only.  It reads pre-computed
    JSON result files (results_all.json, results_acc.json) from per-task
    subdirectories generated by the upstream experiment scripts.
- Short task-name abbreviations are derived automatically from task names and
    de-duplicated where necessary.
- Rows can be filtered by method-name substring via --exclude_methods.
- Significance (on by default, disable with --no-significance):
    (1) Sanity check vs. chance — per (task, hallucination_type) unit (one 4-set
        ranking problem), the exact one-sided Kendall-tau permutation p-value,
        combined per metric via Fisher's method, plus a binomial test on the
        exact-match rate against chance (1/n!). Supports the "not by chance"
        framing of this task, unlike task2's per-dialogue paired tests — here
        each unit ranks only 3-4 sets, so there is no large per-item paired
        sample to draw on.
    (2) FlowKL vs. each baseline (FBD, PRD) — paired Wilcoxon
        signed-rank test and paired t-test (two-sided) on per-(task,
        hallucination_type) scores, Holm-Bonferroni corrected across the 3
        baselines, with bootstrap CIs on the mean gap. n is small (~task count
        x hallucination types) and scores are coarse (few distinct rank-based
        values), so this test has much less power than task2's.

Usage
-----
    python3 methods/task1/tables.py -r results/task1/dialog2flow-joint-dse-base/
"""

import argparse
import os
import json
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau, wilcoxon, ttest_rel, binom, chi2
from itertools import combinations, permutations
from math import factorial
from functools import lru_cache


def compute_ndcg(y_true, y_pred, k=None):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Args:
        y_true: Ground truth ranking (list of items in order of relevance)
        y_pred: Predicted ranking (list of items in predicted order)
        k: Consider only the top k items (if None, consider all)

    Returns:
        NDCG score between 0 and 1
    """
    def dcg_at_k(r, k):
        """Compute DCG@k"""
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    # Convert rankings to relevance scores
    # Items at the top of y_true get higher relevance scores
    n = len(y_true)
    relevance_map = {item: n - i for i, item in enumerate(y_true)}

    # Get relevance scores for predicted ranking
    predicted_relevances = [relevance_map.get(item, 0) for item in y_pred]

    # Compute DCG for predicted ranking
    dcg = dcg_at_k(predicted_relevances, k if k else len(y_pred))

    # Compute ideal DCG (best possible ranking)
    ideal_relevances = sorted(predicted_relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k if k else len(y_pred))

    if idcg == 0:
        return 0.

    return dcg / idcg


@lru_cache(maxsize=None)
def _exact_random_baseline_score(n, metric):
    """Expected score of a uniformly random n-item ranking against a fixed ground truth,
    for the given --metric, via exact enumeration over all n! permutations (n is small
    here -- at most a handful of forked sets -- so this is exact, not a Monte Carlo
    estimate). Reuses the same scoring functions as the real per-(task, hallucination_type)
    computation below, so it always matches whatever those formulas actually compute.
    """
    if n < 2:
        return float("nan")
    ground_truth = list(range(n))
    scores = []
    for perm in permutations(ground_truth):
        predicted = list(perm)
        if metric == "accuracy":
            scores.append(accuracy_score(ground_truth, predicted))
        elif metric == "kendall":
            tau, _ = kendalltau(ground_truth, predicted)
            scores.append(tau if not np.isnan(tau) else 0.0)
        elif metric == "ndcg":
            scores.append(compute_ndcg(ground_truth, predicted))
        else:
            raise ValueError(f"Unsupported metric for random baseline: {metric}")
    return float(np.mean(scores))


parser = argparse.ArgumentParser()

parser.add_argument("-r", "--results_folder", type=str, default="n0.001-e0.001_",
                    help="Path to the folder where results will be saved.")
parser.add_argument("-m", "--metric", type=str, default="kendall",
                    choices=["accuracy", "kendall", "ndcg"],
                    help="Metric to use for evaluation. Options: accuracy (exact match), kendall (Kendall's Tau), ndcg (Normalized Discounted Cumulative Gain).")
parser.add_argument("-ht", "--hallucination_types", nargs="+",
                    choices=["all", "workflow_identity_corruption", "step_selection_policy_corruption",
                             "goal_replacement", "dialogue_state_reset",
                             "conflicting_system_constraints"],
                    default=["all"],
                    # default=["workflow_identity_corruption"],
                    help="List of hallucination types to evaluate. If not provided, all types will be evaluated.")
parser.add_argument("-eht", "--exclude_hallucination_types", nargs="+",
                    choices=["workflow_identity_corruption", "step_selection_policy_corruption",
                             "goal_replacement", "dialogue_state_reset",
                             "conflicting_system_constraints"],
                    default=["dialogue_state_reset"],
                    help="List of hallucination types to exclude from evaluation.")
parser.add_argument("-t", "--tasks", nargs="+", default=["all"],
                    help="List of tasks to evaluate.")
parser.add_argument("-et", "--exclude_tasks", nargs="+", default=[],
                    help="List of tasks to exclude from evaluation.")
parser.add_argument("-s", "--sort", action="store_true",
                    help="Sort final results by AVG column (descending).")
parser.add_argument("-nsn", "--no_short_names", dest="short_names", action="store_false",
                    default=True,
                    help="Disable short task names in result columns.")
parser.add_argument("-tr", "--transpose", action="store_true",
                    help="Print transposed table: rows are tasks and columns are metrics.")
parser.add_argument("-md", "--markdown", action="store_true",
                    help="Print table as markdown format.")
parser.add_argument("-em", "--exclude_methods", nargs="+", default=[],
                    help="Exclude methods (rows) whose names contain any of the provided substrings. Example: -em ai sys")
parser.add_argument("-p", "--print_table", type=str, choices=["hallucination", "tasks"],
                    default="hallucination",
                    help="Which final table to print: 'hallucination' (metric x hallucination type, averaged over tasks) "
                         "or 'tasks' (metric x task, averaged over hallucination types). Default: hallucination.")
parser.add_argument(
    "--significance",
    dest="significance",
    action="store_true",
    default=True,
    help=("Report two significance checks (on by default, disable with --no-significance): "
          "(1) a sanity-check test — per (task, hallucination_type) unit, the exact permutation "
          "p-value of the observed Kendall tau against the random-ordering null, combined across "
          "units per metric via Fisher's method, plus a binomial test on the exact-match rate vs. "
          "chance (1/n! for n forked sets); (2) a paired comparison of FlowKL against each of "
          "FBD, PRD, and the naive length-only controls MeanLenDiff-turns/-words "
          "(paired Wilcoxon signed-rank + paired t-test, two-sided, on per-(task, hallucination_type) "
          "scores under the selected --metric), Holm-Bonferroni corrected across that family of "
          "baselines, with bootstrap CIs on the mean gap."),
)
parser.add_argument(
    "--no-significance",
    dest="significance",
    action="store_false",
    help="Skip the significance-testing sections.",
)
parser.add_argument("--n-bootstrap", type=int, default=10000,
                    help="Number of paired bootstrap resamples for the FlowKL-vs-baseline mean-gap CIs (default: 10000).")
parser.add_argument("--significance-seed", type=int, default=42,
                    help="Random seed for the bootstrap resampling (default: 42).")
parser.add_argument("--significance-alpha", type=float, default=0.05,
                    help="Significance threshold used to flag results in the significance tables (default: 0.05).")
args = parser.parse_args()

HALLUCINATION_ABBREVIATIONS = {
    "workflow_identity_corruption": "WIC",
    "step_selection_policy_corruption": "SSPC",
    "goal_replacement": "GR",
    "conflicting_system_constraints": "CSC",
}


def _shorten_task_name(name: str) -> str:
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    return "".join(p[0] for p in parts)


def _dedupe_names(names):
    counts = {}
    deduped = []
    for n in names:
        counts[n] = counts.get(n, 0) + 1
        if counts[n] == 1:
            deduped.append(n)
        else:
            deduped.append(f"{n}{counts[n]}")
    return deduped


def _move_flowkl_to_end(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical row/column order: the chance-level Random baseline first, then the
    naive length baselines (MeanLenDiff-*), then the embedding-distribution methods
    (FBD, PRD, ...), FlowKL variants last. Relative order within each group is
    preserved (stable sort).
    """
    def _group(idx):
        name = str(idx).lower()
        if name == "random":
            return 0
        if "meanlendiff" in name:
            return 1
        if "flowkl" in name:
            return 3
        return 2

    order = sorted(range(len(df.index)), key=lambda i: (_group(df.index[i]), i))
    return df.iloc[order]


def _filter_methods_by_substrings(df: pd.DataFrame, substrings):
    if not substrings:
        return df

    normalized_substrings = [s.lower() for s in substrings if s]
    if not normalized_substrings:
        return df

    keep_mask = [
        not any(sub in str(idx).lower() for sub in normalized_substrings)
        for idx in df.index
    ]
    return df.loc[keep_mask]


def _strip_metric_name(name: str) -> str:
    if name.startswith("kde-distance-"):
        name = name[len("kde-distance-"):]
    if name.startswith("centroid-similarity-"):
        name = name[len("centroid-similarity-"):]
    return name


def _tie_aware_tau(dist, order):
    """Kendall tau between the distances {set: d} and the ground-truth `order` (most
    similar first, i.e. distance should increase along it). Tied distances count as
    neither concordant nor discordant: the expected tau under random tie-breaking,
    as in Task 2 and evaluate.py."""
    n = len(order)
    score = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = dist[order[i]], dist[order[j]]
            score += (a < b) - (a > b)
    return score / (n * (n - 1) / 2.0)


def _tie_aware_tau_p(dist, order):
    """(tau, exact one-sided p): p = share of the n! assignments of the observed
    distances to the sets whose tau is >= the observed one (ties handled as above)."""
    tau = _tie_aware_tau(dist, order)
    values = [dist[k] for k in order]
    n_ge = n_all = 0
    for perm in permutations(values):
        n_all += 1
        n_ge += _tie_aware_tau(dict(zip(order, perm)), order) >= tau - 1e-12
    return tau, n_ge / n_all


def _kendall_one_sided_p(tau, p_two_sided):
    """Convert scipy's exact two-sided Kendall-tau p-value into a one-sided
    p-value for the a priori direction (metric distance grows with fork
    deviation, i.e. tau > 0 under the true ordering). Valid because the exact
    permutation null distribution of tau is symmetric around 0.
    """
    if tau is None or np.isnan(tau):
        return float("nan")
    return p_two_sided / 2 if tau >= 0 else 1 - p_two_sided / 2


def _format_p(p):
    if np.isnan(p):
        return "?"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fisher_combined_p(pvals):
    """Fisher's method to combine independent one-sided p-values into one
    omnibus test of whether a metric recovers the ground-truth ordering far
    beyond what random chance would produce.
    """
    pvals = [max(p, 1e-300) for p in pvals if not np.isnan(p)]
    if not pvals:
        return float("nan"), float("nan")
    statistic = -2.0 * np.sum(np.log(pvals))
    p_combined = float(chi2.sf(statistic, df=2 * len(pvals)))
    return float(statistic), p_combined


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


def _ht_passes_filter(ht):
    if "all" not in args.hallucination_types and not any(ht.startswith(h) for h in args.hallucination_types):
        return False
    if args.exclude_hallucination_types and any(ht.startswith(h) for h in args.exclude_hallucination_types):
        return False
    return True


def _compute_sanity_summary(sig_records):
    """Sanity check: does each metric recover the ground-truth fork ordering
    far beyond what a random assignment of distances to forks would produce?
    Combines the exact per-unit Kendall-tau permutation p-values via Fisher's
    method, and separately tests the exact-match rate against chance (1/n!).
    """
    by_metric = {}
    for r in sig_records:
        by_metric.setdefault(r["metric"], []).append(r)

    rows = []
    for metric_name, records in sorted(by_metric.items()):
        p_values = [r["p_one_sided"] for r in records if not np.isnan(r["p_one_sided"])]
        fisher_stat, p_combined = _fisher_combined_p(p_values)

        n_units = len(records)
        n_exact = sum(1 for r in records if r["exact_match"])
        n_items_values = [r["n_items"] for r in records if r["n_items"] > 0]
        # Assumes (as is the case for DialFork) that group size is constant across
        # units; if it varies, this averages the per-unit chance rate, which is an
        # approximation rather than an exact test.
        chance_rate = float(np.mean([1.0 / factorial(n) for n in n_items_values])) if n_items_values else float("nan")
        if n_units > 0 and not np.isnan(chance_rate):
            # One-sided P(X >= n_exact) under X ~ Binomial(n_units, chance_rate).
            binom_p = float(binom.sf(n_exact - 1, n_units, chance_rate))
        else:
            binom_p = float("nan")

        taus = [r["tau"] for r in records if not np.isnan(r["tau"])]
        rows.append({
            "metric": metric_name,
            "n_units": n_units,
            "mean_tau": float(np.mean(taus)) if taus else float("nan"),
            "n_exact_match": n_exact,
            "exact_match_rate": n_exact / n_units if n_units else float("nan"),
            "chance_rate": chance_rate,
            "binomial_p": binom_p,
            "fisher_stat": fisher_stat,
            "fisher_p": p_combined,
        })
    return pd.DataFrame(rows)


def _compute_flowkl_significance(paired_records, target="FlowKL",
                                  baselines=("FBD", "PRD", "MeanLenDiff-turns", "MeanLenDiff-words"),
                                  n_boot=10000, seed=42, alpha=0.05):
    """Paired Wilcoxon signed-rank test and paired t-test (both two-sided) comparing
    FlowKL against each baseline on per-(task, hallucination_type) scores, with
    paired bootstrap CIs on the mean gap. P-values for each test are Holm-Bonferroni
    corrected within their own family across baselines.
    """
    rng = np.random.default_rng(seed)
    target_data = paired_records.get(target, {})

    family_rows = []
    for baseline in baselines:
        baseline_data = paired_records.get(baseline, {})
        keys = sorted(set(target_data) & set(baseline_data))
        if len(keys) < 2:
            continue

        x = np.array([target_data[k] for k in keys], dtype=np.float64)
        y = np.array([baseline_data[k] for k in keys], dtype=np.float64)
        # Kendall taus over 4 items are multiples of 1/6: round the differences so that
        # equal gaps are exact ties for the Wilcoxon ranking (float noise otherwise splits
        # them and makes W depend on rounding).
        diff = np.round(x - y, 10)

        if np.all(diff == 0):
            w_stat, w_p_raw = np.nan, 1.0
        else:
            w_stat, w_p_raw = wilcoxon(diff, alternative="two-sided")
        t_stat, t_p_raw = ttest_rel(x, y, alternative="two-sided")

        ci_low, ci_high = _paired_bootstrap_ci_mean(diff, n_boot=n_boot, alpha=alpha, rng=rng)

        family_rows.append({
            "baseline": baseline,
            "n_pairs": len(keys),
            "mean_diff": float(diff.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else float("nan"),
            "wilcoxon_p_raw": float(w_p_raw),
            "ttest_stat": float(t_stat) if not np.isnan(t_stat) else float("nan"),
            "ttest_p_raw": float(t_p_raw),
        })

    if family_rows:
        w_corrected = _holm_bonferroni([r["wilcoxon_p_raw"] for r in family_rows])
        t_corrected = _holm_bonferroni([r["ttest_p_raw"] for r in family_rows])
        for r, w_p_corr, t_p_corr in zip(family_rows, w_corrected, t_corrected):
            r["wilcoxon_p_corrected"] = w_p_corr
            r["wilcoxon_significant"] = w_p_corr < alpha
            r["ttest_p_corrected"] = t_p_corr
            r["ttest_significant"] = t_p_corr < alpha

    return pd.DataFrame(family_rows)


def _print_sanity_table(df_sanity):
    if df_sanity.empty:
        print("No sanity-check significance results available.")
        return
    print("*Per (task, hallucination_type) unit: exact one-sided Kendall tau permutation p-value "
          "(H1: distance grows with fork deviation), combined per metric via Fisher's method. "
          "Exact-match rate is the fraction of units where the full 4-way ranking was recovered "
          "exactly, tested one-sided against chance (1/n!) with a binomial test.*\n")
    display_rows = []
    for _, row in df_sanity.iterrows():
        display_rows.append({
            "Metric": row["metric"],
            "N units": int(row["n_units"]),
            "Mean tau": f"{row['mean_tau']:.3f}",
            "Exact match": f"{int(row['n_exact_match'])}/{int(row['n_units'])} ({100 * row['exact_match_rate']:.1f}%)",
            "Chance rate": f"{100 * row['chance_rate']:.2f}%",
            "Binomial p": _format_p(row["binomial_p"]),
            "Fisher-combined p": _format_p(row["fisher_p"]),
        })
    df_display = pd.DataFrame(display_rows)
    if args.markdown:
        print(df_display.to_markdown(index=False))
    else:
        print(df_display.to_string(index=False))


def _print_flowkl_significance_table(df_sig, metric_label):
    if df_sig.empty:
        print("No FlowKL-vs-baseline significance results available.")
        return
    print(f"*Paired Wilcoxon signed-rank test and paired t-test (both two-sided) comparing FlowKL "
          f"against each baseline on per-(task, hallucination_type) {metric_label} scores, with 95% "
          "paired-bootstrap CIs on the mean gap. P-values for each test are Holm-Bonferroni corrected "
          "within their own family across baselines.*\n")
    display_rows = []
    for _, row in df_sig.iterrows():
        display_rows.append({
            "Baseline": row["baseline"],
            "N": int(row["n_pairs"]),
            "Mean Δ (FlowKL - baseline)": f"{row['mean_diff']:.3f}",
            "95% CI": f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            "W": f"{row['wilcoxon_stat']:,.1f}" if not np.isnan(row["wilcoxon_stat"]) else "?",
            "p_Wilcoxon (Holm)": _format_p(row["wilcoxon_p_corrected"]),
            "t": f"{row['ttest_stat']:.2f}" if not np.isnan(row["ttest_stat"]) else "?",
            "p_ttest (Holm)": _format_p(row["ttest_p_corrected"]),
            "Sig. (Wilcoxon)": "yes" if row["wilcoxon_significant"] else "no",
        })
    df_display = pd.DataFrame(display_rows)
    if args.markdown:
        print(df_display.to_markdown(index=False))
    else:
        print(df_display.to_string(index=False))


if "all" in args.tasks:
    tasks = [d for d in os.listdir(args.results_folder)
             if os.path.isdir(os.path.join(args.results_folder, d))]
else:
    tasks = args.tasks

if args.exclude_tasks:
    tasks = [task for task in tasks if task not in args.exclude_tasks]

sorted_tasks = sorted(tasks)
task_short_name_map = {}
if args.short_names:
    base_short_names = [_shorten_task_name(t) for t in sorted_tasks]
    deduped_short_names = _dedupe_names(base_short_names)
    task_short_name_map = dict(zip(sorted_tasks, deduped_short_names))

results_combined = None
hallucination_tables = []
sig_records = []
paired_records = {}
for task in sorted_tasks:
    if args.print_table == "tasks":
        if args.short_names:
            print(f"Processing task: {task} ({task_short_name_map.get(task, _shorten_task_name(task))})")
        else:
            print(f"Processing task: {task}")

    if not os.path.exists(os.path.join(args.results_folder, task, "results_acc.json")):
        print(f"  - Skipping task {task} as results_acc.json not found.")
        continue

    with open(os.path.join(args.results_folder, task, "results_all.json"), "r") as f:
        results = json.load(f)

    # Define ground truth rank based on hallucination type
    # Get candidate keys from first hallucination type
    first_ht = next(iter(results.keys()))
    candidate_keys = list(results[first_ht].keys())

    # Map known hallucination types to their ground truth ranking
    if all(key.startswith("fork_at_segment_") for key in candidate_keys):
        ground_truth_rank = ["fork_at_segment_4", "fork_at_segment_3", "fork_at_segment_2", "fork_at_segment_1"]
    elif "happy" in candidate_keys:
        # For sentiment-based hallucinations
        ground_truth_rank = ["happy", "unhappy", "multitask"]
    else:
        # Fallback: assume fork segments with unknown count or use first available order
        ground_truth_rank = sorted(
            candidate_keys,
            key=lambda x: int(x.split("_")[-1]) if x.startswith("fork_at_segment_") else 0,
            reverse=True,
        )
    results_rank = {}
    ht = list(results.keys())[0]
    metrics = list(results[ht][list(results[ht].keys())[0]].keys())
    metrics = [m for m in metrics if "no-nc" not in m.lower() and "nonc" not in m.lower()]
    for ht in results:
        results_rank[ht] = {}
        for metric in metrics:
            results_rank[ht][metric] = [dataset for dataset, _ in sorted([(dataset, results[ht][dataset][metric]) for dataset in results[ht]], key=lambda x: x[1])]

    results_accs = {}
    for ht in results:
        current_ground_truth = ground_truth_rank
        results_accs[ht] = {}
        for metric in metrics:
            predicted_rank = results_rank[ht][metric]

            # Always compute Kendall's tau and its exact permutation p-value, independent of
            # --metric, since the sanity-check significance test needs it regardless of which
            # score is used for the main reported table. Computed from the distances, so tied
            # distances count as neither concordant nor discordant (not broken by key order).
            common_items = set(current_ground_truth) & set(predicted_rank)
            dist = {item: results[ht][item][metric] for item in common_items}
            order = [item for item in current_ground_truth if item in common_items]
            tau, p_one_sided = _tie_aware_tau_p(dist, order)
            # an exact recovery needs a strictly correct order: a tie is not a recovery
            strict = all(dist[order[i]] < dist[order[i + 1]] for i in range(len(order) - 1))

            if args.significance and _ht_passes_filter(ht):
                sig_records.append({
                    "task": task,
                    "ht": ht,
                    "metric": _strip_metric_name(metric),
                    "n_items": len(common_items),
                    "tau": float(tau) if not np.isnan(tau) else float("nan"),
                    "p_one_sided": p_one_sided,
                    "exact_match": bool(strict),
                })

            # Compute the selected metric
            if args.metric == "accuracy":
                score = accuracy_score(current_ground_truth, predicted_rank)
            elif args.metric == "kendall":
                # Raw Kendall's tau in [-1, 1] (chance = 0), as reported in the paper
                score = tau
            elif args.metric == "ndcg":
                score = compute_ndcg(current_ground_truth, predicted_rank)
            results_accs[ht][metric] = score

        # "Random" isn't computed from any real distance metric: it's the theoretical
        # chance-level score for a uniformly random ranking of the same size, under the
        # selected --metric, computed exactly (not simulated) via permutation enumeration.
        results_accs[ht]["Random"] = _exact_random_baseline_score(len(current_ground_truth), args.metric)

    df_accs = pd.DataFrame(results_accs)
    df_accs = df_accs.T
    results_acc = df_accs.to_dict()

    results_acc = {k.replace("kde-distance-", "", 1) if k.startswith("kde-distance-") else k: v
                    for k, v in results_acc.items()}
    results_acc = {k.replace("centroid-similarity-", "", 1) if k.startswith("centroid-similarity-") else k: v
                    for k, v in results_acc.items()}

    df_accs = pd.DataFrame(results_acc).T

    if "all" not in args.hallucination_types:
        cols_to_keep = []
        for ht in args.hallucination_types:
            cols_to_keep.extend([col for col in df_accs.columns if col.startswith(ht)])
        df_accs = df_accs[cols_to_keep]
        # print(df_accs.head())
        # print()

    # Exclude specified hallucination types
    if args.exclude_hallucination_types:
        cols_to_exclude = []
        for ht in args.exclude_hallucination_types:
            cols_to_exclude.extend([col for col in df_accs.columns if col.startswith(ht)])
        df_accs = df_accs.drop(columns=cols_to_exclude, errors='ignore')

    # This print a metric x hallucination type table for the current task
    # print(df_accs.round(3).to_markdown())

    if args.significance:
        for metric_name in df_accs.index:
            paired_records.setdefault(metric_name, {})
            for ht_col in df_accs.columns:
                val = df_accs.loc[metric_name, ht_col]
                if pd.notna(val):
                    paired_records[metric_name][(task, ht_col)] = float(val)

    hallucination_tables.append(df_accs.copy())

    df_accs_avg = df_accs.mean(axis=1).to_frame(name=task)

    if results_combined is None:
        results_combined = df_accs_avg
    else:
        results_combined = pd.concat([results_combined, df_accs_avg], axis=1)

# After processing all tasks, compute the average across tasks and print the final combined results
# result_combined is a metric x task table, where each cell is the average score across hallucination types for that metric and task
if results_combined is None:
    raise ValueError("No valid task results were found. Nothing to aggregate.")

results_combined["AVG"] = results_combined.mean(axis=1)

# metric x hallucination-type table where each cell is averaged across tasks
results_by_hallucination = (
    pd.concat(hallucination_tables, axis=0)
    .groupby(level=0)
    .mean(numeric_only=True)
)
results_by_hallucination["AVG"] = results_by_hallucination.mean(axis=1)

if args.exclude_methods:
    results_combined = _filter_methods_by_substrings(results_combined, args.exclude_methods)
    results_by_hallucination = _filter_methods_by_substrings(results_by_hallucination, args.exclude_methods)

if args.print_table == "tasks" and args.short_names and not args.transpose:
    task_columns = [c for c in results_combined.columns if c != "AVG"]
    rename_map = {c: task_short_name_map.get(c, _shorten_task_name(c)) for c in task_columns}
    results_combined = results_combined.rename(columns=rename_map)

if args.print_table == "hallucination":
    results_by_hallucination = results_by_hallucination.rename(columns=HALLUCINATION_ABBREVIATIONS)

if args.sort:
    results_combined = results_combined.sort_values(by="AVG", ascending=False)
    results_by_hallucination = results_by_hallucination.sort_values(by="AVG", ascending=False)
else:
    results_combined = _move_flowkl_to_end(results_combined)
    results_by_hallucination = _move_flowkl_to_end(results_by_hallucination)

if args.transpose:
    results_combined = results_combined.T
    results_combined = results_combined.sort_index()

print()
print(f"Metric used: {args.metric.upper()}")

if args.print_table == "tasks":
    print()
    if args.markdown:
        print(results_combined.round(3).to_markdown())
    else:
        print(results_combined.round(3))
else:
    print()
    if args.markdown:
        print(results_by_hallucination.round(3).to_markdown())
    else:
        print(results_by_hallucination.round(3))

if args.significance:
    df_sanity = _compute_sanity_summary(sig_records)
    print()
    print("### Sanity check: metric-induced ranking vs. chance\n")
    _print_sanity_table(df_sanity)

    df_flowkl_sig = _compute_flowkl_significance(
        paired_records,
        target="FlowKL",
        baselines=("FBD", "PRD", "MeanLenDiff-turns", "MeanLenDiff-words"),
        n_boot=args.n_bootstrap,
        seed=args.significance_seed,
        alpha=args.significance_alpha,
    )
    print()
    print("### Significance: FlowKL vs. each baseline\n")
    _print_flowkl_significance_table(df_flowkl_sig, metric_label=args.metric.upper())
