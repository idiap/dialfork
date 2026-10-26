# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""
Turn-prediction table generator for final manuscript reporting.

This script aggregates turn-prediction outputs from all methods and settings,
then emits paper-ready main and appendix tables with a unified ranking Score.
It is designed as the final reporting layer over per-task JSON artifacts.

The three evaluation regimes below should be read as separate benchmark tracks,
similar to distinct leaderboards with different access assumptions. Within each
track, the methods reported here serve as baseline or reference results for the
paper, rather than as a single global ranking across incomparable settings.

================================================================================
REALISM SETTINGS AND EVALUATION REGIMES
================================================================================

We report three settings with increasing access to privileged information and
decreasing realism, which are intended to contextualize the performance of:

1. Streaming (code: online; most realistic / hardest)
   - Input: only conformant dialogs for calibration/training (in real life could be real dialogs from humans that have no hallucinations by design).
   - Inference: streaming, turn-by-turn.
   - Goal: predict the off-road turn as early as possible while processing one
     turn at a time.

2. Non-Streaming (Dialog-Only) (code: offline; less realistic)
   - Input at inference: full dialog.
   - Goal: predict off-road turn from the complete conversation.
   - In this script, this corresponds to offline runs tagged as "no-workflow"
     (dialog-only condition).

3. Non-Streaming + Workflow (Oracle) (code: offline+workflow;
    least realistic / easiest)
   - Input at inference: full dialog plus ground-truth workflow.
   - This oracle-like condition is typically unavailable in real deployments,
     but is reported to contextualize upper-bound behavior.

================================================================================
METHODS INCLUDED IN TABLES
================================================================================

Streaming methods:
    Step 0: discretize turn embeddings into latent regions.
    Each turn u_t is encoded into an embedding phi(u_t) and quantized into a
    discrete node / region id n_t by clustering the conformant-turn embedding
    space with an agglomerative algorithm, following the Dialog2Flow setting
    (as done in FlowKL). After this step, a dialog is no longer a continuous
    sequence of turns but a discrete trajectory (n_1, ..., n_T) over region tokens.

    Step 1: train a token-level language model on node sequences.
    We fit a simple n-gram LM over the discrete node trajectories, so the model
    estimates the probability of each next region conditioned on the previous
    N-1 regions.

    Let n_t be the node assigned to turn t, and let p_t denote the calibrated
    n-gram flow-LM probability of that node transition.

    Raw turn probability:
        p_t = P_LM(n_t | n_{t-N+1}, ..., n_{t-1})

    Node-confidence (NC) weighted variant (+smooth):
        p_t^(NC) = p_t * c_t,
        c_t = softmax(sim(phi(u_t), c_{n_t^(k)}))_1

    where u_t is the utterance at turn t, phi is the sentence encoder,
    c_{n_t^(k)} are top-k node-centroid candidates, and c_t in [0,1] is the
    top-1 node-assignment confidence.

    Method 1: first_below (Improbable Transition; IT)
        - Decision rule: predict the first turn t such that p_t < theta_off
            (or p_t^(NC) < theta_off in +smooth).
        - Calibration: theta_off is set from conformant dialogs as the
            alpha_off = Q_{0.05}(min_t p_t),
            i.e. the 5th percentile of per-dialog minimum turn probabilities.
        - Intuition: this sets a conservative detection threshold so that only
            about 5% of conformant dialogs are expected to trigger a false off-road
            alarm during evaluation.

    Method 2: max_jump (Surprisal Spike; SS)
        - Define surprisal s_t = -log p_t (or s_t^(NC) = -log p_t^(NC)).
        - Define jump Delta_t = s_t - s_{t-1} for t >= 2.
        - Decision rule: predict the first t such that Delta_t > theta_jump.
        - Calibration: theta_jump is set from conformant dialogs as the
            alpha_jump = Q_{0.95}(max_t Delta_t),
            i.e. the 95th percentile of per-dialog maximum surprisal jumps.
        - Intuition: likewise, this chooses a high jump threshold so that only about 5%
            of conformant dialogs would be expected to exceed it by chance.

    Baselines:
        - first_turn: always predicts turn 1.
        - last_turn: always predicts the final turn.
        - random: predicts a uniformly random turn index.

Non-streaming methods:
    - Only zero-shot LLM methods are reported in Non-Streaming (Dialog-Only)
        and Non-Streaming + Workflow (Oracle).
  - Dedicated prompts are used for each condition:
    * Non-Streaming (Dialog-Only): dialog-only prompt
    * Non-Streaming + Workflow (Oracle): dialog + workflow prompt
  - LLM families/sizes reported in the paper:
      * gemma4:31b
      * qwen3.6:35b
      * gpt-oss:20b

Main manuscript table composition:
  - shared baselines
    - selected best streaming checkpoint rows
    - non-streaming LLM rows (dialog-only and +workflow)

Current manuscript selection uses the streaming encoder with the best Score:
  - LaBSE/lm-3-gram

================================================================================
REPORTED METRICS AND AGGREGATION SCOPE
================================================================================

For each method/setting, this script reports:
  - hallucination-type MAEs: CSC, GR, SSPC, WIC
  - Hallucinated_AVG MAE
  - Hallucinated_Coverage
  - Conformant MAE
  - Conformant_Coverage
  - composite Score (used for ranking)

The script reads AVERAGE.json summaries (plus task-level files when needed to
recover missing coverage/conformant details), harmonizes online/offline schema
differences, computes Score_100 = 100 * Score, and renders manuscript/appendix
tables in markdown or LaTeX.

================================================================================
PAPER-READY DEFINITION OF THE SCORE
================================================================================

The formula below is the canonical ranking definition used by this script.

Let each system i have the following raw metrics:
    - m_h(i): Hallucinated_AVG MAE (lower is better)
    - m_c(i): Conformant MAE (lower is better)
    - r_h(i): Hallucinated_Coverage (higher is better)
    - r_c(i): Conformant_Coverage (lower is better)

MAE terms use fixed anchors derived from baseline policies:
    - U_h = m_h(last_turn)   [task-grounded worst case for hallucinated MAE]
    - U_c = m_c(first_turn)  [task-grounded worst case for conformant MAE]
    - L_h = 0, L_c = 0

Then (with clipping to [0, 1]):
    n_h(i) = clip((m_h(i) - L_h) / (U_h - L_h), 0, 1)
    n_c(i) = clip((m_c(i) - L_c) / (U_c - L_c), 0, 1)

MAE desirability:
    - d_h_mae(i) = 1 - n_h(i)
    - d_c_mae(i) = 1 - n_c(i)

Coverage terms are used on their raw bounded scale (already in [0,1]
conceptually; implementation converts % to [0,1]):
    - d_h_cov(i) = clip(r_h(i), 0, 1)
    - d_c_cov(i) = 1 - clip(r_c(i), 0, 1)

Final Score is a weighted geometric mean:

    Score(i) = d_h_mae(i)^w_h * d_c_mae(i)^w_c * d_h_cov(i)^w_hc * d_c_cov(i)^w_cc

with weights:
    - w_h  = 0.35  (hallucinated MAE)
    - w_c  = 0.35  (conformant MAE)
    - w_hc = 0.15  (hallucinated coverage)
    - w_cc = 0.15  (conformant coverage)
and w_h + w_c + w_hc + w_cc = 1.

Interpretation:
    - Higher Score is better.
    - Ranking in tables is performed by descending Score.
    - Reported table value uses displayed Score(i) = 100 * raw_score(i).
    - Missing required components yield Score = None (displayed as '?').

Rationale:
    - Geometric aggregation penalizes systems that are very weak on any one axis,
      while still allowing weighted trade-offs across all objectives.
    - MAE anchors make scores more stable across runs than per-run min-max.
    - Raw coverage preserves absolute interpretation (0..1 / 0..100%).
    - Asymmetric anchors align with turn-position semantics: predicting last turn
      is maximally wrong for hallucinated targets, while predicting first turn
      is maximally wrong for conformant targets.

Streaming n-gram order note:
  - The paper uses 3-gram models. The order mainly trades false alarms for
    localization (lower N: fewer false alarms; higher N: more), with similar Scores.

"""

import os
import glob
import pandas as pd
import json
import math
import argparse



parser = argparse.ArgumentParser(description="Generate turn-prediction result tables.")
parser.add_argument(
    "--root",
    default="results/task3",
    help="Path containing the dialog-to-dialog results (default: results/task3).",
)
parser.add_argument(
    "--streaming-model",
    default="LaBSE/lm-3-gram",
    help="Streaming experiment shown in the main table (default: the paper's, "
         "LaBSE/lm-3-gram).",
)
parser.add_argument(
    "--latex",
    action="store_true",
    help="Print LaTeX tables (booktabs/multirow style) instead of markdown output.",
)
args = parser.parse_args()

# Composite score weights (must sum to 1.0)
# Equal coverage weights are used by default.
W_HALLU_MAE = 0.35
W_CONF_MAE = 0.35
W_HALLU_COV = 0.15
W_CONF_COV = 0.15
SCORE100_DIGITS = 2

HT_ABBREV = {
    "conflicting_system_constraints": "CSC",
    "goal_replacement": "GR",
    "step_selection_policy_corruption": "SSPC",
    "workflow_identity_corruption": "WIC",
}

# Find all AVERAGE.json summary files (online and offline)
pattern_online = os.path.join(args.root, "**", "AVERAGE.json")
pattern_offline = os.path.join(args.root, "offline", "**", "AVERAGE.json")
files_online = glob.glob(pattern_online, recursive=True)
files_offline = glob.glob(pattern_offline, recursive=True)
files = sorted(set(files_online + files_offline))


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _clip01(value):
    if value is None or pd.isna(value):
        return None
    return min(max(float(value), 0.0), 1.0)


def _normalize_with_anchors(value, lower, upper, eps=1e-12):
    if value is None or pd.isna(value) or lower is None or upper is None:
        return None
    span = float(upper) - float(lower)
    if abs(span) < eps:
        # No discrimination from this component; treated as ideal (n=0 => d=1).
        return 0.0
    raw = (float(value) - float(lower)) / span
    return _clip01(raw)


def _weighted_geomean(components):
    # components: list of (desirability in [0,1], weight)
    # Returns None when any required component is missing.
    eps = 1e-12
    acc = 0.0
    for desirability, weight in components:
        if desirability is None or pd.isna(desirability):
            return None
        d = min(max(float(desirability), eps), 1.0)
        acc += float(weight) * math.log(d)
    return float(math.exp(acc))


def _fmt(value, digits=2):
    if value is None or pd.isna(value):
        return "?"
    return f"{float(value):.{digits}f}"


def _base_method(method_label):
    return method_label.split("+", 1)[0]


def _method_display_and_rank(method_label):
    m = str(method_label)
    fixed = {
        "max_jump": ("max_jump", 0),
        "max_jump+smooth": ("max_jump +smooth", 1),
        "first_below": ("first_below", 2),
        "first_below+smooth": ("first_below +smooth", 3),
    }
    if m in fixed:
        return fixed[m]
    return m, 100


def _pretty_method_label(method_label):
    m = str(method_label)
    pretty = {
        "max_jump": "Surprisal Spike",
        "max_jump +smooth": "Surprisal Spike + node confidence",
        "first_below": "Improbable Transition",
        "first_below +smooth": "Improbable Transition + node confidence",
        "first_turn": "First turn",
        "last_turn": "Last turn",
        "random": "Random",
    }
    if m in pretty:
        return pretty[m]

    # Offline LLM names appear as llm-<family>:<size>, e.g. llm-gemma4:31b.
    # Render with paper-friendly capitalization and size suffix.
    if m.startswith("llm-"):
        body = m[len("llm-"):]
        model = body
        size = None
        if ":" in body:
            model, size = body.rsplit(":", 1)

        model_map = {
            "gemma4": "Gemma4",
            "gpt-oss": "GPT-OSS",
            "qwen3.6": "Qwen3.6",
        }
        model_disp = model_map.get(model, model)
        if size:
            size_disp = size.upper().replace("B", "B")
            return f"{model_disp}-{size_disp}"
        return model_disp

    return m


def _method_family_and_variant(method_label):
    m = str(method_label)
    if m.endswith(" +smooth"):
        return m[: -len(" +smooth")], "+smoothing"
    return m, "raw"


def _extract_benchmark_rows(payload, experiment_name):
    detection = payload.get("detection", {})
    htypes = detection.get("hallucination_types", {})
    happy = detection.get("happy", {}).get("variants", {})

    rows = []
    # Collect all variant keys present in any hallucination type.
    variants = set()
    for ht_data in htypes.values():
        variants.update((ht_data.get("variants") or {}).keys())

    for variant in sorted(variants):
        # Gather methods seen under online/offline metrics for this variant.
        methods_online = set()
        methods_offline = set()
        for ht_data in htypes.values():
            vdata = (ht_data.get("variants") or {}).get(variant, {})
            methods_online.update((vdata.get("online_metrics") or {}).keys())
            methods_offline.update((vdata.get("offline_metrics") or {}).keys())

        for mode, methods in (("online", methods_online), ("offline", methods_offline)):
            for method in sorted(methods):
                hallu_maes = []
                hallu_by_type = {}
                for ht_name, ht_data in htypes.items():
                    vdata = (ht_data.get("variants") or {}).get(variant, {})
                    metrics_map = vdata.get(f"{mode}_metrics") or {}
                    method_entry = metrics_map.get(method, {})
                    mae_val = method_entry.get("mae")
                    hallu_maes.append(mae_val)
                    hallu_by_type[HT_ABBREV.get(ht_name, ht_name)] = float(mae_val) if mae_val is not None else None
                hallucinated = _mean(hallu_maes)

                conformant = None
                happy_variant = happy.get(variant, {})
                happy_metrics = happy_variant.get(f"{mode}_metrics") or {}
                if method in happy_metrics:
                    conformant = happy_metrics[method].get("emae")

                if hallucinated is None:
                    continue

                # AVG is computed from available values to avoid NaN-only rows
                avg = _mean([hallucinated, conformant])

                label = method if variant == "raw" else f"{method}+{variant}"
                rows.append(
                    {
                        "experiment": experiment_name,
                        "setting": mode,
                        "method": label,
                        "CSC": hallu_by_type.get("CSC"),
                        "GR": hallu_by_type.get("GR"),
                        "SSPC": hallu_by_type.get("SSPC"),
                        "WIC": hallu_by_type.get("WIC"),
                        "Hallucinated_AVG": float(hallucinated),
                        "Conformant": float(conformant) if conformant is not None else None,
                        "AVG": float(avg) if avg is not None else None,
                    }
                )

    return rows


rows = []
payload_by_experiment = {}
for f in files:
    rel_dir = os.path.relpath(os.path.dirname(f), args.root)
    try:
        with open(f) as fin:
            payload = json.load(fin)
    except Exception:
        continue

    payload_by_experiment[rel_dir] = payload
    rows.extend(_extract_benchmark_rows(payload, rel_dir))

if not rows:
    print("No results found.")
    exit(0)

results = pd.DataFrame(rows)

# first_turn / random / last_turn are shared baselines and often repeated.
# Collapse them to one row per method label.
shared_methods = {"first_turn", "random", "last_turn"}
shared_mask = results["method"].apply(lambda m: _base_method(m) in shared_methods)

shared_rows = results[shared_mask].copy()
if not shared_rows.empty:
    shared_rows = shared_rows.sort_values(by=["method", "experiment", "setting"]) \
        .groupby("method", as_index=False).first()
    shared_rows["experiment"] = "ALL"
    shared_rows["setting"] = "baseline"

results = pd.concat([results[~shared_mask], shared_rows], ignore_index=True)

results["AVG_sort"] = results["AVG"].fillna(float("inf"))
results["Hallu_sort"] = results["Hallucinated_AVG"].fillna(float("inf"))
results = results.sort_values(by=["AVG_sort", "Hallu_sort"], ascending=[True, True])

# --- Compute coverage columns ---
def _coverage(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _task_files_for_experiment(experiment_name):
    if experiment_name == "ALL":
        return []
    exp_dir = os.path.join(args.root, experiment_name)
    if not os.path.isdir(exp_dir):
        return []
    task_filter = None
    payload = payload_by_experiment.get(experiment_name) or {}
    cfg = payload.get("config") or {}
    cfg_task_names = cfg.get("task_names")
    if isinstance(cfg_task_names, list) and cfg_task_names:
        task_filter = {f"{str(name)}.json" for name in cfg_task_names}

    out = []
    for name in os.listdir(exp_dir):
        if not name.endswith(".json"):
            continue
        if name == "AVERAGE.json":
            continue
        if task_filter is not None and name not in task_filter:
            continue
        out.append(os.path.join(exp_dir, name))
    return sorted(out)


def _aggregate_happy_from_task_files(experiment_name, mode, method, variant):
    # Streaming AVERAGE.json files may omit detection.happy; recover from per-task JSON files.
    task_files = _task_files_for_experiment(experiment_name)
    if not task_files:
        return None, None

    emae_values = []
    n_pred_sum = 0
    n_total_sum = 0
    found_counts = False

    for fp in task_files:
        try:
            payload = json.load(open(fp))
        except Exception:
            continue

        happy = payload.get("detection", {}).get("happy", {}).get("variants", {})
        happy_variant = happy.get(variant, {})
        happy_metrics = happy_variant.get(f"{mode}_metrics") or {}
        entry = happy_metrics.get(method)
        if not isinstance(entry, dict):
            continue

        if entry.get("emae") is not None:
            emae_values.append(float(entry["emae"]))

        n_pred = entry.get("n_pred")
        n_total = entry.get("n_total")
        if n_pred is not None and n_total not in (None, 0):
            n_pred_sum += int(n_pred)
            n_total_sum += int(n_total)
            found_counts = True
        elif entry.get("pred_rate") is not None and n_total not in (None, 0):
            # If n_pred is missing but rate exists with total, reconstruct contribution.
            rate = float(entry["pred_rate"])
            n_total_i = int(n_total)
            n_pred_sum += int(round(rate * n_total_i))
            n_total_sum += n_total_i
            found_counts = True

    conformant = _mean(emae_values) if emae_values else None
    conformant_cov = (100.0 * n_pred_sum / n_total_sum) if found_counts and n_total_sum > 0 else None
    return conformant, conformant_cov

# Add coverage columns for hallucinated and conformant

# --- Fill coverage columns from JSON ---

# Improved: extract Hallucinated_Coverage as mean pred_rate across all hallucinated types;
# extract Conformant_Coverage as mean pred_rate for happy (non-hallucinated) dialogs if available.
def extract_coverage_and_conformant(row, files):
    exp = row["experiment"]
    setting = row["setting"]
    method = row["method"]
    base_method = _base_method(method)
    variant = "raw" if "+" not in method else method.split("+", 1)[1]
    if exp == "ALL":
        # Baselines come from online methods; aggregate across non-offline experiments only.
        payloads = [
            p for name, p in payload_by_experiment.items()
            if not name.startswith("offline/")
        ]
    else:
        payloads = [payload_by_experiment.get(exp)]
    payloads = [p for p in payloads if p is not None]
    if not payloads:
        return None, None, None

    # Hallucinated coverage: mean pred_rate across all hallucination types
    hallu_coverages = []

    # Conformant coverage: mean pred_rate from happy section (if available)
    conformant_coverages = []
    conformants = []

    for payload in payloads:
        detection = payload.get("detection", {})
        htypes = detection.get("hallucination_types", {})
        happy = detection.get("happy", {}).get("variants", {})

        mode_for_row = "online" if setting == "baseline" else setting

        for ht_data in htypes.values():
            vdata = (ht_data.get("variants") or {}).get(variant, {})
            metrics_map = vdata.get(f"{mode_for_row}_metrics") or {}
            method_entry = metrics_map.get(base_method, {})
            if "pred_rate" in method_entry:
                hallu_coverages.append(method_entry["pred_rate"] * 100)

        happy_variant = happy.get(variant, {})
        happy_metrics = happy_variant.get(f"{mode_for_row}_metrics") or {}
        if base_method in happy_metrics:
            entry = happy_metrics[base_method]
            if "pred_rate" in entry:
                conformant_coverages.append(entry["pred_rate"] * 100)
            if "emae" in entry:
                conformants.append(entry["emae"])
    # Fallback for baselines: try all settings and all variants
    if exp == "ALL":
        for payload in payloads:
            detection = payload.get("detection", {})
            htypes = detection.get("hallucination_types", {})
            happy = detection.get("happy", {}).get("variants", {})

            for ht_data in htypes.values():
                for vname, vdata in (ht_data.get("variants") or {}).items():
                    for mode in ["online", "offline"]:
                        metrics_map = vdata.get(f"{mode}_metrics") or {}
                        method_entry = metrics_map.get(base_method, {})
                        if "pred_rate" in method_entry:
                            hallu_coverages.append(method_entry["pred_rate"] * 100)

            for vname, happy_variant in happy.items():
                for mode in ["online", "offline"]:
                    happy_metrics = happy_variant.get(f"{mode}_metrics") or {}
                    if base_method in happy_metrics:
                        entry = happy_metrics[base_method]
                        if "pred_rate" in entry:
                            conformant_coverages.append(entry["pred_rate"] * 100)
                        if "emae" in entry:
                            conformants.append(entry["emae"])
    hallu_cov = _mean(hallu_coverages) if hallu_coverages else None
    conf_cov = _mean(conformant_coverages) if conformant_coverages else None
    conf_val = _mean(conformants) if conformants else None

    # Fallback path for streaming schema where AVERAGE.json may not include detection.happy.
    if setting in {"online", "baseline"} and (conf_cov is None or conf_val is None):
        mode_for_tasks = "online"
        if exp == "ALL":
            # Aggregate baseline over all non-offline experiments.
            task_conf_vals = []
            n_pred_sum = 0
            n_total_sum = 0
            found_cov = False
            for exp_name in payload_by_experiment.keys():
                if exp_name.startswith("offline/"):
                    continue
                cval, ccov = _aggregate_happy_from_task_files(exp_name, mode_for_tasks, base_method, variant)
                if cval is not None:
                    task_conf_vals.append(cval)
                if ccov is not None:
                    # Recover weighted merge using task-level n_pred/n_total by recomputing directly.
                    # Approximation with simple mean is avoided by re-reading task files below.
                    pass

                for fp in _task_files_for_experiment(exp_name):
                    try:
                        payload = json.load(open(fp))
                    except Exception:
                        continue
                    happy = payload.get("detection", {}).get("happy", {}).get("variants", {})
                    happy_variant = happy.get(variant, {})
                    happy_metrics = happy_variant.get(f"{mode_for_tasks}_metrics") or {}
                    entry = happy_metrics.get(base_method)
                    if not isinstance(entry, dict):
                        continue
                    n_pred = entry.get("n_pred")
                    n_total = entry.get("n_total")
                    if n_pred is not None and n_total not in (None, 0):
                        n_pred_sum += int(n_pred)
                        n_total_sum += int(n_total)
                        found_cov = True

            if conf_val is None and task_conf_vals:
                conf_val = _mean(task_conf_vals)
            if conf_cov is None and found_cov and n_total_sum > 0:
                conf_cov = 100.0 * n_pred_sum / n_total_sum
        else:
            cval, ccov = _aggregate_happy_from_task_files(exp, mode_for_tasks, base_method, variant)
            if conf_val is None:
                conf_val = cval
            if conf_cov is None:
                conf_cov = ccov

    return hallu_cov, conf_cov, conf_val

results["Hallucinated_Coverage"] = None
results["Conformant_Coverage"] = None
# Also robustly fill Conformant for baselines if missing
for idx, row in results.iterrows():
    hallu_cov, conf_cov, conf_val = extract_coverage_and_conformant(row, files)
    results.at[idx, "Hallucinated_Coverage"] = hallu_cov
    results.at[idx, "Conformant_Coverage"] = conf_cov
    # Only fill Conformant if missing or NaN
    if (row["Conformant"] is None or pd.isna(row["Conformant"])) and conf_val is not None:
        results.at[idx, "Conformant"] = conf_val

# Recompute AVG from finalized values so AVG matches reporting definition.
results["AVG"] = results.apply(lambda r: _mean([r["Hallucinated_AVG"], r["Conformant"]]), axis=1)
results["AVG_sort"] = results["AVG"].fillna(float("inf"))

# Composite ranking score:
# - MAE: normalized using task-grounded asymmetric anchors
#   (hallucinated -> last_turn, conformant -> first_turn)
# - Coverage: raw bounded desirability (hallu higher is better, conformant lower is better)
# Fixed anchors taken from first occurrence of baseline methods (assume all identical).
_last_turn_mask = results["method"].apply(lambda m: _base_method(m) == "last_turn")
_first_turn_mask = results["method"].apply(lambda m: _base_method(m) == "first_turn")

# Take first occurrence of each baseline
last_turn_rows = results.loc[_last_turn_mask, "Hallucinated_AVG"]
first_turn_rows = results.loc[_first_turn_mask, "Conformant"]

h_mae_upper = last_turn_rows.iloc[0] if len(last_turn_rows) > 0 else None
c_mae_upper = first_turn_rows.iloc[0] if len(first_turn_rows) > 0 else None

# Fallbacks if baselines missing
if h_mae_upper is None or pd.isna(h_mae_upper):
    h_mae_upper = results["Hallucinated_AVG"].max()
if c_mae_upper is None or pd.isna(c_mae_upper):
    c_mae_upper = results["Conformant"].max()

def _row_composite(r):
    # Convert each metric to desirability where higher is better.
    d_h_mae_raw = _normalize_with_anchors(r["Hallucinated_AVG"], 0.0, h_mae_upper)
    d_c_mae_raw = _normalize_with_anchors(r["Conformant"], 0.0, c_mae_upper)
    d_h_cov = _clip01((float(r["Hallucinated_Coverage"]) / 100.0) if r["Hallucinated_Coverage"] is not None and not pd.isna(r["Hallucinated_Coverage"]) else None)
    d_c_cov_raw = _clip01((float(r["Conformant_Coverage"]) / 100.0) if r["Conformant_Coverage"] is not None and not pd.isna(r["Conformant_Coverage"]) else None)

    d_h_mae = None if d_h_mae_raw is None else (1.0 - d_h_mae_raw)
    d_c_mae = None if d_c_mae_raw is None else (1.0 - d_c_mae_raw)
    d_c_cov = None if d_c_cov_raw is None else (1.0 - d_c_cov_raw)

    return _weighted_geomean([
        (d_h_mae, W_HALLU_MAE),
        (d_c_mae, W_CONF_MAE),
        (d_h_cov, W_HALLU_COV),
        (d_c_cov, W_CONF_COV),
    ])

results["Score"] = results.apply(_row_composite, axis=1)
results["Score_sort"] = results["Score"].fillna(float("-inf"))
results["Score_100"] = results["Score"].apply(lambda v: (100.0 * float(v)) if v is not None and not pd.isna(v) else None)

# --- Grouping for sectors ---
def sector(row):
    if row["setting"] == "baseline":
        return 0
    if row["setting"] == "online":
        return 1
    if row["setting"] == "offline" and "no-workflow" in row["experiment"]:
        return 2
    if row["setting"] == "offline" and "no-workflow" not in row["experiment"]:
        return 3
    return 4

results["sector"] = results.apply(sector, axis=1)
results = results.sort_values(by=["sector", "Score_sort", "AVG_sort", "Hallu_sort"], ascending=[True, False, True, True])

# Split by sector for manuscript/appendix-oriented reporting
baseline_df = results[results["sector"] == 0].copy()
online_df = results[results["sector"] == 1].copy()
offline_df = results[results["sector"] == 2].copy()
offline_no_wf_df = results[results["sector"] == 3].copy()


def _as_embedding_label(experiment):
    # e.g. embedding_BAAI-bge-m3/lm-6-gram -> embedding_BAAI-bge-m3
    #      dialog2flow-joint-dse-base/lm-6-gram -> dialog2flow-joint-dse-base
    return str(experiment).split("/", 1)[0] if "/" in str(experiment) else str(experiment)


def _as_lm_order(experiment):
    # e.g. embedding_BAAI-bge-m3/lm-6-gram -> 6
    part = str(experiment).split("/", 1)[1] if "/" in str(experiment) else ""
    if part.startswith("lm-") and "-gram" in part:
        try:
            return int(part.replace("lm-", "").replace("-gram", ""))
        except Exception:
            return None
    return None


if not online_df.empty:
    online_df["embedding_model"] = online_df["experiment"].apply(_as_embedding_label)
    online_df["lm_order"] = online_df["experiment"].apply(_as_lm_order)

    # Manuscript selection: the paper's streaming model (--streaming-model), or, if its
    # results are not there, the single best online model by mean Score.
    if args.streaming_model in set(online_df["experiment"]):
        selected_experiments = [args.streaming_model]
    else:
        model_score = (
            online_df.groupby("experiment", as_index=False)["Score"]
            .mean()
            .sort_values("Score", ascending=False)
        )
        selected_experiments = [model_score.iloc[0]["experiment"]]
else:
    selected_experiments = []

# Main manuscript table: baselines + selected best online model + offline summaries.
paper_main_rows = []
if not baseline_df.empty:
    paper_main_rows.append(baseline_df)
if selected_experiments:
    sel_online = online_df[online_df["experiment"].isin(selected_experiments)].copy()
    paper_main_rows.append(sel_online)
if not offline_df.empty:
    paper_main_rows.append(offline_df)
if not offline_no_wf_df.empty:
    paper_main_rows.append(offline_no_wf_df)

paper_main = pd.concat(paper_main_rows, ignore_index=True) if paper_main_rows else pd.DataFrame()
if not paper_main.empty:
    paper_main = paper_main.sort_values(by=["sector", "Score", "AVG", "Hallucinated_AVG"], ascending=[True, False, True, True])


def _offline_llm_from_experiment(experiment_name):
    exp = str(experiment_name)
    if exp.startswith("offline/"):
        exp = exp[len("offline/"):]
    if exp.endswith("-no-workflow"):
        exp = exp[: -len("-no-workflow")]
    return exp


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
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _fmt_num(value, digits=2, missing="--"):
    if value is None or pd.isna(value):
        return missing
    return f"{float(value):.{digits}f}"


def _best_index_set(series, higher_is_better):
    s = series.dropna()
    if s.empty:
        return set()
    best = s.max() if higher_is_better else s.min()
    tol = 1e-12
    return set(s.index[(s - best).abs() <= tol])


def _style_latex_value(text, is_group_best=False, is_global_best=False):
    if is_global_best:
        return r"\underline{\textbf{" + text + "}}"
    if is_group_best:
        return r"\textbf{" + text + "}"
    return text


def _print_main_table_latex(df, selected_experiments):
    print("% Requires: \\usepackage{booktabs,multirow}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\caption{Main turn-prediction results for baselines, selected streaming model, and non-streaming settings. Lower is better for MAE metrics and conformant coverage; higher is better for hallucinated coverage and overall score. Baselines are reference anchors and are not highlighted. Best values within each non-baseline setting are bold; best overall values are bold+underlined and computed over non-baseline methods only.}")
    print(r"\label{tab:tp_main}")
    print(r"\begin{tabular}{lccccccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{4}{c}{Hallucination MAE $\downarrow$} & \multicolumn{2}{c}{Hallucination} & \multicolumn{2}{c}{Conformance} & \\")
    print(r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    print(r"Method & CSC & GR & SSPC & WIC & Avg. MAE $\downarrow$ & Cov. $\uparrow$ & MAE $\downarrow$ & Cov. $\downarrow$ & Score $\uparrow$ \\")
    print(r"\midrule")

    metrics = [
        ("CSC", 2, False),
        ("GR", 2, False),
        ("SSPC", 2, False),
        ("WIC", 2, False),
        ("Hallucinated_AVG", 2, False),
        ("Hallucinated_Coverage", 2, True),
        ("Conformant", 2, False),
        ("Conformant_Coverage", 2, False),
        ("Score_100", SCORE100_DIGITS, True),
    ]

    non_baseline_df = df[df["setting"] != "baseline"]
    global_best = {
        col: _best_index_set(non_baseline_df[col], higher) for col, _, higher in metrics
    }
    setting_best = {}
    for setting in ["baseline", "Streaming", "Non-Streaming (Dialog-Only)", "Non-Streaming + Workflow (Oracle)"]:
        g = df[df["setting"] == setting]
        for col, _, higher in metrics:
            setting_best[(setting, col)] = _best_index_set(g[col], higher)

    sections = [
        ("baseline", "Baselines"),
        ("Streaming", "Streaming"),
        ("Non-Streaming (Dialog-Only)", "Non-Streaming (Dialog-Only)"),
        ("Non-Streaming + Workflow (Oracle)", "Non-Streaming + Workflow (Oracle)"),
    ]

    for setting_key, setting_title in sections:
        g = df[df["setting"] == setting_key]
        if g.empty:
            continue
        section_title = setting_title
        if setting_key == "Streaming" and selected_experiments:
            section_title = f"{setting_title} ({selected_experiments[0]})"
        print(r"\addlinespace[2pt]")
        print(r"\multicolumn{10}{l}{\textbf{" + _latex_escape(section_title) + r"}} \\")
        print(r"\midrule")
        for idx, row in g.iterrows():
            vals = []
            for col, digits, _ in metrics:
                txt = _fmt_num(row[col], digits=digits)
                if setting_key != "baseline":
                    txt = _style_latex_value(
                        txt,
                        is_group_best=(idx in setting_best[(setting_key, col)]),
                        is_global_best=(idx in global_best[col]),
                    )
                vals.append(txt)
            method = _latex_escape(_pretty_method_label(row["method"]))
            print(method + " & " + " & ".join(vals) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    if selected_experiments:
        print("% Selected streaming model: " + _latex_escape(selected_experiments[0]))
    print(r"\end{table*}")


def _print_online_appendix_latex(df):
    def _appendix_method_compact_label(method_label):
        family, variant = _method_family_and_variant(method_label)
        fam_map = {
            "max_jump": "SS",
            "first_below": "TT",
            "first_turn": "FT",
            "last_turn": "LT",
            "random": "RD",
        }
        fam = fam_map.get(family, family)
        return fam if variant == "raw" else f"{fam}+NC"

    print("% Requires: \\usepackage{booktabs,multirow}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\caption{Streaming methods grouped by embedding model (3-gram lm). Compact method labels are used to reduce table width: SS = Surprisal Spike, TT = Improbable Transition, and +NC = node confidence. Best values within each embedding model are bold; best overall values are bold+underlined and computed over non-baseline methods only.}")
    print(r"\label{tab:tp_online_grouped}")
    print(r"\begin{tabular}{llccccccccc}")
    print(r"\toprule")
    print(r"& & \multicolumn{4}{c}{Hallucination MAE $\downarrow$} & \multicolumn{2}{c}{Hallucination} & \multicolumn{2}{c}{Conformance} & \\")
    print(r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    print(r"Embedding & Method & CSC & GR & SSPC & WIC & Avg. MAE $\downarrow$ & Cov. $\uparrow$ & MAE $\downarrow$ & Cov. $\downarrow$ & Score $\uparrow$ \\")
    print(r"\midrule")

    metrics = [
        ("CSC", 2, False),
        ("GR", 2, False),
        ("SSPC", 2, False),
        ("WIC", 2, False),
        ("Hallucinated_AVG", 2, False),
        ("Hallucinated_Coverage", 2, True),
        ("Conformant", 2, False),
        ("Conformant_Coverage", 2, False),
        ("Score_100", SCORE100_DIGITS, True),
    ]

    global_best = {
        col: _best_index_set(df[col], higher) for col, _, higher in metrics
    }
    emb_best = {}
    for emb, g in df.groupby("embedding_model"):
        for col, _, higher in metrics:
            emb_best[(emb, col)] = _best_index_set(g[col], higher)

    for emb, g in df.groupby("embedding_model", sort=False):
        g = g.copy()
        n = len(g)
        emb_first = True
        for idx, row in g.iterrows():
            method = _appendix_method_compact_label(row["method"])
            vals = []
            for col, digits, _ in metrics:
                txt = _fmt_num(row[col], digits=digits)
                txt = _style_latex_value(
                    txt,
                    is_group_best=(idx in emb_best[(emb, col)]),
                    is_global_best=(idx in global_best[col]),
                )
                vals.append(txt)
            emb_cell = (r"\multirow{" + str(n) + r"}{*}{" + _latex_escape(emb) + "}") if emb_first else ""
            emb_first = False
            print(emb_cell + " & " + _latex_escape(method) + " & " + " & ".join(vals) + r" \\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

display_cols = [
    "setting", "method",
    "CSC", "GR", "SSPC", "WIC", "Hallucinated_AVG", "Hallucinated_Coverage",
    "Conformant", "Conformant_Coverage", "Score",
]

paper_main_out = paper_main.copy()
# Main-table presentation cleanup:
# - remove redundant experiment column from output
# - move no-workflow information into setting
# - for offline rows, show LLM name as method
if not paper_main_out.empty:
    _orig_setting = paper_main_out["setting"].astype(str)
    paper_main_out["setting"] = paper_main_out.apply(
        lambda r: "Non-Streaming (Dialog-Only)"
        if str(r.get("setting")) == "offline" and "no-workflow" in str(r.get("experiment"))
        else (
            "Non-Streaming + Workflow (Oracle)"
            if str(r.get("setting")) == "offline"
            else ("Streaming" if str(r.get("setting")) == "online" else str(r.get("setting")))
        ),
        axis=1,
    )
    paper_main_out["method"] = [
        _offline_llm_from_experiment(exp) if setting == "offline" else method
        for exp, setting, method in zip(
            paper_main_out["experiment"],
            _orig_setting,
            paper_main_out["method"],
        )
    ]
    m_disp_rank = paper_main_out["method"].apply(_method_display_and_rank)
    paper_main_out["method"] = m_disp_rank.apply(lambda x: x[0])
    paper_main_out["__method_rank"] = m_disp_rank.apply(lambda x: x[1])
    paper_main_out = paper_main_out.sort_values(
        by=["sector", "__method_rank", "Score", "AVG"],
        ascending=[True, True, False, True],
    )

online_out = online_df.copy()
if not online_out.empty:
    if "lm_order" in online_out.columns:
        online_out = online_out[online_out["lm_order"] == 6].copy()
    if "embedding_model" not in online_out.columns:
        online_out["embedding_model"] = online_out["experiment"].apply(_as_embedding_label)
    m_disp_rank = online_out["method"].apply(_method_display_and_rank)
    online_out["method"] = m_disp_rank.apply(lambda x: x[0])
    online_out["__method_rank"] = m_disp_rank.apply(lambda x: x[1])
    online_out["_embedding_best"] = online_out.groupby("embedding_model")["Score"].transform("max")
    online_out = online_out.sort_values(by=["_embedding_best", "embedding_model", "__method_rank"], ascending=[False, True, True])
    online_out = online_out.drop(columns=["_embedding_best"])

if args.latex:
    if not paper_main_out.empty:
        _print_main_table_latex(paper_main_out, selected_experiments)
    else:
        print("% No rows available for manuscript table.")

    print()
    if not online_out.empty:
        _print_online_appendix_latex(online_out)
    else:
        print("% No online rows found.")
else:
    for col in ["CSC", "GR", "SSPC", "WIC", "Hallucinated_AVG", "Conformant", "AVG", "Hallucinated_Coverage", "Conformant_Coverage"]:
        if col in paper_main_out.columns:
            paper_main_out[col] = paper_main_out[col].apply(_fmt)
    if "Score_100" in paper_main_out.columns:
        paper_main_out["Score"] = paper_main_out["Score_100"].apply(lambda v: _fmt(v, digits=SCORE100_DIGITS))

    print("### Turn Prediction: Main Manuscript Table\n")
    if selected_experiments:
        print("*Main paper: baselines, selected best streaming model, and non-streaming methods.*\n")
        print(f"*Selected streaming model: {selected_experiments[0]}*\n")
    else:
        print("*Main paper: baselines and non-streaming methods (no streaming rows available).*\n")
    if not paper_main_out.empty:
        print(paper_main_out[display_cols].to_markdown(index=False))
    else:
        print("No rows available for manuscript table.")

    print("\n\n<details><summary>Appendix: Streaming Methods Grouped by Embedding Model</summary>\n")
    if not online_out.empty:
        for col in ["CSC", "GR", "SSPC", "WIC", "Hallucinated_AVG", "Conformant", "AVG", "Hallucinated_Coverage", "Conformant_Coverage"]:
            online_out[col] = online_out[col].apply(_fmt)
        online_out["Score"] = online_out["Score_100"].apply(lambda v: _fmt(v, digits=SCORE100_DIGITS))
        appendix_online_cols = [
            "embedding_model", "method",
            "CSC", "GR", "SSPC", "WIC", "Hallucinated_AVG", "Hallucinated_Coverage",
            "Conformant", "Conformant_Coverage", "Score",
        ]
        print(online_out[appendix_online_cols].to_markdown(index=False))
    else:
        print("No online rows found.")
    print("\n</details>")
