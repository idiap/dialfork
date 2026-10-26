# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

import os
import json
import argparse
import random
import numpy as np
import pandas as pd

from sdialog import Dialog
from sdialog.datasets import STAR
from sdialog.evaluation import DatasetComparator
from sdialog.evaluation import FrechetBERTDistanceEvaluator
from sdialog.evaluation import DialogFlowPPL
from sdialog.evaluation import KDEDistanceEvaluator, PrecisionRecallDistanceEvaluator
from sdialog.evaluation.base import BaseDatasetEvaluator

from sklearn.metrics import accuracy_score

# import sdialog
# sdialog.config.cache(True)

STAR.set_path("datasets/STAR/")
PATH_DIALFORK = "datasets/dialfork"
PATH_OUTPUT = "results/task1"
STAR_AS_REFERENCE = False
D2F_KWARGS = {
    "nodes_prune_threshold": 0.001,
    "edges_prune_threshold": 0.001,
}

def load_task_hallucinated_dialogs(task_name, hallucination_type:str = None, path_dataset: str = PATH_DIALFORK):
    """Load hallucinated dialogs for a specific task."""
    path_output = os.path.join(path_dataset, task_name)

    output = {
        "original": Dialog.from_folder(os.path.join(path_output, "originals")),
        "hallucinations": {}
    }

    output["original_star"] = [STAR.get_dialog(d.id) for d in output["original"]]

    if hallucination_type is None:
        # Load all hallucination types
        hallucination_types = os.listdir(path_output)
    else:
        # Load only the specified hallucination type
        hallucination_types = [hallucination_type]

    for ht in hallucination_types:
        if ht in ["originals", "dialogue_state_reset"] or os.path.isfile(os.path.join(path_output, ht)):
            continue
        # if len(hallucination_types) > 1:
        output["hallucinations"][ht] = {}
        hallucinated = output["hallucinations"][ht]
        # else:
        #     hallucinated = output["hallucinations"]

        if os.path.exists(os.path.join(path_output, ht, "n_hallucinations_4")):
            path_dialogs = os.path.join(path_output, ht, "n_hallucinations_4")
        else:
            path_dialogs = os.path.join(path_output, ht)

        for fork_segment in os.listdir(path_dialogs):
            hallucinated[fork_segment] = Dialog.from_folder(os.path.join(path_dialogs, fork_segment))

    return output


class MeanLengthDiffEvaluator(BaseDatasetEvaluator):
    """Naive length-only control: |mean dialog length of candidate set - mean dialog
    length of the reference (original) set|, in turns or words. No embeddings involved.

    Used as a construction-artifact control for Task 1 (§4.1): since DialFork forks
    share an exact prefix with the original up to the fork point, this checks whether a
    trivial length-difference heuristic alone can recover the fork-position ordering
    that FBD/PRD/FlowKL are evaluated against.
    """
    def __init__(self, reference_dialogs, unit="turns", name=None):
        if unit not in {"turns", "words"}:
            raise ValueError(f"Invalid unit: {unit}. Must be 'turns' or 'words'.")
        self.unit = unit
        self.name = name or f"MeanLenDiff-{unit}"
        self.ref_mean = self._mean_length(reference_dialogs)

    def _mean_length(self, dialogs):
        if self.unit == "turns":
            lengths = [len(d.turns) for d in dialogs]
        else:
            lengths = [
                sum(len(t.text.split()) for t in d.turns if t.text)
                for d in dialogs
            ]
        return float(np.mean(lengths)) if lengths else 0.0

    def __call__(self, dialogues, dataset_name=None, **kwargs):
        return abs(self._mean_length(dialogues) - self.ref_mean)


def evaluator_already_computed(ev, computed_evaluators):
    if ev.name in computed_evaluators:
        return True
    elif hasattr(ev, "metric"):
        if f"{ev.name}-{ev.metric}" in computed_evaluators:
            return True
        elif f"{ev.name}-kl" in computed_evaluators:
            return True
        elif f"{ev.name}-cs" in computed_evaluators:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dialog flow metrics on hallucinated dialogs")
    parser.add_argument("task", help="Task name (e.g., 'apartment_schedule')", default="apartment_schedule", type=str.lower)
    parser.add_argument("node_thresholds", nargs="?", type=float, default=0.01)
    parser.add_argument("-i", "--input-path", type=str, default=PATH_DIALFORK,
                        help="Path to hallucinated STAR dataset")
    parser.add_argument("-o", "--output-path", type=str, default=PATH_OUTPUT,
                        help="Base output path for experiment results")
    parser.add_argument("-m", "--node-embedding-model", type=str, default="sergioburdisso/dialog2flow-joint-dse-base", help="Node embedding model for dialog2flow")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--baselines-only", action="store_true",
                        help=("Only compute the naive length-only baseline controls (MeanLenDiff-turns/-words). "
                              "Skips FBD/PRD/FlowKL entirely, so no dialog2flow graph or BERT embeddings are "
                              "built and nothing needs a GPU. Results are written to the same "
                              "results_all.json/results_acc.json as a full run (under the same "
                              "--node-embedding-model folder), so a later full run just fills in the "
                              "remaining FBD/PRD/FlowKL columns instead of recomputing the baselines."))

    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed
        set_seed(args.seed)
    except ImportError:
        pass

    print(f"Random seed set to: {args.seed}")

    print(f"Loading hallucinated dialogs for task: {args.task}")

    if args.baselines_only:
        print("--baselines-only: skipping FBD/PRD/FlowKL (no graph/embeddings will be built).")
    else:
        print(f"Clustering node threshold: {args.node_thresholds}")
        D2F_KWARGS["node_thresholds"] = args.node_thresholds
        print(f"Using node embedding model: {args.node_embedding_model}")
        D2F_KWARGS["node_embedding_model"] = args.node_embedding_model

    # Extract model name from path (e.g., "path/to/model" -> "model"); used for the output
    # folder even in --baselines-only mode so results merge with a later full run of the
    # same --node-embedding-model.
    model_name = args.node_embedding_model.split("/")[-1]

    output_path = os.path.join(args.output_path, model_name, args.task)
    os.makedirs(output_path, exist_ok=True)

    dialogs = load_task_hallucinated_dialogs(args.task, path_dataset=args.input_path)

    # Define ground truth rank based on hallucination type
    # Get candidate keys from first hallucination type
    first_ht = next(iter(dialogs["hallucinations"].keys()))
    candidate_keys = list(dialogs["hallucinations"][first_ht].keys())

    # Map known hallucination types to their ground truth ranking
    if all(key.startswith("fork_at_segment_") for key in candidate_keys):
        ground_truth_rank = ["fork_at_segment_4", "fork_at_segment_3", "fork_at_segment_2", "fork_at_segment_1"]
    elif "happy" in candidate_keys:
        # For sentiment-based hallucinations
        ground_truth_rank = ["happy", "unhappy", "multitask"]
    else:
        # Fallback: assume fork segments with unknown count or use first available order
        ground_truth_rank = sorted(candidate_keys, key=lambda x: int(x.split("_")[-1]) if x.startswith("fork_at_segment_") else 0, reverse=True)

    if STAR_AS_REFERENCE:
        reference_dialogs = dialogs["original_star"]
        ground_truth_rank = ["no_forks"] + ground_truth_rank
    else:
        reference_dialogs = dialogs["original"]

    evaluators = [
        MeanLengthDiffEvaluator(reference_dialogs, unit="turns"),
        MeanLengthDiffEvaluator(reference_dialogs, unit="words"),
    ]

    if not args.baselines_only:
        graph_folder = "graph"
        score_name = "FlowKL"

        print(f"\nBuilding {score_name}...")

        from sdialog.evaluation import DialogFlowScore

        d2f = DialogFlowScore(  # Already lunched the ones with ai_speaker="System", NEED TO TRY WITHOUT IT AND DialogFlowScore instead of PPL.
        # d2f = DialogFlowPPL(
            reference_dialogs,
            name=score_name,
            # ai_speaker="System",
            use_softmax=False,
            use_only_known_edges=True,
            # use_only_known_edges=False,
            graph=None, nodes=None,
            output_path=os.path.join(output_path, graph_folder),
            out_png=True,
            **D2F_KWARGS
        )

        evaluators.append(KDEDistanceEvaluator(d2f, metric="kl"))

        evaluators += [
            FrechetBERTDistanceEvaluator(reference_dialogs,
                                         enable_plotting=False,
                                         name="FBD"),
            PrecisionRecallDistanceEvaluator(reference_dialogs, name="PRD"),
        ]

    comparator = DatasetComparator(
        evaluators=evaluators
    )
    original_evaluators = comparator.evaluators

    if os.path.exists(os.path.join(output_path, f"results_all.json")):
        print("Loading existing results...")
        with open(os.path.join(output_path, f"results_all.json"), "r") as f:
            results = json.load(f)
    else:
        print("No existing results found. Evaluating from scratch...")
        results = {}

    hallucination_types = list(dialogs["hallucinations"].keys())
    for ht in hallucination_types:
        comparator.evaluators = original_evaluators
        if STAR_AS_REFERENCE:
            candidates_dialogs = dialogs["hallucinations"][ht].copy()
            candidates_dialogs["no_forks"] = dialogs["original"]
        else:
            candidates_dialogs = dialogs["hallucinations"][ht]

        print(f"Evaluating hallucination type: {ht.upper()}")
        if ht in results:
            eval_already_done = list(results[ht].get(next(iter(candidates_dialogs.keys())), {}).keys())
            # Filter out metrics that are None in ANY dataset
            eval_already_done = [k for k in eval_already_done
                                if all(results[ht][dataset].get(k) is not None
                                      for dataset in results[ht])]
            comparator.evaluators = [ev for ev in original_evaluators
                                     if not evaluator_already_computed(ev, eval_already_done)]

        if len(comparator.evaluators) > 0:
            new_results = comparator(candidates_dialogs)

            for dataset in new_results:
                new_results[dataset] = {k: 1 - v if "similarity" in k else v
                                        for k, v in new_results[dataset].items()}

            if ht not in results:
                results[ht] = new_results
            else:
                for dataset in new_results:
                    results[ht][dataset].update(new_results[dataset])

            with open(os.path.join(output_path, f"results_all.json"), "w") as f:
                json.dump(results, f, indent=4)
            try:
                comparator.plot(save_folder_path=os.path.join(output_path, "plots", ht))
            except Exception as e:
                print(f"Error plotting results for hallucination type {ht}: {e}")
        else:
            print(f"All evaluations already computed for hallucination type: {ht.upper()}")

    metrics = list(results[ht][list(results[ht].keys())[0]].keys())
    results_rank = {}
    for ht in hallucination_types:
        results_rank[ht] = {}
        for metric in metrics:
            results_rank[ht][metric] = [dataset for dataset, _ in sorted([(dataset, results[ht][dataset][metric]) for dataset in results[ht]], key=lambda x: x[1])]

    results_accs = {}
    for ht in results:
        # Use reversed ground truth rank for "reset" hallucination type
        current_ground_truth = list(reversed(ground_truth_rank)) if "reset" in ht.lower()  else ground_truth_rank
        results_accs[ht] = {}
        for metric in metrics:
            results_accs[ht][metric] = accuracy_score(current_ground_truth, results_rank[ht][metric])

    df_accs = pd.DataFrame(results_accs)
    df_accs = df_accs.T

    print("Rank accuracy results (rows: hallucination types, columns: metrics):")
    print(df_accs.to_markdown(floatfmt=".3f"))

    df_accs_avg = df_accs.mean(0)

    print("\nAverage rank accuracy results (across hallucination types):")
    print(df_accs_avg.to_markdown(floatfmt=".3f"))

    with open(os.path.join(output_path, f"results_acc.json"), "w") as f:
        json.dump(df_accs.to_dict(), f)

    with open(os.path.join(output_path, f"results_acc.txt"), "w") as f:
        f.write(df_accs.to_markdown(floatfmt=".3f"))

    with open(os.path.join(output_path, f"results_acc_avg.json"), "w") as f:
        json.dump(df_accs_avg.to_dict(), f)

    with open(os.path.join(output_path, f"results_acc_avg.txt"), "w") as f:
        f.write(df_accs_avg.to_markdown(floatfmt=".3f"))
