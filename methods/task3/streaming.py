# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""
Node-sequence n-gram language model over dialog flow graphs.

Given a task-root folder, this script:
    1. Loads original dialogs from originals/.
    2. Builds a dialog flow graph from those original dialogs.
    3. Converts each dialog into a sequence of graph nodes via
         BaseDialogFlowScore.get_node_sequence().
    4. Trains an n-gram language model over the resulting node sequences.
    5. Evaluates off-road turn prediction separately for each hallucination type
         folder against annotations["fork_assistant_turn"].

Expected task-root layout
-------------------------
    task_root/
        originals/
        conflicting_system_constraints/
        goal_replacement/
        step_selection_policy_corruption/
        workflow_identity_corruption/

Hallucination folders may contain nested subfolders; all dialogs found under
each hallucination type are loaded recursively.

Usage example
-------------
    python methods/task3/streaming.py --task-root path/to/task_root --output results.json

================================================================================
BUILT-IN OFF-ROAD TURN PREDICTION METHODS
================================================================================

All methods share the same evaluation protocol: given a dialog, predict the
index of the first "off-road" turn — the earliest turn at which the dialog
deviates from the expected flow encoded in the n-gram language model.

Two principal evaluation variants are defined for each method:
  - raw:    Turn probability is the raw n-gram LM score p(node_t | node_{t-n+1}...t-1).
  - smooth: Turn probability is the raw score multiplied by prob_correct_node,
            the soft node-assignment confidence from the graph embedding lookup
            (softmax of 1 - cosine distances to the k nearest graph nodes).
            This down-weights turns whose utterance was ambiguously mapped to a
            graph node, reducing false positives caused by noisy node assignment.

Thresholds for both methods are calibrated on the original (training) dialogs
in an unsupervised way, requiring no hallucination labels:
  - offroad_threshold: the OFFROAD_QUANTILE (default 5th) percentile of per-dialog
    minimum turn probabilities over original dialogs.  Chosen so that only ~5% of
    training dialogs would trigger a false off-road alert.
  - jump_threshold: the JUMP_QUANTILE (default 95th) percentile of per-dialog
    maximum surprisal jumps over original dialogs.  Chosen so that ~5% of training
    dialogs would produce a spurious large jump.

------------------------------------------------------------
METHOD: first_below  (alias in papers: "Improbable Transition" / IT)
------------------------------------------------------------
  Intuition: An off-road turn is one whose n-gram transition probability falls
  below a calibrated floor — i.e., the LM considers the observed node sequence
  locally improbable.

  Algorithm:
    1. Compute per-turn probabilities p_t = P_LM(node_t | context) for t = 1..T.
    2. Scan left-to-right; return the index of the first t where p_t < offroad_threshold.
    3. If no turn crosses the threshold, return None (no off-road event predicted).

  Threshold calibration (training):
    offroad_threshold = quantile(min_t p_t over original dialogs, q=OFFROAD_QUANTILE).

  Characteristics:
    - Detects the *first* locally anomalous transition.
    - Sensitive to single rare node tokens (n-gram sparsity may produce very low
      probabilities even for legitimate transitions; mitigated by KneserNey smoothing
      and by the "smooth" variant which down-weights uncertain node assignments).
    - Expected to over-predict early when the LM vocabulary is sparse.

------------------------------------------------------------
METHOD: max_jump  (alias in papers: "Surprisal Spike" / SS)
------------------------------------------------------------
  Intuition: A hallucination does not merely produce a low-probability turn; it
  produces a *sudden increase* in surprisal — a spike relative to the preceding
  turns.  This makes the method more robust to dialogs that are globally unusual
  but locally consistent.

  Algorithm:
    1. Compute per-turn surprisals s_t = -log(p_t) for t = 1..T.
    2. Compute consecutive surprisal differences (jumps): Δ_t = s_t - s_{t-1} for t = 2..T.
    3. Scan left-to-right; return the index of the first t where Δ_t > jump_threshold.
    4. If no jump exceeds the threshold, return turn index 1 (the first turn) as a
       conservative fallback (the model must predict *something*).

  Threshold calibration (training):
    jump_threshold = quantile(max_t Δ_t over original dialogs, q=JUMP_QUANTILE).

  Characteristics:
    - Detects the first *relative* anomaly rather than an absolute low-probability
      event, making it more robust to globally low-probability dialogs.
    - The surprisal-jump formulation is analogous to change-point detection in
      time-series literature.
    - Less sensitive to cold-start effects (early turns with sparse context).

------------------------------------------------------------
BASELINE METHODS (sanity checks)
------------------------------------------------------------
  first_turn  — Always predicts turn 1.  Lower bound for early detection.
  last_turn   — Always predicts the final turn.  Upper bound / trivial late detector.
  random      — Predicts a uniformly random turn index.  Expected-value baseline.

================================================================================
HOW TO ADD YOUR OWN OFF-ROAD TURN PREDICTION METHOD
================================================================================

This script provides a clean extension point for researchers to add custom
off-road turn prediction methods. You only need to implement two functions:

  1. train_fn(dialogs) -> model
     Input: List of original (training) dialogs
     Output: A model (can be anything you want, it will be passed to your predict_fn as-is, see below)
     Task: Train/calibrate your method on the original dialogs

  2. predict_fn(model, dialog) -> index or None
     Input: Your model and a single dialog to evaluate
     Output: The predicted turn index (int) or None if no prediction
     Task: Use your model to predict the off-road turn for this dialog

Example custom method:

    def my_train(dialogs):
        '''Train your method on original dialogs.'''
        # Example: compute some statistics
        dialog_lengths = [len(d) for d in dialogs]
        return {
            'mean_length': sum(dialog_lengths) / len(dialog_lengths),
        }

    def my_predict(model, dialog):
        '''Predict turn index for a dialog using your trained model.'''
        # Example: always predict the turn at position mean_length
        return int(model['mean_length'])

Then, edit the built-in train_fn/predict_fn implementations below to include
your method, or pass them when running the main task.

================================================================================
"""
import os
import sys
import json
import time
import argparse
import random
import gc
import shutil
import subprocess
import tempfile
from datetime import timedelta
from typing import List, Tuple, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Some encoder models need a newer `transformers` than the one pinned inside
# the apptainer image (currently 4.51.3). Detected so far:
#
#   - google/embeddinggemma-300m sets `use_bidirectional_attention=True` in
#     its config, which transformers only started honoring in 4.56.2. On
#     4.51.3, sentence-transformers logs a warning and the model silently
#     falls back to causal (left-to-right) attention, which produces poorly
#     separated embeddings for every turn. Downstream, dialog2flow's
#     turn-clustering step (fixed cosine distance_threshold) then merges
#     almost everything into a handful of nodes (observed: 4 nodes instead
#     of ~30-60), the n-gram LM never sees an "improbable" transition, and
#     first_below/max_jump never fire (0% prediction coverage -> Conformant
#     eMAE is undefined/n/a).
#
# We can't touch the container image or sdialog, so instead of upgrading
# transformers globally (which would silently change embeddings/results for
# every other encoder too), pip-install the newer version into an isolated,
# on-disk target directory and prepend it to sys.path *before* transformers
# is imported anywhere (by sdialog or otherwise) - but only when the
# requested --model actually needs it, so every other model keeps running
# against the exact pinned version its existing results were computed with.
# ---------------------------------------------------------------------------
_MODELS_NEEDING_NEWER_TRANSFORMERS = {
    "google/embeddinggemma-300m": "4.56.2",
}


def _version_tuple(v):
    parts = []
    for chunk in v.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _requested_model_from_argv():
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok in ("-m", "--model") and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--model="):
            return tok.split("=", 1)[1]
    return None


def _ensure_transformers_version_for_model():
    model_name = _requested_model_from_argv()
    min_version = _MODELS_NEEDING_NEWER_TRANSFORMERS.get(model_name)
    if not min_version:
        return

    import importlib

    try:
        installed_version = importlib.import_module("transformers").__version__
    except ImportError:
        installed_version = "0"
    if _version_tuple(installed_version) >= _version_tuple(min_version):
        return

    target_dir = os.path.join(os.getcwd(), ".task3_deps", f"transformers-{min_version}")
    marker = os.path.join(target_dir, ".install_ok")
    if not os.path.exists(marker):
        print(f"[methods/task3/streaming.py] {model_name} needs transformers>={min_version} "
              f"(found {installed_version}); installing an isolated copy into "
              f"{target_dir} (does not affect other models/runs)...", file=sys.stderr)
        os.makedirs(target_dir, exist_ok=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--target", target_dir, f"transformers>={min_version},<5",
        ])
        with open(marker, "w") as f:
            f.write(min_version)

    sys.path.insert(0, target_dir)
    # Drop any transformers submodules a prior (pinned-version) import may have cached.
    for mod_name in list(sys.modules):
        if mod_name == "transformers" or mod_name.startswith("transformers."):
            del sys.modules[mod_name]


_ensure_transformers_version_for_model()

# ---------------------------------------------------------------------------
# Per-model encode-time fix: Qwen/Qwen3-Embedding-8B is a causal decoder-only
# embedding model that only ships one named prompt ("query", per its model
# card - meant to be applied to both sides for symmetric similarity/
# clustering use, not just retrieval queries). Called with no prompt (as
# sdialog's node-clustering step and our own online encoding below both do
# by default), its raw embeddings are severely anisotropic: measured on real
# DialFork turns, EVERY pair of distinct system turns had cosine distance
# below sdialog's clustering threshold (0.55), vs ~33% for a normal sentence
# encoder (LaBSE) on the same turns - so every turn collapses into one flow
# node (observed: 2 nodes total) and first_below/max_jump never fire (0%
# coverage). Applying prompt_name="query" brings the distribution back in
# line with LaBSE's (~31% of pairs under threshold). Gated by --model so
# every other encoder's embeddings/results are unaffected.
# ---------------------------------------------------------------------------
_MODEL_ENCODE_PROMPT_NAME = {
    "Qwen/Qwen3-Embedding-8B": "query",
}

_encode_prompt_name_override = _MODEL_ENCODE_PROMPT_NAME.get(_requested_model_from_argv())
if _encode_prompt_name_override:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _orig_st_encode = _SentenceTransformer.encode

    def _patched_st_encode(self, *st_args, **st_kwargs):
        if st_kwargs.get("prompt_name") is None and st_kwargs.get("prompt") is None:
            st_kwargs["prompt_name"] = _encode_prompt_name_override
        return _orig_st_encode(self, *st_args, **st_kwargs)

    _SentenceTransformer.encode = _patched_st_encode

from sdialog import Dialog
from sdialog.evaluation import DialogFlowPPL          # concrete BaseDialogFlowScore subclass
from sdialog.evaluation.base import DEFAULT_TOKEN_START   # "<START>" sentinel already used by the graph

# ---------------------------------------------------------------------------
# NLTK n-gram LM helpers (nltk.lm)
# ---------------------------------------------------------------------------
import nltk
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_sequence

# ---------------------------------------------------------------------------
# Per-model load-time fixes, applied regardless of how --model reaches
# transformers.PreTrainedModel.from_pretrained (sentence-transformers loads
# node_embedding_model through a few different internal paths):
#
#   - aws-ai/dse-bert-base predates safetensors and only ships
#     pytorch_model.bin, with no modules.json. Since it isn't a native
#     sentence-transformers checkpoint, sentence-transformers loads it
#     through its bare-encoder fallback, which forces use_safetensors=True
#     and makes transformers hard-fail instead of falling back to .bin.
#
#   - Qwen/Qwen3-Embedding-8B has no torch_dtype requested, so transformers
#     defaults to float32 (~32GB for 8B params), which OOMs on a 32GB V100
#     (and on a 24GB RTX 3090) once activations are added. Forcing float16
#     (safe on both V100/Volta and RTX3090/Ampere, unlike bfloat16 which
#     Volta lacks native support for) halves that to ~16GB.
# ---------------------------------------------------------------------------
import transformers as _transformers

_MODEL_LOAD_OVERRIDES = {
    "aws-ai/dse-bert-base": {"use_safetensors": False},
    "Qwen/Qwen3-Embedding-8B": {"torch_dtype": torch.float16} if torch else {},
}

_orig_from_pretrained = _transformers.PreTrainedModel.from_pretrained.__func__


@classmethod
def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    overrides = _MODEL_LOAD_OVERRIDES.get(str(pretrained_model_name_or_path))
    if overrides:
        kwargs.update(overrides)
    return _orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

_transformers.PreTrainedModel.from_pretrained = _patched_from_pretrained

# Special boundary tokens (mirroring dialog2flow conventions)
BOS = DEFAULT_TOKEN_START   # beginning-of-sequence
EOS = "<END>"               # end-of-sequence

# Internal defaults for automatic threshold calibration
LM_MODEL_TYPE = "kneser_ney"
LM_FALLBACK_CHAIN = ["kneser_ney", "laplace", "mle"]  # Ordered fallback chain for LM model types
OFFROAD_QUANTILE = 0.05
JUMP_QUANTILE = 0.95
NODE_PRUNE_THRESHOLD = 0.001
EDGE_PRUNE_THRESHOLD = 0.001
ORIGINALS_SUBDIR = "originals"
HALLUCINATION_TYPES = [
    "conflicting_system_constraints",
    "goal_replacement",
    "step_selection_policy_corruption",
    "workflow_identity_corruption",
]
METHOD_ABBREV = {
    "first_below": "fb",
    "max_jump":    "mj",
    "first_turn":  "ft",
    "last_turn":   "lt",
    "random":      "rd",
}
BASELINE_METHODS = {"first_turn", "last_turn", "random"}
HT_ABBREV = {
    "conflicting_system_constraints":    "csc",
    "goal_replacement":  "gdr",
    "step_selection_policy_corruption":   "spi",
    "workflow_identity_corruption":      "wic",
}
# Methods are read from these modes when rendering summary tables.
METHOD_MODE = {
    "first_below": "online",
    "max_jump":    "online",
    "first_turn":  "online",
    "last_turn":   "online",
    "random":      "online",
}

# Set to True in main() when --verbose is passed
_verbose = False


# ============================================================================
# SECTION 0: HELPER FUNCTIONS FOR DIALOG PROCESSING AND METRICS
# ============================================================================

def _softmax(values: List[float]) -> np.ndarray:
    """Numerically stable softmax for a 1D list."""
    if not values:
        return np.array([], dtype=float)
    arr = np.array(values, dtype=float)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    denom = np.sum(exp_arr)
    if denom <= 0:
        return np.zeros_like(exp_arr)
    return exp_arr / denom


def train_nltk_lm(sequences: List[List[str]], n: int, model_type: str):
    """Train an NLTK n-gram LM. Returns the fitted model."""
    train_data, padded_vocab = padded_everygram_pipeline(n, sequences)

    if model_type == "mle":
        lm = MLE(n)
    elif model_type == "laplace":
        lm = Laplace(n)
    elif model_type == "kneser_ney":
        lm = KneserNeyInterpolated(n)
    else:
        raise ValueError(f"Unknown NLTK model type: {model_type}")

    lm.fit(train_data, padded_vocab)
    return lm


def nltk_perplexity(lm, sequences: List[List[str]], n: int) -> float:
    """Compute average per-sequence perplexity using an NLTK lm."""
    ppls = []
    for seq in sequences:
        test_ngrams = list(
            pad_sequence(seq, n,
                         pad_left=True, pad_right=True,
                         left_pad_symbol=BOS, right_pad_symbol=EOS)
        )
        test_ngrams = list(nltk.ngrams(test_ngrams, n))
        try:
            ppl = lm.perplexity(test_ngrams)
            if np.isfinite(ppl):
                ppls.append(ppl)
        except Exception:
            pass
    return float(np.mean(ppls)) if ppls else float("inf")


def train_nltk_lm_with_fallback(sequences: List[List[str]], n: int, model_type: str):
    """
    Train an NLTK n-gram LM, falling back to simpler models if the requested
    type fails (e.g. KneserNey on short/sparse sequences with large n).
    Returns (fitted_model, model_type_used).
    """
    # Filter sequences too short for this n-gram order
    valid_sequences = [s for s in sequences if len(s) >= n]
    if not valid_sequences:
        raise ValueError(
            f"No sequences long enough to train a {n}-gram model. "
            f"Longest sequence has {max(len(s) for s in sequences)} tokens."
        )
    if len(valid_sequences) < len(sequences):
        print(
            f"  [WARN] Dropped {len(sequences) - len(valid_sequences)}/{len(sequences)} "
            f"sequences shorter than n={n}.",
            file=sys.stderr,
        )

    # Build fallback chain starting from the requested model type
    chain = [model_type] + [m for m in LM_FALLBACK_CHAIN if m != model_type]

    last_exc = None
    for candidate in chain:
        try:
            lm = train_nltk_lm(valid_sequences, n=n, model_type=candidate)
            if candidate != model_type:
                print(
                    f"  [WARN] Requested LM type '{model_type}' failed for n={n}. "
                    f"Automatically switched to '{candidate}'.",
                    file=sys.stderr,
                )
            return lm, candidate
        except Exception as exc:
            print(
                f"  [WARN] LM type '{candidate}' failed for n={n}: {exc}. "
                f"Trying next fallback...",
                file=sys.stderr,
            )
            last_exc = exc

    raise RuntimeError(
        f"All LM model types {chain} failed for n={n}. Last error: {last_exc}"
    )


def dialogs_to_node_sequences(
    dialogs: List[Dialog],
    flow_score: DialogFlowPPL,
    add_boundaries: bool = True,
) -> List[List[str]]:
    """
    Convert a list of Dialog objects to lists of node-ID strings.

    :param dialogs: Dialogs to convert.
    :param flow_score: Fitted BaseDialogFlowScore instance.
    :param add_boundaries: If True, prepend BOS and append EOS.
    :return: List of node-sequence lists.
    """
    sequences = []
    for dialog in dialogs:
        try:
            node_seq = flow_score.get_node_sequence(dialog)
            if add_boundaries:
                node_seq = [BOS] + list(node_seq) + [EOS]
            sequences.append(node_seq)
        except ValueError as exc:
            print(f"  [WARN] Skipping dialog {dialog.id}: {exc}", file=sys.stderr)
    return sequences


def get_effective_turn_indices(dialog: Dialog, flow_score: DialogFlowPPL) -> List[int]:
    """
    Return original 1-based turn indices used by get_node_sequence().

    If only AI/system turns are enabled, this returns only those turn indices.
    """
    if flow_score.only_system:
        ai_speaker = flow_score.ai_speaker.lower()
        return [idx + 1 for idx, turn in enumerate(dialog) if turn.speaker.lower() == ai_speaker]
    return list(range(1, len(dialog) + 1))


def get_node_sequence_and_correct_node_probs(
    dialog: Dialog,
    flow_score: DialogFlowPPL,
    batch_size: int = 16,
) -> Tuple[List[str], List[float]]:
    """
    Reconstruct node sequence plus per-turn node assignment confidence.

    Mirrors BaseDialogFlowScore.get_node_sequence() behavior for prob_correct_node:
      prob_correct_node = softmax(1 - distance_to_neighbors)[0] (or 1.0 in hard mode).
    """
    node_sequence = []
    prob_correct_sequence = []

    if flow_score.only_system:
        dialog = [turn for turn in dialog if turn.speaker.lower() == flow_score.ai_speaker.lower()]

    if not dialog:
        return node_sequence, prob_correct_sequence

    normalized_speakers: List[str] = []
    texts: List[str] = []
    for turn in dialog:
        speaker = turn.speaker.lower()
        if speaker in flow_score.speakers:
            normalized_speakers.append(flow_score.speakers[speaker])
            texts.append(turn.text)
        else:
            raise ValueError(
                f"WARNING: speaker '{turn.speaker}' not found in the graph metadata, expected one of "
                f"{list(flow_score.speakers.keys())}"
            )

    # Encode all turns for this dialog in one call so batch_size is effective.
    utt_embs = encode_with_adaptive_batch(texts, flow_score, batch_size=batch_size)

    # Group embeddings by speaker for batch KNN lookup
    # Map: speaker -> (turn_indices, embeddings)
    speaker_to_indices_and_embs = {}
    for turn_idx, (speaker, utt_emb) in enumerate(zip(normalized_speakers, utt_embs)):
        if speaker not in speaker_to_indices_and_embs:
            speaker_to_indices_and_embs[speaker] = ([], [])
        speaker_to_indices_and_embs[speaker][0].append(turn_idx)
        speaker_to_indices_and_embs[speaker][1].append(utt_emb)

    # Query neighbors for each speaker's embeddings in batch
    # Pre-allocate results arrays
    node_sequence = [None] * len(normalized_speakers)
    prob_correct_sequence = [None] * len(normalized_speakers)

    for speaker, (turn_indices, speaker_embs) in speaker_to_indices_and_embs.items():
        # Convert to numpy array for batch processing
        speaker_embs_array = np.array(speaker_embs)

        # Get neighbors for all turns of this speaker at once
        all_neighbors = flow_score.knn_models[speaker](
            speaker_embs_array,
            k=None if flow_score.use_softmax else 1
        )

        # Scatter results back to correct positions
        for batch_idx, turn_idx in enumerate(turn_indices):
            neighbors = all_neighbors[batch_idx]
            current_node, _ = neighbors[0]
            node_sequence[turn_idx] = current_node

            if flow_score.use_softmax:
                probs = _softmax([1 - dist for _, dist in neighbors])
                prob_correct = float(probs[0]) if len(probs) else 0.0
            else:
                prob_correct = 1.0
            prob_correct_sequence[turn_idx] = prob_correct

    return node_sequence, prob_correct_sequence


def release_cuda_memory() -> None:
    """Best-effort GPU memory cleanup between dialogs."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom_error(exc: Exception) -> bool:
    """Return True when an exception is a CUDA out-of-memory failure."""
    if torch is not None and isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "outofmemory" in msg


class DialogEncodingSkipped(RuntimeError):
    """Raised when a dialog cannot be encoded after adaptive CUDA OOM backoff."""


def encode_with_adaptive_batch(texts: List[str], flow_score: DialogFlowPPL, batch_size: int):
    """Encode texts with adaptive batch-size backoff on CUDA OOM."""
    current_bs = max(1, int(batch_size))
    while True:
        try:
            return flow_score.encoder.encode(
                texts,
                show_progress_bar=False,
                batch_size=current_bs,
            )
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise

            if current_bs > 1:
                next_bs = max(1, current_bs // 2)
                print(
                    f"  [WARN] CUDA OOM during encoding (batch_size={current_bs}). "
                    f"Retrying with batch_size={next_bs}...",
                    file=sys.stderr,
                )
                release_cuda_memory()
                current_bs = next_bs
                continue

            print(
                "  [WARN] CUDA OOM during encoding at batch_size=1. "
                "Skipping this dialog.",
                file=sys.stderr,
            )
            release_cuda_memory()
            raise DialogEncodingSkipped(
                "Dialog skipped after CUDA OOM at batch_size=1"
            )


def per_turn_transition_probs(
    node_sequence: List[str],
    n: int,
    lm,
    smooth_prob: bool = False,
    correct_node_probs: Optional[List[float]] = None,
) -> List[float]:
    """Compute per-turn probabilities, optionally weighted by node assignment confidence."""
    if not node_sequence:
        return []

    if n > len(node_sequence) + 1:  # +1 for BOS padding
        print(f"  [WARN] Sequence length {len(node_sequence)} < n={n}, probabilities may be unreliable.")

    if smooth_prob:
        if correct_node_probs is None:
            raise ValueError("smooth_prob=True requires correct_node_probs")
        if len(correct_node_probs) != len(node_sequence):
            raise ValueError("correct_node_probs length must match node_sequence length")

    context = [BOS] * max(n - 1, 0)
    probs = []
    for i, node in enumerate(node_sequence):
        ctx = tuple(context[-(n - 1):]) if n > 1 else tuple()
        p = float(lm.score(node, ctx))
        if smooth_prob:
            p *= float(correct_node_probs[i])
        probs.append(p)
        context.append(node)
    return probs


def perplexity_from_prob_lists(prob_lists: List[List[float]]) -> float:
    """Compute perplexity from a list of per-turn probability lists."""
    total_log_p = 0.0
    total_tokens = 0
    for probs in prob_lists:
        for p in probs:
            total_log_p += float(np.log(max(float(p), 1e-12)))
            total_tokens += 1
    if total_tokens == 0:
        return float("inf")
    return float(np.exp(-total_log_p / total_tokens))


def compute_online_params(
    dialog_probs_list: List[List[float]],
) -> float:
    """Compute online threshold for max_jump from training probabilities.

    Calibrated at the per-dialog level: threshold = JUMP_QUANTILE percentile of
    per-dialog maximum jumps, so only ~(1-JUMP_QUANTILE)*100% of training dialogs
    trigger (rather than (1-JUMP_QUANTILE)*100% of training turns).
    """
    if not dialog_probs_list:
        return float("inf")

    per_dialog_max_jumps = []
    for probs in dialog_probs_list:
        if len(probs) < 2:
            continue
        surprise = -np.log(np.maximum(np.array(probs, dtype=float), 1e-12))
        jumps = np.diff(surprise)
        per_dialog_max_jumps.append(float(np.max(jumps)))

    if not per_dialog_max_jumps:
        return float("inf")

    return float(np.quantile(per_dialog_max_jumps, JUMP_QUANTILE))


def vprint(*args, **kwargs):
    """Print only when verbose mode is enabled."""
    if _verbose:
        print(*args, **kwargs)


def render_summary_table(
    hallucination_types: List[str],
    all_type_metrics: dict,
    methods: List[str],
    happy_metrics: Optional[dict] = None,
) -> str:
    """
    Print a MAE/eMAE summary table.

    Hallucination columns show  MAE(eMAE/dMAE)  at 2 decimal places.
    For display only, if eMAE/dMAE is undefined because there were zero
    early/delayed events, it is shown as 0.00 (JSON remains null).
    Conformant column shows eMAE (error for non-hallucinated dialogs).
    Hallu Avg shows the mean MAE across hallucination types.
    Coverage% shows prediction coverage rates (% dialogs with real early predictions).
    AVG column shows the overall score.
    AVG = (Hallu Avg MAE + Conformant eMAE) / 2  -- macro-average
    treating hallucinated and conformant dialogs as two equal groups.
    Smooth sub-rows show signed diffs vs raw in the same triplet format.
    Modes used for table reporting follow METHOD_MODE.
    """
    D = 2  # decimal places
    # Minimum column widths:
    #   hallu: "X.XX(X.XX/X.XX)" = 15 chars; smooth: "+X.XX(+X.XX/+X.XX)" = 18
    hallu_val_w = 3 * (D + 3) + 3   # e.g. D=2 → 15
    orig_val_w  = D + 2              # e.g. D=2 → 4
    rate_val_w = 6                   # e.g. "100.0%"
    avg_val_w   = D + 2

    ht_labels = [ht.replace("_", " ") for ht in hallucination_types]
    has_happy = happy_metrics is not None
    col_labels = ht_labels + ["AVG", "Coverage%"] + (["Conformant"] if has_happy else []) + (["Coverage%"] if has_happy else []) + ["AVG"]

    col_widths = [max(len(lbl), hallu_val_w) for lbl in ht_labels]
    col_widths.append(max(len("AVG"), avg_val_w))
    col_widths.append(max(len("Coverage%"), rate_val_w))
    if has_happy:
        col_widths.append(max(len("Conformant"), orig_val_w))
    if has_happy:
        col_widths.append(max(len("Coverage%"), rate_val_w))
    col_widths.append(max(len("AVG"), avg_val_w))

    row_w = max(max((len(m) for m in methods), default=8), len("+ smooth"))

    sep = f"+{'-' * (row_w + 2)}" + "".join(f"+{'-' * (w + 2)}" for w in col_widths) + "+"
    hdr = (
        f"| {'method'.ljust(row_w)} "
        + "".join(f"| {lbl.center(w)} " for lbl, w in zip(col_labels, col_widths))
        + "|"
    )
    lines = []
    lines.append(
        f"  hallu cols: MAE(eMAE/dMAE) | Conformant col: eMAE"
        f" | Coverage%: prediction coverage"
        f" | AVG: mean MAE across hallucination types"
        f" | AVG = (Hallu AVG + Conformant eMAE) / 2"
    )
    lines.append(f"  {sep}")
    lines.append(f"  {hdr}")
    lines.append(f"  {sep}")

    def _v(v):
        """Format a non-negative value at D decimal places."""
        return "n/a" if v is None else f"{v:.{D}f}"

    def _dv(v):
        """Format a signed diff value at D decimal places."""
        return "n/a" if v is None else f"{v:+.{D}f}"

    def _pct(v):
        return "n/a" if v is None else f"{100.0 * v:.1f}%"

    def _disp_error_component(value, count):
        """Display-only error component: 0.00 if undefined due to zero events."""
        if value is not None:
            return float(value)
        if count == 0:
            return 0.0
        return None

    def _entry_rate(entry: dict) -> Optional[float]:
        if not isinstance(entry, dict):
            return None
        if entry.get("pred_rate") is not None:
            return float(entry["pred_rate"])
        n_pred = entry.get("n_pred")
        n_total = entry.get("n_total")
        if n_pred is None or n_total in (None, 0):
            return None
        return float(n_pred) / float(n_total)

    def _ensure_counts(entry: dict, n_eval_fallback: Optional[int]):
        if not isinstance(entry, dict):
            return
        if entry.get("n_total") is None and n_eval_fallback is not None and entry.get("mae") is not None:
            entry["n_total"] = int(n_eval_fallback)
        if entry.get("n_pred") is None and entry.get("n_total") is not None and entry.get("mae") is not None:
            entry["n_pred"] = int(entry["n_total"])
        if entry.get("pred_rate") is None and entry.get("n_total") not in (None, 0) and entry.get("n_pred") is not None:
            entry["pred_rate"] = float(entry["n_pred"]) / float(entry["n_total"])

    def _hallu_cell(mae, emae, dmae):
        if mae is None:
            return "n/a"
        # Keep a consistent triplet shape even when one side is undefined.
        emae_txt = _v(emae)
        dmae_txt = _v(dmae)
        return f"{mae:.{D}f}({emae_txt}/{dmae_txt})"

    def _hallu_diff_cell(r_mae, s_mae, r_emae, s_emae, r_dmae, s_dmae):
        if r_mae is None or s_mae is None:
            return "n/a"
        mae_diff_txt = f"{s_mae - r_mae:+.{D}f}"
        emae_diff = None if (r_emae is None or s_emae is None) else (s_emae - r_emae)
        dmae_diff = None if (r_dmae is None or s_dmae is None) else (s_dmae - r_dmae)
        return f"{mae_diff_txt}({_dv(emae_diff)}/{_dv(dmae_diff)})"

    preferred_order = ["first_below", "max_jump", "first_turn", "last_turn", "random"]
    ordered_methods = [m for m in preferred_order if m in methods] + [m for m in methods if m not in preferred_order]

    def _get(ht, variant, key):
        mode_key = METHOD_MODE.get(method, "online")
        return (
            all_type_metrics.get(ht, {}).get(variant, {})
            .get(mode_key, {}).get(method, {}).get(key)
        )

    for method in ordered_methods:
        mode_key = METHOD_MODE.get(method, "online")

        # Gather per-hallucination-type raw values
        rv = {ht: {k: _get(ht, "raw",    k) for k in ("mae", "emae", "dmae", "n_early", "n_delay", "n_pred", "n_total", "pred_rate")} for ht in hallucination_types}
        sv = {ht: {k: _get(ht, "smooth", k) for k in ("mae", "emae", "dmae", "n_early", "n_delay", "n_pred", "n_total", "pred_rate")} for ht in hallucination_types}
        for ht in hallucination_types:
            _ensure_counts(rv[ht], all_type_metrics.get(ht, {}).get("raw", {}).get("n_eval"))
            _ensure_counts(sv[ht], all_type_metrics.get(ht, {}).get("smooth", {}).get("n_eval"))

        # Happy (originals) eMAE
        happy_raw_emae = happy_smo_emae = None
        happy_raw_entry = {}
        happy_smo_entry = {}
        if has_happy:
            happy_raw_entry = happy_metrics.get("raw", {}).get(mode_key, {}).get(method, {})
            happy_smo_entry = happy_metrics.get("smooth", {}).get(mode_key, {}).get(method, {})
            _ensure_counts(happy_raw_entry, happy_metrics.get("raw", {}).get("n_eval"))
            _ensure_counts(happy_smo_entry, happy_metrics.get("smooth", {}).get("n_eval"))
            happy_raw_emae = happy_raw_entry.get("emae")
            happy_smo_emae = happy_smo_entry.get("emae")

        # Hallucination and originals prediction coverage
        raw_h_pred_num = sum(float(rv[ht].get("n_pred")) for ht in hallucination_types if rv[ht].get("n_pred") is not None and rv[ht].get("n_total") not in (None, 0))
        raw_h_pred_den = sum(float(rv[ht].get("n_total")) for ht in hallucination_types if rv[ht].get("n_pred") is not None and rv[ht].get("n_total") not in (None, 0))
        smo_h_pred_num = sum(float(sv[ht].get("n_pred")) for ht in hallucination_types if sv[ht].get("n_pred") is not None and sv[ht].get("n_total") not in (None, 0))
        smo_h_pred_den = sum(float(sv[ht].get("n_total")) for ht in hallucination_types if sv[ht].get("n_pred") is not None and sv[ht].get("n_total") not in (None, 0))
        hallu_rate_raw = (raw_h_pred_num / raw_h_pred_den) if raw_h_pred_den > 0 else None
        hallu_rate_smo = (smo_h_pred_num / smo_h_pred_den) if smo_h_pred_den > 0 else None
        orig_rate_raw = _entry_rate(happy_raw_entry) if has_happy else None
        orig_rate_smo = _entry_rate(happy_smo_entry) if has_happy else None

        # AVG = (avg hallucination MAE + originals eMAE) / 2
        raw_mae_numbers = [rv[ht]["mae"] for ht in hallucination_types if rv[ht]["mae"] is not None]
        avg_hallu_raw = float(np.mean(raw_mae_numbers)) if raw_mae_numbers else None
        if avg_hallu_raw is not None and happy_raw_emae is not None:
            avg_raw = (avg_hallu_raw + happy_raw_emae) / 2.0
        else:
            avg_raw = avg_hallu_raw

        # Raw row
        cells = [
            _hallu_cell(
                rv[ht]["mae"],
                _disp_error_component(rv[ht]["emae"], rv[ht].get("n_early")),
                _disp_error_component(rv[ht]["dmae"], rv[ht].get("n_delay")),
            ).center(w)
            for ht, w in zip(hallucination_types, col_widths)
        ]
        next_idx = len(hallucination_types)
        cells.append(_v(avg_hallu_raw).center(col_widths[next_idx]))
        next_idx += 1
        cells.append(_pct(hallu_rate_raw).center(col_widths[next_idx]))
        next_idx += 1
        if has_happy:
            cells.append(_v(happy_raw_emae).center(col_widths[next_idx]))
            next_idx += 1
            cells.append(_pct(orig_rate_raw).center(col_widths[next_idx]))
            next_idx += 1
        cells.append(_v(avg_raw).center(col_widths[next_idx]))
        lines.append(f"  | {method.ljust(row_w)} " + "".join(f"| {c} " for c in cells) + "|")

        # Smooth sub-row — signed diffs (shown only for non-baseline methods)
        if method not in BASELINE_METHODS:
            smo_mae_numbers = [sv[ht]["mae"] for ht in hallucination_types if sv[ht]["mae"] is not None]
            avg_hallu_smo = float(np.mean(smo_mae_numbers)) if smo_mae_numbers else None
            if avg_hallu_smo is not None and happy_smo_emae is not None:
                avg_smo = (avg_hallu_smo + happy_smo_emae) / 2.0
            else:
                avg_smo = avg_hallu_smo
            avg_hallu_diff = (avg_hallu_smo - avg_hallu_raw) if avg_hallu_smo is not None and avg_hallu_raw is not None else None
            avg_diff = (avg_smo - avg_raw) if avg_smo is not None and avg_raw is not None else None

            cells = [
                _hallu_diff_cell(
                    rv[ht]["mae"],  sv[ht]["mae"],
                    _disp_error_component(rv[ht]["emae"], rv[ht].get("n_early")),
                    _disp_error_component(sv[ht]["emae"], sv[ht].get("n_early")),
                    _disp_error_component(rv[ht]["dmae"], rv[ht].get("n_delay")),
                    _disp_error_component(sv[ht]["dmae"], sv[ht].get("n_delay")),
                ).center(w)
                for ht, w in zip(hallucination_types, col_widths)
            ]
            next_idx = len(hallucination_types)
            cells.append(_dv(avg_hallu_diff).center(col_widths[next_idx]))
            next_idx += 1
            cells.append(_dv(None if (hallu_rate_raw is None or hallu_rate_smo is None)
                             else (100.0 * (hallu_rate_smo - hallu_rate_raw))).center(col_widths[next_idx]))
            next_idx += 1
            if has_happy:
                cells.append(_dv(None if (happy_raw_emae is None or happy_smo_emae is None)
                                 else happy_smo_emae - happy_raw_emae).center(col_widths[next_idx]))
                next_idx += 1
                cells.append(_dv(None if (orig_rate_raw is None or orig_rate_smo is None)
                                 else (100.0 * (orig_rate_smo - orig_rate_raw))).center(col_widths[next_idx]))
                next_idx += 1
            cells.append(_dv(avg_diff).center(col_widths[next_idx]))
            lines.append(f"  | {'+ smooth'.ljust(row_w)} " + "".join(f"| {c} " for c in cells) + "|")

    lines.append(f"  {sep}")
    return "\n".join(lines)


def render_benchmark_summary_table(
    hallucination_types: List[str],
    all_type_metrics: dict,
    methods: List[str],
    happy_metrics: Optional[dict] = None,
) -> str:
    """Render a compact benchmark-ready summary table with absolute raw/smooth metrics."""
    D = 2
    has_happy = happy_metrics is not None
    row_w = max(max((len(m) for m in methods), default=8), len("method"))

    col_labels = ["Hallucinated"] + (["Conformant"] if has_happy else []) + ["AVG"]
    col_widths = [max(len(lbl), D + 2) for lbl in col_labels]

    sep = f"+{'-' * (row_w + 2)}" + "".join(f"+{'-' * (w + 2)}" for w in col_widths) + "+"
    hdr = (
        f"| {'method'.ljust(row_w)} "
        + "".join(f"| {lbl.center(w)} " for lbl, w in zip(col_labels, col_widths))
        + "|"
    )
    lines = []
    lines.append(
        "  benchmark-ready summary: lower is better | "
        "Hallucinated = mean MAE across hallucination types | "
        "Conformant = eMAE on non-hallucinated dialogs | "
        "AVG = (Hallucinated + Conformant) / 2"
    )
    lines.append(f"  {sep}")
    lines.append(f"  {hdr}")
    lines.append(f"  {sep}")

    def _v(v):
        return "n/a" if v is None else f"{v:.{D}f}"

    preferred_order = ["first_below", "max_jump", "first_turn", "last_turn", "random"]
    ordered_methods = [m for m in preferred_order if m in methods] + [m for m in methods if m not in preferred_order]

    def _variant_metrics(method: str, variant: str):
        mode_key = METHOD_MODE.get(method, "online")
        mae_numbers = []
        for ht in hallucination_types:
            mae = (
                all_type_metrics.get(ht, {})
                .get(variant, {})
                .get(mode_key, {})
                .get(method, {})
                .get("mae")
            )
            if mae is not None:
                mae_numbers.append(float(mae))
        hallu_avg = float(np.mean(mae_numbers)) if mae_numbers else None

        conformant = None
        if has_happy:
            conformant = (
                happy_metrics.get(variant, {})
                .get(mode_key, {})
                .get(method, {})
                .get("emae")
            )

        if hallu_avg is not None and conformant is not None:
            overall_avg = (hallu_avg + conformant) / 2.0
        else:
            overall_avg = hallu_avg

        return hallu_avg, conformant, overall_avg

    for method in ordered_methods:
        hallu_avg, conformant, overall_avg = _variant_metrics(method, "raw")

        cells = [_v(hallu_avg).center(col_widths[0])]
        next_idx = 1
        if has_happy:
            cells.append(_v(conformant).center(col_widths[next_idx]))
            next_idx += 1
        cells.append(_v(overall_avg).center(col_widths[next_idx]))
        lines.append(f"  | {method.ljust(row_w)} " + "".join(f"| {c} " for c in cells) + "|")

        if method not in BASELINE_METHODS:
            smooth_hallu_avg, smooth_conformant, smooth_overall_avg = _variant_metrics(method, "smooth")
            smooth_cells = [_v(smooth_hallu_avg).center(col_widths[0])]
            smooth_idx = 1
            if has_happy:
                smooth_cells.append(_v(smooth_conformant).center(col_widths[smooth_idx]))
                smooth_idx += 1
            smooth_cells.append(_v(smooth_overall_avg).center(col_widths[smooth_idx]))
            lines.append(f"  | {'+ smooth'.ljust(row_w)} " + "".join(f"| {c} " for c in smooth_cells) + "|")

    lines.append(f"  {sep}")
    return "\n".join(lines)


def print_summary_table(
    hallucination_types: List[str],
    all_type_metrics: dict,
    methods: List[str],
    happy_metrics: Optional[dict] = None,
) -> None:
    print("\n" + render_benchmark_summary_table(hallucination_types, all_type_metrics, methods, happy_metrics=happy_metrics))
    print("\n" + render_summary_table(hallucination_types, all_type_metrics, methods, happy_metrics=happy_metrics))


# ============================================================================
# SECTION 1: TRAIN/PREDICT IMPLEMENTATIONS FOR BUILT-IN METHODS
# ============================================================================
# Each method is implemented as (train_fn, predict_fn) pair.
# Users can add their own by registering in METHOD_REGISTRY
# (see beginning of this file: "HOW TO ADD YOUR OWN OFF-ROAD TURN PREDICTION METHOD" section).

def _train_generic(dialogs, context):
    """Generic train function for built-in methods. Stores context in model."""
    return {
        "args": context["args"],
        "flow_score": context["flow_score"],
        "lm": context["lm"],
        "thresholds_by_variant": context["thresholds_by_variant"],
    }


def _predict_first_below(model, dialog, precomputed=None):
    """Predict: first turn with probability below threshold."""
    variant = model.get("_variant", "raw")
    threshold = model["thresholds_by_variant"][variant]["offroad_threshold"]
    n = model["args"].n
    lm = model["lm"]
    flow_score = model["flow_score"]

    if precomputed is None:
        node_seq, correct_node_probs = get_node_sequence_and_correct_node_probs(
            dialog,
            flow_score,
            batch_size=model["args"].batch_size,
        )
        turn_indices = get_effective_turn_indices(dialog, flow_score)
    else:
        node_seq = precomputed["node_sequence"]
        correct_node_probs = precomputed["correct_node_probs"]
        turn_indices = precomputed["turn_indices"]

    is_smooth = variant == "smooth"
    probs = per_turn_transition_probs(
        node_seq,
        n=n,
        lm=lm,
        smooth_prob=is_smooth,
        correct_node_probs=(correct_node_probs if is_smooth else None),
    )
    for i, prob in enumerate(probs):
        if prob < threshold:
            return turn_indices[i]
    return None


def _predict_max_jump(model, dialog, precomputed=None):
    """Predict: first turn with surprise jump above threshold."""
    variant = model.get("_variant", "raw")
    jump_threshold = model["thresholds_by_variant"][variant]["jump_threshold"]
    n = model["args"].n
    lm = model["lm"]
    flow_score = model["flow_score"]

    if precomputed is None:
        node_seq, correct_node_probs = get_node_sequence_and_correct_node_probs(
            dialog,
            flow_score,
            batch_size=model["args"].batch_size,
        )
        turn_indices = get_effective_turn_indices(dialog, flow_score)
    else:
        node_seq = precomputed["node_sequence"]
        correct_node_probs = precomputed["correct_node_probs"]
        turn_indices = precomputed["turn_indices"]

    is_smooth = variant == "smooth"
    probs = per_turn_transition_probs(
        node_seq,
        n=n,
        lm=lm,
        smooth_prob=is_smooth,
        correct_node_probs=(correct_node_probs if is_smooth else None),
    )
    surprise = -np.log(np.maximum(np.array(probs, dtype=float), 1e-12))
    if len(surprise) <= 1:
        return turn_indices[0]
    jumps = np.diff(surprise)
    for i, jump in enumerate(jumps, start=1):
        if float(jump) > jump_threshold:
            return turn_indices[i]
    return None


def _predict_first_turn(model, dialog, precomputed=None):
    """Predict: always the first turn."""
    flow_score = model["flow_score"]
    turn_indices = precomputed["turn_indices"] if precomputed is not None else get_effective_turn_indices(dialog, flow_score)
    return turn_indices[0]


def _predict_last_turn(model, dialog, precomputed=None):
    """Predict: always the last turn."""
    flow_score = model["flow_score"]
    turn_indices = precomputed["turn_indices"] if precomputed is not None else get_effective_turn_indices(dialog, flow_score)
    return turn_indices[-1]


def _predict_random(model, dialog, precomputed=None):
    """Predict: random turn."""
    flow_score = model["flow_score"]
    turn_indices = precomputed["turn_indices"] if precomputed is not None else get_effective_turn_indices(dialog, flow_score)
    return random.choice(turn_indices)


def resolve_task_names(task_parent: str, requested_tasks: Optional[List[str]]) -> List[str]:
    """Resolve task names from CLI input or by listing all immediate subfolders."""
    if not os.path.isdir(task_parent):
        raise ValueError(f"Task parent folder does not exist: {task_parent}")

    if requested_tasks:
        task_names = [t.strip() for t in requested_tasks if t and t.strip()]
    else:
        task_names = sorted(
            [d for d in os.listdir(task_parent) if os.path.isdir(os.path.join(task_parent, d))]
        )

    if not task_names:
        raise ValueError(f"No task folders found under: {task_parent}")

    missing = [t for t in task_names if not os.path.isdir(os.path.join(task_parent, t))]
    if missing:
        raise ValueError(f"Requested task folders not found under {task_parent}: {missing}")

    return task_names


def load_dialogs(path: str) -> List[Dialog]:
    """Load dialogs from a file or folder."""
    if os.path.isdir(path):
        return Dialog.from_folder(path)
    return Dialog.from_file(path)


def load_dialogs_recursive(root_path: str) -> List[Dialog]:
    """Recursively load dialogs from all files under a directory tree."""
    if not os.path.isdir(root_path):
        raise ValueError(f"Path '{root_path}' is not a directory.")

    dialogs = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in sorted(filenames):
            file_path = os.path.join(dirpath, filename)
            try:
                loaded = Dialog.from_file(file_path)
            except Exception:
                continue

            if isinstance(loaded, list):
                dialogs.extend(loaded)
            else:
                dialogs.append(loaded)
    return dialogs


def load_split(split_file: str) -> dict:
    """dialog_id (source dialogue) -> "train" | "test", from task3_splits.csv."""
    import csv
    with open(split_file, encoding="utf-8", newline="") as fh:
        return {int(r["dialog_id"]): r["split"] for r in csv.DictReader(fh)}


def load_task_datasets(task_root: str) -> Tuple[List[Dialog], dict]:
    """Load originals and hallucination datasets from the fixed task-root structure."""
    originals_path = os.path.join(task_root, ORIGINALS_SUBDIR)
    if not os.path.isdir(originals_path):
        raise ValueError(f"Missing required originals folder: {originals_path}")

    train_dialogs = load_dialogs_recursive(originals_path)
    hallucination_dialogs = {}
    for hallucination_type in HALLUCINATION_TYPES:
        hallucination_path = os.path.join(task_root, hallucination_type)
        if not os.path.isdir(hallucination_path):
            raise ValueError(f"Missing required hallucination folder: {hallucination_path}")
        hallucination_dialogs[hallucination_type] = load_dialogs_recursive(hallucination_path)

    return train_dialogs, hallucination_dialogs


def _is_valid_turn(turn) -> bool:
    """Return True if a turn has usable speaker/text fields for graph extraction."""
    speaker = getattr(turn, "speaker", None)
    text = getattr(turn, "text", None)
    if not isinstance(speaker, str) or not speaker.strip():
        return False
    if not isinstance(text, str) or not text.strip():
        return False
    return True


def sanitize_dialogs(dialogs: List[Dialog], split_name: str, task_name: str) -> List[Dialog]:
    """Drop dialogs containing malformed turns that break dialog2flow graph extraction."""
    clean_dialogs = []
    dropped = 0
    for dialog in dialogs:
        valid = True
        for turn in dialog:
            if not _is_valid_turn(turn):
                valid = False
                break
        if valid:
            clean_dialogs.append(dialog)
        else:
            dropped += 1
            dialog_id = getattr(dialog, "id", "?")
            print(
                f"  [WARN] Dropping malformed dialog {dialog_id} in {task_name}/{split_name} "
                "(missing/invalid turn speaker or text)",
                file=sys.stderr,
            )

    if dropped:
        print(
            f"  [WARN] Dropped {dropped}/{len(dialogs)} malformed dialogs in {task_name}/{split_name}",
            file=sys.stderr,
        )
    return clean_dialogs


def build_result_path(args, result_filename: str) -> Optional[str]:
    """Derive the structured result path under the output root folder."""
    if not args.output:
        return None
    model_name = args.model.split("/")[-1] if args.model else "default"
    ai_part = "_ai_only" if args.ai_speaker else ""
    setting = f"lm-{args.n}-gram{ai_part}"
    output_folder = os.path.join(args.output, model_name, setting)
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, result_filename)


def metrics_to_detection_artifact(all_type_metrics: dict) -> dict:
    """Convert internal metrics map to the persisted detection artifact format."""
    detection = {"hallucination_types": {}}
    for ht in HALLUCINATION_TYPES:
        detection["hallucination_types"][ht] = {"variants": {}}
        for variant in ["raw", "smooth"]:
            vdata = all_type_metrics.get(ht, {}).get(variant, {})
            detection["hallucination_types"][ht]["variants"][variant] = {
                "dialogs_evaluated": int(vdata.get("n_eval", 0)),
                "offline_metrics": vdata.get("offline", {}),
                "online_metrics": vdata.get("online", {}),
            }
    return detection


def extract_task_metrics(payload: dict) -> Tuple[dict, List[str], dict]:
    """Extract all_type_metrics, available methods, and happy_metrics from a payload."""
    cfg = payload.get("config", {})
    detection = payload.get("detection") or {}
    hallucination_types = cfg.get("hallucination_types", HALLUCINATION_TYPES)
    all_methods = ["first_below", "max_jump", "first_turn", "last_turn", "random"]

    all_type_metrics = {}
    eval_methods = all_methods
    for ht in hallucination_types:
        ht_data = detection.get("hallucination_types", {}).get(ht, {})
        all_type_metrics[ht] = {}
        for variant in ["raw", "smooth"]:
            vdata = ht_data.get("variants", {}).get(variant, {})
            n_eval = vdata.get("dialogs_evaluated", 0)
            offline_metrics = vdata.get("offline_metrics", {})
            online_metrics = vdata.get("online_metrics", {})
            all_type_metrics[ht][variant] = {
                "n_eval": n_eval,
                "offline": offline_metrics,
                "online": online_metrics,
            }
            if variant == "raw":
                eval_methods = [m for m in all_methods if m in offline_metrics] or all_methods

    # Extract happy (originals) metrics if present
    happy_metrics = {}
    happy_data = detection.get("happy", {})
    for variant in ["raw", "smooth"]:
        vdata = happy_data.get("variants", {}).get(variant, {})
        happy_metrics[variant] = {
            "n_eval": vdata.get("n_eval", 0),
            "offline": vdata.get("offline_metrics", {}),
            "online": vdata.get("online_metrics", {}),
        }

    return all_type_metrics, eval_methods, happy_metrics


def aggregate_metrics_across_tasks(
    task_metrics_list: List[dict],
    happy_metrics_list: List[dict],
    methods: List[str],
) -> Tuple[dict, dict]:
    """Aggregate metrics across tasks.

    Returns (aggregated_hallu_metrics, aggregated_happy_metrics).
    """
    aggregated = {}
    for ht in HALLUCINATION_TYPES:
        aggregated[ht] = {}
        for variant in ["raw", "smooth"]:
            aggregated[ht][variant] = {
                "n_eval": 0,
                "offline": {},
                "online": {},
            }
            n_eval_sum = 0
            for tmetrics in task_metrics_list:
                n_eval_sum += int(tmetrics.get(ht, {}).get(variant, {}).get("n_eval", 0))
            aggregated[ht][variant]["n_eval"] = n_eval_sum

            for method in methods:
                mode_key = METHOD_MODE.get(method, "online")
                method_entry = {}
                for metric_key in ("mae", "emae", "dmae"):
                    values = [
                        float(v) for tmetrics in task_metrics_list
                        if (v := (
                            tmetrics.get(ht, {})
                            .get(variant, {})
                            .get(mode_key, {})
                            .get(method, {})
                            .get(metric_key)
                        )) is not None
                    ]
                    if values:
                        method_entry[metric_key] = float(np.mean(values))

                # Aggregate support counts for display semantics and coverage rates.
                for count_key in ("n_early", "n_delay", "n_pred", "n_total"):
                    csum = 0
                    has_count = False
                    for tmetrics in task_metrics_list:
                        val = (
                            tmetrics.get(ht, {})
                            .get(variant, {})
                            .get(mode_key, {})
                            .get(method, {})
                            .get(count_key)
                        )
                        if val is not None:
                            csum += int(val)
                            has_count = True
                    if has_count:
                        method_entry[count_key] = int(csum)

                if method_entry.get("n_total", 0) > 0 and method_entry.get("n_pred") is not None:
                    method_entry["pred_rate"] = float(method_entry["n_pred"]) / float(method_entry["n_total"])
                if method_entry:
                    aggregated[ht][variant][mode_key][method] = method_entry

    # Aggregate happy (originals) metrics across tasks
    aggregated_happy: dict = {"raw": {"offline": {}, "online": {}}, "smooth": {"offline": {}, "online": {}}}
    for method in methods:
        mode_key = METHOD_MODE.get(method, "online")
        for variant in ["raw", "smooth"]:
            values = []
            method_entry = {}
            for hm in happy_metrics_list:
                val = hm.get(variant, {}).get(mode_key, {}).get(method, {}).get("emae")
                if val is not None:
                    values.append(float(val))
            if values:
                method_entry["emae"] = float(np.mean(values))

            for count_key in ("n_early", "n_pred", "n_total"):
                csum = 0
                has_count = False
                for hm in happy_metrics_list:
                    val = hm.get(variant, {}).get(mode_key, {}).get(method, {}).get(count_key)
                    if val is not None:
                        csum += int(val)
                        has_count = True
                if has_count:
                    method_entry[count_key] = int(csum)

            if method_entry.get("n_total", 0) > 0 and method_entry.get("n_pred") is not None:
                method_entry["pred_rate"] = float(method_entry["n_pred"]) / float(method_entry["n_total"])

            if method_entry:
                aggregated_happy[variant][mode_key][method] = method_entry

    return aggregated, aggregated_happy


def _reprint_results(payload: dict) -> None:
    """Reprint summary tables from a previously saved output JSON."""
    cfg = payload.get("config", {})
    all_type_metrics, eval_methods, happy_metrics = extract_task_metrics(payload)

    vprint(f"Task root:  {cfg.get('task_root', '?')}")
    vprint(f"n-gram: {cfg.get('n', '?')}  |  model: {cfg.get('lm_model', '?')}")

    hallucination_types = cfg.get("hallucination_types", HALLUCINATION_TYPES)
    vprint("\nDialogs evaluated per hallucination type:")
    for ht in hallucination_types:
        raw_n = all_type_metrics[ht].get("raw", {}).get("n_eval", 0)
        smo_n = all_type_metrics[ht].get("smooth", {}).get("n_eval", 0)
        vprint(f"  {ht}: raw={raw_n}  smooth={smo_n}")

    print_summary_table(
        hallucination_types=hallucination_types,
        all_type_metrics=all_type_metrics,
        methods=eval_methods,
        happy_metrics=happy_metrics if any(happy_metrics.get(v, {}) for v in ["raw", "smooth"]) else None,
    )

    runtime = cfg.get("total_runtime_seconds")
    if runtime is not None:
        runtime_str = f" (original run: {timedelta(seconds=round(runtime))})"
    else:
        runtime_str = ""
    vprint(f"\nDone (loaded from cache{runtime_str}).")


# ============================================================================
# SECTION 2: METHOD REGISTRY - CONFIGURATION AND EXTENSIBILITY
# ============================================================================
# Centralized registry for all off-road turn prediction methods.
# Each method is a (train_fn, predict_fn) pair.

METHOD_REGISTRY = {
    "first_below": (_train_generic, _predict_first_below),
    "max_jump": (_train_generic, _predict_max_jump),
    "first_turn": (_train_generic, _predict_first_turn),
    "last_turn": (_train_generic, _predict_last_turn),
    "random": (_train_generic, _predict_random),
}


# ============================================================================
# SECTION 3: TASK RUNNER - MAIN EVALUATION PIPELINE
# ============================================================================

def run_single_task(args, task_name: str) -> Tuple[Optional[dict], List[str], Optional[dict]]:
    """Run one task end-to-end (or load cache), print its table, and return metrics for aggregation."""
    task_start_time = time.perf_counter()
    task_root = os.path.join(args.task_root, task_name)
    output_path = build_result_path(args, f"{task_name}.json")
    temp_output_root = os.path.join(os.getcwd(), ".dialog2flow")
    temp_output_path = None
    flow_score = None  # set once the encoder is loaded below; released in `finally`

    # ------------------------------------------------------------------
    # Fast path: results already exist — just reprint them
    # ------------------------------------------------------------------
    if output_path and os.path.isfile(output_path):
        vprint(f"Results file already exists: {output_path}")
        vprint("Skipping computation and loading cached results...\n")
        with open(output_path) as fh:
            payload = json.load(fh)
        _reprint_results(payload)
        all_type_metrics, eval_methods, happy_metrics = extract_task_metrics(payload)
        return all_type_metrics, eval_methods, happy_metrics

    try:
        os.makedirs(temp_output_root, exist_ok=True)
        temp_output_path = tempfile.mkdtemp(prefix=f"{task_name}-", dir=temp_output_root)

        # Reproducibility per task
        random.seed(args.seed)
        np.random.seed(args.seed)

        # ------------------------------------------------------------------
        # 1. Load dialogs
        # ------------------------------------------------------------------
        vprint(f"Loading task datasets from: {task_root}")
        train_dialogs, hallucination_dialogs = load_task_datasets(task_root)

        train_dialogs = sanitize_dialogs(train_dialogs, "originals", task_name)
        for hallucination_type in HALLUCINATION_TYPES:
            hallucination_dialogs[hallucination_type] = sanitize_dialogs(
                hallucination_dialogs[hallucination_type],
                hallucination_type,
                task_name,
            )

        # Task 3 split (default): learn the graph, the n-gram LM and the thresholds from the
        # train split's conformant dialogues only; evaluate on the test split's conformant
        # dialogues and the forks of its sources (a fork shares its source's id).
        # --in-sample: learn and evaluate on all conformant dialogues (not the benchmark protocol).
        eval_originals = train_dialogs
        if not args.in_sample:
            split_of = load_split(os.path.join(args.task_root, "task3_splits.csv"))
            eval_originals = [d for d in train_dialogs if split_of[int(d.id)] == "test"]
            train_dialogs = [d for d in train_dialogs if split_of[int(d.id)] == "train"]
            for hallucination_type in HALLUCINATION_TYPES:
                hallucination_dialogs[hallucination_type] = [
                    d for d in hallucination_dialogs[hallucination_type] if split_of[int(d.id)] == "test"]
            vprint(f"  Task 3 split: {len(train_dialogs)} train / {len(eval_originals)} test original dialogs.")

        if not train_dialogs:
            print(f"  [WARN] Skipping task {task_name}: no valid original dialogs after sanitization.")
            return None, ["first_below", "max_jump", "first_turn", "last_turn", "random"], None

        vprint(f"  Loaded {len(train_dialogs)} original dialogs.")
        for hallucination_type in HALLUCINATION_TYPES:
            vprint(f"  Loaded {len(hallucination_dialogs[hallucination_type])} dialogs for {hallucination_type}.")

        # ------------------------------------------------------------------
        # 2. Build flow graph from input dialogs
        # ------------------------------------------------------------------
        vprint("Using input dialogs as the flow graph reference.")
        d2f_kwargs = {
            "nodes_prune_threshold": NODE_PRUNE_THRESHOLD,
            "edges_prune_threshold": EDGE_PRUNE_THRESHOLD,
            "output_path": temp_output_path,
        }
        if args.model:
            d2f_kwargs["node_embedding_model"] = args.model

        vprint(f"Building dialog flow graph (n-gram order: {args.n}, model: {LM_MODEL_TYPE}) ...")
        flow_score = DialogFlowPPL(
            reference_dialogues=train_dialogs,
            ai_speaker=args.ai_speaker,
            use_only_ai_speaker=args.ai_speaker is not None,
            verbose=False,
            **d2f_kwargs,
        )
        n_nodes = len([k for k in flow_score.nodes.keys() if not k.startswith("_")])
        n_edges = flow_score.graph.number_of_edges()
        vprint(f"  Flow graph built: {n_nodes} nodes, {n_edges} edges.")

        # ------------------------------------------------------------------
        # 3. Convert dialogs → node sequences
        # ------------------------------------------------------------------
        vprint("Converting dialogs to node sequences ...")
        train_sequences = dialogs_to_node_sequences(train_dialogs, flow_score, add_boundaries=True)
        vprint(f"  {len(train_sequences)} sequences extracted (avg length: "
               f"{np.mean([len(s) for s in train_sequences]):.1f} tokens).")

        vocab = set(tok for seq in train_sequences for tok in seq)
        vprint(f"  Vocabulary size: {len(vocab)} unique node tokens.")

        # ------------------------------------------------------------------
        # 4. Train n-gram LM
        # ------------------------------------------------------------------
        vprint(f"\nTraining {args.n}-gram LM ({LM_MODEL_TYPE}) ...")

        # lm = train_nltk_lm(train_sequences, n=args.n, model_type=LM_MODEL_TYPE)
        lm, lm_model_used = train_nltk_lm_with_fallback(train_sequences, n=args.n, model_type=LM_MODEL_TYPE)
        if lm_model_used != LM_MODEL_TYPE:
            print(f"  [WARN] Results for n={args.n} use '{lm_model_used}' instead of '{LM_MODEL_TYPE}'.", file=sys.stderr)

        vprint("  NLTK LM trained.")

        # Compute raw and smooth probabilities in a single per-dialog pass
        train_dialog_probs_by_variant = {"raw": [], "smooth": []}
        for dialog in train_dialogs:
            try:
                node_sequence, correct_node_probs = get_node_sequence_and_correct_node_probs(
                    dialog,
                    flow_score,
                    batch_size=args.batch_size,
                )
            except (ValueError, DialogEncodingSkipped) as exc:
                print(f"  [WARN] Skipping dialog {dialog.id} in train probability computation: {exc}", file=sys.stderr)
                continue

            raw_probs = per_turn_transition_probs(
                node_sequence=node_sequence,
                n=args.n,
                lm=lm,
                smooth_prob=False,
                correct_node_probs=None,
            )
            smooth_probs = [float(p) * float(c) for p, c in zip(raw_probs, correct_node_probs)]

            train_dialog_probs_by_variant["raw"].append(raw_probs)
            train_dialog_probs_by_variant["smooth"].append(smooth_probs)

        train_ppl_by_variant = {
            variant: perplexity_from_prob_lists(train_dialog_probs_by_variant[variant])
            for variant in ["raw", "smooth"]
        }
        vprint(f"  Train perplexity [raw]:   {train_ppl_by_variant['raw']:.4f}")
        vprint(f"  Train perplexity [smooth]: {train_ppl_by_variant['smooth']:.4f}")

        thresholds_by_variant = {}
        for variant in ["raw", "smooth"]:
            dialog_probs = train_dialog_probs_by_variant[variant]
            # Calibrate at per-dialog level: threshold = OFFROAD_QUANTILE percentile of
            # per-dialog *minimum* turn probs. Only ~OFFROAD_QUANTILE fraction of training
            # dialogs trigger, regardless of dialog length.
            per_dialog_min_probs = [min(probs) for probs in dialog_probs if probs]
            offroad_threshold = float(np.quantile(per_dialog_min_probs, OFFROAD_QUANTILE)) if per_dialog_min_probs else 0.0
            jump_threshold = compute_online_params(dialog_probs)
            thresholds_by_variant[variant] = {
                "offroad_threshold": offroad_threshold,
                "jump_threshold": jump_threshold,
            }
            vprint(f"  [{variant}] off-road threshold: {offroad_threshold:.8f}")
            vprint(f"  [{variant}] online max_jump threshold: {jump_threshold:.8f}")

        detection_artifact = None
        all_methods = ["first_below", "max_jump", "first_turn", "last_turn", "random"]
        all_type_metrics = {}
        happy_metrics = None  # populated inside the hallucination_dialogs block

        # --- Train models for each method using METHOD_REGISTRY ---
        models = {}
        context = {
            "args": args,
            "flow_score": flow_score,
            "lm": lm,
            "thresholds_by_variant": thresholds_by_variant,
        }
        for method in all_methods:
            if method not in METHOD_REGISTRY:
                raise ValueError(f"Method '{method}' not found in METHOD_REGISTRY. Add it first!")
            train_fn_impl, predict_fn_impl = METHOD_REGISTRY[method]
            model = train_fn_impl(train_dialogs, context)
            # Store both model state and the predict function for efficient lookup
            model["_predict_fn"] = predict_fn_impl
            models[method] = model

        # ------------------------------------------------------------------
        # 5. Off-road turn prediction on another dataset
        # ------------------------------------------------------------------
        vprint("\nPredicting off-road turns by hallucination type...")
        if hallucination_dialogs:
            requested_methods = all_methods
            online_eval_methods = requested_methods

            # ------------------------------------------------------------------
            # 5b. Predictions on original (happy) dialogs
            # ------------------------------------------------------------------
            vprint("\nComputing predictions on original (happy) dialogs...")
            happy_buffers = {
                variant: {
                    "y_true_online": {m: [] for m in online_eval_methods},
                    "y_pred_online": {m: [] for m in online_eval_methods},
                }
                for variant in ["raw", "smooth"]
            }
            happy_eval_total_by_variant = {variant: 0 for variant in ["raw", "smooth"]}
            happy_per_dialog = {variant: [] for variant in ["raw", "smooth"]}

            for dialog in eval_originals:
                _dialog_id = getattr(dialog, "id", None)
                try:
                    _node_seq, _correct_probs = get_node_sequence_and_correct_node_probs(
                        dialog,
                        flow_score,
                        batch_size=args.batch_size,
                    )
                    _turn_indices = get_effective_turn_indices(dialog, flow_score)
                    if not _turn_indices:
                        continue
                    _precomputed = {
                        "node_sequence": _node_seq,
                        "correct_node_probs": _correct_probs,
                        "turn_indices": _turn_indices,
                    }
                except (ValueError, DialogEncodingSkipped) as exc:
                    print(f"  [WARN] Skipping dialog {_dialog_id} in happy evaluation: {exc}", file=sys.stderr)
                    continue

                _gt_happy = _turn_indices[-1] + 1  # one beyond the last system turn: no fork should occur

                for variant in ["raw", "smooth"]:
                    _h_pred_online = {}
                    happy_eval_total_by_variant[variant] += 1

                    # Keep baseline predictions identical across raw/smooth variants for fair comparison.
                    for method in online_eval_methods:
                        if variant == "smooth" and method in BASELINE_METHODS:
                            # Baselines intentionally have no smooth variant.
                            continue
                        models[method]["_variant"] = variant
                        _pt = models[method]["_predict_fn"](models[method], dialog, precomputed=_precomputed)
                        if _pt is not None:
                            _h_pred_online[method] = _pt

                    for method in online_eval_methods:
                        if variant == "smooth" and method in BASELINE_METHODS:
                            continue
                        if method in _h_pred_online:
                            happy_buffers[variant]["y_true_online"][method].append(_gt_happy)
                            happy_buffers[variant]["y_pred_online"][method].append(_h_pred_online[method])
                    happy_per_dialog[variant].append({
                        "dialog_id": _dialog_id,
                        "ground_truth": _gt_happy,
                        "pred_online": {m: _h_pred_online[m] for m in online_eval_methods if m in _h_pred_online},
                    })
                release_cuda_memory()

            # Compute happy eMAE (all predictions should be early: pred <= last_turn < gt_happy)
            happy_metrics = {}
            for variant in ["raw", "smooth"]:
                happy_metrics[variant] = {"n_eval": 0, "offline": {}, "online": {}}
                happy_metrics[variant]["n_eval"] = int(happy_eval_total_by_variant[variant])
                for method in online_eval_methods:
                    _yt = np.array(happy_buffers[variant]["y_true_online"][method])
                    _yp = np.array(happy_buffers[variant]["y_pred_online"][method])
                    _n_total = int(happy_eval_total_by_variant[variant])
                    if _yt.size == 0:
                        happy_metrics[variant]["online"][method] = {
                            "emae": None,
                            "n_early": 0,
                            "n_exact": 0,
                            "n_pred": 0,
                            "n_total": _n_total,
                            "pred_rate": (0.0 if _n_total > 0 else None),
                        }
                        continue
                    _err = _yp - _yt
                    _em = _err < 0
                    happy_metrics[variant]["online"][method] = {
                        "emae": float(np.mean(np.abs(_err[_em]))) if _em.any() else None,
                        "n_early": int(np.sum(_em)),
                        "n_exact": int(np.sum(_err == 0)),
                        "n_pred": int(_yp.size),
                        "n_total": _n_total,
                        "pred_rate": (float(_yp.size) / float(_n_total)) if _n_total > 0 else None,
                    }

            # Initialize combined absolute error pool with happy dialog errors (grows with each hallucination type)
            combined_abs_errors_pool = {
                variant: {
                    "online": {
                        m: list(np.abs(
                            np.array(happy_buffers[variant]["y_pred_online"][m]) -
                            np.array(happy_buffers[variant]["y_true_online"][m])
                        )) for m in online_eval_methods
                    },
                }
                for variant in ["raw", "smooth"]
            }

            detection_artifact = {"hallucination_types": {}}
            # Store per-dialog predictions for future metric computation
            per_dialog_predictions = {ht: {variant: [] for variant in ["raw", "smooth"]} for ht in HALLUCINATION_TYPES}
            has_any = False
            for hallucination_type in HALLUCINATION_TYPES:
                detect_dialogs = hallucination_dialogs[hallucination_type]
                metrics_buffers = {
                    variant: {
                        "y_true_online": {m: [] for m in online_eval_methods},
                        "y_pred_online": {m: [] for m in online_eval_methods},
                        "is_real_pred_online": {m: [] for m in online_eval_methods},  # Track which online preds are real (not penalties)
                    }
                    for variant in ["raw", "smooth"]
                }

                for dialog in detect_dialogs:
                    dialog_id = getattr(dialog, "id", None)
                    gt = None
                    if hasattr(dialog, "annotations") and isinstance(dialog.annotations, dict):
                        gt = dialog.annotations.get("fork_assistant_turn")
                    if gt is None:
                        print(
                            f"  [WARN] Skipping dialog {dialog.id} in {hallucination_type}: "
                            "missing annotations['fork_assistant_turn']",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        gt = int(gt)
                    except (TypeError, ValueError):
                        print(
                            f"  [WARN] Skipping dialog {dialog.id} in {hallucination_type}: invalid fork_assistant_turn={gt}",
                            file=sys.stderr,
                        )
                        continue

                    try:
                        node_sequence, correct_node_probs = get_node_sequence_and_correct_node_probs(
                            dialog,
                            flow_score,
                            batch_size=args.batch_size,
                        )
                        turn_indices = get_effective_turn_indices(dialog, flow_score)
                        precomputed = {
                            "node_sequence": node_sequence,
                            "correct_node_probs": correct_node_probs,
                            "turn_indices": turn_indices,
                        }
                    except (ValueError, DialogEncodingSkipped) as exc:
                        print(f"  [WARN] Skipping dialog {dialog.id} in {hallucination_type}: {exc}", file=sys.stderr)
                        continue

                    last_turn_index = turn_indices[-1]

                    for variant in ["raw", "smooth"]:
                        pred_turns_online = {}

                        for method in online_eval_methods:
                            if variant == "smooth" and method in BASELINE_METHODS:
                                # Baselines intentionally have no smooth variant.
                                continue
                            models[method]["_variant"] = variant
                            pred_turn = models[method]["_predict_fn"](models[method], dialog, precomputed=precomputed)
                            # Online methods can now return None if no early condition is triggered
                            # (instead of falling back to last_pos)
                            pred_turns_online[method] = pred_turn

                        for method in online_eval_methods:
                            if variant == "smooth" and method in BASELINE_METHODS:
                                continue
                            if pred_turns_online[method] is not None:
                                # Only accumulate when prediction was successfully triggered
                                metrics_buffers[variant]["y_true_online"][method].append(gt)
                                metrics_buffers[variant]["y_pred_online"][method].append(pred_turns_online[method])
                                metrics_buffers[variant]["is_real_pred_online"][method].append(True)
                            else:
                                # No early prediction: penalty is last turn + 1
                                metrics_buffers[variant]["y_true_online"][method].append(gt)
                                metrics_buffers[variant]["y_pred_online"][method].append((last_turn_index + 1))
                                metrics_buffers[variant]["is_real_pred_online"][method].append(False)  # Mark as penalty
                        # Store per-dialog predictions after they are computed (bug fix: was before)
                        per_dialog_predictions[hallucination_type][variant].append({
                            "dialog_id": dialog_id,
                            "ground_truth": gt,
                            "last_turn_index": last_turn_index,
                            "pred_online": {m: pred_turns_online[m] for m in online_eval_methods if m in pred_turns_online},
                        })
                    release_cuda_memory()

                vprint(f"  Hallucination type: {hallucination_type}")
                hallucination_artifact = {"variants": {}}
                type_has_any = False
                for variant in ["raw", "smooth"]:
                    n_eval = 0
                    if online_eval_methods:
                        first_method = online_eval_methods[0]
                        n_eval = len(metrics_buffers[variant]["y_true_online"][first_method])
                    if n_eval == 0:
                        vprint(f"    [{variant}] no dialogs were evaluated.")
                        hallucination_artifact["variants"][variant] = {
                            "dialogs_evaluated": 0,
                            "offline_metrics": {},
                            "online_metrics": {},
                        }
                        continue

                    has_any = True
                    type_has_any = True
                    vprint(f"    [{variant}] dialogs evaluated: {n_eval}")

                    def split_early_delay_errors(y_true, y_pred):
                        errors = np.array(y_pred) - np.array(y_true)
                        early_mask = errors < 0
                        delay_mask = errors > 0
                        early_errors = errors[early_mask]
                        delay_errors = errors[delay_mask]
                        return early_errors, delay_errors

                    metrics_online = {}
                    for method in online_eval_methods:
                        y_true = np.array(metrics_buffers[variant]["y_true_online"][method])
                        y_pred = np.array(metrics_buffers[variant]["y_pred_online"][method])
                        is_real = np.array(metrics_buffers[variant]["is_real_pred_online"][method])
                        if y_true.size == 0:
                            metrics_online[method] = {
                                "mse": None,
                                "mae": None,
                                "emse": None,
                                "dmse": None,
                                "emae": None,
                                "dmae": None,
                                "n_early": 0,
                                "n_delay": 0,
                                "n_pred": 0,
                                "n_total": 0,
                                "pred_rate": None,
                            }
                            continue
                        abs_errors = np.abs(y_pred - y_true)
                        mse = float(np.mean(abs_errors ** 2))
                        mae = float(np.mean(abs_errors))
                        early_errors, delay_errors = split_early_delay_errors(y_true, y_pred)
                        eMSE = float(np.mean(early_errors ** 2)) if early_errors.size > 0 else None
                        dMSE = float(np.mean(delay_errors ** 2)) if delay_errors.size > 0 else None
                        eMAE = float(np.mean(np.abs(early_errors))) if early_errors.size > 0 else None
                        dMAE = float(np.mean(np.abs(delay_errors))) if delay_errors.size > 0 else None
                        n_real_preds = int(np.sum(is_real))  # Only count actual predictions, not penalties
                        metrics_online[method] = {
                            "mse": mse,
                            "mae": mae,
                            "emse": eMSE,
                            "dmse": dMSE,
                            "emae": eMAE,
                            "dmae": dMAE,
                            "n_early": int(early_errors.size),
                            "n_delay": int(delay_errors.size),
                            "n_pred": n_real_preds,  # Count only real predictions (not penalties)
                            "n_total": int(y_true.size),
                            "pred_rate": float(n_real_preds) / float(y_true.size) if y_true.size > 0 else None,
                        }
                        combined_abs_errors_pool[variant]["online"][method].extend(abs_errors.tolist())

                    if hallucination_type not in all_type_metrics:
                        all_type_metrics[hallucination_type] = {}
                    all_type_metrics[hallucination_type][variant] = {
                        "n_eval": n_eval,
                        "offline": {},
                        "online": metrics_online,
                    }

                    hallucination_artifact["variants"][variant] = {
                        "dialogs_evaluated": n_eval,
                        "offline_metrics": {},
                        "online_metrics": metrics_online,
                        "per_dialog": per_dialog_predictions[hallucination_type][variant],
                    }

                if not type_has_any:
                    vprint(f"    No dialogs were evaluated for {hallucination_type}.")
                detection_artifact["hallucination_types"][hallucination_type] = hallucination_artifact

            # Compute combined MAE across happy + all hallucination types
            combined_metrics = {}
            for variant in ["raw", "smooth"]:
                combined_metrics[variant] = {"offline": {}, "online": {}}
                for method in online_eval_methods:
                    errs = combined_abs_errors_pool[variant]["online"][method]
                    combined_metrics[variant]["online"][method] = {"mae": float(np.mean(errs)) if errs else None}

            detection_artifact["happy"] = {
                "variants": {
                    variant: {
                        "n_eval": happy_metrics[variant]["n_eval"],
                        "offline_metrics": happy_metrics[variant]["offline"],
                        "online_metrics": happy_metrics[variant]["online"],
                        "per_dialog": happy_per_dialog[variant],
                    }
                    for variant in ["raw", "smooth"]
                }
            }
            detection_artifact["combined"] = {
                "variants": {
                    variant: {
                        "offline_metrics": combined_metrics[variant]["offline"],
                        "online_metrics": combined_metrics[variant]["online"],
                    }
                    for variant in ["raw", "smooth"]
                }
            }

            if not has_any:
                print("  No hallucination dialogs were evaluated (missing or invalid fork_assistant_turn).")
                all_type_metrics = {}
            else:
                print_summary_table(
                    hallucination_types=HALLUCINATION_TYPES,
                    all_type_metrics=all_type_metrics,
                    methods=online_eval_methods,
                    happy_metrics=happy_metrics,
                )

        if output_path:
            total_runtime_seconds = time.perf_counter() - task_start_time
            output_payload = {
                "config": {
                    "task_root": task_root,
                    "task_name": task_name,
                    "task_parent": args.task_root,
                    "in_sample": args.in_sample,
                    "originals_subdir": ORIGINALS_SUBDIR,
                    "hallucination_types": HALLUCINATION_TYPES,
                    "n": args.n,
                    "lm_model": lm_model_used,   # was: LM_MODEL_TYPE
                    "lm_model_requested": LM_MODEL_TYPE,
                    "ai_speaker": args.ai_speaker,
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "node_prune_threshold": NODE_PRUNE_THRESHOLD,
                    "edge_prune_threshold": EDGE_PRUNE_THRESHOLD,
                    "offroad_quantile": OFFROAD_QUANTILE,
                    "jump_quantile": JUMP_QUANTILE,
                    "total_runtime_seconds": float(total_runtime_seconds),
                },
                "thresholds": thresholds_by_variant,
                "detection": detection_artifact,
            }
            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            with open(output_path, "w") as fh:
                json.dump(output_payload, fh, indent=2)
            vprint(f"  Results saved to: {output_path}")

        vprint(f"Task {task_name} runtime: {timedelta(seconds=round(time.perf_counter() - task_start_time))}")
        return (all_type_metrics if all_type_metrics else None), online_eval_methods, happy_metrics
    finally:
        if temp_output_path and os.path.isdir(temp_output_path):
            shutil.rmtree(temp_output_path, ignore_errors=True)
        if os.path.isdir(temp_output_root) and not os.listdir(temp_output_root):
            os.rmdir(temp_output_root)
        # Each task's DialogFlowPPL loads its own encoder onto the GPU. Without
        # explicitly moving its weights off-GPU here, the next task's encoder can
        # end up loaded while this one is still resident (observed: two Qwen3-
        # Embedding-8B copies, ~16GB each in fp16, together OOMing a 32GB GPU),
        # since dropping the last Python reference alone doesn't guarantee timely
        # CUDA memory release across the many places (e.g. `models[method]`) that
        # end up holding a reference to the same `flow_score`/encoder object.
        if flow_score is not None:
            encoder = getattr(flow_score, "encoder", None)
            if encoder is not None:
                if hasattr(encoder, "to"):
                    try:
                        encoder.to("cpu")
                    except Exception:
                        pass
                # Relocating frees the GPU copy even if some other alias (e.g.
                # `models[method]["flow_score"]`) still references the same
                # object; dropping our own reference then avoids also leaving
                # the now-CPU-resident copy sitting in host RAM until return.
                flow_score.encoder = None
                del encoder
        release_cuda_memory()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train/evaluate each task under a parent folder and report per-task + global averages."
    )
    p.add_argument(
        "--task-root", "-d", default="datasets/dialfork/",
        help="Parent folder containing one subfolder per task."
    )
    p.add_argument(
        "--tasks", "-t", nargs="+", default=None,
        help="Optional list of task folder names under --task-root. If omitted, all immediate subfolders are used."
    )
    p.add_argument(
        "--n", "-n", type=int, default=3,
        help="N-gram order (default: 3, as in the paper)."
    )
    p.add_argument(
        "--ai-speaker", "-a", default=None,
        help="If set, restrict graph construction and scoring to turns from this "
             "AI/system speaker only (name is case-insensitive)."
    )
    p.add_argument(
        "--model", "-m", default="sergioburdisso/dialog2flow-joint-dse-base",
        help="Sentence-transformer model for node embeddings "
             "(overrides dialog2flow default)."
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size used by the sentence encoder (default: 16)."
    )
    p.add_argument(
        "--output", "-o", default="results/task3",
        help="Root folder for results. Outputs are saved to "
             "<output>/<model>/lm-{n}-gram[_ai_only]/<task>.json plus "
             "<output>/<model>/lm-{n}-gram[_ai_only]/AVERAGE.json."
    )
    p.add_argument(
        "--in-sample", action="store_true",
        help="Learn from and evaluate on all conformant dialogues, ignoring the Task 3 split "
             "(<task-root>/task3_splits.csv). Not the benchmark protocol; for comparison only."
    )
    p.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)."
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed progress during computation (default: off)."
    )
    return p.parse_args()


def print_run_settings(args, task_names: List[str]) -> None:
    """Print the effective run configuration once at startup."""
    requested_tasks = ", ".join(args.tasks) if args.tasks else "ALL"
    resolved_tasks = ", ".join(task_names)

    print("\n=== Run Settings ===")
    print(f"Task root: {args.task_root}")
    print(f"Requested tasks: {requested_tasks}")
    print(f"Resolved tasks ({len(task_names)}): {resolved_tasks}")
    print(f"Embedding model: {args.model}")
    print(f"LM order: {args.n}")
    print(f"LM model type: {LM_MODEL_TYPE}")
    print(f"AI speaker filter: {args.ai_speaker if args.ai_speaker else 'disabled'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output root: {args.output if args.output else 'disabled'}")
    print(f"Task 3 split: {'disabled (in-sample)' if args.in_sample else 'train -> test'}")
    print(f"Seed: {args.seed}")
    print(f"Verbose: {args.verbose}")
    print(f"Node prune threshold: {NODE_PRUNE_THRESHOLD}")
    print(f"Edge prune threshold: {EDGE_PRUNE_THRESHOLD}")
    print(f"Off-road quantile: {OFFROAD_QUANTILE}")
    print(f"Jump quantile: {JUMP_QUANTILE}")


def main():
    global _verbose
    start_time = time.perf_counter()
    args = parse_args()
    _verbose = args.verbose

    task_names = resolve_task_names(args.task_root, args.tasks)
    print_run_settings(args, task_names)
    all_tasks_metrics = []
    all_happy_metrics = []
    methods_for_final = ["first_below", "max_jump", "first_turn", "last_turn", "random"]

    for task_name in task_names:
        print(f"\n=== Task: {task_name} ===")
        task_metrics, methods, task_happy_metrics = run_single_task(args, task_name)
        if methods:
            methods_for_final = methods
        if task_metrics:
            all_tasks_metrics.append(task_metrics)
        if task_happy_metrics:
            all_happy_metrics.append(task_happy_metrics)

    avg_table_text = None
    if all_tasks_metrics:
        print("\n=== Average Across Tasks ===")
        avg_metrics, avg_happy_metrics = aggregate_metrics_across_tasks(
            all_tasks_metrics, all_happy_metrics, methods_for_final
        )
        avg_benchmark_text = render_benchmark_summary_table(
            hallucination_types=HALLUCINATION_TYPES,
            all_type_metrics=avg_metrics,
            methods=methods_for_final,
            happy_metrics=avg_happy_metrics if avg_happy_metrics else None,
        )
        avg_extended_text = render_summary_table(
            hallucination_types=HALLUCINATION_TYPES,
            all_type_metrics=avg_metrics,
            methods=methods_for_final,
            happy_metrics=avg_happy_metrics if avg_happy_metrics else None,
        )
        avg_table_text = avg_benchmark_text + "\n\n" + avg_extended_text
        print("\n" + avg_table_text)
    else:
        avg_metrics = None
        avg_happy_metrics = None
        print("\nNo task produced evaluable metrics.")

    total_runtime_seconds = time.perf_counter() - start_time

    global_output_path = build_result_path(args, "AVERAGE.json")
    if global_output_path and avg_metrics is not None:
        global_payload = {
            "config": {
                "task_parent": args.task_root,
                "task_names": task_names,
                "in_sample": args.in_sample,
                "originals_subdir": ORIGINALS_SUBDIR,
                "hallucination_types": HALLUCINATION_TYPES,
                "n": args.n,
                "lm_model": LM_MODEL_TYPE,
                "ai_speaker": args.ai_speaker,
                "model": args.model,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "node_prune_threshold": NODE_PRUNE_THRESHOLD,
                "edge_prune_threshold": EDGE_PRUNE_THRESHOLD,
                "offroad_quantile": OFFROAD_QUANTILE,
                "jump_quantile": JUMP_QUANTILE,
                "total_runtime_seconds": float(total_runtime_seconds),
                "is_global_average": True,
            },
            "detection": metrics_to_detection_artifact(avg_metrics),
        }
        with open(global_output_path, "w") as fh:
            json.dump(global_payload, fh, indent=2)
        vprint(f"  Global average results saved to: {global_output_path}")

    average_txt_path = build_result_path(args, "AVERAGE.txt")
    if average_txt_path and avg_metrics is not None and avg_table_text is not None:
        with open(average_txt_path, "w") as fh:
            fh.write("Average Across Tasks\n")
            fh.write(avg_table_text)
            fh.write("\n")
        vprint(f"  Global average table saved to: {average_txt_path}")

    print(f"\nDone. Total computation time: {timedelta(seconds=round(total_runtime_seconds))}")


if __name__ == "__main__":
    main()
