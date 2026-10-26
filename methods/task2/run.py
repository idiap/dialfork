# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

import os
import json
import time
import gc
import torch
import random
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

from sdialog import Dialog
from sdialog.datasets import STAR
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, set_seed
from sdialog.evaluation import SentenceTransformerDialogEmbedder

# import sdialog
# sdialog.config.cache(True)

STAR.set_path("datasets/STAR/")
PATH_HALLUCINATED_DATASET = "datasets/dialfork"
PATH_OUTPUT = "results/task2"
STAR_AS_REFERENCE = False
DEFAULT_DISTANCE_METHOD = "mean-turn-emb"
DEFAULT_BERTSCORE_MODEL = "allenai/longformer-base-4096"
DEFAULT_BERTSCORE_BERT_MODEL = "bert-base-uncased"


def _fork_segment_order_key(name: str):
    """Sort fork segments by segment index in descending order (4,3,2,1)."""
    if name.startswith("fork_at_segment_"):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return -1
    return -1


def _build_dialog_index(dialogs):
    """Create a dialog-id -> Dialog lookup map."""
    output = {}
    for dialog in dialogs:
        output[str(dialog.id)] = dialog
    return output


def _safe_cosine_distance(v1, v2):
    """Cosine distance with a fallback for degenerate vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 1.0
    similarity = float(np.dot(v1, v2) / (norm_v1 * norm_v2))
    distance = 1.0 - similarity
    if np.isnan(distance):
        return 1.0
    return distance


def _position_accuracy(ground_truth_rank, predicted_rank):
    """Fraction of positions with the correct segment label."""
    if not ground_truth_rank:
        return 0.0
    matches = sum(g == p for g, p in zip(ground_truth_rank, predicted_rank))
    return float(matches / len(ground_truth_rank))


def _kendall_tau_from_distances(ground_truth_rank, distances):
    """Kendall's tau between the ground-truth order (most similar first) and the
    order induced by `distances`, with tied distances counted as neither
    concordant nor discordant (the expected tau under random tie-breaking).
    Unlike ranking with a stable sort, ties are never resolved in favor of the
    ground truth, which matters for integer-valued controls such as turn counts."""
    n = len(ground_truth_rank)
    if n < 2:
        return 0.0
    score = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = distances[ground_truth_rank[i]], distances[ground_truth_rank[j]]
            score += (a < b) - (a > b)
    return float(score / (n * (n - 1) / 2))


def _kendall_tau_from_orders(ground_truth_rank, predicted_rank):
    """Kendall's tau for permutations without ties."""
    n = len(ground_truth_rank)
    if n < 2:
        return 0.0

    gt_pos = {name: i for i, name in enumerate(ground_truth_rank)}
    pred = [gt_pos[name] for name in predicted_rank]

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred[i] < pred[j]:
                concordant += 1
            else:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    return float((concordant - discordant) / total_pairs)


def _dialog_turn_texts(dialog: Dialog):
    """Return non-empty turn texts for a dialog."""
    if dialog.turns is None:
        return []
    return [t.text.strip() for t in dialog.turns if t.text is not None and t.text.strip()]


def _dialog_as_string(dialog: Dialog):
    """Serialize a dialog as a single string with speaker tags."""
    if dialog.turns is None:
        return ""
    parts = []
    for t in dialog.turns:
        if t.text is None:
            continue
        text = t.text.strip()
        if not text:
            continue
        speaker = (t.speaker or "UNK").strip()
        parts.append(f"{speaker}: {text}")
    return "\n".join(parts)


def _safe_f1(precision, recall):
    """Compute harmonic mean with numeric safeguards."""
    if precision + recall == 0.0:
        return 0.0
    value = float((2.0 * precision * recall) / (precision + recall))
    if np.isnan(value):
        return 0.0
    return value


def _turn_sequence_bertscore_distance(turn_embs_a, turn_embs_b):
    """BERTScore-like distance over two turn-embedding sequences."""
    if turn_embs_a.shape[0] == 0 and turn_embs_b.shape[0] == 0:
        return 0.0
    if turn_embs_a.shape[0] == 0 or turn_embs_b.shape[0] == 0:
        return 1.0

    # Embeddings are normalized, so dot-product equals cosine similarity.
    sim = turn_embs_a @ turn_embs_b.T
    precision = float(np.max(sim, axis=1).mean())
    recall = float(np.max(sim, axis=0).mean())
    f1 = _safe_f1(precision, recall)
    return float(1.0 - f1)


def _dtw_distance_from_cost_matrix(cost_matrix):
    """Compute normalized DTW distance from a local-cost matrix."""
    n, m = cost_matrix.shape
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = (
                (dp[i - 1, j], steps[i - 1, j]),
                (dp[i, j - 1], steps[i, j - 1]),
                (dp[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            best_cost, best_steps = min(candidates, key=lambda x: x[0])
            dp[i, j] = cost_matrix[i - 1, j - 1] + best_cost
            steps[i, j] = best_steps + 1

    path_len = int(steps[n, m])
    if path_len == 0:
        return 1.0
    return float(dp[n, m] / path_len)


def _turn_sequence_dtw_cosine_distance(turn_embs_a, turn_embs_b):
    """DTW distance over turn embeddings using cosine local costs."""
    if turn_embs_a.shape[0] == 0 and turn_embs_b.shape[0] == 0:
        return 0.0
    if turn_embs_a.shape[0] == 0 or turn_embs_b.shape[0] == 0:
        return 1.0

    # Embeddings are normalized, so dot-product equals cosine similarity.
    sim = turn_embs_a @ turn_embs_b.T
    local_costs = 1.0 - sim
    return _dtw_distance_from_cost_matrix(local_costs)


def _token_sequence_bertscore_distance(token_embs_a, token_embs_b):
    """Vanilla BERTScore-style distance over token embeddings."""
    if token_embs_a.shape[0] == 0 and token_embs_b.shape[0] == 0:
        return 0.0
    if token_embs_a.shape[0] == 0 or token_embs_b.shape[0] == 0:
        return 1.0

    sim = token_embs_a @ token_embs_b.T
    precision = float(np.max(sim, axis=1).mean())
    recall = float(np.max(sim, axis=0).mean())
    f1 = _safe_f1(precision, recall)
    return float(1.0 - f1)


def _safe_tokenizer_max_length(tokenizer, model, fallback=4096):
    """Return a bounded tokenizer max_length to avoid overflow errors."""
    max_pos = int(getattr(model.config, "max_position_embeddings", fallback) or fallback)
    tok_max = getattr(tokenizer, "model_max_length", None)

    # Some tokenizers use huge sentinel values for "unbounded" max length.
    if tok_max is None:
        return max_pos
    if not isinstance(tok_max, int):
        try:
            tok_max = int(tok_max)
        except Exception:
            return max_pos
    if tok_max <= 0 or tok_max > 1_000_000:
        return max_pos
    return min(tok_max, max_pos)


def _lru_cache_get(cache, key):
    """Fetch from OrderedDict cache and mark as recently used."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _lru_cache_put(cache, key, value, max_entries):
    """Insert into OrderedDict cache and evict least-recently-used if capped."""
    cache[key] = value
    cache.move_to_end(key)
    if max_entries > 0 and len(cache) > max_entries:
        cache.popitem(last=False)


def _merged_encode_kwargs(base_kwargs, overrides=None):
    """Merge encode kwargs while ignoring None override values."""
    merged = dict(base_kwargs)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


def _model_supported_extra_encode_kwargs(model):
    """Return model-level encode kwargs accepted by SentenceTransformer internals."""
    get_kwargs = getattr(model, "get_model_kwargs", None)
    if not callable(get_kwargs):
        return set()
    try:
        values = get_kwargs()
    except Exception:
        return set()
    return {str(v) for v in values}


def _sanitize_encode_kwargs_for_model(model, encode_kwargs):
    """Keep only model-supported extra kwargs to avoid SentenceTransformer errors."""
    supported = _model_supported_extra_encode_kwargs(model)
    if not supported:
        # If model doesn't expose extra kwargs, keep defaults only.
        return {
            k: v for k, v in encode_kwargs.items()
            if k in {"batch_size", "prompt_name"}
        }

    safe = {}
    for key, value in encode_kwargs.items():
        if key in {"batch_size", "prompt_name"}:
            safe[key] = value
        elif key in supported:
            safe[key] = value
    return safe


def _force_max_seq_length(model, max_length):
    """Override a SentenceTransformer's truncation window directly.

    `encode(..., max_length=...)` is silently dropped by
    `_sanitize_encode_kwargs_for_model()` for models whose SentenceTransformer
    module doesn't expose `get_model_kwargs()` (older custom checkpoints, e.g.
    dialog2flow, dse-bert-base). Those models then fall back to whatever
    `max_seq_length` is baked into their `sentence_bert_config.json` (as low
    as 64 tokens for dialog2flow, vs. 256-8192 for other models here), which
    silently truncates full-dialog-emb to a couple of turns and confounds the
    model comparison. Setting the attribute directly bypasses that kwarg path
    and applies uniformly to every SentenceTransformer-backed model.
    """
    if max_length is not None and max_length > 0:
        model.max_seq_length = int(max_length)


def build_dialog2dialog_distance(
    distance_method,
    embedding_model,
    encode_batch_size=8,
    max_cache_entries=1000,
    encode_truncation=True,
    encode_max_length=512,
):
    """Return a callable: (dialog_a, dialog_b) -> distance."""
    # Naive length-only controls: no embedding model involved, so build and return
    # before any SentenceTransformer machinery is touched.
    if distance_method == "turn-count-diff":
        def _distance(dialog_a, dialog_b, **_ignored):
            return float(abs(len(_dialog_turn_texts(dialog_a)) - len(_dialog_turn_texts(dialog_b))))
        return _distance

    if distance_method == "shared-prefix":
        # Construction-cue control: number of opening turns two dialogs share verbatim
        # (negated, so a longer shared prefix means a smaller distance). It exploits how
        # DialFork is built and has no counterpart for real dialogs; reported only to
        # show the proposed metrics do not reduce to it.
        def _distance(dialog_a, dialog_b, **_ignored):
            shared = 0
            for ta, tb in zip(_dialog_turn_texts(dialog_a), _dialog_turn_texts(dialog_b)):
                if ta.strip() != tb.strip():
                    break
                shared += 1
            return float(-shared)
        return _distance

    if distance_method == "word-count-diff":
        def _word_count(dialog):
            return sum(len(t.split()) for t in _dialog_turn_texts(dialog))

        def _distance(dialog_a, dialog_b, **_ignored):
            return float(abs(_word_count(dialog_a) - _word_count(dialog_b)))
        return _distance

    encode_query_kwargs = {}
    encode_document_kwargs = {}
    if "qwen3-embedding" in embedding_model.lower():
        encode_query_kwargs["prompt_name"] = "query"

    if encode_batch_size is not None and encode_batch_size > 0:
        # SentenceTransformerDialogEmbedder routes kwargs directly to model.encode().
        encode_query_kwargs["batch_size"] = int(encode_batch_size)
        encode_document_kwargs["batch_size"] = int(encode_batch_size)

    if encode_truncation:
        encode_query_kwargs["truncation"] = True
        encode_document_kwargs["truncation"] = True
    if encode_max_length is not None and encode_max_length > 0:
        encode_query_kwargs["max_length"] = int(encode_max_length)
        encode_document_kwargs["max_length"] = int(encode_max_length)

    if distance_method == "mean-turn-emb":
        embedder = SentenceTransformerDialogEmbedder(
            model_name=embedding_model,
            mean=True,
            verbose=False,
        )
        _force_max_seq_length(embedder.model, encode_max_length)

        def _distance(dialog_a, dialog_b, query_encode_overrides=None, document_encode_overrides=None):
            query_kwargs = _sanitize_encode_kwargs_for_model(
                embedder.model,
                _merged_encode_kwargs(encode_query_kwargs, query_encode_overrides),
            )
            doc_kwargs = _sanitize_encode_kwargs_for_model(
                embedder.model,
                _merged_encode_kwargs(encode_document_kwargs, document_encode_overrides),
            )
            emb_a = embedder(dialog_a, **query_kwargs)
            emb_b = embedder(dialog_b, **doc_kwargs)
            return _safe_cosine_distance(emb_a, emb_b)

        return _distance

    if distance_method == "full-dialog-emb":
        embedder = SentenceTransformerDialogEmbedder(
            model_name=embedding_model,
            mean=False,
            verbose=False,
        )
        _force_max_seq_length(embedder.model, encode_max_length)

        def _distance(dialog_a, dialog_b, query_encode_overrides=None, document_encode_overrides=None):
            query_kwargs = _sanitize_encode_kwargs_for_model(
                embedder.model,
                _merged_encode_kwargs(encode_query_kwargs, query_encode_overrides),
            )
            doc_kwargs = _sanitize_encode_kwargs_for_model(
                embedder.model,
                _merged_encode_kwargs(encode_document_kwargs, document_encode_overrides),
            )
            emb_a = embedder(dialog_a, **query_kwargs)
            emb_b = embedder(dialog_b, **doc_kwargs)
            return _safe_cosine_distance(emb_a, emb_b)

        return _distance

    if distance_method == "flowscore":
        model = SentenceTransformer(
            embedding_model,
            model_kwargs={"use_safetensors": True}
        )
        _force_max_seq_length(model, encode_max_length)
        turns_emb_cache = OrderedDict()

        def _get_turn_embeddings(dialog, **encode_kwargs):
            turns = _dialog_turn_texts(dialog)
            cache_key = (tuple(turns), tuple(sorted(encode_kwargs.items())))
            cached = _lru_cache_get(turns_emb_cache, cache_key)
            if cached is not None:
                return cached

            if len(turns) == 0:
                dim = int(model.get_sentence_embedding_dimension())
                embs = np.zeros((0, dim), dtype=np.float32)
            else:
                local_encode_kwargs = dict(encode_kwargs)
                local_batch_size = int(local_encode_kwargs.pop("batch_size", encode_batch_size))
                embs = model.encode(
                    turns,
                    batch_size=max(1, local_batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    **local_encode_kwargs
                )
            _lru_cache_put(turns_emb_cache, cache_key, embs, max_cache_entries)
            return embs

        def _distance(dialog_a, dialog_b, query_encode_overrides=None, document_encode_overrides=None):
            query_kwargs = _sanitize_encode_kwargs_for_model(
                model,
                _merged_encode_kwargs(encode_query_kwargs, query_encode_overrides),
            )
            doc_kwargs = _sanitize_encode_kwargs_for_model(
                model,
                _merged_encode_kwargs(encode_document_kwargs, document_encode_overrides),
            )
            embs_a = _get_turn_embeddings(dialog_a, **query_kwargs)
            embs_b = _get_turn_embeddings(dialog_b, **doc_kwargs)
            return _turn_sequence_bertscore_distance(embs_a, embs_b)

        return _distance

    if distance_method == "flowdtw":
        model = SentenceTransformer(
            embedding_model,
            model_kwargs={"use_safetensors": True}
        )
        _force_max_seq_length(model, encode_max_length)
        turns_emb_cache = OrderedDict()

        def _get_turn_embeddings(dialog, **encode_kwargs):
            turns = _dialog_turn_texts(dialog)
            cache_key = (tuple(turns), tuple(sorted(encode_kwargs.items())))
            cached = _lru_cache_get(turns_emb_cache, cache_key)
            if cached is not None:
                return cached

            if len(turns) == 0:
                dim = int(model.get_sentence_embedding_dimension())
                embs = np.zeros((0, dim), dtype=np.float32)
            else:
                local_encode_kwargs = dict(encode_kwargs)
                local_batch_size = int(local_encode_kwargs.pop("batch_size", encode_batch_size))
                embs = model.encode(
                    turns,
                    batch_size=max(1, local_batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    **local_encode_kwargs
                )
            _lru_cache_put(turns_emb_cache, cache_key, embs, max_cache_entries)
            return embs

        def _distance(dialog_a, dialog_b, query_encode_overrides=None, document_encode_overrides=None):
            query_kwargs = _sanitize_encode_kwargs_for_model(
                model,
                _merged_encode_kwargs(encode_query_kwargs, query_encode_overrides),
            )
            doc_kwargs = _sanitize_encode_kwargs_for_model(
                model,
                _merged_encode_kwargs(encode_document_kwargs, document_encode_overrides),
            )
            embs_a = _get_turn_embeddings(dialog_a, **query_kwargs)
            embs_b = _get_turn_embeddings(dialog_b, **doc_kwargs)
            return _turn_sequence_dtw_cosine_distance(embs_a, embs_b)

        return _distance

    if distance_method in {"bertscore", "bertscore-bert"}:
        if distance_method == "bertscore":
            bertscore_model = DEFAULT_BERTSCORE_MODEL
            max_length_cap = 4096
        else:
            bertscore_model = DEFAULT_BERTSCORE_BERT_MODEL
            max_length_cap = 512

        tokenizer = AutoTokenizer.from_pretrained(bertscore_model)
        model = AutoModel.from_pretrained(bertscore_model)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        max_length = min(
            _safe_tokenizer_max_length(tokenizer, model, fallback=max_length_cap),
            max_length_cap,
        )
        token_emb_cache = {}

        def _get_token_embeddings(dialog):
            text = _dialog_as_string(dialog)
            cache_key = text
            if cache_key in token_emb_cache:
                return token_emb_cache[cache_key]

            if not text:
                hidden = int(getattr(model.config, "hidden_size", 768))
                embs = np.zeros((0, hidden), dtype=np.float32)
                token_emb_cache[cache_key] = embs
                return embs

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            # encoded["global_attention_mask"] = global_attention_mask
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)
                last_hidden = outputs.last_hidden_state[0]  # [seq_len, hidden]

            attention_mask = encoded.get("attention_mask", None)
            if attention_mask is not None:
                valid = attention_mask[0].bool()
                last_hidden = last_hidden[valid]

            input_ids = encoded.get("input_ids", None)
            if input_ids is not None:
                ids = input_ids[0]
                special = torch.zeros_like(ids, dtype=torch.bool)
                for sid in tokenizer.all_special_ids:
                    special = special | (ids == sid)
                keep = ~special
                last_hidden = last_hidden[keep]

            if last_hidden.shape[0] == 0:
                hidden = int(getattr(model.config, "hidden_size", 768))
                embs = np.zeros((0, hidden), dtype=np.float32)
            else:
                last_hidden = torch.nn.functional.normalize(last_hidden, p=2, dim=1)
                embs = last_hidden.detach().cpu().numpy().astype(np.float32)

            token_emb_cache[cache_key] = embs
            return embs

        def _distance(dialog_a, dialog_b, query_encode_overrides=None, document_encode_overrides=None):
            token_embs_a = _get_token_embeddings(dialog_a)
            token_embs_b = _get_token_embeddings(dialog_b)
            return _token_sequence_bertscore_distance(token_embs_a, token_embs_b)

        return _distance

    raise ValueError(
        f"Unsupported distance method: {distance_method}. "
        "Add your custom method inside build_dialog2dialog_distance()."
    )


def evaluate_hallucination_type(
    dialog2dialog_distance,
    originals,
    hallucinations_by_fork,
    ground_truth_rank,
    cuda_empty_cache_every=0,
    oom_retry_batch_size=1,
    oom_retry_max_length=256,
    oom_fail_fast=False,
):
    """Evaluate one hallucination type by ranking fork segments per original dialog."""
    results = {}

    original_index = _build_dialog_index(originals)
    fork_indexes = {fork: _build_dialog_index(dialogs)
                    for fork, dialogs in hallucinations_by_fork.items()}

    available_forks = [fork for fork in ground_truth_rank if fork in fork_indexes]
    if not available_forks:
        summary = {
            "num_dialogs": 0,
            "exact_match_rate": 0.0,
            "position_accuracy_mean": 0.0,
            "kendall_tau_mean": 0.0,
        }
        diagnostics = {
            "oom_errors": 0,
            "oom_retry_success": 0,
            "oom_skipped_dialogs": 0,
        }
        return {}, summary, diagnostics

    exact_matches = []
    position_accuracies = []
    kendall_taus = []
    oom_total = 0
    oom_retry_success = 0
    oom_skipped_dialogs = 0

    retry_overrides = {
        "batch_size": max(1, int(oom_retry_batch_size)),
        "truncation": True,
        "max_length": max(1, int(oom_retry_max_length)),
    }

    processed_dialogs = 0
    for dialog_id, original_dialog in original_index.items():
        if any(dialog_id not in fork_indexes[fork] for fork in available_forks):
            continue

        distances = {}
        dialog_failed = False
        for fork in available_forks:
            candidate_dialog = fork_indexes[fork][dialog_id]
            try:
                distances[fork] = float(dialog2dialog_distance(original_dialog, candidate_dialog))
            except torch.OutOfMemoryError:
                oom_total += 1
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                can_retry = oom_retry_batch_size is not None and oom_retry_batch_size > 0
                if can_retry:
                    try:
                        distances[fork] = float(
                            dialog2dialog_distance(
                                original_dialog,
                                candidate_dialog,
                                query_encode_overrides=retry_overrides,
                                document_encode_overrides=retry_overrides,
                            )
                        )
                        oom_retry_success += 1
                        continue
                    except torch.OutOfMemoryError:
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()

                if oom_fail_fast:
                    raise

                dialog_failed = True
                break

        if dialog_failed:
            oom_skipped_dialogs += 1
            continue

        predicted_rank = sorted(available_forks, key=lambda x: distances[x])

        pos_acc = _position_accuracy(available_forks, predicted_rank)
        tau = _kendall_tau_from_distances(available_forks, distances)
        exact = float(predicted_rank == available_forks)

        exact_matches.append(exact)
        position_accuracies.append(pos_acc)
        kendall_taus.append(tau)

        results[dialog_id] = {
            "distances": distances,
            "predicted_rank": predicted_rank,
            "ground_truth_rank": available_forks,
            "position_accuracy": pos_acc,
            "kendall_tau": tau,
            "exact_match": exact,
        }

        processed_dialogs += 1
        if (
            cuda_empty_cache_every > 0
            and torch.cuda.is_available()
            and processed_dialogs % cuda_empty_cache_every == 0
        ):
            gc.collect()
            torch.cuda.empty_cache()

    summary = {
        "num_dialogs": int(len(exact_matches)),
        "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "position_accuracy_mean": float(np.mean(position_accuracies)) if position_accuracies else 0.0,
        "kendall_tau_mean": float(np.mean(kendall_taus)) if kendall_taus else 0.0,
    }
    diagnostics = {
        "oom_errors": int(oom_total),
        "oom_retry_success": int(oom_retry_success),
        "oom_skipped_dialogs": int(oom_skipped_dialogs),
    }

    return results, summary, diagnostics

def load_task_hallucinated_dialogs(task_name, hallucination_type:str = None, path_dataset: str = PATH_HALLUCINATED_DATASET):
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


def load_all_task_summaries_from_disk(experiment_output_path):
    """Scan `experiment_output_path` for per-task `results_d2d_summary.json`
    files and load them into a {task -> summaries} dict.

    Each Slurm job submitted by methods/task2/run_all.sh only processes a single task, so
    the in-process `all_task_summaries` dict only ever reflects that one
    task. Rebuilding it from disk instead ensures the "overall" aggregate
    files reflect every task that has completed so far, regardless of which
    job (re)writes them last.
    """
    all_task_summaries = {}
    if not os.path.isdir(experiment_output_path):
        return all_task_summaries
    for entry in sorted(os.listdir(experiment_output_path)):
        task_dir = os.path.join(experiment_output_path, entry)
        summary_json_path = os.path.join(task_dir, "results_d2d_summary.json")
        if os.path.isdir(task_dir) and os.path.isfile(summary_json_path):
            try:
                with open(summary_json_path) as f:
                    all_task_summaries[entry] = json.load(f)
            except Exception:
                continue
    return all_task_summaries


if __name__ == "__main__":
    run_start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Evaluate dialog to dialog distance metrics")
    parser.add_argument("tasks", nargs="*", type=str.lower,
                        help="Task name(s) to evaluate (e.g., 'apartment_schedule'). If omitted, all tasks are processed.")
    parser.add_argument("-i", "--input-path", type=str, default=PATH_HALLUCINATED_DATASET,
                        help="Path to hallucinated STAR dataset")
    parser.add_argument("-o", "--output-path", type=str, default=PATH_OUTPUT,
                        help="Base output path for experiment results")
    parser.add_argument("-m", "--embedding-model", type=str, default="sentence-transformers/LaBSE",
                        help="SentenceTransformer model used for per-turn averaged dialog embeddings")
    parser.add_argument("-dm", "--distance-method", type=str, default=DEFAULT_DISTANCE_METHOD,
                        help=("Dialog-to-dialog distance method. "
                        "Options: mean-turn-emb, full-dialog-emb, flowscore, flowdtw, bertscore, bertscore-bert, "
                        "turn-count-diff, word-count-diff, shared-prefix. "
                        "The last three are controls that need no embedding model "
                        "(-m/--embedding-model is ignored for them). "
                        "Default: mean-turn-emb"))
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--encode-batch-size", type=int, default=16,
                        help="Batch size for SentenceTransformer.encode() (default: 16)")
    parser.add_argument("--encode-max-length", type=int, default=512,
                        help="Max token length for SentenceTransformer.encode(), enforced uniformly across all "
                             "models via model.max_seq_length so results are comparable under the same context "
                             "budget regardless of a model's own default (default: 512)")
    parser.add_argument("--no-encode-truncation", action="store_true",
                        help="Disable truncation in SentenceTransformer.encode() kwargs")
    parser.add_argument("--oom-truncation-only", dest="oom_truncation_only", action="store_true", default=True,
                        help="Do not truncate on first pass; apply truncation only in OOM retry overrides (default: enabled)")
    parser.add_argument("--no-oom-truncation-only", dest="oom_truncation_only", action="store_false",
                        help="Disable OOM-only truncation policy and use base truncation settings")
    parser.add_argument("--max-cache-entries", type=int, default=1000,
                        help="Maximum number of cached turn-embedding entries per flow method (default: 1000, 0 disables cap)")
    parser.add_argument("--cuda-empty-cache-every", type=int, default=50,
                        help="Run gc.collect() + torch.cuda.empty_cache() every N processed dialogs (default: 50, 0 disables)")
    parser.add_argument("--oom-retry-batch-size", type=int, default=1,
                        help="Retry batch size after CUDA OOM; <=0 disables retry (default: 1)")
    parser.add_argument("--oom-retry-max-length", type=int, default=256,
                        help="Retry max token length after CUDA OOM (default: 256)")
    parser.add_argument("--oom-fail-fast", action="store_true",
                        help="Raise immediately on OOM instead of skipping the failing dialog after retry")

    args = parser.parse_args()

    # Resolve task list
    if args.tasks:
        task_list = args.tasks
    else:
        task_list = sorted(
            entry for entry in os.listdir(args.input_path)
            if os.path.isdir(os.path.join(args.input_path, entry))
        )
        print(f"No tasks specified. Processing all {len(task_list)} tasks: {task_list}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    print(f"Random seed set to: {args.seed}")
    print(f"Embedding model: {args.embedding_model}")
    if args.distance_method == "bertscore":
        print(f"BERTScore model (fixed): {DEFAULT_BERTSCORE_MODEL}")
    elif args.distance_method == "bertscore-bert":
        print(f"BERTScore model (fixed): {DEFAULT_BERTSCORE_BERT_MODEL}")
    print(f"Distance method: {args.distance_method}")
    print(f"Encode batch size: {args.encode_batch_size}")
    print(f"Encode max length: {args.encode_max_length}")
    if args.oom_truncation_only:
        print("Encode truncation policy: OOM retry only")
    else:
        print(f"Encode truncation: {not args.no_encode_truncation}")
    print(f"Max cache entries: {args.max_cache_entries}")
    print(f"CUDA cache cleanup every: {args.cuda_empty_cache_every}")
    print(f"OOM retry batch size: {args.oom_retry_batch_size}")
    print(f"OOM retry max length: {args.oom_retry_max_length}")
    print(f"OOM fail fast: {args.oom_fail_fast}")

    model_name = args.embedding_model.split("/")[-1]
    if args.distance_method in {"turn-count-diff", "word-count-diff", "shared-prefix"}:
        # No embedding model is involved; use a fixed placeholder "model" level so the
        # folder layout still matches the <method>/<model>/<task> pattern that
        # methods/task2/tables.py globs for (results_d2d_overall_by_ht.json two levels below ROOT).
        experiment_folder = f"{args.distance_method}/none"
    elif args.distance_method in {"bertscore", "bertscore-bert"}:
        experiment_folder = args.distance_method
    else:
        experiment_folder = f"{args.distance_method}/{model_name}"

    base_encode_truncation = (not args.no_encode_truncation) and (not args.oom_truncation_only)

    # Build the distance function once (shared across tasks)
    dialog2dialog_distance = build_dialog2dialog_distance(
        distance_method=args.distance_method,
        embedding_model=args.embedding_model,
        encode_batch_size=max(1, args.encode_batch_size),
        max_cache_entries=max(0, args.max_cache_entries),
        encode_truncation=base_encode_truncation,
        encode_max_length=(args.encode_max_length if args.encode_max_length > 0 else None),
    )

    all_task_summaries = {}  # task -> {ht -> summary_dict}
    all_task_diagnostics = {}  # task -> {ht -> runtime diagnostics}

    for task in task_list:
        print(f"\n{'='*60}")
        print(f"Task: {task}")

        output_path = os.path.join(args.output_path, experiment_folder, task)
        os.makedirs(output_path, exist_ok=True)
        summary_json_path = os.path.join(output_path, "results_d2d_summary.json")

        if os.path.exists(summary_json_path):
            print("Already computed. Loading results from existing JSON...")
            with open(summary_json_path, "r") as f:
                summaries = json.load(f)

            df_summary = pd.DataFrame(summaries).T.sort_index()
            metric_cols = ["exact_match_rate", "position_accuracy_mean", "kendall_tau_mean"]
            existing_metric_cols = [c for c in metric_cols if c in df_summary.columns]
            if existing_metric_cols:
                df_summary = df_summary[existing_metric_cols]
            if not df_summary.empty:
                df_summary.loc["AVG"] = df_summary.mean(axis=0)

            print("\nPer-hallucination ranking summary:")
            if df_summary.empty:
                print("No valid dialog pairs were found for evaluation.")
            else:
                print(df_summary.to_markdown(floatfmt=".4f"))

            all_task_summaries[task] = summaries
            continue

        os.makedirs(output_path, exist_ok=True)

        print(f"Loading hallucinated dialogs...")
        dialogs = load_task_hallucinated_dialogs(task, path_dataset=args.input_path)

        # Define ground truth rank based on hallucination type
        # Get candidate keys from first hallucination type
        first_ht = next(iter(dialogs["hallucinations"].keys()))
        candidate_keys = list(dialogs["hallucinations"][first_ht].keys())

        # Ground truth rank: later fork point means longer shared prefix with original.
        ground_truth_rank = sorted(candidate_keys, key=_fork_segment_order_key, reverse=True)

        reference_dialogs = dialogs["original"]

        print("Evaluating from scratch...")
        results = {}
        summaries = {}
        diagnostics_by_ht = {}

        hallucination_types = list(dialogs["hallucinations"].keys())
        for ht in hallucination_types:
            print(f"Evaluating hallucination type: {ht}")
            per_dialog_results, summary, diagnostics = evaluate_hallucination_type(
                dialog2dialog_distance=dialog2dialog_distance,
                originals=reference_dialogs,
                hallucinations_by_fork=dialogs["hallucinations"][ht],
                ground_truth_rank=ground_truth_rank,
                cuda_empty_cache_every=max(0, args.cuda_empty_cache_every),
                oom_retry_batch_size=args.oom_retry_batch_size,
                oom_retry_max_length=args.oom_retry_max_length,
                oom_fail_fast=args.oom_fail_fast,
            )
            results[ht] = per_dialog_results
            summaries[ht] = summary
            diagnostics_by_ht[ht] = diagnostics

        df_summary = pd.DataFrame(summaries).T.sort_index()
        metric_cols = ["exact_match_rate", "position_accuracy_mean", "kendall_tau_mean"]
        existing_metric_cols = [c for c in metric_cols if c in df_summary.columns]
        if existing_metric_cols:
            df_summary = df_summary[existing_metric_cols]
        if not df_summary.empty:
            df_summary.loc["AVG"] = df_summary.mean(axis=0)

        df_diagnostics = pd.DataFrame(diagnostics_by_ht).T.sort_index()
        if not df_diagnostics.empty:
            df_diagnostics.loc["TOTAL"] = df_diagnostics.sum(axis=0)

        print("\nPer-hallucination ranking summary:")
        if df_summary.empty:
            print("No valid dialog pairs were found for evaluation.")
        else:
            print(df_summary.to_markdown(floatfmt=".4f"))

        with open(os.path.join(output_path, "results_d2d_rankings.json"), "w") as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(output_path, "results_d2d_summary.json"), "w") as f:
            json.dump(summaries, f, indent=2)

        with open(os.path.join(output_path, "results_d2d_runtime_diagnostics.json"), "w") as f:
            json.dump(diagnostics_by_ht, f, indent=2)

        with open(os.path.join(output_path, "results_d2d_summary.txt"), "w") as f:
            if df_summary.empty:
                f.write("No valid dialog pairs were found for evaluation.\n")
            else:
                f.write(df_summary.to_markdown(floatfmt=".4f"))

        with open(os.path.join(output_path, "results_d2d_runtime_diagnostics.txt"), "w") as f:
            if df_diagnostics.empty:
                f.write("No runtime diagnostics available.\n")
            else:
                f.write(df_diagnostics.to_markdown(floatfmt=".0f"))

        all_task_summaries[task] = summaries
        all_task_diagnostics[task] = diagnostics_by_ht

    # ------------------------------------------------------------------ #
    # Overall report: per-task average (across hallucination types) and   #
    # grand average (across all tasks)                                     #
    # ------------------------------------------------------------------ #
    # Rebuild from disk (not just this invocation's task_list) since each
    # Slurm job only processes one task and shares this output directory
    # with the jobs for every other task.
    all_task_summaries = load_all_task_summaries_from_disk(
        os.path.join(args.output_path, experiment_folder)
    )

    if all_task_summaries:
        metrics_cols = ["exact_match_rate", "position_accuracy_mean", "kendall_tau_mean"]
        overall_rows = {}
        for task, summaries in all_task_summaries.items():
            task_df = pd.DataFrame(summaries).T
            task_df = task_df[[c for c in metrics_cols if c in task_df.columns]]
            if not task_df.empty:
                overall_rows[task] = task_df.mean(axis=0)

        if overall_rows:
            df_overall = pd.DataFrame(overall_rows).T.sort_index()
            df_overall.loc["AVG"] = df_overall.mean(axis=0)

            print(f"\n{'='*60}")
            print("Overall results (rows: tasks, columns: metrics — averaged across hallucination types):")
            print(df_overall.to_markdown(floatfmt=".4f"))

            # Average by hallucination type (across tasks)
            ht_records = []
            for task, summaries in all_task_summaries.items():
                for ht, summary in summaries.items():
                    record = {"task": task, "hallucination_type": ht}
                    for c in metrics_cols:
                        record[c] = summary.get(c, np.nan)
                    ht_records.append(record)

            df_ht = pd.DataFrame()
            if ht_records:
                df_ht = pd.DataFrame(ht_records)
                df_ht = (
                    df_ht.groupby("hallucination_type", as_index=True)[metrics_cols]
                    .mean()
                    .sort_index()
                )
                if not df_ht.empty:
                    df_ht.loc["AVG"] = df_ht.mean(axis=0)

                    print(f"\n{'='*60}")
                    print("Overall results (rows: hallucination types, columns: metrics — averaged across tasks):")
                    print(df_ht.to_markdown(floatfmt=".4f"))

            overall_output_path = os.path.join(args.output_path, experiment_folder)
            os.makedirs(overall_output_path, exist_ok=True)
            with open(os.path.join(overall_output_path, "results_d2d_overall.json"), "w") as f:
                json.dump(df_overall.to_dict(), f, indent=2)
            with open(os.path.join(overall_output_path, "results_d2d_overall.txt"), "w") as f:
                f.write(df_overall.to_markdown(floatfmt=".4f"))

            if not df_ht.empty:
                with open(os.path.join(overall_output_path, "results_d2d_overall_by_ht.json"), "w") as f:
                    json.dump(df_ht.to_dict(), f, indent=2)
                with open(os.path.join(overall_output_path, "results_d2d_overall_by_ht.txt"), "w") as f:
                    f.write(df_ht.to_markdown(floatfmt=".4f"))

    elapsed_seconds = time.perf_counter() - run_start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    print(f"\nTotal computation time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
