# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

import sdialog
import json
import re
import os
import argparse
import time
import numpy as np
from datetime import timedelta
from typing import List, Tuple, Optional

from tqdm.auto import tqdm
from jinja2 import Template
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from sdialog import Dialog
from sdialog.util import get_llm_model, set_generator_seed
from sdialog.config import config

# Constants
ORIGINALS_SUBDIR = "originals"
HALLUCINATION_TYPES = [
    "conflicting_system_constraints",
    "goal_replacement",
    "step_selection_policy_corruption",
    "workflow_identity_corruption",
]
METHOD_NAME = "llm_prediction"

# Methods are read from these modes when rendering summary tables.
METHOD_MODE = {
    METHOD_NAME: "offline",
}
BASELINE_METHODS = set()  # No baseline methods for LLM detector

# Set to True in main() when --verbose is passed
_verbose = False


def vprint(*args, **kwargs):
    """Print only when verbose mode is enabled."""
    if _verbose:
        print(*args, **kwargs)


class LLMWorkflowHallucationDetector_Dummy:
    """Dummy version of the detector that always returns 1 for testing."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, dialog, max_attempts=3, seed=None) -> int | None:
        return 1


class LLMWorkflowHallucationDetector:
    """Detects when a dialog diverges from an expected workflow graph.

    Takes a dialog and workflow definition as input and returns the 1-based index
    of the first system turn that goes offroad, or None if the dialog follows the workflow.
    """
    def __init__(
        self,
        instruction: str = None,
        system_prompt: str = None,
        model_name: str = None,
        output_format: BaseModel = None,
        use_workflow: bool = True,
        **llm_kwargs,
    ):

        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}
        if model_name is None:
            model_name = config["llm"]["model"]

        if output_format is None:
            class Output(BaseModel):
                hallucination_turn: int | None = None
            output_format = Output

        self.use_workflow = use_workflow

        if system_prompt is None:
            if self.use_workflow:
                system_prompt = """You are an expert system that analyzes task-oriented dialogs and detects hallucinations.

Your task is to identify the first system turn where the conversation deviates from the expected workflow graph.

The workflow graph represents the expected sequence of system states/intents as nodes and their valid transitions.
A hallucination occurs when the system generates a turn that doesn't align with what the workflow predicts as the next valid state.

Analyze the dialog transcript carefully:
1. Track which workflow node the conversation is at after each exchange
2. Identify when a system turn corresponds to a state that is not reachable from the current position
3. Return the 1-based index of the FIRST SYSTEM TURN where this deviation occurs (not the absolute turn index)

Return the result in JSON format:
```json
{
  "hallucination_turn": <system_turn_index_or_null>
}
```

Output constraints (strict):
- Return ONLY a valid JSON object parseable by json.loads().
- Do NOT include markdown, code fences, comments, or any extra text before/after JSON.
- Use double quotes for keys/strings and valid JSON null when needed.

Where `hallucination_turn` is:
- The system turn counter (e.g., if hallucination is at "System K" in the transcript, return K)
- null if the dialog follows the workflow correctly throughout"""
            else:
                system_prompt = """You are an expert system that analyzes task-oriented dialogs and detects hallucinations.

Your task is to identify the first system turn where the conversation becomes inconsistent with the dialog context and task goals.

No workflow graph is provided. Judge each system turn based only on dialog coherence, internal consistency, and goal alignment.

Analyze the dialog transcript carefully:
1. Track user constraints, intents, and confirmed facts across turns
2. Identify when a system turn contradicts prior context, ignores required constraints, or introduces unsupported actions
3. Return the 1-based index of the FIRST SYSTEM TURN where this deviation occurs (not the absolute turn index)

Return the result in JSON format:
```json
{
  "hallucination_turn": <system_turn_index_or_null>
}
```

Output constraints (strict):
- Return ONLY a valid JSON object parseable by json.loads().
- Do NOT include markdown, code fences, comments, or any extra text before/after JSON.
- Use double quotes for keys/strings and valid JSON null when needed.

Where `hallucination_turn` is:
- The system turn counter (e.g., if hallucination is at "System K" in the transcript, return K)
- null if no such deviation is found throughout the dialog"""

        self.output_format = output_format
        if instruction is None:
            if self.use_workflow:
                self.instruction = """GRAPH_START
    {{ workflow_dot }}
    GRAPH_END

    DIALOG_START

    {% for turn_label, turn_text in dialog_turns %}{{ turn_label }}: {{ turn_text }}
    {% endfor %}
    DIALOG_END

    TASK_START
    Goal:
    - Find the FIRST system turn where the dialog goes offroad with respect to the workflow graph.

    Definition of offroad:
    - At each step, the next system action/state must be consistent with valid workflow transitions.
    - Offroad starts at the first system turn that is not reachable/consistent from the current workflow position.

    Indexing rule:
    - Use SYSTEM TURN INDEX ONLY (System 1, System 2, ..., System K).
    - Do NOT use absolute turn index across both speakers.
    - If first deviation is at "System K", return K.

    No-deviation case:
    - If the dialog follows the workflow throughout, return null.
    TASK_END

    OUTPUT_FORMAT_START
    Return ONLY one valid raw JSON object parseable by json.loads():
    {"hallucination_turn": <int_or_null>}

    Strict constraints:
    - No markdown, no code fences, no comments.
    - No extra text before or after JSON.
    - Use valid JSON only (double quotes, null for missing value).
    OUTPUT_FORMAT_END"""
            else:
                self.instruction = """DIALOG_START

    {% for turn_label, turn_text in dialog_turns %}{{ turn_label }}: {{ turn_text }}
    {% endfor %}
    DIALOG_END

    TASK_START
    Goal:
    - Find the FIRST system turn where the dialog becomes inconsistent with context, constraints, or task goals.

    Definition of offroad (dialog-only):
    - The next system action/response must be coherent with prior turns, user requirements, and established facts.
    - Offroad starts at the first system turn that introduces contradiction, unsupported assumptions, or goal drift.

    Indexing rule:
    - Use SYSTEM TURN INDEX ONLY (System 1, System 2, ..., System K).
    - Do NOT use absolute turn index across both speakers.
    - If first deviation is at "System K", return K.

    No-deviation case:
    - If the dialog remains coherent and goal-consistent throughout, return null.
    TASK_END

    OUTPUT_FORMAT_START
    Return ONLY one valid raw JSON object parseable by json.loads():
    {"hallucination_turn": <int_or_null>}

    Strict constraints:
    - No markdown, no code fences, no comments.
    - No extra text before or after JSON.
    - Use valid JSON only (double quotes, null for missing value).
    OUTPUT_FORMAT_END"""
        else:
            self.instruction = instruction

        self.output_format = output_format
        # No output_format/structured-output here on purpose: JSON-schema-constrained
        # decoding + think=False breaks some models (empty completions), and think=True
        # is uniformly slow and makes cross-model comparison unfair (some models would
        # reason, others wouldn't get an equivalent boost). Instead every model gets the
        # same plain-text prompt + think=False, and the regex fallback below
        # (_extract_json_from_text) parses the JSON out of the response text.
        self.llm = get_llm_model(model_name,
                                 think=False,
                                 **llm_kwargs)
        self.memory = [SystemMessage(content=system_prompt), HumanMessage(content="")]

    def __call__(self, dialog, max_attempts=3, seed=None) -> int | None:
        """
        Args:
            dialog: A Dialog object with turns, transcript, and workflow annotations

        Returns:
            Output object with hallucination_turn (int | None)
        """
        dialog_turns = self._dialog_to_turns(dialog)
        if self.use_workflow:
            workflow = dialog.annotations.get("task", {}).get("graph")
            if workflow is None:
                raise ValueError("Workflow graph missing in dialog annotations while use_workflow=True.")
            workflow_dot = self._dict_to_dot(workflow)
            content = Template(self.instruction).render(
                workflow_dot=workflow_dot,
                dialog_turns=dialog_turns,
            )
        else:
            content = Template(self.instruction).render(dialog_turns=dialog_turns)
        self.memory[1].content = content
        for attempt in range(1, max_attempts + 1):
            set_generator_seed(self, seed=seed)

            # 0) Invoking the LLM can itself raise (e.g. the model returns an empty/
            #    malformed completion and LangChain's structured-output parser blows up
            #    on it). Treat that as a failed attempt so it gets retried like any other
            #    parsing failure, instead of aborting the whole call immediately.
            try:
                response = self.llm.invoke(self.memory)
            except Exception as e:
                print(f"[DEBUG] LLM invocation failed on attempt {attempt}/{max_attempts}: {e!r}, retrying...")
                continue

            # 1) Try direct validation first (works for dict / structured outputs).
            try:
                return self.output_format.model_validate(response).hallucination_turn
            except Exception as e:
                pass

            # 2) Regex fallback: extract JSON from markdown block first, then generic object.
            try:
                json_str = self._extract_json_from_text(response.content)
                return self.output_format.model_validate(json.loads(json_str)).hallucination_turn
            except Exception as e:
                pass

            print(f"[DEBUG] Parsing failed on attempt {attempt}/{max_attempts}, retrying...")

        print(
            f"[WARN] Failed to parse LLM output after {max_attempts} attempts. Returning no prediction for dialog {dialog.id}."
        )
        return None

    def _extract_json_from_text(self, text):
        """Extract JSON object from markdown code block or free text using regex."""
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"(\{[\s\S]*\})", text)
        if not match:
            raise ValueError("Could not extract JSON from model response.")
        return match.group(1)

    def _dialog_to_turns(self, dialog) -> list:
        """Convert Dialog object to a list of (turn_label, turn_text) tuples.

        Each speaker (User/System) has its own 1-based index counter.

        Args:
            dialog: A Dialog object

        Returns:
            List of tuples (turn_label, turn_text) where turn_label includes speaker and index
        """
        turns = []
        user_counter = 1
        system_counter = 1

        for turn in dialog.turns:
            text = turn.text

            if turn.speaker.lower() == "user":
                turn_label = f"User {user_counter}"
                user_counter += 1
            else:
                turn_label = f"System {system_counter}"
                system_counter += 1

            turns.append((turn_label, text))

        return turns

    def _dict_to_dot(self, workflow: dict) -> str:
        """Convert workflow dictionary to DOT notation for better readability.

        Args:
            workflow: Dict mapping source nodes to target nodes

        Returns:
            DOT format string representation of the workflow graph
        """
        dot_edges = ";\n    ".join(f"{source} -> {target}" for source, target in workflow.items())
        return f"digraph workflow {{\n    {dot_edges};\n}}"


# ============================================================================
# SECTION 1: HELPER FUNCTIONS FOR PIPELINE
# ============================================================================

def resolve_task_names(task_parent: str, requested_tasks: Optional[List[str]]) -> List[str]:
    """Resolve task names from CLI input or by listing all immediate subfolders."""
    if not os.path.isdir(task_parent):
        raise ValueError(f"Task parent folder does not exist: {task_parent}")

    if requested_tasks:
        task_names = requested_tasks
    else:
        task_names = [d for d in os.listdir(task_parent) if os.path.isdir(os.path.join(task_parent, d))]

    if not task_names:
        raise ValueError(f"No tasks found in: {task_parent}")

    missing = [t for t in task_names if not os.path.isdir(os.path.join(task_parent, t))]
    if missing:
        raise ValueError(f"Task(s) not found: {missing}")

    return task_names


def load_dialogs_recursive(root_path: str) -> List[Dialog]:
    """Recursively load dialogs from all files under a directory tree."""
    if not os.path.isdir(root_path):
        raise ValueError(f"Path does not exist: {root_path}")

    dialogs = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith((".json", ".jsonl")):
                filepath = os.path.join(dirpath, filename)
                try:
                    file_dialogs = Dialog.from_file(filepath)
                    if isinstance(file_dialogs, list):
                        dialogs.extend(file_dialogs)
                    else:
                        dialogs.append(file_dialogs)
                except Exception as e:
                    vprint(f"Warning: Failed to load {filepath}: {e}")
    return dialogs


def load_task_datasets(task_root: str) -> Tuple[List[Dialog], dict]:
    """Load originals and hallucination datasets from the fixed task-root structure."""
    originals_path = os.path.join(task_root, ORIGINALS_SUBDIR)
    if not os.path.isdir(originals_path):
        raise ValueError(f"Originals folder not found: {originals_path}")

    train_dialogs = load_dialogs_recursive(originals_path)
    hallucination_dialogs = {}
    for hallucination_type in HALLUCINATION_TYPES:
        hallu_path = os.path.join(task_root, hallucination_type)
        if os.path.isdir(hallu_path):
            hallucination_dialogs[hallucination_type] = load_dialogs_recursive(hallu_path)
        else:
            hallucination_dialogs[hallucination_type] = []

    return train_dialogs, hallucination_dialogs


def _is_valid_turn(turn) -> bool:
    """Return True if a turn has usable speaker/text fields."""
    speaker = getattr(turn, "speaker", None)
    text = getattr(turn, "text", None)
    if not isinstance(speaker, str) or not speaker.strip():
        return False
    if not isinstance(text, str) or not text.strip():
        return False
    return True


def sanitize_dialogs(dialogs: List[Dialog], split_name: str, task_name: str) -> List[Dialog]:
    """Drop dialogs containing malformed turns that break graph extraction."""
    clean_dialogs = []
    dropped = 0
    for dialog in dialogs:
        try:
            if all(_is_valid_turn(turn) for turn in dialog.turns):
                clean_dialogs.append(dialog)
            else:
                dropped += 1
        except Exception:
            dropped += 1

    if dropped:
        vprint(
            f"  [{split_name}][{task_name}] Dropped {dropped} / {len(dialogs)} dialogs "
            f"due to malformed turns."
        )
    return clean_dialogs


def build_result_path(args, result_filename: str) -> Optional[str]:
    """Derive the structured result path under the output root folder."""
    if not args.output:
        return None
    model_name = args.model.split("/")[-1] if args.model else "default"
    modality_suffix = "-no-workflow" if args.no_workflow else ""
    setting = f"llm-{model_name}{modality_suffix}"
    output_folder = os.path.join(args.output, setting)
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, result_filename)


def metrics_to_detection_artifact(all_type_metrics: dict, happy_metrics: dict) -> dict:
    """Convert internal metrics map to the persisted detection artifact format."""
    detection = {"hallucination_types": {}}
    for ht in HALLUCINATION_TYPES:
        ht_data = all_type_metrics.get(ht, {})
        detection["hallucination_types"][ht] = {
            "variants": {
                "raw": {
                    "dialogs_evaluated": int(ht_data.get("raw", {}).get("n_eval", 0)),
                    "offline_metrics": ht_data.get("raw", {}).get("offline", {}),
                }
            }
        }
    detection["happy"] = {
        "variants": {
            "raw": {
                "n_eval": int(happy_metrics.get("raw", {}).get("n_eval", 0)),
                "offline_metrics": happy_metrics.get("raw", {}).get("offline", {}),
            }
        }
    }
    return detection


def extract_task_metrics(payload: dict, cache_file: Optional[str] = None) -> Tuple[dict, List[str], dict]:
    """Extract all_type_metrics, available methods, and happy_metrics from a payload."""
    cfg = payload.get("config", {})
    detection = payload.get("detection") or {}
    hallucination_types = cfg.get("hallucination_types", HALLUCINATION_TYPES)
    all_methods = [METHOD_NAME]

    all_type_metrics = {}
    for ht in hallucination_types:
        ht_data = detection.get("hallucination_types", {}).get(ht, {})
        all_type_metrics[ht] = {
            "raw": {
                "n_eval": ht_data.get("variants", {}).get("raw", {}).get("dialogs_evaluated", 0),
                "offline": ht_data.get("variants", {}).get("raw", {}).get("offline_metrics", {}),
            }
        }

    happy_data = detection.get("happy", {})
    happy_metrics = {
        "raw": {
            "n_eval": happy_data.get("variants", {}).get("raw", {}).get("n_eval", 0),
            "offline": happy_data.get("variants", {}).get("raw", {}).get("offline_metrics", {}),
        }
    }

    happy_metrics = repair_happy_metrics_from_cache(happy_metrics, cache_file)

    return all_type_metrics, all_methods, happy_metrics


def load_prediction_cache(cache_file: str, cache_type: str = None) -> dict:
    """Load existing predictions from JSONL cache file.

    Args:
        cache_file: Path to cache file
        cache_type: Optional type filter ("originals", hallucination type name, etc.)
                   If provided, only returns entries matching that type.

    Returns a dict mapping dialog_id (conformant dialogues) or (dialog_id, fork turn)
    (forks) to {"ground_truth": ..., "prediction": ..., "type": ...}.
    """
    cache = {}
    if not os.path.isfile(cache_file):
        return cache

    try:
        with open(cache_file, "r") as fh:
            for line in fh:
                if line.strip():
                    entry = json.loads(line.strip())
                    # Filter by type if specified
                    if cache_type and entry.get("type") != cache_type:
                        continue
                    # the four forks of a dialogue share its dialog_id and type: key them by
                    # their fork turn too, or a resumed run would reuse one fork's prediction
                    # for the others (conformant dialogues are unique per dialog_id)
                    key = entry.get("dialog_id") if entry.get("type") == "originals" \
                        else (entry.get("dialog_id"), entry.get("ground_truth"))
                    cache[key] = {
                        "ground_truth": entry.get("ground_truth"),
                        "prediction": entry.get("prediction"),
                        "type": entry.get("type"),
                    }
        vprint(f"  Loaded {len(cache)} cached predictions from: {cache_file}")
    except Exception as e:
        vprint(f"  Warning: Failed to load cache from {cache_file}: {e}")

    return cache


def save_prediction_to_cache(cache_file: str, dialog_id, ground_truth, prediction, cache_type: str) -> None:
    """Append a single prediction to JSONL cache file.

    Args:
        cache_file: Path to cache file
        dialog_id: Dialog identifier
        ground_truth: Ground truth value
        prediction: Predicted value
        cache_type: Type label ("originals", hallucination type name, etc.)
    """
    try:
        entry = {
            "dialog_id": dialog_id,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "type": cache_type,
        }
        with open(cache_file, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as e:
        vprint(f"  Warning: Failed to save prediction to cache: {e}")


def repair_happy_metrics_from_cache(happy_metrics: dict, cache_file: Optional[str]) -> dict:
    """Repair conformant coverage from the originals cache when available."""
    if not cache_file or not os.path.isfile(cache_file):
        return happy_metrics

    cache = load_prediction_cache(cache_file, cache_type="originals")
    if not cache:
        return happy_metrics

    raw_metrics = happy_metrics.get("raw", {})
    offline_metrics = raw_metrics.get("offline", {})
    n_total = int(raw_metrics.get("n_eval", 0) or 0)
    if n_total <= 0:
        return happy_metrics

    n_pred = sum(1 for entry in cache.values() if entry.get("prediction") is not None)
    pred_rate = float(n_pred) / float(n_total)
    for method_entry in offline_metrics.values():
        if isinstance(method_entry, dict):
            method_entry["n_pred"] = n_pred
            method_entry["n_total"] = n_total
            method_entry["pred_rate"] = pred_rate

    return happy_metrics


def render_summary_table(
    hallucination_types: List[str],
    all_type_metrics: dict,
    methods: List[str],
    happy_metrics: Optional[dict] = None,
) -> str:
    """
    Print a MAE/eMAE summary table.
    (Copied from methods/task3/streaming.py for output parity.)
    """
    D = 2
    hallu_val_w = 3 * (D + 3) + 3
    orig_val_w  = D + 2
    rate_val_w = 6
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

    row_w = max(max((len(m) for m in methods), default=8), len("method"))

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
        return "n/a" if v is None else f"{v:.{D}f}"

    def _pct(v):
        return "n/a" if v is None else f"{100.0 * v:.1f}%"

    def _disp_error_component(value, count):
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
        emae_txt = _v(emae)
        dmae_txt = _v(dmae)
        return f"{mae:.{D}f}({emae_txt}/{dmae_txt})"

    ordered_methods = list(methods)

    def _get(ht, variant, key):
        mode_key = METHOD_MODE.get(method, "online")
        return (
            all_type_metrics.get(ht, {}).get(variant, {})
            .get(mode_key, {}).get(method, {}).get(key)
        )

    for method in ordered_methods:
        mode_key = METHOD_MODE.get(method, "online")

        rv = {ht: {k: _get(ht, "raw", k) for k in ("mae", "emae", "dmae", "n_early", "n_delay", "n_pred", "n_total", "pred_rate")} for ht in hallucination_types}
        for ht in hallucination_types:
            _ensure_counts(rv[ht], all_type_metrics.get(ht, {}).get("raw", {}).get("n_eval"))

        happy_raw_emae = None
        happy_raw_entry = {}
        if has_happy:
            happy_raw_entry = happy_metrics.get("raw", {}).get(mode_key, {}).get(method, {})
            _ensure_counts(happy_raw_entry, happy_metrics.get("raw", {}).get("n_eval"))
            happy_raw_emae = happy_raw_entry.get("emae")

        raw_h_pred_num = sum(float(rv[ht].get("n_pred")) for ht in hallucination_types if rv[ht].get("n_pred") is not None and rv[ht].get("n_total") not in (None, 0))
        raw_h_pred_den = sum(float(rv[ht].get("n_total")) for ht in hallucination_types if rv[ht].get("n_pred") is not None and rv[ht].get("n_total") not in (None, 0))
        hallu_rate_raw = (raw_h_pred_num / raw_h_pred_den) if raw_h_pred_den > 0 else None
        orig_rate_raw = _entry_rate(happy_raw_entry) if has_happy else None

        raw_mae_numbers = [rv[ht]["mae"] for ht in hallucination_types if rv[ht]["mae"] is not None]
        avg_hallu_raw = float(np.mean(raw_mae_numbers)) if raw_mae_numbers else None
        if avg_hallu_raw is not None and happy_raw_emae is not None:
            avg_raw = (avg_hallu_raw + happy_raw_emae) / 2.0
        else:
            avg_raw = avg_hallu_raw

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

    lines.append(f"  {sep}")
    return "\n".join(lines)


def render_benchmark_summary_table(
    hallucination_types: List[str],
    all_type_metrics: dict,
    methods: List[str],
    happy_metrics: Optional[dict] = None,
) -> str:
    """Render a compact benchmark-ready summary table.
    (Copied from methods/task3/streaming.py for output parity.)
    """
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

    for method in list(methods):
        hallu_avg, conformant, overall_avg = _variant_metrics(method, "raw")

        cells = [_v(hallu_avg).center(col_widths[0])]
        next_idx = 1
        if has_happy:
            cells.append(_v(conformant).center(col_widths[next_idx]))
            next_idx += 1
        cells.append(_v(overall_avg).center(col_widths[next_idx]))
        lines.append(f"  | {method.ljust(row_w)} " + "".join(f"| {c} " for c in cells) + "|")

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
# SECTION 2: TASK RUNNER - MAIN EVALUATION PIPELINE
# ============================================================================

def run_single_task(args, task_name: str) -> Tuple[Optional[dict], List[str], Optional[dict]]:
    """Run one task end-to-end (or load cache), print its table, and return metrics for aggregation."""
    task_start_time = time.perf_counter()
    task_root = os.path.join(args.task_root, task_name)
    output_path = build_result_path(args, f"{task_name}.json")
    modality = "dialog_only" if args.no_workflow else "workflow+dialog"

    # Fast path: results already exist — just reprint them
    if output_path and os.path.isfile(output_path):
        try:
            with open(output_path, "r") as fh:
                payload = json.load(fh)
            payload_modality = payload.get("config", {}).get("modality", "workflow+dialog")
            if payload_modality != modality:
                raise ValueError(
                    f"Cached payload modality mismatch: file has {payload_modality}, current run expects {modality}."
                )
            vprint(f"Loading cached results from: {output_path}")
            cache_file = output_path.replace(".json", "_cache.jsonl")
            all_type_metrics, eval_methods, happy_metrics = extract_task_metrics(payload, cache_file=cache_file)
            print_summary_table(
                hallucination_types=HALLUCINATION_TYPES,
                all_type_metrics=all_type_metrics,
                methods=eval_methods,
                happy_metrics=happy_metrics,
            )
            return all_type_metrics, eval_methods, happy_metrics
        except Exception as e:
            vprint(f"Warning: Could not load cached results: {e}. Re-processing task.")

    # ------------------------------------------------------------------
    # 1. Load dialogs
    # ------------------------------------------------------------------
    vprint(f"Loading task datasets from: {task_root}")
    train_dialogs, hallucination_dialogs = load_task_datasets(task_root)

    train_dialogs = sanitize_dialogs(train_dialogs, "originals", task_name)
    for hallucination_type in HALLUCINATION_TYPES:
        hallucination_dialogs[hallucination_type] = sanitize_dialogs(
            hallucination_dialogs[hallucination_type], hallucination_type, task_name
        )

    if not train_dialogs:
        vprint(f"  No original dialogs found. Skipping task {task_name}.")
        return None, [], None

    vprint(f"  Loaded {len(train_dialogs)} original dialogs.")
    for hallucination_type in HALLUCINATION_TYPES:
        vprint(f"  {hallucination_type}: {len(hallucination_dialogs[hallucination_type])} dialogs")

    # ------------------------------------------------------------------
    # 2. Create LLM detector
    # ------------------------------------------------------------------
    vprint(f"\nInitializing LLM detector with model: {args.model}")
    # detector = LLMWorkflowHallucationDetector_Dummy(model_name=args.model)
    detector = LLMWorkflowHallucationDetector(
        model_name=args.model,
        use_workflow=not args.no_workflow,
    )
    vprint("  LLM detector initialized.")

    # Setup single unified cache file for this task
    unified_cache_file = None
    if output_path:
        unified_cache_file = output_path.replace(".json", "_cache.jsonl")

    # ------------------------------------------------------------------
    # 3. Predictions on original (happy) dialogs
    # ------------------------------------------------------------------
    vprint("\nEvaluating original dialogs...")
    y_true = []
    y_pred = []
    is_real_pred = []
    per_dialog_preds = []
    cache = load_prediction_cache(unified_cache_file, cache_type="originals") if unified_cache_file else {}
    n_cached = len(cache)
    n_skipped = 0

    for dialog in tqdm(train_dialogs, desc="originals"):
        dialog_id = getattr(dialog, "id", None)
        # gt = last system turn index + 1 (one beyond the dialog: no hallucination expected)
        system_turn_count = sum(1 for turn in dialog.turns if turn.speaker.lower() in ['system', 'assistant', 'agent'])
        gt_fork_turn = system_turn_count + 1

        if dialog_id in cache:
            cached_pred = cache[dialog_id]["prediction"]
            pred_value = gt_fork_turn if cached_pred is None else cached_pred
            per_dialog_preds.append({"dialog_id": dialog_id, "ground_truth": gt_fork_turn, "prediction": cached_pred, "cached": True})
            y_true.append(gt_fork_turn)
            y_pred.append(pred_value)
            is_real_pred.append(cached_pred is not None)
            n_skipped += 1
            continue

        try:
            pred_turn = detector(dialog)
            pred_value = gt_fork_turn if pred_turn is None else pred_turn
            per_dialog_preds.append({"dialog_id": dialog_id, "ground_truth": gt_fork_turn, "prediction": pred_turn, "cached": False})
            if unified_cache_file:
                save_prediction_to_cache(unified_cache_file, dialog_id, gt_fork_turn, pred_turn, cache_type="originals")
            y_true.append(gt_fork_turn)
            y_pred.append(pred_value)
            is_real_pred.append(pred_turn is not None)
        except Exception as e:
            print(f"    Warning: Failed to predict on dialog {dialog_id}: {e}")

    # Compute happy metrics (eMAE only — early = false positive predictions)
    n_eval_orig = len(train_dialogs)
    if y_true:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        early_mask = y_pred_arr < y_true_arr
        n_early = int(np.sum(early_mask))
        # If predictions exist but none are early, conformant eMAE is 0.0 (not n/a).
        emae = float(np.mean(y_true_arr[early_mask] - y_pred_arr[early_mask])) if n_early > 0 else 0.0
        n_pred = int(np.sum(is_real_pred))
        pred_rate = float(n_pred) / float(n_eval_orig)
    else:
        emae = None
        n_early = 0
        n_pred = 0
        pred_rate = 0.0

    happy_metrics = {
        "raw": {
            "n_eval": n_eval_orig,
            "offline": {
                METHOD_NAME: {
                    "emae": emae,
                    "n_early": n_early,
                    "n_pred": n_pred,
                    "n_total": n_eval_orig,
                    "pred_rate": pred_rate,
                }
            },
        }
    }

    cache_info = f" ({n_cached} cached, {n_eval_orig - n_skipped} new)" if unified_cache_file else ""
    emae_str = f"{emae:.4f}" if emae is not None else "N/A"
    vprint(f"  Originals eMAE: {emae_str}{cache_info}")

    # ------------------------------------------------------------------
    # 4. Predictions on hallucination dialogs
    # ------------------------------------------------------------------
    vprint("\nPredicting off-road turns by hallucination type...")
    all_type_metrics = {}

    for hallucination_type in HALLUCINATION_TYPES:
        hallu_dialogs = hallucination_dialogs[hallucination_type]
        n_eval_ht = len(hallu_dialogs)
        if not hallu_dialogs:
            vprint(f"  {hallucination_type}: No dialogs to evaluate.")
            all_type_metrics[hallucination_type] = {"raw": {"n_eval": 0, "offline": {METHOD_NAME: {"mae": None, "emae": None, "dmae": None, "n_early": 0, "n_delay": 0, "n_pred": 0, "n_total": 0}}}}
            continue

        cache = load_prediction_cache(unified_cache_file, cache_type=hallucination_type) if unified_cache_file else {}
        n_cached = len(cache)
        y_true = []
        y_pred = []
        is_real_pred = []
        per_dialog_preds = []
        n_skipped = 0

        vprint(f"  Evaluating {hallucination_type}...")
        for dialog in tqdm(hallu_dialogs, desc=hallucination_type):
            dialog_id = getattr(dialog, "id", None)
            gt_fork_turn = dialog.annotations.get("fork_assistant_turn")
            if gt_fork_turn is None:
                continue
            # Penalty for no prediction: last system turn index + 1
            system_turn_count = sum(1 for turn in dialog.turns if turn.speaker.lower() in ['system', 'assistant', 'agent'])
            penalty = system_turn_count + 1

            if (dialog_id, gt_fork_turn) in cache:
                cached_pred = cache[(dialog_id, gt_fork_turn)]["prediction"]
                pred_value = penalty if cached_pred is None else cached_pred
                per_dialog_preds.append({"dialog_id": dialog_id, "ground_truth": gt_fork_turn, "prediction": cached_pred, "cached": True})
                y_true.append(gt_fork_turn)
                y_pred.append(pred_value)
                is_real_pred.append(cached_pred is not None)
                n_skipped += 1
                continue

            try:
                pred_turn = detector(dialog)
                pred_value = penalty if pred_turn is None else pred_turn
                per_dialog_preds.append({"dialog_id": dialog_id, "ground_truth": gt_fork_turn, "prediction": pred_turn, "cached": False})
                if unified_cache_file:
                    save_prediction_to_cache(unified_cache_file, dialog_id, gt_fork_turn, pred_turn, cache_type=hallucination_type)
                y_true.append(gt_fork_turn)
                y_pred.append(pred_value)
                is_real_pred.append(pred_turn is not None)
            except Exception as e:
                print(f"    Warning: Failed to predict on dialog {dialog_id}: {e}")

        # Compute MAE, eMAE, dMAE
        if y_true:
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)
            mae = float(np.mean(np.abs(y_pred_arr - y_true_arr)))
            early_mask = y_pred_arr < y_true_arr
            delay_mask = y_pred_arr > y_true_arr
            n_early = int(np.sum(early_mask))
            n_delay = int(np.sum(delay_mask))
            emae = float(np.mean(y_true_arr[early_mask] - y_pred_arr[early_mask])) if n_early > 0 else None
            dmae = float(np.mean(y_pred_arr[delay_mask] - y_true_arr[delay_mask])) if n_delay > 0 else None
            n_pred = int(np.sum(is_real_pred))  # only real (non-penalty) predictions
            pred_rate = float(n_pred) / float(n_eval_ht)
        else:
            mae = emae = dmae = None
            n_early = n_delay = n_pred = 0
            pred_rate = 0.0

        all_type_metrics[hallucination_type] = {
            "raw": {
                "n_eval": n_eval_ht,
                "offline": {
                    METHOD_NAME: {
                        "mae": mae,
                        "emae": emae,
                        "dmae": dmae,
                        "n_early": n_early,
                        "n_delay": n_delay,
                        "n_pred": n_pred,
                        "n_total": n_eval_ht,
                        "pred_rate": pred_rate,
                    }
                },
            }
        }

        cache_info = f" ({n_cached} cached, {n_eval_ht - n_skipped} new)" if unified_cache_file else ""
        mae_str = f"{mae:.4f}" if mae is not None else "N/A"
        vprint(f"    {hallucination_type} MAE: {mae_str}{cache_info}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    if output_path:
        result_payload = {
            "config": {
                "task_root": task_root,
                "task_name": task_name,
                "originals_subdir": ORIGINALS_SUBDIR,
                "hallucination_types": HALLUCINATION_TYPES,
                "llm_model": args.model,
                "method": METHOD_NAME,
                "use_workflow": not args.no_workflow,
                "modality": modality,
                "total_runtime_seconds": float(time.perf_counter() - task_start_time),
            },
            "detection": metrics_to_detection_artifact(all_type_metrics, happy_metrics),
        }
        with open(output_path, "w") as fh:
            json.dump(result_payload, fh, indent=2)
        vprint(f"  Task results saved to: {output_path}")

    # Print summary
    print_summary_table(
        hallucination_types=HALLUCINATION_TYPES,
        all_type_metrics=all_type_metrics,
        methods=[METHOD_NAME],
        happy_metrics=happy_metrics,
    )

    vprint(f"Task {task_name} runtime: {timedelta(seconds=round(time.perf_counter() - task_start_time))}")
    return all_type_metrics, [METHOD_NAME], happy_metrics


def aggregate_metrics_across_tasks(
    task_metrics_list: List[dict],
    happy_metrics_list: List[dict],
    methods: List[str],
) -> Tuple[dict, dict]:
    """Aggregate metrics across tasks. Returns (aggregated_hallu, aggregated_happy)."""
    aggregated = {}
    for ht in HALLUCINATION_TYPES:
        aggregated[ht] = {
            "raw": {
                "n_eval": 0,
                "offline": {},
            }
        }
        aggregated[ht]["raw"]["n_eval"] = sum(
            int(tm.get(ht, {}).get("raw", {}).get("n_eval", 0)) for tm in task_metrics_list
        )
        for method in methods:
            metric_keys = ("mae", "emae", "dmae")
            count_keys = ("n_early", "n_delay", "n_pred", "n_total")
            method_entry = {}
            for key in metric_keys:
                vals = [
                    tm.get(ht, {}).get("raw", {}).get("offline", {}).get(method, {}).get(key)
                    for tm in task_metrics_list
                ]
                vals = [v for v in vals if v is not None]
                method_entry[key] = float(np.mean(vals)) if vals else None
            for key in count_keys:
                method_entry[key] = sum(
                    int(tm.get(ht, {}).get("raw", {}).get("offline", {}).get(method, {}).get(key, 0) or 0)
                    for tm in task_metrics_list
                )
            if method_entry.get("n_total", 0) > 0 and method_entry.get("n_pred") is not None:
                method_entry["pred_rate"] = float(method_entry["n_pred"]) / float(method_entry["n_total"])
            aggregated[ht]["raw"]["offline"][method] = method_entry

    aggregated_happy = {
        "raw": {
            "n_eval": sum(int(hm.get("raw", {}).get("n_eval", 0)) for hm in happy_metrics_list),
            "offline": {},
        }
    }
    for method in methods:
        method_entry = {}
        emae_vals = [
            hm.get("raw", {}).get("offline", {}).get(method, {}).get("emae")
            for hm in happy_metrics_list
        ]
        emae_vals = [v for v in emae_vals if v is not None]
        method_entry["emae"] = float(np.mean(emae_vals)) if emae_vals else None
        for key in ("n_early", "n_pred", "n_total"):
            method_entry[key] = sum(
                int(hm.get("raw", {}).get("offline", {}).get(method, {}).get(key, 0) or 0)
                for hm in happy_metrics_list
            )
        if method_entry.get("n_total", 0) > 0 and method_entry.get("n_pred") is not None:
            method_entry["pred_rate"] = float(method_entry["n_pred"]) / float(method_entry["n_total"])
        aggregated_happy["raw"]["offline"][method] = method_entry

    return aggregated, aggregated_happy


# ============================================================================
# MAIN
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLM-based off-road turn detection across tasks."
    )
    p.add_argument(
        "--task-root", "-d", required=True,
        help="Parent folder containing one subfolder per task."
    )
    p.add_argument(
        "--tasks", "-t", nargs="+", default=None,
        help="Optional list of task folder names under --task-root. If omitted, all immediate subfolders are used."
    )
    p.add_argument(
        "--model", "-m", default="gpt-oss:20b",
        help="LLM model name to use for hallucination detection (default: gpt-oss:20b)."
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Root folder for results. Outputs are saved to <output>/llm-<model>/<task>.json."
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed progress during computation (default: off)."
    )
    p.add_argument(
        "--no-workflow", action="store_true",
        help="Disable workflow-graph conditioning and run dialog-only detection mode."
    )
    return p.parse_args()


def check_all_tasks_complete(args, task_names: List[str]) -> bool:
    """Check if all task JSON result files exist and are readable."""
    for task_name in task_names:
        task_json_path = build_result_path(args, f"{task_name}.json")
        if not task_json_path or not os.path.isfile(task_json_path):
            return False
        try:
            with open(task_json_path, "r") as fh:
                json.load(fh)
        except Exception:
            return False
    return True


def load_all_task_metrics(args, task_names: List[str]) -> Tuple[List[dict], List[dict], bool]:
    """Load metrics from all readable task JSON files.

    Returns (hallu_metrics_list, happy_metrics_list, all_complete).
    """
    all_tasks_metrics = []
    all_happy_metrics = []
    for task_name in task_names:
        task_json_path = build_result_path(args, f"{task_name}.json")
        if not task_json_path or not os.path.isfile(task_json_path):
            return all_tasks_metrics, all_happy_metrics, False
        try:
            with open(task_json_path, "r") as fh:
                payload = json.load(fh)
            cache_file = task_json_path.replace(".json", "_cache.jsonl")
            task_metrics, _, happy_metrics = extract_task_metrics(payload, cache_file=cache_file)
            all_tasks_metrics.append(task_metrics)
            all_happy_metrics.append(happy_metrics)
        except Exception as e:
            vprint(f"  Warning: Failed to load metrics from {task_json_path}: {e}")
            return all_tasks_metrics, all_happy_metrics, False
    return all_tasks_metrics, all_happy_metrics, True


def main():
    global _verbose
    start_time = time.perf_counter()
    args = parse_args()
    _verbose = args.verbose

    task_names = resolve_task_names(args.task_root, args.tasks)

    print("\n=== Selected Settings ===")
    print(f"task_root: {args.task_root}")
    print(f"tasks: {', '.join(task_names)}")
    print(f"model: {args.model}")
    print(f"output: {args.output if args.output else '(disabled)'}")
    print(f"verbose: {args.verbose}")
    print(f"no_workflow: {args.no_workflow}")
    print(f"modality: {'dialog_only' if args.no_workflow else 'workflow+dialog'}")

    # Always use the full task set for global completion checks and AVERAGE files.
    all_task_names = resolve_task_names(args.task_root, None)
    all_tasks_metrics = []
    all_happy_metrics = []
    methods_for_final = [METHOD_NAME]

    for task_name in task_names:
        print(f"\n=== Task: {task_name} ===")
        task_metrics, methods, task_happy = run_single_task(args, task_name)
        if methods:
            methods_for_final = methods
        if task_metrics:
            all_tasks_metrics.append(task_metrics)
        if task_happy:
            all_happy_metrics.append(task_happy)

    # Check global completion across all tasks under task-root (parallel jobs may run subsets).
    all_complete = check_all_tasks_complete(args, all_task_names)

    avg_table_text = None
    avg_metrics = None
    avg_happy_metrics = None

    if all_complete:
        # Load metrics from the full task set when computing global averages.
        all_tasks_metrics, all_happy_metrics, _ = load_all_task_metrics(args, all_task_names)

        if all_tasks_metrics:
            print("\n=== Average Across Tasks ===")
            avg_metrics, avg_happy_metrics = aggregate_metrics_across_tasks(
                all_tasks_metrics, all_happy_metrics, methods_for_final
            )
            avg_benchmark_text = render_benchmark_summary_table(
                hallucination_types=HALLUCINATION_TYPES,
                all_type_metrics=avg_metrics,
                methods=methods_for_final,
                happy_metrics=avg_happy_metrics,
            )
            avg_extended_text = render_summary_table(
                hallucination_types=HALLUCINATION_TYPES,
                all_type_metrics=avg_metrics,
                methods=methods_for_final,
                happy_metrics=avg_happy_metrics,
            )
            avg_table_text = avg_benchmark_text + "\n\n" + avg_extended_text
            print("\n" + avg_table_text)
        else:
            print("\nNo task produced evaluable metrics.")
    else:
        if all_tasks_metrics:
            print("\nNot all tasks are complete yet. Skipping average computation.")
        else:
            print("\nNo task produced evaluable metrics.")

    total_runtime_seconds = time.perf_counter() - start_time

    # Only save average files if all tasks are complete
    if all_complete and avg_metrics is not None:
        global_output_path = build_result_path(args, "AVERAGE.json")
        if global_output_path:
            global_payload = {
                "config": {
                    "task_parent": args.task_root,
                    "task_names": all_task_names,
                    "originals_subdir": ORIGINALS_SUBDIR,
                    "hallucination_types": HALLUCINATION_TYPES,
                    "llm_model": args.model,
                    "method": METHOD_NAME,
                    "use_workflow": not args.no_workflow,
                    "modality": "dialog_only" if args.no_workflow else "workflow+dialog",
                    "total_runtime_seconds": float(total_runtime_seconds),
                    "is_global_average": True,
                },
                "detection": metrics_to_detection_artifact(avg_metrics, avg_happy_metrics),
            }
            with open(global_output_path, "w") as fh:
                json.dump(global_payload, fh, indent=2)
            vprint(f"  Global average results saved to: {global_output_path}")

        average_txt_path = build_result_path(args, "AVERAGE.txt")
        if average_txt_path and avg_table_text is not None:
            with open(average_txt_path, "w") as fh:
                fh.write("Average Across Tasks\n")
                fh.write(avg_table_text)
                fh.write("\n")
            vprint(f"  Global average table saved to: {average_txt_path}")

    print(f"\nDone. Total computation time: {timedelta(seconds=round(total_runtime_seconds))}")


if __name__ == "__main__":
    main()
