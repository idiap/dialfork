# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Read-only access to the DialFork dataset for the annotation tool.

Nothing here writes to datasets/dialfork/ -- the released files are never
mutated.

The dataset quirks this module exists to encapsulate:
  - Turns do NOT strictly alternate System/User, so the fork can never be
    located as 2*(fork_assistant_turn-1). It is the fork_assistant_turn-th System entry.
  - 209 forks were never actually injected (fork_assistant_turn-1 >= #system turns).
     resolve_fork_index() returns None for those; the sampler drops them.
"""
import ast
import glob
import json
import os

from .workflow import TASK_DESCRIPTIONS, parse_wic_injection

HALLUCINATION_TYPES = (
    "workflow_identity_corruption",
    "step_selection_policy_corruption",
    "goal_replacement",
    "conflicting_system_constraints",
)
TYPE_ABBREV = {
    "workflow_identity_corruption": "WIC",
    "step_selection_policy_corruption": "SSPC",
    "goal_replacement": "GR",
    "conflicting_system_constraints": "CSC",
}
SEGMENTS = (1, 2, 3, 4)

_POOL_CONSTANTS = {
    "step_selection_policy_corruption": "STEP_SELECTION_POLICIES",
    "goal_replacement": "GOAL_DRIFT_OBJECTIVES",
    "conflicting_system_constraints": "CONFLICTING_CONSTRAINTS",
}


def load_variant_pools(dataset_py="generation/generate.py"):
    """(type, variant_id) -> expected-behaviour sentence, parsed from generation/generate.py.

    Parsed rather than imported: generation/generate.py imports sdialog, which is only
    available inside the apptainer image. Parsing keeps generation/generate.py the single
    source of truth for the variant text shown to annotators.
    """
    with open(dataset_py, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    wanted = {v: k for k, v in _POOL_CONSTANTS.items()}
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in wanted:
                htype = wanted[tgt.id]
                for vid, policy, effect in ast.literal_eval(node.value):
                    out[(htype, vid)] = {"policy": policy, "effect": effect}
    missing = set(_POOL_CONSTANTS) - {k[0] for k in out}
    if missing:
        raise RuntimeError("variant pools not found in %s: %s" % (dataset_py, missing))
    return out


def load_domain_tasks(dataset_py="generation/generate.py"):
    """domain -> [task, ...], parsed from generation/generate.py (see load_variant_pools)."""
    with open(dataset_py, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "DOMAIN_TASKS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("DOMAIN_TASKS not found in %s" % dataset_py)


def cue_sentence(htype, variant_id, pools):
    """The concrete 'look for this' shown to the annotator in step 2.

    Returns {lead, instruction} so the UI can bold the instruction, or None if
    the variant is unknown. The pool sentences are imperative ("Ask the same
    questions ..."), so they are placed after "it was meant to:" verbatim rather
    than conjugated into a clause.

    WIC has no fixed pool -- variant_id is the name of the substitute task --
    so its instruction is that task's one-line description.
    """
    if htype == "workflow_identity_corruption":
        desc = TASK_DESCRIPTIONS.get(variant_id)
        if not desc:
            return None
        return {
            "lead": "At some point, the assistant was told to stop doing the task "
                    "above and, from then on, switch to a completely different one:",
            "instruction": desc,
        }
    entry = pools.get((htype, variant_id))
    if not entry:
        return None
    return {
        # "from then on": the instruction changes behaviour from a point onwards,
        # and that turning point is what the annotator is asked to find
        "lead": "At some point, the assistant was given an extra instruction. "
                "From then on, it was meant to:",
        "instruction": entry["effect"],
    }


# --- dialogue access --------------------------------------------------------

def load_dialog(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def system_turn_indices(dialog):
    """Absolute indices into dialog['turns'] of the System turns."""
    return [i for i, t in enumerate(dialog.get("turns") or [])
            if t.get("speaker") == "System"]


def resolve_fork_index(dialog):
    """Absolute turn index of the first hallucinated System turn, or None.

    None means the injection never fired for this file (one of the 209).
    """
    ann = dialog.get("annotations") or {}
    ft = ann.get("fork_assistant_turn")
    if ft is None:
        return None
    sys_idx = system_turn_indices(dialog)
    if ft - 1 >= len(sys_idx) or ft < 1:
        return None
    return sys_idx[ft - 1]


def injection_event(dialog):
    for ev in dialog.get("events") or []:
        if ev.get("action") == "instruct-persist":
            return ev
    return None


def was_injected(dialog):
    return injection_event(dialog) is not None


def substitute_workflow(dialog):
    """(name, graph, replies) of a WIC fork's substitute workflow, or None.

    Read from the injection event, NOT from datasets/STAR/ -- verified over all
    24 distinct variant_ids that the event carries the full DOT plus replies,
    including for trivia_score_reporting and weather_alerts which have no STAR
    directory.
    """
    ev = injection_event(dialog)
    if ev is None:
        return None
    name, graph, replies = parse_wic_injection(ev.get("content") or "")
    if not graph:
        return None
    return name, graph, replies


# --- paths ------------------------------------------------------------------

def fork_path(root, task, htype, segment, dialog_id):
    return os.path.join(root, task, htype, "n_hallucinations_4",
                        "fork_at_segment_%d" % segment, "dialog_%s.json" % dialog_id)


def parent_path(root, task, dialog_id):
    return os.path.join(root, task, "originals", "dialog_%s.json" % dialog_id)


def list_tasks(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def list_parent_ids(root, task):
    ids = []
    for p in glob.glob(os.path.join(root, task, "originals", "dialog_*.json")):
        ids.append(os.path.basename(p)[len("dialog_"):-len(".json")])
    return sorted(ids, key=lambda s: (len(s), s))


def quartet_paths(root, task, htype, dialog_id):
    """The 4 segment files plus the conformant parent, or None if incomplete.

    A quartet is only usable if all four forks exist AND all four resolve a
    fork turn (so none of the 209 never-injected files is in it).
    """
    parent = parent_path(root, task, dialog_id)
    if not os.path.exists(parent):
        return None
    forks = {}
    for k in SEGMENTS:
        p = fork_path(root, task, htype, k, dialog_id)
        if not os.path.exists(p):
            return None
        d = load_dialog(p)
        if resolve_fork_index(d) is None or not was_injected(d):
            return None
        forks[k] = p
    return {"parent": parent, "forks": forks}
