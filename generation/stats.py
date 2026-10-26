# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

import argparse
import csv
import os
import re
import statistics
from collections import Counter, defaultdict

from sdialog import Dialog
from tqdm.auto import tqdm


DOMAIN_TASKS = {
    "apartment": ["apartment_schedule", "apartment_search"],
    "bank": ["bank_balance", "bank_fraud_report"],
    "doctor": ["doctor_followup", "doctor_schedule"],
    "hotel": ["hotel_book", "hotel_search", "hotel_service_request"],
    "party": ["party_plan", "party_rsvp"],
    "plane": ["plane_book", "plane_search"],
    "restaurant": ["restaurant_book", "restaurant_search"],
    "ride": ["ride_book", "ride_change", "ride_status"],
    "spaceship": ["spaceship_access_codes", "spaceship_life_support"],
    "meeting_trip": ["meeting_schedule", "trip_directions"],
    "trivia": ["trivia", "trivia_score_reporting"],
    "weather": ["weather", "weather_alerts"],
}

TASK_TO_DOMAIN = {
    task_name: domain_name
    for domain_name, task_names in DOMAIN_TASKS.items()
    for task_name in task_names
}

EXPECTED_SEGMENTS = [f"fork_at_segment_{idx}" for idx in range(1, 5)]
HALLUCINATION_DISPLAY = {
    "workflow_identity_corruption": "WIC forked",
    "step_selection_policy_corruption": "SSPC forked",
    "goal_replacement": "GR forked",
    "conflicting_system_constraints": "CSC forked",
}

# Heuristics for turns that look like leaked chain-of-thought, tags, or meta blocks.
SUSPICIOUS_UTTERANCE_PATTERNS = [
    (re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>"), "xml_or_html_tag"),
    (re.compile(r"<\s*channel\s*\|", re.IGNORECASE), "channel_tag"),
    (re.compile(r"<\s*thought", re.IGNORECASE), "thought_tag"),
    # High-precision capture of leaked control/meta notes in plain utterances.
    (re.compile(r"^\s*[\(\[]\s*system\s+note\s*:", re.IGNORECASE), "system_note_prefix"),
    (re.compile(r"^\s*system\s+note\s*:", re.IGNORECASE), "system_note_prefix"),
    (re.compile(r"\bthinking\s*process\b", re.IGNORECASE), "thinking_process"),
    (re.compile(r"```"), "code_fence"),
    (re.compile(r"\bself-?correction\b", re.IGNORECASE), "self_correction_marker"),
    (re.compile(r"\bsystem\s*note\s*:", re.IGNORECASE), "system_note_meta"),
    (re.compile(r"^\s*\(\s*system\s*note\s*:[\s\S]*\)\s*$", re.IGNORECASE), "parenthesized_system_note"),
    (
        re.compile(
            r"\b(happy\s*path|role-?play|instructions?\s+stated|only\s+one\s+task)\b",
            re.IGNORECASE,
        ),
        "prompt_meta_language",
    ),
]

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


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Comprehensive statistics for STAR hallucination dataset."
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        nargs="?",
        default="datasets/dialfork",
        help="Root directory containing task folders of the dataset.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also print markdown tables after console summaries.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print paper-ready LaTeX tables (with booktabs/multicolumn grouping).",
    )
    parser.add_argument(
        "--top-k-tasks",
        type=int,
        default=0,
        help="If >0, only print top-k tasks by total dialogues in task table.",
    )
    parser.add_argument(
        "--sections",
        type=str,
        default="all",
        help=(
            "Comma-separated sections to print. "
            "Options: overview,split,type,segment,matrix,task,domain,coverage,length,divergence,all"
        ),
    )
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Exit with code 1 if integrity checks detect missing/orphan/duplicate entries.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full diagnostics and all analysis tables (debug mode).",
    )
    parser.add_argument(
        "--issues-output-csv",
        type=str,
        default="",
        help=(
            "Optional path to save a CSV file (columns: issue_type,path) covering "
            "dialogs with potential non-utterance turn content."
        ),
    )
    return parser


def parse_sections(raw_sections):
    allowed = {
        "overview",
        "split",
        "type",
        "segment",
        "matrix",
        "task",
        "domain",
        "coverage",
        "length",
        "divergence",
        "all",
    }
    selected = {item.strip().lower() for item in raw_sections.split(",") if item.strip()}
    invalid = selected.difference(allowed)
    if invalid:
        raise ValueError(f"Invalid sections: {sorted(invalid)}")
    if "all" in selected or not selected:
        return allowed.difference({"all"})
    return selected


def safe_int_segment(segment_name):
    match = re.search(r"(\d+)$", segment_name)
    return int(match.group(1)) if match else None


def dialog_key(dialog):
    if getattr(dialog, "id", None) is not None:
        return str(dialog.id)
    path = getattr(dialog, "_path", None)
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    return "unknown"


def as_text(value):
    if value is None:
        return ""
    return str(value)


def speaker_kind(speaker):
    token = as_text(speaker).strip().lower()
    if "system" in token or token in {"assistant", "wizard"}:
        return "system"
    if "user" in token or token in {"customer", "client"}:
        return "user"
    return "other"


def percentile(values, q):
    if not values:
        return 0.0
    ranked = sorted(values)
    if len(ranked) == 1:
        return float(ranked[0])
    pos = (len(ranked) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ranked) - 1)
    frac = pos - low
    return ranked[low] + (ranked[high] - ranked[low]) * frac


def summarize_numeric(values):
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(title, headers, rows):
    print_header(title)
    rendered_rows = [[format_value(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rendered_rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def print_markdown_table(title, headers, rows):
    print(f"\n### {title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(cell) for cell in row) + " |")


def _latex_escape(value):
    text = str(value)
    replacements = {
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
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _latex_fmt(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _latex_fmt_int(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _latex_fmt_pct(value):
    try:
        return f"{float(value):.2f}\\%"
    except Exception:
        return str(value)


def _latex_bold(text):
    return r"\textbf{" + str(text) + "}"


def _print_latex_table_1(rows):
    print("% Requires: \\usepackage{booktabs}")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\caption{Dataset composition across conformant and forked splits.}")
    print(r"\label{tab:forkbench_composition}")
    print(r"\begin{tabular}{lrrcc}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{\textbf{Coverage}} & \multicolumn{2}{c}{\textbf{Dialog Length}} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"\textbf{Split} & \textbf{\#Dialogs} & \textbf{\%Total} & \textbf{\#Turns} & \textbf{\#Words} \\")
    print(r"\midrule")
    body_rows = rows[:-1]
    total_row = rows[-1] if rows else None

    latex_split_map = {
        "WIC forked": r"\WIC forked",
        "SSPC forked": r"\SSPC forked",
        "GR forked": r"\GR forked",
        "CSC forked": r"\CSC forked",
    }

    for split, count, pct, mean_turns, mean_words in body_rows:
        split_label = latex_split_map.get(split, _latex_escape(split))
        print(
            f"{split_label} & {_latex_fmt_int(count)} & {_latex_fmt_pct(pct)} & "
            f"{_latex_fmt(mean_turns)} & {_latex_fmt(mean_words)} " + r"\\")

    if total_row is not None:
        split, count, pct, mean_turns, mean_words = total_row
        print(r"\midrule")
        print(
            f"{_latex_bold(_latex_escape(split))} & {_latex_bold(_latex_fmt_int(count))} & "
            f"{_latex_bold(_latex_fmt_pct(pct))} & {_latex_bold(_latex_fmt(mean_turns))} & "
            f"{_latex_bold(_latex_fmt(mean_words))} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _print_latex_table_2(rows):
    print("% Requires: \\usepackage{booktabs,multirow}")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\caption{Task coverage snapshot. \#Turns and \#Words denote average dialog length in turns and words, respectively. Detailed split-aware statistics are reported in Table~\ref{tab:forkbench_task_details}.}")
    print(r"\label{tab:forkbench_task_snapshot}")
    print(r"\begin{tabular}{lrrcc}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{\textbf{Coverage}} & \multicolumn{2}{c}{\textbf{Dialog Length}} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"\textbf{Task} & \textbf{\#Dialogs} & \textbf{\%Total} & \textbf{\#Turns} & \textbf{\#Words} \\")
    print(r"\midrule")
    body_rows = rows[:-1]
    total_row = rows[-1] if rows else None
    for task, count, pct, mean_turns, mean_words in body_rows:
        print(
            f"{_latex_escape(task)} & {_latex_fmt_int(count)} & {_latex_fmt_pct(pct)} & "
            f"{_latex_fmt(mean_turns)} & {_latex_fmt(mean_words)} \\\\")
    if total_row is not None:
        task, count, pct, mean_turns, mean_words = total_row
        print(r"\midrule")
        print(
            f"{_latex_bold(_latex_escape(task))} & {_latex_bold(_latex_fmt_int(count))} & "
            f"{_latex_bold(_latex_fmt_pct(pct))} & {_latex_bold(_latex_fmt(mean_turns))} & "
            f"{_latex_bold(_latex_fmt(mean_words))} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _print_latex_table_3(rows):
    print("% Requires: \\usepackage{booktabs,multirow}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\caption{Task-level detailed statistics. Task labels are shown as task\_name (abbr). In each group, \# denotes number of dialogs, Mean Turns denotes average dialog length in turns, and Mean Words denotes average dialog length in words. Conformant corresponds to non-hallucinated dialogs; Forks corresponds to all forked variants; Total aggregates conformant and forks.}")
    print(r"\label{tab:forkbench_task_details}")
    print(r"\begin{tabular}{lccccccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{Conformant}} & \multicolumn{3}{c}{\textbf{Forks}} & \multicolumn{3}{c}{\textbf{Total}} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    print(
        r"\textbf{Task} & \textbf{\#} & \textbf{Mean Turns} & \textbf{Mean Words} & \textbf{\#} & \textbf{Mean Turns} & \textbf{Mean Words} & \textbf{\#} & \textbf{Mean Turns} & \textbf{Mean Words} \\")
    print(r"\midrule")
    body_rows = rows[:-1]
    total_row = rows[-1] if rows else None
    for row in body_rows:
        _domain, task, c_n, c_turns, c_words, f_n, f_turns, f_words, t_n, t_turns, t_words = row
        print(
            f"{_latex_escape(task)} & "
            f"{_latex_fmt_int(c_n)} & {_latex_fmt(c_turns)} & {_latex_fmt(c_words)} & "
            f"{_latex_fmt_int(f_n)} & {_latex_fmt(f_turns)} & {_latex_fmt(f_words)} & "
            f"{_latex_fmt_int(t_n)} & {_latex_fmt(t_turns)} & {_latex_fmt(t_words)} " + r"\\")
    if total_row is not None:
        _domain, task, c_n, c_turns, c_words, f_n, f_turns, f_words, t_n, t_turns, t_words = total_row
        print(r"\midrule")
        print(
            f"{_latex_bold(_latex_escape(task))} & "
            f"{_latex_bold(_latex_fmt_int(c_n))} & {_latex_bold(_latex_fmt(c_turns))} & {_latex_bold(_latex_fmt(c_words))} & "
            f"{_latex_bold(_latex_fmt_int(f_n))} & {_latex_bold(_latex_fmt(f_turns))} & {_latex_bold(_latex_fmt(f_words))} & "
            f"{_latex_bold(_latex_fmt_int(t_n))} & {_latex_bold(_latex_fmt(t_turns))} & {_latex_bold(_latex_fmt(t_words))} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def first_difference_turn(original_turns, fork_turns):
    min_len = min(len(original_turns), len(fork_turns))
    for idx in range(min_len):
        if as_text(original_turns[idx]) != as_text(fork_turns[idx]):
            return idx + 1
    if len(original_turns) != len(fork_turns):
        return min_len + 1
    return 0


def to_words(text):
    return len(as_text(text).split())


def detect_turn_issues(text):
    text = as_text(text)
    reasons = []

    if not text.strip():
        reasons.append("empty_text")

    # Control characters are almost never valid in normal utterances.
    if any(ord(ch) < 32 and ch not in {"\n", "\t", "\r"} for ch in text):
        reasons.append("control_char")

    for pattern, label in SUSPICIOUS_UTTERANCE_PATTERNS:
        if pattern.search(text):
            reasons.append(label)

    # Safer guard: parenthetical note with instruction-like keywords is likely prompt leakage.
    if text.strip().startswith("(") and "system note" in text.lower():
        if re.search(r"\b(happy\s*path|persona|assistant\s+should|proceed\s+with\s+the\s+booking)\b", text, re.IGNORECASE):
            reasons.append("instructional_parenthetical_note")

    # Additional guardrail: explicit meta-instruction language that commonly appears
    # inside leaked system notes (kept conjunctive to reduce false positives).
    if re.search(r"\bsystem\s+note\b", text, re.IGNORECASE) and re.search(
        r"\b(persona\s+must|assistant\s+should|happy\s+path|follow\s+the\s+instructions?)\b",
        text,
        re.IGNORECASE,
    ):
        reasons.append("meta_instruction_text")

    # Typical long planning blocks often leak as numbered lists with meta headers.
    if re.search(r"\n\s*\d+\.\s+", text) and len(text.split()) >= 40:
        reasons.append("long_numbered_meta_block")

    return sorted(set(reasons))


def collect_records(root):
    records = []
    original_turns_by_task_and_id = {}
    duplicate_entries = Counter()
    orphan_forks = 0
    missing_segment_dirs = Counter()
    issue_rows = []
    issue_dialog_ids = set()
    issue_reason_counts = Counter()
    issue_turns_total = 0

    tasks = [
        name
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]

    total_folders = len(tasks)
    with tqdm(total=total_folders, desc="Processing tasks", unit="task") as pbar:
        for task in tasks:
            task_path = os.path.join(root, task)
            domain = TASK_TO_DOMAIN.get(task, "unknown")

            originals_path = os.path.join(task_path, "originals")
            if os.path.isdir(originals_path):
                pbar.set_description(f"Loading originals/{task}")
                dialogs = Dialog.from_folder(originals_path)
                for dialog in dialogs:
                    d_key = dialog_key(dialog)
                    n_turns = len(dialog.turns)
                    n_words = 0
                    n_turns_system = 0
                    n_turns_user = 0
                    n_words_system = 0
                    n_words_user = 0
                    turns_text = []
                    dialog_issue_turn_count = 0
                    dialog_issue_reasons = set()

                    for turn in dialog.turns:
                        text = as_text(turn.text)
                        turns_text.append(text)
                        issue_reasons = detect_turn_issues(text)
                        if issue_reasons:
                            dialog_issue_turn_count += 1
                            issue_turns_total += 1
                            dialog_issue_reasons.update(issue_reasons)
                            for reason in issue_reasons:
                                issue_reason_counts[reason] += 1
                        words = to_words(text)
                        n_words += words
                        role = speaker_kind(getattr(turn, "speaker", None))
                        if role == "system":
                            n_turns_system += 1
                            n_words_system += words
                        elif role == "user":
                            n_turns_user += 1
                            n_words_user += words

                    complete_value = bool(getattr(dialog, "complete", False))
                    path_value = as_text(getattr(dialog, "_path", ""))
                    if dialog_issue_turn_count > 0:
                        issue_dialog_ids.add((task, d_key, "original"))
                        if path_value:
                            for reason in dialog_issue_reasons:
                                issue_rows.append((reason, path_value))
                    records.append(
                        {
                            "task": task,
                            "domain": domain,
                            "split": "original",
                            "hallucination_type": "originals",
                            "segment": 0,
                            "dialog_id": d_key,
                            "parent_id": "",
                            "file_path": path_value,
                            "complete": complete_value,
                            "n_turns": n_turns,
                            "n_turns_system": n_turns_system,
                            "n_turns_user": n_turns_user,
                            "n_words": n_words,
                            "n_words_system": n_words_system,
                            "n_words_user": n_words_user,
                            "delta_turns": 0,
                            "delta_words": 0,
                            "first_diff_turn": 0,
                            "divergence_depth": 0.0,
                            "has_turn_issue": dialog_issue_turn_count > 0,
                            "n_turn_issues": dialog_issue_turn_count,
                        }
                    )
                    original_turns_by_task_and_id[(task, d_key)] = {
                        "turns": turns_text,
                        "n_turns": n_turns,
                        "n_words": n_words,
                    }

            hallucination_types = [
                name
                for name in sorted(os.listdir(task_path))
                if os.path.isdir(os.path.join(task_path, name)) and name != "originals"
            ]

            for hallucination_type in hallucination_types:
                h_path = os.path.join(task_path, hallucination_type, "n_hallucinations_4")
                if not os.path.isdir(h_path):
                    continue

                segments = [
                    name for name in sorted(os.listdir(h_path)) if os.path.isdir(os.path.join(h_path, name))
                ]
                for expected in EXPECTED_SEGMENTS:
                    if expected not in segments:
                        missing_segment_dirs[(task, hallucination_type)] += 1

                for segment_name in segments:
                    segment_value = safe_int_segment(segment_name)
                    segment_path = os.path.join(h_path, segment_name)
                    pbar.set_description(f"Loading {task}/{hallucination_type}/{segment_name}")
                    dialogs = Dialog.from_folder(segment_path)

                    for dialog in dialogs:
                        d_key = dialog_key(dialog)
                        parent_id = as_text(getattr(dialog, "parentId", ""))
                        if parent_id in {"None", "none", "null"}:
                            parent_id = ""

                        n_turns = len(dialog.turns)
                        n_words = 0
                        n_turns_system = 0
                        n_turns_user = 0
                        n_words_system = 0
                        n_words_user = 0
                        turns_text = []
                        dialog_issue_turn_count = 0
                        dialog_issue_reasons = set()

                        for turn in dialog.turns:
                            text = as_text(turn.text)
                            turns_text.append(text)
                            issue_reasons = detect_turn_issues(text)
                            if issue_reasons:
                                dialog_issue_turn_count += 1
                                issue_turns_total += 1
                                dialog_issue_reasons.update(issue_reasons)
                                for reason in issue_reasons:
                                    issue_reason_counts[reason] += 1
                            words = to_words(text)
                            n_words += words
                            role = speaker_kind(getattr(turn, "speaker", None))
                            if role == "system":
                                n_turns_system += 1
                                n_words_system += words
                            elif role == "user":
                                n_turns_user += 1
                                n_words_user += words

                        ref = original_turns_by_task_and_id.get((task, d_key))
                        if ref is None:
                            orphan_forks += 1
                            delta_turns = 0
                            delta_words = 0
                            first_diff = 0
                            divergence_depth = 0.0
                        else:
                            delta_turns = n_turns - ref["n_turns"]
                            delta_words = n_words - ref["n_words"]
                            first_diff = first_difference_turn(ref["turns"], turns_text)
                            if first_diff > 0 and n_turns > 0:
                                divergence_depth = (n_turns - first_diff + 1) / n_turns
                            else:
                                divergence_depth = 0.0

                        complete_value = bool(getattr(dialog, "complete", False))
                        path_value = as_text(getattr(dialog, "_path", ""))
                        if dialog_issue_turn_count > 0:
                            issue_dialog_ids.add((task, d_key, f"fork:{hallucination_type}:seg{segment_value}"))
                            if path_value:
                                for reason in dialog_issue_reasons:
                                    issue_rows.append((reason, path_value))
                        entry_key = (task, hallucination_type, segment_value, d_key)
                        duplicate_entries[entry_key] += 1

                        records.append(
                            {
                                "task": task,
                                "domain": domain,
                                "split": "fork",
                                "hallucination_type": hallucination_type,
                                "segment": segment_value if segment_value is not None else -1,
                                "dialog_id": d_key,
                                "parent_id": parent_id,
                                "file_path": path_value,
                                "complete": complete_value,
                                "n_turns": n_turns,
                                "n_turns_system": n_turns_system,
                                "n_turns_user": n_turns_user,
                                "n_words": n_words,
                                "n_words_system": n_words_system,
                                "n_words_user": n_words_user,
                                "delta_turns": delta_turns,
                                "delta_words": delta_words,
                                "first_diff_turn": first_diff,
                                "divergence_depth": divergence_depth,
                                "has_turn_issue": dialog_issue_turn_count > 0,
                                "n_turn_issues": dialog_issue_turn_count,
                            }
                        )
            pbar.update(1)

    duplicate_count = sum(1 for count in duplicate_entries.values() if count > 1)
    issue_rows_sorted = sorted(set(issue_rows), key=lambda item: (item[0], item[1]))
    issue_report = {
        "dialog_count": len(issue_dialog_ids),
        "turn_count": issue_turns_total,
        "reason_counts": issue_reason_counts,
        "rows": issue_rows_sorted,
    }
    return (
        records,
        orphan_forks,
        duplicate_count,
        missing_segment_dirs,
        issue_report,
    )


def rows_from_counter(counter, total, key_name):
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        pct = (count / total * 100.0) if total > 0 else 0.0
        rows.append([key, count, pct])
    return rows


def split_display_label(split_value):
    if split_value == "original":
        return "conformant"
    return split_value


def task_with_abbrev(task_name):
    abbr = TASK_ABBREV.get(task_name)
    if not abbr:
        return task_name
    return f"{task_name} ({abbr})"


def group_summary(records, key_name, value_name):
    groups = defaultdict(list)
    for rec in records:
        groups[rec[key_name]].append(rec[value_name])
    rows = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        s = summarize_numeric(values)
        rows.append([group, s["n"], s["mean"], s["median"], s["std"], s["min"], s["max"], s["p25"], s["p75"]])
    return rows


def main():
    args = build_argument_parser().parse_args()
    sections = parse_sections(args.sections)

    root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(root):
        raise ValueError(f"Dataset root does not exist: {root}")

    (
        records,
        orphan_forks,
        duplicate_count,
        missing_segment_dirs,
        issue_report,
    ) = collect_records(root)
    if not records:
        raise ValueError(f"No dialogues found under: {root}")

    total_dialogs = len(records)
    total_turns_all = sum(rec["n_turns"] for rec in records)
    total_words_all = sum(rec["n_words"] for rec in records)
    split_counts = Counter(rec["split"] for rec in records)
    type_counts = Counter(rec["hallucination_type"] for rec in records if rec["split"] == "fork")
    segment_counts = Counter(rec["segment"] for rec in records if rec["split"] == "fork")
    task_counts = Counter(rec["task"] for rec in records)
    domain_counts = Counter(rec["domain"] for rec in records)

    task_original_ids = defaultdict(set)
    task_fork_coverage = defaultdict(lambda: defaultdict(set))
    forks_per_original = defaultdict(set)
    for rec in records:
        if rec["split"] == "original":
            task_original_ids[rec["task"]].add(rec["dialog_id"])
        else:
            combo = (rec["hallucination_type"], rec["segment"])
            task_fork_coverage[rec["task"]][combo].add(rec["dialog_id"])
            forks_per_original[(rec["task"], rec["dialog_id"])].add(combo)

    realized_forks_distribution = Counter()
    missing_combinations = []
    expected_combo_count = 0
    for task, original_ids in task_original_ids.items():
        combos = sorted(task_fork_coverage[task].keys(), key=lambda item: (item[0], item[1]))
        expected_combo_count += len(original_ids) * 16
        for original_id in original_ids:
            realized = len(forks_per_original.get((task, original_id), set()))
            realized_forks_distribution[realized] += 1
        for combo in combos:
            observed = len(task_fork_coverage[task][combo])
            missing = max(0, len(original_ids) - observed)
            if missing > 0:
                missing_combinations.append([task, combo[0], combo[1], missing])

    realized_forks_total = sum(len(combos) for combos in forks_per_original.values())
    complete_flags = Counter(rec["complete"] for rec in records)

    # Default mode: only paper-ready tables (main manuscript + appendix).
    paper_split_rows = []
    originals_count = split_counts.get("original", 0)
    conformant_records = [rec for rec in records if rec["split"] == "original"]
    paper_split_rows.append([
        "Conformant",
        originals_count,
        (originals_count / total_dialogs * 100.0) if total_dialogs else 0.0,
        summarize_numeric([rec["n_turns"] for rec in conformant_records])["mean"],
        summarize_numeric([rec["n_words"] for rec in conformant_records])["mean"],
    ])
    for key in [
        "workflow_identity_corruption",
        "step_selection_policy_corruption",
        "goal_replacement",
        "conflicting_system_constraints",
    ]:
        value = type_counts.get(key, 0)
        fork_type_records = [
            rec for rec in records
            if rec["split"] == "fork" and rec["hallucination_type"] == key
        ]
        paper_split_rows.append([
            HALLUCINATION_DISPLAY.get(key, key),
            value,
            (value / total_dialogs * 100.0) if total_dialogs else 0.0,
            summarize_numeric([rec["n_turns"] for rec in fork_type_records])["mean"],
            summarize_numeric([rec["n_words"] for rec in fork_type_records])["mean"],
        ])
    paper_split_rows.append([
        "Total",
        total_dialogs,
        100.0,
        summarize_numeric([rec["n_turns"] for rec in records])["mean"],
        summarize_numeric([rec["n_words"] for rec in records])["mean"],
    ])

    task_orig_turns = defaultdict(list)
    task_fork_turns = defaultdict(list)
    task_orig_words = defaultdict(list)
    task_fork_words = defaultdict(list)
    task_total_turns = defaultdict(list)
    task_total_words = defaultdict(list)
    for rec in records:
        task_total_turns[rec["task"]].append(rec["n_turns"])
        task_total_words[rec["task"]].append(rec["n_words"])
        if rec["split"] == "original":
            task_orig_turns[rec["task"]].append(rec["n_turns"])
            task_orig_words[rec["task"]].append(rec["n_words"])
        elif rec["split"] == "fork":
            task_fork_turns[rec["task"]].append(rec["n_turns"])
            task_fork_words[rec["task"]].append(rec["n_words"])

    appendix_rows = []
    for task in sorted(task_counts.keys(), key=lambda item: (TASK_TO_DOMAIN.get(item, "unknown"), item)):
        originals_task = len(task_original_ids.get(task, set()))
        forks_task = task_counts[task] - originals_task
        conf_turns = task_orig_turns.get(task, [])
        fork_turns = task_fork_turns.get(task, [])
        conf_words = task_orig_words.get(task, [])
        fork_words = task_fork_words.get(task, [])
        appendix_rows.append(
            [
                TASK_TO_DOMAIN.get(task, "unknown"),
                task_with_abbrev(task),
                originals_task,
                summarize_numeric(conf_turns)["mean"],
                summarize_numeric(conf_words)["mean"],
                forks_task,
                summarize_numeric(fork_turns)["mean"],
                summarize_numeric(fork_words)["mean"],
                task_counts[task],
                summarize_numeric(conf_turns + fork_turns)["mean"],
                summarize_numeric(conf_words + fork_words)["mean"],
            ]
        )

    total_conformant = split_counts.get("original", 0)
    total_forks = split_counts.get("fork", 0)
    total_count = total_conformant + total_forks
    global_conformant_turns_mean = summarize_numeric(
        [rec["n_turns"] for rec in records if rec["split"] == "original"]
    )["mean"]
    global_fork_turns_mean = summarize_numeric(
        [rec["n_turns"] for rec in records if rec["split"] == "fork"]
    )["mean"]
    global_conformant_words_mean = summarize_numeric(
        [rec["n_words"] for rec in records if rec["split"] == "original"]
    )["mean"]
    global_fork_words_mean = summarize_numeric(
        [rec["n_words"] for rec in records if rec["split"] == "fork"]
    )["mean"]
    global_total_turns_mean = summarize_numeric([rec["n_turns"] for rec in records])["mean"]
    global_total_words_mean = summarize_numeric([rec["n_words"] for rec in records])["mean"]

    paper_task_rows = []
    for task in sorted(task_counts.keys(), key=lambda item: (TASK_TO_DOMAIN.get(item, "unknown"), item)):
        count = task_counts[task]
        pct = (count / total_dialogs * 100.0) if total_dialogs else 0.0
        paper_task_rows.append(
            [
                task_with_abbrev(task),
                count,
                pct,
                summarize_numeric(task_total_turns.get(task, []))["mean"],
                summarize_numeric(task_total_words.get(task, []))["mean"],
            ]
        )
    paper_task_rows.append(["TOTAL", total_dialogs, 100.0, global_total_turns_mean, global_total_words_mean])

    appendix_rows.append(
        [
            "ALL",
            "TOTAL",
            total_conformant,
            global_conformant_turns_mean,
            global_conformant_words_mean,
            total_forks,
            global_fork_turns_mean,
            global_fork_words_mean,
            total_count,
            global_total_turns_mean,
            global_total_words_mean,
        ]
    )

    if args.latex:
        _print_latex_table_1(paper_split_rows)
        print()
        _print_latex_table_2(paper_task_rows)
        print()
        _print_latex_table_3(appendix_rows)
    else:
        print_table(
            "Table 2 (Main Manuscript): Dataset Composition",
            ["Split", "#Dialogs", "%Total", "MeanDialogLenTurns", "MeanDialogLenWords"],
            paper_split_rows,
        )
        print_table(
            "Table 3 (Main Manuscript): Task Coverage Snapshot",
            ["Task", "#Dialogs", "%Total", "MeanDialogLenTurns", "MeanDialogLenWords"],
            paper_task_rows,
        )
        print_table(
            "Table 7 (Appendix): Task-Level Composition and Length",
            [
                "Domain",
                "Task",
                "#Conformant",
                "MeanDialogLenTurns Conf",
                "MeanDialogLenWords Conf",
                "#Forks",
                "MeanDialogLenTurns Fork",
                "MeanDialogLenWords Fork",
                "#Total",
                "MeanDialogLenTurns Total",
                "MeanDialogLenWords Total",
            ],
            appendix_rows,
        )
        print("\nNote: Table 7 reports detailed per-task split and dialog-length statistics.")

    print("\nCoverage checks:")
    print(f"  missing combinations: {max(0, expected_combo_count - realized_forks_total)}")
    print(f"  orphan forks: {orphan_forks}")
    print(f"  duplicate fork entries: {duplicate_count}")
    print(f"  missing segment dirs: {sum(missing_segment_dirs.values())}")

    print("\nPotential turn-content issues:")
    print(f"  dialogs with issue turns: {issue_report['dialog_count']}")
    print(f"  issue turns total: {issue_report['turn_count']}")
    if issue_report["reason_counts"]:
        top_reasons = sorted(issue_report["reason_counts"].items(), key=lambda item: (-item[1], item[0]))
        for reason, count in top_reasons:
            print(f"  - {reason}: {count}")
    else:
        print("  no suspicious turn content detected")

    if args.issues_output_csv:
        output_path = os.path.abspath(args.issues_output_csv)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["issue_type", "path"])
            writer.writerows(issue_report["rows"])
        print(f"\nSaved issue report ({len(issue_report['rows'])} rows) to: {output_path}")

    if args.markdown and not args.latex:
        print("\n\n# Markdown Tables")
        print_markdown_table(
            "Table 2 (Main Manuscript): Dataset Composition",
            ["Split", "#Dialogs", "%Total", "MeanDialogLenTurns", "MeanDialogLenWords"],
            paper_split_rows,
        )
        print_markdown_table(
            "Table 3 (Main Manuscript): Task Coverage Snapshot",
            ["Task", "#Dialogs", "%Total", "MeanDialogLenTurns", "MeanDialogLenWords"],
            paper_task_rows,
        )
        print_markdown_table(
            "Table 7 (Appendix): Task-Level Composition and Length",
            [
                "Domain",
                "Task",
                "#Conformant",
                "MeanDialogLenTurns Conf",
                "MeanDialogLenWords Conf",
                "#Forks",
                "MeanDialogLenTurns Fork",
                "MeanDialogLenWords Fork",
                "#Total",
                "MeanDialogLenTurns Total",
                "MeanDialogLenWords Total",
            ],
            appendix_rows,
        )

    has_coverage_issue = (
        orphan_forks > 0
        or duplicate_count > 0
        or (expected_combo_count - realized_forks_total) > 0
        or sum(missing_segment_dirs.values()) > 0
    )
    if args.strict_coverage and has_coverage_issue:
        raise SystemExit(1)

    if not args.verbose:
        print("\nOverall totals:")
        print(f"  total turns: {total_turns_all}")
        print(f"  total words: {total_words_all}")
        return

    if "overview" in sections:
        overview_rows = [
            ["Root", root],
            ["Tasks", len(task_counts)],
            ["Domains", len(domain_counts)],
            ["Total Dialogues", total_dialogs],
            ["Conformant Dialogues", split_counts.get("original", 0)],
            ["Forked Dialogues", split_counts.get("fork", 0)],
            ["Conformant/Fork Ratio", f"{split_counts.get('original', 0)}/{split_counts.get('fork', 0)}"],
            ["Complete=True", complete_flags.get(True, 0)],
            ["Complete=False", complete_flags.get(False, 0)],
        ]
        print_table("Dataset Overview", ["Metric", "Value"], overview_rows)

    split_rows = rows_from_counter(split_counts, total_dialogs, "split")
    split_rows = [[split_display_label(row[0]), row[1], row[2]] for row in split_rows]
    if "split" in sections:
        print_table("Split Composition", ["Split", "Count", "Percent"], split_rows)

    type_rows = rows_from_counter(type_counts, split_counts.get("fork", 0), "hallucination_type")
    if "type" in sections:
        print_table("Fork Composition by Hallucination Type", ["Hallucination Type", "Count", "Percent"], type_rows)

    segment_rows = rows_from_counter(segment_counts, split_counts.get("fork", 0), "segment")
    if "segment" in sections:
        print_table("Fork Composition by Segment", ["Segment", "Count", "Percent"], segment_rows)

    matrix_rows = []
    segment_labels = sorted(segment_counts.keys())
    for h_type in sorted(type_counts.keys()):
        row = [h_type]
        h_total = 0
        for segment in segment_labels:
            value = sum(
                1
                for rec in records
                if rec["split"] == "fork"
                and rec["hallucination_type"] == h_type
                and rec["segment"] == segment
            )
            h_total += value
            row.append(value)
        row.append(h_total)
        matrix_rows.append(row)
    if "matrix" in sections:
        matrix_headers = ["Hallucination Type"] + [f"seg_{seg}" for seg in segment_labels] + ["total"]
        print_table("Hallucination Type x Segment", matrix_headers, matrix_rows)

    task_rows = []
    for task, count in sorted(task_counts.items(), key=lambda item: (-item[1], item[0])):
        originals_count = len(task_original_ids.get(task, set()))
        forks_count = count - originals_count
        forks_per_original_mean = (forks_count / originals_count) if originals_count > 0 else 0.0
        task_rows.append([task, TASK_TO_DOMAIN.get(task, "unknown"), count, originals_count, forks_count, forks_per_original_mean])
    if args.top_k_tasks > 0:
        task_rows = task_rows[: args.top_k_tasks]
    if "task" in sections:
        print_table(
            "Per-Task Composition",
            ["Task", "Domain", "Total", "Conformant", "Forks", "Avg forks/conformant"],
            task_rows,
        )

    domain_rows = []
    for domain, count in sorted(domain_counts.items(), key=lambda item: (-item[1], item[0])):
        domain_originals = sum(
            1 for rec in records if rec["domain"] == domain and rec["split"] == "original"
        )
        domain_forks = count - domain_originals
        domain_rows.append([domain, count, domain_originals, domain_forks])
    if "domain" in sections:
        print_table("Per-Domain Composition", ["Domain", "Total", "Conformant", "Forks"], domain_rows)

    if "coverage" in sections:
        coverage_rows = [
            ["Expected fork combinations total (16 per conformant)", expected_combo_count],
            ["Realized fork combinations total", realized_forks_total],
            ["Missing combinations total", max(0, expected_combo_count - realized_forks_total)],
            ["Forks missing conformant (orphans)", orphan_forks],
            ["Duplicate fork entries (task,type,segment,id)", duplicate_count],
            ["Missing expected segment directories", sum(missing_segment_dirs.values())],
        ]
        print_table("Coverage and Integrity", ["Metric", "Value"], coverage_rows)

        dist_rows = [[count, freq] for count, freq in sorted(realized_forks_distribution.items())]
        print_table("Realized Forks Per Conformant", ["Fork combos per conformant", "Conformant dialogues"], dist_rows)

        missing_rows = sorted(missing_combinations, key=lambda item: (-item[3], item[0], item[1], item[2]))
        if missing_rows:
            print_table(
                "Missing Combinations by Task/Type/Segment",
                ["Task", "Hallucination Type", "Segment", "Missing conformant"],
                missing_rows,
            )
        else:
            print("\nNo missing combinations detected.")

    if "length" in sections:
        length_rows = []
        for split in sorted(split_counts.keys()):
            split_records = [rec for rec in records if rec["split"] == split]
            turns_summary = summarize_numeric([rec["n_turns"] for rec in split_records])
            words_summary = summarize_numeric([rec["n_words"] for rec in split_records])
            sys_turn_ratio = (
                sum(rec["n_turns_system"] for rec in split_records)
                / max(1, sum(rec["n_turns"] for rec in split_records))
            )
            user_turn_ratio = (
                sum(rec["n_turns_user"] for rec in split_records)
                / max(1, sum(rec["n_turns"] for rec in split_records))
            )
            length_rows.append(
                [
                    split,
                    turns_summary["mean"],
                    turns_summary["median"],
                    turns_summary["std"],
                    turns_summary["min"],
                    turns_summary["max"],
                    words_summary["mean"],
                    words_summary["median"],
                    words_summary["std"],
                    words_summary["min"],
                    words_summary["max"],
                    sys_turn_ratio,
                    user_turn_ratio,
                ]
            )
        print_table(
            "Length and Speaker Balance by Split",
            [
                "Split",
                "turn_mean",
                "turn_med",
                "turn_std",
                "turn_min",
                "turn_max",
                "word_mean",
                "word_med",
                "word_std",
                "word_min",
                "word_max",
                "sys_turn_ratio",
                "user_turn_ratio",
            ],
            length_rows,
        )

        fork_records = [rec for rec in records if rec["split"] == "fork"]
        by_type_turns = group_summary(fork_records, "hallucination_type", "n_turns")
        print_table(
            "Fork Turn Distribution by Hallucination Type",
            ["Hallucination Type", "n", "mean", "median", "std", "min", "max", "p25", "p75"],
            by_type_turns,
        )
        by_segment_turns = group_summary(fork_records, "segment", "n_turns")
        print_table(
            "Fork Turn Distribution by Segment",
            ["Segment", "n", "mean", "median", "std", "min", "max", "p25", "p75"],
            by_segment_turns,
        )

    if "divergence" in sections:
        fork_records = [rec for rec in records if rec["split"] == "fork"]
        diff_records = [rec for rec in fork_records if rec["first_diff_turn"] > 0]

        summary_rows = [
            ["Fork dialogues", len(fork_records)],
            ["Fork dialogues with detected difference", len(diff_records)],
            [
                "Difference detection ratio",
                (len(diff_records) / len(fork_records) * 100.0) if fork_records else 0.0,
            ],
            ["Delta turns mean", summarize_numeric([rec["delta_turns"] for rec in fork_records])["mean"]],
            ["Delta words mean", summarize_numeric([rec["delta_words"] for rec in fork_records])["mean"]],
            [
                "Divergence depth mean",
                summarize_numeric([rec["divergence_depth"] for rec in diff_records])["mean"],
            ],
        ]
        print_table("Fork Divergence Summary", ["Metric", "Value"], summary_rows)

        by_type_rows = []
        for h_type in sorted(type_counts.keys()):
            subset = [rec for rec in fork_records if rec["hallucination_type"] == h_type]
            subset_diff = [rec for rec in subset if rec["first_diff_turn"] > 0]
            by_type_rows.append(
                [
                    h_type,
                    len(subset),
                    len(subset_diff),
                    summarize_numeric([rec["first_diff_turn"] for rec in subset_diff])["mean"],
                    summarize_numeric([rec["delta_turns"] for rec in subset])["mean"],
                    summarize_numeric([rec["delta_words"] for rec in subset])["mean"],
                    summarize_numeric([rec["divergence_depth"] for rec in subset_diff])["mean"],
                ]
            )
        print_table(
            "Divergence by Hallucination Type",
            [
                "Hallucination Type",
                "n",
                "n_diff",
                "first_diff_mean",
                "delta_turns_mean",
                "delta_words_mean",
                "div_depth_mean",
            ],
            by_type_rows,
        )

        by_segment_rows = []
        for segment in sorted(segment_counts.keys()):
            subset = [rec for rec in fork_records if rec["segment"] == segment]
            subset_diff = [rec for rec in subset if rec["first_diff_turn"] > 0]
            by_segment_rows.append(
                [
                    segment,
                    len(subset),
                    len(subset_diff),
                    summarize_numeric([rec["first_diff_turn"] for rec in subset_diff])["mean"],
                    summarize_numeric([rec["delta_turns"] for rec in subset])["mean"],
                    summarize_numeric([rec["delta_words"] for rec in subset])["mean"],
                    summarize_numeric([rec["divergence_depth"] for rec in subset_diff])["mean"],
                ]
            )
        print_table(
            "Divergence by Segment",
            [
                "Segment",
                "n",
                "n_diff",
                "first_diff_mean",
                "delta_turns_mean",
                "delta_words_mean",
                "div_depth_mean",
            ],
            by_segment_rows,
        )

    if args.markdown:
        print("\n\n# Markdown Tables")
        print_markdown_table("Split Composition", ["Split", "Count", "Percent"], split_rows)
        print_markdown_table(
            "Fork Composition by Hallucination Type",
            ["Hallucination Type", "Count", "Percent"],
            type_rows,
        )
        print_markdown_table("Fork Composition by Segment", ["Segment", "Count", "Percent"], segment_rows)
        print_markdown_table(
            "Per-Task Composition",
            ["Task", "Domain", "Total", "Conformant", "Forks", "Avg forks/conformant"],
            task_rows,
        )
        print_markdown_table("Per-Domain Composition", ["Domain", "Total", "Conformant", "Forks"], domain_rows)

        if "matrix" in sections:
            matrix_headers = ["Hallucination Type"] + [f"seg_{seg}" for seg in segment_labels] + ["total"]
            print_markdown_table("Hallucination Type x Segment", matrix_headers, matrix_rows)

    print("\nOverall totals:")
    print(f"  total turns: {total_turns_all}")
    print(f"  total words: {total_words_all}")

    # strict coverage already handled before optional verbose block.


if __name__ == "__main__":
    main()
