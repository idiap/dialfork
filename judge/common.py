# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Paths, the selection split, and reading the judge's answer files."""
import glob
import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the repository
DATASET = os.path.join(ROOT, "datasets", "dialfork")
HUMAN_DB = os.path.join(ROOT, "annotool", "data", "dialfork_human_study.sqlite")
HUMAN_CSV = os.path.join(ROOT, "annotool", "data", "dialfork_human_judgments.csv")
GENERATE_PY = os.path.join(ROOT, "generation", "generate.py")

# released with the code: the answers of the paper's judges, and the selected judge
ANSWERS = os.path.join(ROOT, "judge", "answers")
STUDY_ANSWERS = os.path.join(ANSWERS, "study")
FULL_ANSWERS = os.path.join(ANSWERS, "full")
FROZEN = os.path.join(ROOT, "judge", "frozen_judge.json")

# built or written by the scripts (not versioned)
WORK = os.path.join(ROOT, "results", "judge")
ITEMS = os.path.join(WORK, "items.jsonl")
FULL_ITEMS = os.path.join(WORK, "full_items.jsonl")
OUTPUTS = os.path.join(WORK, "outputs")

# the judge is selected on three tasks of the study, and reported on the other two
SELECT_TASKS = ("hotel_book", "bank_fraud_report", "weather")
REPORT_TASKS = ("plane_book", "party_plan")

# Gemma4-31B generated DialFork: excluded from the selection, to avoid self-verification
GENERATOR = "gemma4:31b"


def split_of(task):
    return "select" if task in SELECT_TASKS else "report" if task in REPORT_TASKS else None


def model_dir(model):
    return model.replace("/", "_").replace(":", "_")


def answer_files(folder):
    """A run's answer files: judgments.jsonl, its shards, or their .gz release copies."""
    return sorted(glob.glob(os.path.join(folder, "judgments*.jsonl"))
                  + glob.glob(os.path.join(folder, "judgments*.jsonl.gz")))


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_frozen():
    with open(FROZEN, encoding="utf-8") as fh:
        return json.load(fh)
