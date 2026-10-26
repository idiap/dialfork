# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Smoke tests for the release: evaluate.py on the shipped prediction files, and the
LLM judge's released answers.

    python scripts/download_data.py --no-star   # the ground truth, once
    python tests/test_release.py

Standard library only. Exits with a non-zero status on the first failure.
"""
import csv
import glob
import gzip
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(task, path):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "scores.json")
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluate.py"), "--task", str(task),
                        path, "--json", out], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        with open(out) as fh:
            return json.load(fh)


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main():
    if not os.path.isdir(os.path.join(ROOT, "datasets", "dialfork")):
        sys.exit("run `python scripts/download_data.py --no-star` first")

    print("Task 1")
    s = evaluate(1, "predictions/task1/flowkl.csv.gz")
    check(round(s["avg"], 3) == 0.882, "FlowKL averages tau = 0.882, as in the paper (%.3f)" % s["avg"])
    s = evaluate(1, "predictions/task1/fbd.csv.gz")
    check(round(s["avg"], 3) == 0.913, "FBD averages tau = 0.913, as in the paper (%.3f)" % s["avg"])

    print("Task 2")
    s = evaluate(2, "predictions/task2/flowscore__bge-m3.csv.gz")
    check(round(s["avg"], 3) == 0.845, "FlowScore + BGE-M3 averages tau = 0.845, as in the paper (%.3f)" % s["avg"])
    s = evaluate(2, "predictions/task2/shared-prefix.csv.gz")
    check(round(s["avg"], 3) == 0.682, "the shared-prefix control averages tau = 0.682, as in the paper (%.3f)" % s["avg"])

    print("Task 3")
    s = evaluate(3, "predictions/task3/first-turn.csv.gz")
    check(round(s["score"], 2) == 0.0 and round(s["hall_cov"], 2) == 100.0,
          "the first-turn baseline flags every dialogue and scores 0")
    files = sorted(glob.glob(os.path.join(ROOT, "predictions", "task3", "*.csv.gz")))
    for p in files:
        s = evaluate(3, p)
        check(0.0 <= s["score"] <= 100.0, "%s is scored (0-100)" % os.path.basename(p))

    # the LLM judge on the whole corpus (docs/llm_judge.md), as in the paper's appendix
    with gzip.open(os.path.join(ROOT, "judge", "answers", "dialfork_llm_judge.csv.gz"), "rt",
                   encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    check(len(rows) == 36069 and len({r["id"] for r in rows}) == len(rows),
          "the LLM judge answered 36,069 dialogues, once each")
    check(all(0 <= int(r["slider"]) <= 100 for r in rows), "every judge rating is in 0-100")
    true = [r for r in rows if r["kind"] == "fork" and r["is_foil"] == "0"]
    foil = [r for r in rows if r["kind"] == "conformant"]
    rate = lambda rs: 100.0 * sum(r["appears"] in ("yes", "partly") for r in rs) / len(rs)  # noqa: E731
    check(round(rate(true)) == 84 and round(rate(foil), 1) == 2.0,
          "the judge confirms 84% of true instructions and 2.0% of foils on conformant dialogues")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
