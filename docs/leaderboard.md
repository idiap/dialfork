<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# 🏆 Leaderboard

[← back to the README](../README.md)

This is where methods on DialFork meet. It starts with the methods from our paper, and we hope it doesn't stay that way for long. Built something that beats them, or that fails in an interesting way? We'd love to add it.

📄 = from the [DialFork paper](https://arxiv.org/abs/2610.21561) · 🌍 = community submission

## How to add your method

1. **Score it** with [`evaluate.py`](../evaluate.py) on the full test set (no `--allow-missing`).
2. **Open a pull request** that adds:
   - your prediction file to `predictions/taskN/` (a `.csv.gz` is fine);
   - a row to the table below, with a link to your paper or code.
3. **Tell us what your method used.** For Task 3, give the regime: streaming, non-streaming, or non-streaming + workflow. For every task, say whether it learned from or was calibrated on DialFork dialogues, and on which ones.

We re-run `evaluate.py` on your file before merging. One rule: **never train or tune on the labels** (fork position, fork turn, hallucination type). DialFork has a single test split, and those labels are the answers. Using the conformant dialogues as reference data is fine (our streaming detectors do it), as long as you say so.

## Task 1 · Set-to-set distance

Kendall τ, higher is better. [Task page](task1.md)

| | method | WIC | SSPC | GR | CSC | **Avg.** | link |
|---|---|---:|---:|---:|---:|---:|---|
| 📄 | FBD | 0.972 | 0.792 | 0.986 | 0.903 | **0.913** | [Xiang et al., 2021](https://aclanthology.org/2021.findings-acl.193/) |
| 📄 | FlowKL | 0.847 | 0.903 | 0.944 | 0.833 | 0.882 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | PRD | 0.931 | 0.764 | 0.986 | 0.833 | 0.878 | [Xiang et al., 2021](https://aclanthology.org/2021.findings-acl.193/) |
| | *Word-count diff. (control)* | *0.222* | *0.208* | *0.583* | *−0.208* | *0.201* | |
| | *Random (control)* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | |
| | *Turn-count diff. (control)* | *−0.076* | *−0.340* | *0.278* | *−0.208* | *−0.087* | |

## Task 2 · Dialog-to-dialog distance

Kendall τ, higher is better. One row per method, with its best encoder; [the task page](task2.md) has every (method, encoder) pair.

| | method | encoder | WIC | SSPC | GR | CSC | **Avg.** | link |
|---|---|---|---:|---:|---:|---:|---:|---|
| 📄 | FlowScore | BGE-M3 | 0.924 | 0.814 | 0.942 | 0.700 | **0.845** | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Mean-turn emb. | D2F | 0.848 | 0.743 | 0.895 | 0.561 | 0.762 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Full-dialog emb. | D2F | 0.817 | 0.724 | 0.843 | 0.603 | 0.747 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | FlowDTW | D2F | 0.864 | 0.702 | 0.913 | 0.489 | 0.742 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| | *Shared prefix (control)* | – | *0.688* | *0.668* | *0.712* | *0.662* | *0.682* | |
| | *Word-count diff. (control)* | – | *0.220* | *0.242* | *0.390* | *0.040* | *0.223* | |
| | *Turn-count diff. (control)* | – | *0.025* | *0.251* | *0.226* | *0.002* | *0.126* | |
| | *Random (control)* | – | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | |

## Task 3 · Hallucination turn prediction

On the Task 3 `test` split (18,241 dialogues). Score, higher is better (MAE: lower is better). [Task page](task3.md)

| | method | regime | MAE | Cov. | False alarms | **Score** | link |
|---|---|---|---:|---:|---:|---:|---|
| 📄 | Gemma4-31B | non-streaming + workflow | 2.10 | 93.03 | 30.48% | **77.06** | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | SS (LaBSE) | **streaming** | 2.19 | 93.24 | 12.02% | 73.69 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | IT (LaBSE) | **streaming** | 1.98 | 93.86 | 12.49% | 73.60 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | IT + NC (LaBSE) | **streaming** | 1.94 | 94.80 | 16.31% | 73.23 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | SS + NC (LaBSE) | **streaming** | 2.20 | 93.56 | 15.47% | 72.45 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Qwen3.6-35B | non-streaming + workflow | 2.33 | 96.71 | 53.96% | 71.87 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Gemma4-31B | non-streaming | 5.85 | 73.33 | 9.79% | 65.62 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Qwen3.6-35B | non-streaming | 5.62 | 82.94 | 29.92% | 64.56 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | GPT-OSS-20B | non-streaming | 6.48 | 72.38 | 20.04% | 62.65 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | GPT-OSS-20B | non-streaming + workflow | 2.60 | 96.88 | 81.64% | 62.31 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| | *Random (baseline)* | – | *6.60* | *100.00* | *100.00%* | *0.94* | |

The regimes are not equally hard. A streaming method sees one turn at a time, as a live guardrail would. A "+ workflow" method is handed the ground-truth workflow, which you rarely have in practice. Compare within a regime first.

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
