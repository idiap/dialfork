<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# 🏆 Leaderboard

[← back to the README](../README.md)

This is where methods on DialFork meet. It starts with the methods from our paper, and we hope it doesn't stay that way for long. Built something that beats them, or that fails in an interesting way? We'd love to add it.

🤗 **Browse it on [Hugging Face](https://huggingface.co/spaces/DialFork/leaderboard)**, with sortable columns. It shows the same rows as the tables below, read from [`leaderboard/v1.json`](../leaderboard/v1.json).

📄 = from the [DialFork paper](https://arxiv.org/abs/2610.21561) · 🌍 = community submission

## How to add your method

All it takes is a pull request with your prediction file. We take care of the rest.

1. **Run your method** on one of the three tasks and save its predictions as a simple CSV file. New to DialFork? The [Quickstart](../README.md#-quickstart) gets you from the data to a first score in a few minutes, and each task page details its format: [Task 1](task1.md#evaluate-your-method) · [Task 2](task2.md#evaluate-your-method) · [Task 3](task3.md#evaluate-your-method).
2. **Score your CSV file** with [`evaluate.py`](../evaluate.py), where `N` is your task's number:
   ```bash
   python evaluate.py --task N my_method.csv
   ```
3. **Open a pull request** that adds your file to `predictions/taskN/` (a `.csv.gz` is fine). The pull request form asks a few quick questions about your method: its name, a link to your paper or code, and what it used.

That's it! 🎉 Once it's merged, your method shows up here and on [Hugging Face](https://huggingface.co/spaces/DialFork/leaderboard).

<!-- Maintainers: add a submission's row with `python scripts/make_leaderboard.py --add predictions/taskN/<file> --method ... --link TEXT URL` (plus --encoder for Task 2, --regime and --learns-from for Task 3), and commit it to the pull request. The tables below are written from leaderboard/v1.json: don't edit them by hand. -->

## Task 1 · Set-to-set distance

Kendall τ, higher is better. [Task page](task1.md)

<!-- leaderboard:task1 -->

| | method | WIC | SSPC | GR | CSC | **Avg.** | link |
|---|---|---:|---:|---:|---:|---:|---|
| 📄 | FBD | 0.972 | 0.792 | 0.986 | 0.903 | **0.913** | [Xiang et al., 2021](https://aclanthology.org/2021.findings-acl.193/) |
| 📄 | FlowKL | 0.847 | 0.903 | 0.944 | 0.833 | 0.882 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | PRD | 0.931 | 0.764 | 0.986 | 0.833 | 0.878 | [Xiang et al., 2021](https://aclanthology.org/2021.findings-acl.193/) |
| | *Word-count diff. (control)* | *0.222* | *0.208* | *0.583* | *−0.208* | *0.201* | |
| | *Random (control)* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | |
| | *Turn-count diff. (control)* | *−0.076* | *−0.340* | *0.278* | *−0.208* | *−0.087* | |

<!-- /leaderboard:task1 -->

## Task 2 · Dialog-to-dialog distance

Kendall τ, higher is better. One row per method, with its best encoder; [the task page](task2.md) has every (method, encoder) pair.

<!-- leaderboard:task2 -->

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

<!-- /leaderboard:task2 -->

## Task 3 · Hallucination turn prediction

On the Task 3 `test` split (18,241 dialogues). Score, higher is better (MAE and false alarms: lower is better). [Task page](task3.md)

The three regimes give a method different information, each strictly more than the one before, so **each regime has its own table**: a better rank across regimes would mostly reflect what the method was given. `evaluate.py` cannot check the regime, so say which one your method follows, and link its code.

| regime | the decision on turn *t* may use | the ground-truth workflow |
|---|---|---|
| **streaming** | turns 1 … *t* only | not given |
| **non-streaming** | the whole dialogue | not given |
| **non-streaming + workflow** | the whole dialogue | given: an oracle reference, rarely available in practice |

In every regime, a method may learn only from the conformant dialogues of the `train` split (see the [Task 3 split](dataset.md#task-3-split)); the *learns from* column says what it used. Trivial baselines, for reference: random turn 0.94, first turn 0, last turn 0.

### Streaming

The main track: a live guardrail, flagging a deviation as soon as it happens.

<!-- leaderboard:task3-streaming -->

| | method | learns from | MAE | Cov. | False alarms | **Score** | link |
|---|---|---|---:|---:|---:|---:|---|
| 📄 | SS (LaBSE) | train conformant | 2.19 | 93.24 | 12.02% | **73.69** | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | IT (LaBSE) | train conformant | 1.98 | 93.86 | 12.49% | 73.60 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | IT + NC (LaBSE) | train conformant | 1.94 | 94.80 | 16.31% | 73.23 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | SS + NC (LaBSE) | train conformant | 2.20 | 93.56 | 15.47% | 72.45 | [DialFork paper](https://arxiv.org/abs/2610.21561) |

<!-- /leaderboard:task3-streaming -->

### Non-streaming

<!-- leaderboard:task3-non-streaming -->

| | method | learns from | MAE | Cov. | False alarms | **Score** | link |
|---|---|---|---:|---:|---:|---:|---|
| 📄 | Mistral-Small-3.2-24B | none (zero-shot) | 5.23 | 78.20 | 12.12% | **68.14** | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Llama-3.3-70B | none (zero-shot) | 4.50 | 80.62 | 18.36% | 67.09 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Gemma4-31B | none (zero-shot) | 5.85 | 73.33 | 9.79% | 65.62 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Qwen3.6-35B | none (zero-shot) | 5.62 | 82.94 | 29.92% | 64.56 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | GPT-OSS-20B | none (zero-shot) | 6.48 | 72.38 | 20.04% | 62.65 | [DialFork paper](https://arxiv.org/abs/2610.21561) |

<!-- /leaderboard:task3-non-streaming -->

### Non-streaming + workflow (oracle reference)

<!-- leaderboard:task3-non-streaming+workflow -->

| | method | learns from | MAE | Cov. | False alarms | **Score** | link |
|---|---|---|---:|---:|---:|---:|---|
| 📄 | Llama-3.3-70B | none (zero-shot) | 1.83 | 93.89 | 21.06% | **77.09** | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Gemma4-31B | none (zero-shot) | 2.10 | 93.03 | 30.48% | 77.06 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Mistral-Small-3.2-24B | none (zero-shot) | 2.57 | 92.91 | 16.68% | 76.38 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | Qwen3.6-35B | none (zero-shot) | 2.33 | 96.71 | 53.96% | 71.87 | [DialFork paper](https://arxiv.org/abs/2610.21561) |
| 📄 | GPT-OSS-20B | none (zero-shot) | 2.60 | 96.88 | 81.64% | 62.31 | [DialFork paper](https://arxiv.org/abs/2610.21561) |

<!-- /leaderboard:task3-non-streaming+workflow -->

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
