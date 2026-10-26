<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# Task 3 · Hallucination turn prediction

[← back to the README](../README.md)

> *Where did this conversation go off-track?*

A guardrail watching a live conversation should step in as soon as the agent leaves its intended process, not after the damage is done. It should also stay silent while the agent behaves. That's exactly what Task 3 asks: **at which turn does the conversation first go off-track**? Or *none*, if it never does.

|  |  |
|---|---|
| **Input** | one dialogue (turn by turn, or complete; see the regimes below) |
| **Output** | the first turn where the conversation goes off-track, given as its position in the conversation (turns counted from 0), or *none* |
| **Ground truth** | for a fork, the turn where the deviation starts: the first assistant turn under the new instruction (the dataset's `fork_turn_index`); *none* for conformant dialogues |
| **Metrics** | hallucination MAE and coverage on forks; MAE and false-alarm rate on conformant dialogues; a composite **Score** (0–100) |
| **Units** | the 18,241 dialogues of the `test` split: 1,073 conformant dialogues and their 17,168 forks |
| **Training data** | only the 1,061 conformant dialogues of the `train` split, for methods that learn from data (not its forks or labels); see the [Task 3 split](dataset.md#task-3-split) |

## Three regimes

| regime | what the method sees | realism |
|---|---|---|
| **Streaming** | turns arrive one at a time; it must decide after each turn, with no access to future turns, and with **no workflow** | the most realistic, the hardest |
| **Non-streaming** | the complete dialogue, with **no workflow** | |
| **Non-streaming + workflow** | the complete dialogue **and** its ground-truth workflow | an oracle reference: the workflow is rarely available in practice |

In every regime, a method may also learn from the conformant dialogues of the `train` split (see the [Task 3 split](dataset.md#task-3-split)).

## Methods in the paper

**Streaming detectors.**

1. Cluster the turns of the `train` split's conformant dialogues into workflow-step nodes with a sentence encoder, as in FlowKL.
2. Fit an *N*-gram (*N* = 6) language model over the node sequences.
3. While a new dialogue streams in, map each turn to a node and track how probable each transition is.

On top of this, two detectors decide when to flag:

- **IT (Improbable Transition)** flags the first transition whose probability falls below a threshold.
- **SS (Surprisal Spike)** flags the first sudden jump in surprisal.

Both thresholds are calibrated on the same `train` conformant dialogues, for a target false-alarm rate of about 5% on them. The **+NC** variants weight each transition by a *node confidence*, a softmax over the turn's similarity to the nearest centroids, which discounts turns that lie between clusters.

**Non-streaming (LLMs).** Zero-shot prompting of open-weight LLMs (Gemma4-31B, Qwen3.6-35B, GPT-OSS-20B, Mistral-Small-3.2-24B, Llama-3.3-70B) asks for the system turn where the dialogue first deviates, or *none*. In the workflow regime, the prompt also includes the ground-truth workflow.

**Baselines.** Always predict the first, the last, or a random turn.

## Metrics

| metric | computed on | definition | better |
|---|---|---|---|
| **Hallucination MAE** | forks | \|predicted − true turn\|, in turns. An unflagged fork counts as predicted at *n*, just past its last turn (*n* = number of turns). Averaged per (task, type), then over tasks, then over the 4 types | lower |
| **Hallucination coverage** | forks | % of forks flagged | higher |
| **Conformant MAE** | flagged conformant dialogues | \|predicted − *n*\|: how early the false alarm fires. Averaged per task, over the tasks with at least one flag | lower |
| **Conformant coverage** | conformant dialogues | % flagged: the **false-alarm rate** | lower |
| **Score** | all | 100 × weighted geometric mean of four desirabilities in [0, 1]: 1 − MAE/anchor for both MAEs (weights 0.35 each), hallucination coverage, and 1 − false-alarm rate (0.15 each). The anchors are the worst credible MAEs, those of the trivial baselines: last turn on forks, first turn on conformant dialogues | higher |

Localization is the defining output of the task, so the MAEs carry the most weight. The coverage terms mainly rule out degenerate strategies: always flag, or never flag. A geometric mean means a method must do well on every term. Always flagging the first turn scores 0, no matter how good its hallucination MAE is.

## Results

As reported in the paper, on the 18,241 dialogues of the Task 3 `test` split; MAEs are in turns. The streaming rows use the encoder with the best Score, LaBSE, and a 3-gram model, learned from the `train` split's conformant dialogues.

| regime | method | CSC | GR | SSPC | WIC | **MAE** ↓ | Cov. ↑ | Conf. MAE ↓ | False alarms ↓ | **Score** ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *baseline* | *Random* | *6.31* | *7.92* | *5.74* | *6.44* | *6.60* | *100.00* | *7.50* | *100.00%* | *0.94* |
| *baseline* | *First turn* | *5.64* | *5.64* | *5.64* | *5.64* | *5.64* | *100.00* | *14.12* | *100.00%* | *0.00* |
| *baseline* | *Last turn* | *12.01* | *15.57* | *10.26* | *12.22* | *12.52* | *100.00* | *1.00* | *100.00%* | *0.00* |
| streaming | **SS** | 4.21 | 1.56 | 1.30 | 1.68 | 2.19 | 93.24 | 6.33 | 12.02% | **73.69** |
| streaming | SS + NC | 4.13 | 1.61 | 1.33 | 1.71 | 2.20 | 93.56 | 6.58 | 15.47% | 72.45 |
| streaming | IT | 4.13 | 0.93 | 1.25 | 1.63 | 1.98 | 93.86 | 6.52 | 12.49% | 73.60 |
| streaming | IT + NC | 3.94 | 0.92 | 1.29 | 1.60 | 1.94 | 94.80 | 6.55 | 16.31% | 73.23 |
| non-streaming | GPT-OSS-20B | 10.10 | 6.70 | 4.77 | 4.34 | 6.48 | 72.38 | 4.40 | 20.04% | 62.65 |
| non-streaming | Qwen3.6-35B | 8.45 | 6.46 | 3.28 | 4.28 | 5.62 | 82.94 | 4.86 | 29.92% | 64.56 |
| non-streaming | Gemma4-31B | 10.13 | 7.12 | 2.26 | 3.88 | 5.85 | 73.33 | 4.63 | 9.79% | 65.62 |
| non-streaming | Llama-3.3-70B | 8.10 | 5.00 | 2.04 | 2.85 | 4.50 | 80.62 | 5.69 | 18.36% | 67.09 |
| non-streaming | Mistral-Small-3.2-24B | 8.51 | 5.56 | 2.88 | 3.98 | 5.23 | 78.20 | 4.60 | 12.12% | 68.14 |
| + workflow | GPT-OSS-20B | 5.18 | 1.37 | 1.40 | 2.45 | 2.60 | 96.88 | 4.45 | 81.64% | 62.31 |
| + workflow | Qwen3.6-35B | 4.96 | 1.14 | 1.36 | 1.85 | 2.33 | 96.71 | 4.57 | 53.96% | 71.87 |
| + workflow | Mistral-Small-3.2-24B | 5.03 | 1.36 | 1.74 | 2.14 | 2.57 | 92.91 | 4.94 | 16.68% | 76.38 |
| + workflow | Gemma4-31B | 5.30 | 0.51 | 1.31 | 1.28 | 2.10 | 93.03 | 4.41 | 30.48% | 77.06 |
| + workflow | Llama-3.3-70B | 3.83 | 0.69 | 1.19 | 1.61 | 1.83 | 93.89 | 5.18 | 21.06% | **77.09** |

**Best streaming Score, by encoder** (paper appendix): LaBSE 73.69 (SS) · D2F 73.28 (IT+NC) · Qwen3-Emb. 72.06 (IT) · EmbGemma 71.93 (IT+NC) · SBERT 70.61 (IT) · BGE-M3 66.20 (IT) · DSE 66.02 (IT).

**What the numbers say.**

- **Streaming detectors** run turn by turn, with no future context and no ground-truth workflow, learning only from past conformant dialogues. They localize the fork within about 1.9–2.2 turns, flagging 93–95% of forks at a false-alarm rate of 12–16%. IT localizes better than SS; NC weighting sharpens IT further (MAE 1.94, the lowest of the methods that need no ground-truth workflow) and flags more forks, at the cost of more false alarms.
- **LLMs without the workflow localize far worse** (MAE 4.5–6.5) and flag only 72–83% of forks, despite seeing the whole dialogue.
- **Given the workflow**, LLMs localize about as well as streaming (MAE 1.83–2.60; Llama-3.3-70B localizes best of all methods, 1.83) but raise more false alarms on conformant dialogues: from 17% (Mistral-Small-3.2-24B) and 21% (Llama-3.3-70B) up to 82%. Llama-3.3-70B + workflow has the best overall Score (77.09), tied with Gemma4-31B (77.06).
- A lightweight detector that learns the workflow from past dialogues thus localizes deviations as they happen nearly as well as the best LLM given the true workflow and the whole dialogue, with fewer false alarms (12–16%). **No method is close to a perfect Score**: real-time localization of procedural hallucination is open.

To compare a new method with the paper's, score the paper's prediction files in [`predictions/task3/`](../predictions/task3/) with `evaluate.py`. Every (detector, encoder) pair and every LLM has one.

## Evaluate your method

Got a detector of your own? Write one row per dialogue of the Task 3 `test` split: **18,241 rows** (rows for `train` dialogues are ignored, so a file covering all 36,278 is fine too). If it learns anything from data, it may learn only from the conformant dialogues of the `train` split, not from its forks or labels (see the [Task 3 split](dataset.md#task-3-split)). Each dialogue is identified by its dataset `id` (`<dialog_id>-conformant` or `<dialog_id>-<TYPE>-k<k>`). `predicted_turn_index` is the position of the first turn your method flags as off-track, counting the conversation's turns from 0, or **empty** if it flags nothing. It uses the same convention as the dataset's `fork_turn_index`: on Hugging Face, it indexes a row's `turns` list.

```csv
id,predicted_turn_index
30-conformant,
30-GR-k1,0
30-GR-k2,3
30-WIC-k3,
...
```

```bash
python evaluate.py --task 3 my_method.csv.gz              # the test split
python evaluate.py --task 3 my_method.csv.gz --split all  # every dialogue (not comparable with the leaderboard)
```

The Score's default weights are the paper's and the leaderboard's. If your setting weighs errors differently, for example a guardrail where a false alarm costs more than a late detection, pass your own with `--weights`, four numbers in this order:

| position | weight of | default |
|---|---|---:|
| 1 | hallucination MAE | 0.35 |
| 2 | conformant MAE | 0.35 |
| 3 | hallucination coverage | 0.15 |
| 4 | conformant coverage (the false-alarm rate) | 0.15 |

```bash
python evaluate.py --task 3 my_method.csv.gz --weights 0.3,0.2,0.1,0.4   # false alarms weigh most
```

The four components are always printed next to the Score.

- **Any turn can be flagged,** the assistant's or the user's: user turns are evidence too (a user who keeps asking for the forecast is a strong hint). A streaming method reports the turn at which it fires; in deployment, the guardrail would act on the next assistant turn.
- **Every `test` dialogue needs a row.** A prediction must fall in 0…*n*−1, where *n* is the dialogue's number of turns. `--allow-missing` scores a subset, for development only.
- **Declare your regime.** The evaluator cannot check whether a method used future turns or the workflow. When reporting, say whether it is streaming, non-streaming, or non-streaming + workflow, and which conformant dialogues it was calibrated on.

When you're happy with the result, add it to the **[leaderboard](leaderboard.md)**. We'd love to see it there! 🏆

## Reproducing the paper

Want to rerun the paper's Task 3 experiments? Everything is in **[Reproducing the paper → Task 3](reproducing.md#task-3--turn-prediction)**: the setup, the local and Slurm commands, all the options, and the script that builds the paper's tables and tests.

## Optional: tuning with cross-validation on `train`

<details>
<summary>A suggestion for those who tune hyperparameters, or want an estimate of how well a configuration generalizes before looking at <code>test</code>.</summary>

<br>

If you choose among configurations (an encoder, an n-gram order, a threshold), one way to do it without the `test` split is **k-fold cross-validation over the `train` sources**:

1. Split each task's `train` sources into **k = 4** folds, keeping each source with its 16 forks (a fork replays its source before the fork turn).
2. For each fold, learn from the **conformant** dialogues of the other three folds, and predict every dialogue of the held-out fold, conformant dialogues and forks, so that false alarms count too.
3. Put the four folds' predictions in one file: `evaluate.py --split train` then gives the cross-validated Score.
4. Learn the chosen configuration from all `train` conformant dialogues and report its `test` Score.

**Why k = 4:** the smallest task, *ride_status*, has 4 `train` sources, so k = 4 is the largest k that leaves no fold empty, while each model still learns from 75% of the training data (k = 2 would leave only two dialogues to learn from in the smallest tasks). The forks are used only to compare configurations, never to train a method. A large gap between the cross-validated and the `test` Score suggests the configuration does not generalize.

```python
import csv, random
from collections import defaultdict

K = 4
train = defaultdict(list)                          # task -> its train source dialogues
for r in csv.DictReader(open("datasets/dialfork/task3_splits.csv")):
    if r["split"] == "train":
        train[r["task"]].append(int(r["dialog_id"]))

fold = {}                                          # dialog_id -> 0..K-1
for task, ids in sorted(train.items()):
    ids.sort()
    random.Random(f"dialfork-cv:{task}").shuffle(ids)
    for i, dialog_id in enumerate(ids):
        fold[dialog_id] = i % K                    # every task has >= 4 train sources: no empty fold

# fold f: learn from the conformant dialogues with fold[dialog_id] != f,
#         predict every dialogue (conformant and forks) with fold[dialog_id] == f
```

</details>

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
