<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# Task 1 · Set-to-set distance ranking

[← back to the README](../README.md)

> *Do these conversations, as a whole, unfold like the reference ones?*

A company deploys an LLM agent to handle routine customer requests, or upgrades its assistant to a newer model. Before the switch, it needs to know whether the new agent's conversations unfold like the reference ones: past conversations handled by human agents, or by the previous assistant. After the switch, it needs to know whether they drift over time. Neither question is about a single dialogue: you care about the conversations *as a whole*. That's Task 1: how far does a **set** of dialogues deviate from a reference set?

|  |  |
|---|---|
| **Input** | two sets of dialogues: a reference set and a target set |
| **Output** | a scalar distance between them |
| **Ground truth** | for each task and hallucination type: d(conformant, forks at *k*=1) > d(…, *k*=2) > d(…, *k*=3) > d(…, *k*=4) |
| **Metric** | Kendall's τ between the predicted and expected order (chance = 0, perfect = 1), averaged over the 24 tasks |
| **Units** | 24 tasks × 4 types = 96 rankings of 4 sets each |

## Setup

For each of the 24 tasks and each hallucination type *h*, the forks are grouped by fork position into four sets 𝒟<sup>h</sup><sub>1</sub> … 𝒟<sup>h</sup><sub>4</sub>, and the task's conformant dialogues 𝒟<sup>+</sup> are the reference. An earlier fork shares less of the conformant conversation, so a good set distance μ satisfies μ(𝒟<sup>+</sup>, 𝒟<sup>h</sup><sub>1</sub>) > … > μ(𝒟<sup>+</sup>, 𝒟<sup>h</sup><sub>4</sub>).

This ordering follows directly from how the forks are built. Task 1 is therefore also a **sanity check of DialFork itself**: if independent metrics recover the order, the forks deviate as intended.

## Methods in the paper

| method | idea |
|---|---|
| **FBD** (Fréchet BERT Distance; [Xiang et al., 2021](https://aclanthology.org/2021.findings-acl.193/)) | fits a Gaussian to each set's embeddings of (system, user) turn pairs; reports the Fréchet distance between them |
| **PRD** (Precision–Recall Distance; same paper) | a precision–recall trade-off between the two embedding distributions |
| **FlowKL** (*ours*) | interpretable, built on an explicit **dialog-flow graph**. (1) Cluster the reference set's utterances with [Dialog2Flow](https://huggingface.co/sergioburdisso/dialog2flow-joint-dse-base) embeddings into workflow-step nodes, with transition probabilities between them. (2) Score every dialogue by the geometric mean of its transition probabilities under that graph. (3) Report the KL divergence between the two sets' score distributions (kernel density estimates) |
| *controls* | **Turn-/Word-count diff.**: the difference in mean length between the sets (forks are longer than conformant dialogues); **Random**: exact chance |

FBD and PRD compare opaque embedding distributions. FlowKL instead tells *why* two sets differ: the graph is human-readable, and one can inspect the transitions that the target set takes and the reference set does not.

## Results

Kendall's τ (higher is better), produced by `evaluate.py` on the files in [`predictions/task1/`](../predictions/task1/):

| method | WIC | SSPC | GR | CSC | **Avg.** |
|---|---:|---:|---:|---:|---:|
| *Random* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| *Turn-count diff.* | *−0.076* | *−0.340* | *0.278* | *−0.208* | *−0.087* |
| *Word-count diff.* | *0.222* | *0.208* | *0.583* | *−0.208* | *0.201* |
| FBD | **0.972** | 0.792 | **0.986** | **0.903** | **0.913** |
| PRD | 0.931 | 0.764 | **0.986** | 0.833 | 0.878 |
| FlowKL | 0.847 | **0.903** | 0.944 | 0.833 | 0.882 |

- **All three metrics recover the fork order far above chance.** Fisher-combined *p* < 0.001. The full 4-way order is recovered exactly in 77% (FBD), 69% (PRD) and 68% (FlowKL) of the 96 units, against 4.2% by chance.
- **Length does not explain it.** Turn-count diff. is not significant (Fisher *p* = 0.995), and Word-count diff. is barely so (*p* = 0.048).
- **FlowKL is on par with FBD and PRD** (Holm-corrected Wilcoxon *p* = 0.43 and 0.79) while being interpretable. It is the best metric on SSPC, the type that corrupts step order.

## Evaluate your method

Got a set distance of your own? Here's how to score it. Write one row per (task, hallucination type, fork position): 24 × 4 × 4 = **384 rows**.

```csv
task,hallucination_type,fork_position,distance
apartment_schedule,WIC,1,0.8731
apartment_schedule,WIC,2,0.6402
...
```

`hallucination_type` may be the short (`WIC`) or the full name (`workflow_identity_corruption`). Then:

```bash
python evaluate.py --task 1 my_method.csv            # .csv or .csv.gz
python evaluate.py --task 1 my_method.csv --per-task # also the per-task breakdown
```

The reference set for a task is all its conformant dialogues. The target set for (*h*, *k*) is all its forks of type *h* at position *k*. Ties count as neither concordant nor discordant, so a constant distance scores 0, not 1.

Your distance should come from the dialogues alone: the fork position is the answer, so don't let your method see it (nor tune it to the fork order).

When you're happy with the result, add it to the **[leaderboard](leaderboard.md)**. We'd love to see it there! 🏆

## Reproducing the paper

Want to rerun the paper's Task 1 experiments? Everything is in **[Reproducing the paper → Task 1](reproducing.md#task-1--set-to-set-distance)**: the setup, the local and Slurm commands, all the options, and the script that builds the paper's tables and tests.

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
