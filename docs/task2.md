<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# Task 2 · Dialog-to-dialog distance ranking

[← back to the README](../README.md)

> *Did this conversation unfold like that one?*

Retrieving past conversations that *unfolded* like a given one is useful: to see how similar cases were handled, to audit an agent, to mine examples. Content similarity does not do this. Two conversations about the weather can follow very different processes, while a hotel booking and a flight booking can follow the same one. What you need is a **procedural** distance, and that's what Task 2 asks for: given two dialogues, how differently did they unfold?

|  |  |
|---|---|
| **Input** | two dialogues |
| **Output** | a scalar distance between them |
| **Ground truth** | for each conformant dialogue *D*<sup>+</sup> and type *h*: d(*D*<sup>+</sup>, fork *k*=1) > d(*D*<sup>+</sup>, *k*=2) > d(*D*<sup>+</sup>, *k*=3) > d(*D*<sup>+</sup>, *k*=4) |
| **Metric** | Kendall's τ between the predicted and expected order (chance = 0), averaged over dialogues, then tasks |
| **Units** | 2,134 conformant dialogues × 4 types = 8,536 rankings of 4 forks each |

<p align="center"><img src="../assets/forking.png" width="70%" alt="Forking a conformant dialogue at four positions gives four forks with a known order of similarity"></p>

## Methods in the paper

Every method turns the dialogue into turn embeddings *E* = (*e*<sub>1</sub>…*e*<sub>n</sub>) with a sentence encoder. Each method was run with **seven encoders**: SBERT, LaBSE, BGE-M3, Qwen3-Embedding-8B and EmbeddingGemma, which are general-purpose, and DSE and Dialog2Flow (D2F), which are dialogue-specific.

| method | distance |
|---|---|
| **Mean-turn emb.** | cosine distance between the mean turn embeddings |
| **Full-dialog emb.** | cosine distance between embeddings of the whole dialogue as one text |
| **FlowDTW** (*ours*) | Dynamic Time Warping over the two turn sequences (cosine cost), normalized by the warping-path length |
| **FlowScore** (*ours*) | BERTScore applied to **turns instead of tokens**: each turn is greedily matched to its most similar turn in the other dialogue; precision and recall are combined into F<sub>1</sub>; distance = 1 − F<sub>1</sub> |
| *controls* | **Turn-/Word-count diff.**: difference in length. **Shared prefix**: the number of opening turns the two dialogues share verbatim. It reads the fork position off the construction itself (a strong but imperfect cue, since replayed openings are sometimes reworded) and has no meaning outside DialFork; it is reported to show that the metrics do not reduce to it |

FlowScore and FlowDTW are implemented in [`methods/task2/run.py`](../methods/task2/run.py) and depend only on the turn embeddings, so they can be applied to any pair of dialogues.

## Results

Average Kendall's τ over the four types (higher is better), from the files in [`predictions/task2/`](../predictions/task2/):

| method | SBERT | LaBSE | BGE-M3 | Qwen3-Emb. | EmbGemma | DSE | D2F |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean-turn emb. | 0.728 | 0.759 | 0.748 | 0.648 | 0.739 | 0.741 | 0.762 |
| Full-dialog emb. | 0.623 | 0.675 | 0.711 | 0.519 | 0.676 | 0.704 | 0.747 |
| FlowDTW | 0.738 | 0.742 | 0.727 | 0.722 | 0.721 | 0.737 | 0.742 |
| **FlowScore** | **0.843** | **0.840** | **0.845** | **0.812** | **0.835** | **0.833** | **0.834** |

| control | WIC | SSPC | GR | CSC | Avg. |
|---|---:|---:|---:|---:|---:|
| *Random* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| *Turn-count diff.* | *0.025* | *0.251* | *0.226* | *0.002* | *0.126* |
| *Word-count diff.* | *0.220* | *0.242* | *0.390* | *0.040* | *0.223* |
| *Shared prefix* | *0.688* | *0.668* | *0.712* | *0.662* | *0.682* |
| FlowScore + BGE-M3 (best) | 0.924 | 0.814 | 0.942 | 0.700 | 0.845 |

- **FlowScore is the best method with every encoder.** It is significantly better than every other method and every control, for every encoder (one-sided Wilcoxon, Holm-corrected, *p* < 0.001).
- **FlowScore makes the encoder almost irrelevant** (τ 0.81–0.85). Qwen3-Embedding is a strong semantic encoder. It collapses to τ = 0.519 when whole dialogues are embedded, yet reaches 0.812 with FlowScore. **Semantic similarity is not procedural similarity.**
- **No control explains the results.** Length is close to chance. Shared prefix, which exploits the construction itself, reaches 0.682 and is still beaten by FlowScore with every encoder.
- **CSC is the hard type** (τ ≤ 0.70 for every method). Its forks keep every step of the workflow and only add a competing behaviour on top.

Per-type and per-task tables for every (method, encoder) pair are in the paper's appendix. To regenerate them, run `methods/task2/tables.py` ([how](reproducing.md#task-2--dialog-to-dialog-distance)), or run `evaluate.py --per-task` on any prediction file.

## Evaluate your method

Got a dialogue distance of your own? Write one row per fork, giving its distance to its conformant original: **34,144 rows**.

```csv
id,distance
2115-GR-k1,0.4127
2115-GR-k2,0.3311
...
```

`id` is the fork's id in the dataset (the `id` field on Hugging Face): `<dialog_id>-<type>-k<k>`. The distance is measured to its conformant original, `<dialog_id>-conformant`; for `2115-GR-k2`, that is `2115-conformant`. Then:

```bash
python evaluate.py --task 2 my_method.csv.gz
python evaluate.py --task 2 my_method.csv.gz --per-task --json scores.json
```

Your distance should come from the two dialogues alone: a fork's `id` gives away its position, which is the answer, so don't let your method see it (nor tune it to the fork order). Ties count as neither concordant nor discordant, so they are never resolved in favour of the ground truth. Every file in [`predictions/task2/`](../predictions/task2/) can serve as a format example, e.g. `flowscore__bge-m3.csv.gz`.

When you're happy with the result, add it to the **[leaderboard](leaderboard.md)**. We'd love to see it there! 🏆

## Reproducing the paper

Want to rerun the paper's Task 2 experiments? Everything is in **[Reproducing the paper → Task 2](reproducing.md#task-2--dialog-to-dialog-distance)**: the setup, the local and Slurm commands, all the options, and the script that builds the paper's tables and tests.

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
