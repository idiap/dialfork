<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# 🤖 `judge`: an LLM judge calibrated on the human study

[← back to the README](../README.md) · [the LLM judge: method and results](../docs/llm_judge.md) · [the human study](../docs/human_study.md)

The human study checks DialFork's labels on 428 conversations. This folder extends those checks to every dialogue: an LLM answers the annotators' three questions, in the same order, and is scored against them before it is trusted on the rest.

- **The study's protocol, replayed.** Same workflow text, same cues, same foils, and the same order: the 0–100 rating is given before anything is revealed.
- **Selected against people, then frozen.** 13 judges compared on three of the study's tasks, the selected one checked on the other two.
- **Everything released.** The answers of all 13 judges, and of the selected one on all 36,069 dialogues, so every number can be recomputed without a GPU.
- **Reusable.** Score your own dialogues, or put another LLM in the annotators' seat.

## 🚀 Quick start

From the repository root, with the data downloaded (`python scripts/download_data.py`). Running a judge needs an [Ollama](https://ollama.com) server and a GPU; `run.sbatch` starts the server for you, with Slurm or locally ([set-up](../docs/reproducing.md#2-running-the-scripts-locally-or-on-a-slurm-cluster)):

```bash
python -m judge.items                                   # the 428 study conversations, as the judge sees them
bash judge/run.sbatch -m qwen3.6:35b -p P0 --limit 10   # ten of them, to try it out
python -m judge.score --runs results/judge/outputs      # agreement with the annotators
```

A run writes to `results/judge/outputs/<model>/<prompt>/seed<S>/`, one line per (conversation, question), as it goes; rerun the same command to resume. Keep one request at a time (`OLLAMA_NUM_PARALLEL=1`, the default of `run.sbatch`): with several, some Ollama builds crash on mixture-of-experts models such as `qwen3.6:35b`.

- **The paper's numbers**, from the released answers and without a GPU, and the full reruns: [reproducing the paper](../docs/reproducing.md#the-llm-judge).
- **Your own dialogues**, in three commands: [judge your own dialogues](../docs/llm_judge.md#judge-your-own-dialogues).

## 📁 What each file does

| file | what it does |
|---|---|
| `prompts.py` | The prompts: the system message, the three questions, and the reveal shown before question 3. **P0** is the annotators' wording, **P1** adds rubric anchors, **P2** is P1 with reasoning enabled. |
| `items.py` | Builds the judge's inputs for the 428 study conversations from the released human-study files: the workflow as numbered steps, the user's brief, the transcript, the cue, and the annotators' answers. Checks every count of the study. |
| `full_items.py` | The same, for every dialogue of a corpus (`-d` for another one), with the cue the study's sampler gives each dialogue. |
| `run.py` | Asks one judge (model × prompt × seed) the three questions, through Ollama's `/api/chat`. Writes one JSONL line per (conversation, question) as it goes. |
| `score.py` | Scores runs against the annotators: agreement (Gwet's AC1, Krippendorff's α) next to the annotator-to-annotator reference, the study's measures (RQ1–RQ4), leave-one-task-out, and self-consistency. `--validate` scores the annotators alone and must reproduce the paper. |
| `full_score.py` | Scores a full-corpus run against the construction labels, overall, per type and per task, and lists the dialogues where the two disagree. |
| `export_answers.py` | Packs runs into the released layout of `answers/`, and writes the flat CSV. |
| `common.py` | Paths, the selection split, and reading answer files (`.jsonl` or `.jsonl.gz`). |
| `frozen_judge.json` | The selected judge, and how it was chosen. |
| `run.sbatch` | One judge run: starts Ollama, then `run.py` with the given arguments. |
| `run_all.sh` | The 13 judges on the study, one Slurm job each. |
| `run_full.sh` | The selected judge on a whole corpus, as a Slurm array. |

New inputs, runs and scores go to `results/judge/`, which git ignores.

## 📦 The released answers

```
answers/
├── dialfork_llm_judge.csv.gz          # the selected judge on every dialogue: one row each, by dataset id
├── full/qwen3.6_35b/P0/               # the same run, raw
└── study/<model>/<prompt>/seed0/      # the 13 judges on the 428 study conversations
    (and qwen3.6_35b/P0_t0.7/seed{0,1,2}/: the selected judge sampled, for self-consistency)
```

Each raw file is a gzipped `judgments.jsonl.gz`, one line per (conversation, question): the parsed `answer`, the model's `raw` reply, its `thinking` (P2), the number of `attempts`, and the time and tokens it took. Study conversations are identified by the `item_id` of the [human-study files](../docs/human_study.md#released-data), and corpus dialogues by a `task/type/k/dialog_id` reference (e.g. `apartment_schedule/WIC/k1/26`). The CSV's columns are in the [docs](../docs/llm_judge.md#released-data).

## 🧠 Design choices

- **One chat, three turns.** The questions are asked one after the other in the same chat, each after the previous answer, as the annotation server enforced for people: the cue is never in context when the rating is given, and the reveal is never in context when the cue is answered.
- **Plain-text prompts, no constrained decoding.** The JSON answer is parsed out of the reply, as in `methods/task3/llm.py`. A reply that does not parse is asked again, up to three times; a reply that never parses is recorded as such, and scored as missing.
- **Greedy decoding.** At temperature 0 the seed changes nothing, so seeds only matter for sampled runs (`--temperature`, written to `<prompt>_t<T>/`).
- **Resumable, and safe to shard.** A rerun skips the questions already answered and rebuilds each chat from them. Every shard writes its own file, and reads the others' answers as done. A failed call to the server is retried, and never stored as an answer.
- **The generator is not a candidate.** `gemma4:31b` generated DialFork, so it is run and reported, but excluded from the selection to avoid self-verification.
- **Same statistics as the study.** Agreement and the study's measures come from its own code (`annotool/stats.py`, `annotool/analyze.py`; Krippendorff's α is recomputed in linear time for the bootstrap, and checked against it), and `score.py --validate` checks them against the paper's numbers before any judge is scored.

## 🔧 Using and extending it

- **Your own dialogues:** any dataset in DialFork's folder layout, e.g. one built with the [generation pipeline](../docs/dataset.md#regenerating-dialfork), goes through `full_items.py -d <dataset>`, `run_full.sh` and `full_score.py --items`. The blind rating alone flags deviating dialogues: [commands and how well it works](../docs/llm_judge.md#judge-your-own-dialogues).
- **Another model:** any Ollama model name works with `run.sbatch -m <model>`; then `python -m judge.score --runs results/judge/outputs` compares it with the annotators.
- **Another prompt:** add its name to `PROMPTS` in `prompts.py` and its text to the question builders (`rubric()` shows how P1 extends P0), keeping the three questions in the same order. Reasoning is switched on by name, for P2, in `run.py`.
