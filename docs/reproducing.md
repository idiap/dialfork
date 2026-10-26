<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# 🔬 Reproducing the paper

[← back to the README](../README.md)

Only want to **score your own method**? You don't need any of this: `evaluate.py` runs with the Python standard library alone (see the [quickstart](../README.md#-quickstart)). This page is for rerunning the paper's experiments, end to end.

**Common setup**
- [1. Install](#1-install)
- [2. Running the scripts: locally or on a Slurm cluster](#2-running-the-scripts-locally-or-on-a-slurm-cluster)

**Result by result**
- [Task 1 · set-to-set distance](#task-1--set-to-set-distance)
- [Task 2 · dialog-to-dialog distance](#task-2--dialog-to-dialog-distance)
- [Task 3 · turn prediction](#task-3--turn-prediction)
- [The human study](#the-human-study)
- [The LLM judge](#the-llm-judge)
- [The dataset: statistics and regeneration](#the-dataset-statistics-and-regeneration)

## 1. Install

```bash
git clone https://github.com/idiap/dialfork && cd dialfork
pip install -r requirements.txt     # sdialog, scipy, pandas, ... (a GPU is recommended for the encoders)
python scripts/download_data.py      # -> datasets/dialfork/ and datasets/STAR/
```

Run everything **from the repository root**: all scripts use paths relative to it (`datasets/dialfork`, `results/taskN`, …).

**Two extras, only for some steps:**

- **Ollama**, for the LLM experiments of Task 3, the LLM judge, and generating the data. [Install it](https://ollama.com/download), then pull the models you need, e.g. `ollama pull gemma4:31b` (the paper also uses `qwen3.6:35b` and `gpt-oss:20b`, and the judge also `mistral-small3.2:24b` and `llama3.3:70b`).
- **A Hugging Face token**, only for gated models such as `google/embeddinggemma-300m`: accept the model's terms on its Hub page, then `export HF_TOKEN=hf_...`.

## 2. Running the scripts: locally or on a Slurm cluster

Each experiment has a Python entry point (`methods/.../*.py`) and a `.sbatch` wrapper next to it. The wrapper runs the same command and passes its arguments through, so all options are the same. It works in two ways:

```bash
sbatch methods/task3/streaming.sbatch -m sentence-transformers/LaBSE   # as a Slurm job
bash   methods/task3/streaming.sbatch -m sentence-transformers/LaBSE   # right here, no Slurm
```

The wrappers needing an LLM (Task 3's LLM experiments and the data generation) start an Ollama server for the job on their own. For the full runs, the `run_all*.sh` launchers submit one job per STAR task: they show what they are about to submit, ask for confirmation, skip the tasks already done, and create the `logs/` folders. If you call `sbatch` directly, create those first: `mkdir -p logs/{dataset,task1,task2,task3/offline}`.

### Where the code runs: your environment, or an Apptainer image

**The easy path: your own environment.** Install the requirements in a conda or virtual environment (section 1), activate it, and run or submit from there. Slurm passes your environment on to the jobs. Nothing to configure.

**Or: the sdialog Apptainer image**, which has everything installed, Ollama included. It's how the paper's experiments were run.

```bash
wget https://huggingface.co/datasets/sdialog/apptainer/resolve/main/sdialog.sif
export DIALFORK_SIF=$PWD/sdialog.sif       # every wrapper now runs inside it
```

| variable | what it is | default |
|---|---|---|
| `DIALFORK_SIF` | the Apptainer image to run in | empty: no container |
| `DIALFORK_BIND` | extra folders the container must see, comma-separated (e.g. a model cache on another disk) | none |

Export them in your shell, or set them in [`scripts/slurm.env`](../scripts/slurm.env), which every wrapper reads.

**Either way**, the usual variables work as usual and reach the container too: `HF_TOKEN` (gated models), `HF_HOME` (where models are cached), `OLLAMA_MODELS` (where Ollama keeps its models).

### Slurm settings

Account, partition and GPU type differ on every cluster, so they are not in the scripts. Pass them as usual, per job or once per shell:

```bash
export SBATCH_ACCOUNT=my_account SBATCH_PARTITION=gpu    # or: sbatch --account=... --partition=... --constraint=...
```

The wrappers ask for one GPU, 32 GB of memory and 8 hours; override any of these the same way. The largest models (e.g. Qwen3-Embedding-8B, the 30B+ LLMs) need a GPU with 32 GB or more.

## Task 1 · set-to-set distance

*What it computes:* for each STAR task, FBD, PRD and FlowKL between the conformant dialogues and each set of forks, plus the length controls, all in one run. [What the task is](task1.md).

```bash
python methods/task1/run.py weather -i datasets/dialfork -o results/task1      # locally, one STAR task
bash methods/task1/run_all.sh -i datasets/dialfork -o results/task1           # Slurm, all 24 tasks (one job each)
```

The length controls need no GPU and no embeddings. To compute only those, for all tasks, in a few minutes:

```bash
bash methods/task1/run_all_controls.sh -i datasets/dialfork -o results/task1  # or: run.py <task> ... --baselines-only
```

`run.py` builds FlowKL's flow graph with Dialog2Flow (`-m/--node-embedding-model`, default `sergioburdisso/dialog2flow-joint-dse-base`, the paper's). Results go to `results/task1/<encoder>/<task>/`.

**The paper's table and tests:**

```bash
python methods/task1/tables.py -r results/task1/dialog2flow-joint-dse-base/
```

Besides the table, it prints the statistics behind the paper's claims:
- **Against chance:** an exact permutation test of τ per unit, combined with Fisher's method, and a binomial test of the exact-recovery rate.
- **FlowKL against each other metric:** paired Wilcoxon and t-tests, Holm-corrected, with bootstrap confidence intervals.

## Task 2 · dialog-to-dialog distance

*What it computes:* for every fork, its distance to its conformant original, with one distance method and one encoder per run. [What the task is](task2.md).

```bash
# locally: one method and encoder, all STAR tasks (or name some tasks first)
python methods/task2/run.py -dm flowscore -m sentence-transformers/LaBSE -o results/task2
python methods/task2/run.py weather -dm flowscore -m sentence-transformers/LaBSE -o results/task2

# Slurm: one job per STAR task
bash methods/task2/run_all.sh sentence-transformers/LaBSE flowscore
```

The paper's full grid is four methods × seven encoders:

```bash
for enc in sentence-transformers/all-mpnet-base-v2 sentence-transformers/LaBSE BAAI/bge-m3 \
           Qwen/Qwen3-Embedding-8B google/embeddinggemma-300m aws-ai/dse-bert-base \
           sergioburdisso/dialog2flow-joint-dse-base; do
  for dm in mean-turn-emb full-dialog-emb flowdtw flowscore; do
    bash methods/task2/run_all.sh "$enc" "$dm"
  done
done
```

The controls need no encoder and run on a CPU in seconds:

```bash
python methods/task2/run.py -dm turn-count-diff -o results/task2
python methods/task2/run.py -dm word-count-diff -o results/task2
python methods/task2/run.py -dm shared-prefix   -o results/task2
```

| option | meaning |
|---|---|
| `-dm, --distance-method` | `mean-turn-emb`, `full-dialog-emb`, `flowdtw`, `flowscore`, `turn-count-diff`, `word-count-diff`, `shared-prefix` |
| `-m, --embedding-model` | any Sentence-Transformers model on the Hub |
| `--encode-batch-size`, `--encode-max-length` | encoder settings (defaults 16 and 512). Out-of-memory batches are retried with smaller settings automatically |

`google/embeddinggemma-300m` is gated: set `HF_TOKEN` (see [Install](#1-install)). Results go to `results/task2/<method>/<encoder>/<task>/`.

**The paper's tables and tests:**

```bash
python methods/task2/tables.py --root results/task2      # add --latex for LaTeX
```

It prints the main table, the controls, the per-task appendix table, and the one-sided Wilcoxon and t-tests of FlowScore against every other method and control, per encoder (Holm-corrected, with bootstrap confidence intervals).

## Task 3 · turn prediction

*What it computes:* for every dialogue, the first turn each method flags as off-track, or none. [What the task is](task3.md).

**Streaming detectors.** One run fits the 3-gram flow model on each task's conformant dialogues of the [Task 3 `train` split](dataset.md#task-3-split) and applies the four detectors (SS, SS+NC, IT, IT+NC) to the `test` split. No Ollama needed.

```bash
python methods/task3/streaming.py -m sentence-transformers/LaBSE    # locally: all tasks, the main table's encoder, 3-gram
sbatch methods/task3/streaming.sbatch -m sentence-transformers/LaBSE # Slurm, same thing

# the paper's appendix compares seven encoders:
sbatch methods/task3/streaming.sbatch                               # D2F (the default encoder)
sbatch methods/task3/streaming.sbatch -m sentence-transformers/all-mpnet-base-v2
sbatch methods/task3/streaming.sbatch -m jeonseonjin/embedding_BAAI-bge-m3
sbatch methods/task3/streaming.sbatch -m google/embeddinggemma-300m
sbatch methods/task3/streaming.sbatch -m aws-ai/dse-bert-base
sbatch methods/task3/streaming.sbatch -m Qwen/Qwen3-Embedding-8B          # needs a 32 GB GPU

# another N for the N-gram model (the appendix also reports N = 2, 6, 10)
sbatch methods/task3/streaming.sbatch -m sentence-transformers/LaBSE -n 6
```

Results go to `results/task3/<encoder>/lm-<N>-gram/`.

**LLMs.** Zero-shot prompting through Ollama, with and without the workflow, for the paper's three models. Pull the models first (`ollama pull gemma4:31b`). The wrapper starts an Ollama server for the job; if you call `llm.py` directly, start one yourself (`ollama serve`).

```bash
# locally, one model and setting
python methods/task3/llm.py -d datasets/dialfork -m gemma4:31b -o results/task3/offline
python methods/task3/llm.py -d datasets/dialfork -m gemma4:31b -o results/task3/offline --no-workflow

# Slurm, one job per STAR task
for model in gemma4:31b qwen3.6:35b gpt-oss:20b; do
  bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m "$model"
  bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m "$model" --no-workflow
done
```

Results go to `results/task3/offline/llm-<model>[-no-workflow]/`.

**The paper's tables:**

```bash
python methods/task3/tables.py           # main table and the per-encoder appendix table (--latex for LaTeX)
```

**Scoring with `evaluate.py`.** To turn your runs into prediction files and score them like any other method:

```bash
python scripts/export_predictions.py -o my_predictions --tasks 3
python evaluate.py --task 3 my_predictions/task3/streaming-it-nc__d2f.csv.gz
```

The streaming scripts store 1-based positions over all turns, and the LLMs an assistant-turn number; the export turns both into the 0-based `predicted_turn_index` that `evaluate.py` expects.

## The human study

No experiment to run: the judgments are released. The report and the figure are computed from them:

```bash
python -m annotool.analyze --db annotool/data/dialfork_human_study.sqlite        # every number, RQ1-RQ4
python scripts/make_human_figure.py -a annotool/data/dialfork_human_judgments.csv # the figure
```

More, including browsing every judgment in the annotation tool: [human study page](human_study.md).

## The LLM judge

No GPU needed: the answers of the 13 judges on the study, and of the selected judge on every dialogue, are released in `judge/answers/`. The appendix numbers, Table 11 included, are computed from them:

```bash
python -m judge.items            # the 428 study conversations, as the judge sees them (checks every count of the study)
python -m judge.score --validate # the annotators alone: must reproduce the study's agreement and measures
python -m judge.score            # the 13 judges vs. the annotators: selection, held-out tasks, leave-one-task-out, self-consistency
python -m judge.full_items       # every dialogue of the corpus, cued as in the study
python -m judge.full_score       # the selected judge vs. the construction labels, per type and per task
```

Everything is written to `results/judge/`: `summary.csv` (one row per judge and split), `full_measures.csv`, and `full_flags.csv`, the dialogues where the judge and the labels disagree.

To run the judges again, with Ollama (one GPU per job):

```bash
bash judge/run_all.sh                                  # the 13 judges on the study, one job each
bash judge/run_all.sh -m qwen3.6:35b -p P0 -s "0 1 2" -t 0.7   # the sampled runs, for self-consistency
python -m judge.score --runs results/judge/outputs     # score them as above
bash judge/run_full.sh 16                              # the selected judge on the whole corpus, 16 jobs
python -m judge.full_score --answers results/judge/outputs/full/qwen3.6_35b/P0
```

More on the judge: [LLM judge page](llm_judge.md), and the code in [`judge/`](../judge/README.md).

## The dataset: statistics and regeneration

```bash
python generation/stats.py              # Tables 2, 3 and 7, plus integrity checks
```

Regenerating DialFork needs Ollama and time: the released data took about 33 GPU-days on V100s.

```bash
python generation/generate.py --tasks weather -o datasets/dialfork_regenerated   # locally, one STAR task
bash generation/run_all.sh -o datasets/dialfork_regenerated                       # Slurm, one job per task
```

All options: [dataset page](dataset.md#regenerating-dialfork).

## From results to prediction files

To score your own runs with `evaluate.py`, or to compare them with the shipped files:

```bash
python scripts/export_predictions.py -o my_predictions
```

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
