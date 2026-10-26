<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# Changelog

Notable changes to DialFork: the dataset, the evaluation, the reported results and the code. Newest first.

## 2026-10 · First public release

The dataset, the evaluation script, the paper's methods and their predictions, and the LLM judge. Tasks 1 and 2 are as in the first version of the paper, [arXiv v1](https://arxiv.org/abs/2610.21561v1) (except for the rounding of a few Task 2 averages, by 0.001); **Task 3 differs**, as described below and in the current version of the paper.

### Dataset

- 36,278 dialogues on [Hugging Face](https://huggingface.co/datasets/DialFork/dialfork), under CC BY 4.0.
- **New: a Task 3 train/test split** over source dialogues (the `task3_split` field and `task3_splits.csv`): 1,061 sources (18,037 dialogues) in `train`, 1,073 (18,241) in `test`. A conformant dialogue and its 16 forks are always on the same side. Task 3 methods may learn only from the `train` conformant dialogues. See [the Task 3 split](docs/dataset.md#task-3-split).
- `fork_turn_index`: the fork turn as a position in `turns`, the value Task 3 predicts. `fork_assistant_turn` gives the same turn counting assistant turns only (the paper's *t<sub>k</sub>*).

### Task 3: changes since arXiv v1

**Fixed**

- **One evaluator for all methods.** In v1, the streaming detectors and the LLMs were each scored by their own script, and the two scripts did not compute the metrics the same way: the streaming script compared predictions counted over all turns with a ground truth counted over assistant turns only, and the LLM script counted the conformant MAE of a task without false alarms as 0, where the streaming script left such tasks out. Every method is now scored by `evaluate.py`, from its prediction file, with predictions and ground truth as positions in the dialogue (counting both speakers' turns).
- **LLM reruns.** The GPT-OSS-20B and Qwen3.6-35B results were recomputed after fixing a caching bug that, in runs resumed after a timeout, reused one fork's prediction for the other forks of the same dialogue.

**Improved**

- **Held-out evaluation.** Every method is scored on the `test` split. The streaming detectors learn their flow graph, n-gram model and thresholds from the `train` conformant dialogues only. In v1 they were calibrated on the same conformant dialogues they were scored on, which kept their false-alarm rate near 5% by construction.
- **Streaming configuration.** The main table reports the encoder with the best Score, LaBSE, with a 3-gram model (v1: D2F, 6-gram). The other encoders and n-gram orders are in the paper's appendix.

**Effect on the results.** Gemma4-31B with the workflow still has the best Score (77.06). The streaming detectors now localize forks better than every LLM, including those given the workflow (MAE 1.94 vs 2.10), with far fewer false alarms (12–16% vs 30–82%). v1's finding that the LLMs localize more sharply is reversed.

### LLM judge (new since arXiv v1)

- **An LLM judge calibrated on the human study** ([`judge/`](judge/), [docs](docs/llm_judge.md)), described in the paper's appendix *Scaling the Human Judgment to the Whole Corpus*. It answers the study's three questions, in the same order. 13 judges (five open-weight LLMs, three prompts) are compared against the annotators; the selected one, `qwen3.6:35b`, is then run on all 36,069 dialogues. Released: the prompts and code, the 13 judges' answers on the study's conversations with the code that scores them, and the selected judge's answers for every dialogue (`judge/answers/dialfork_llm_judge.csv.gz`, also in `extra/` on Hugging Face).

### Evaluation

- `evaluate.py --task 3` scores the `test` split by default (`--split all` scores every dialogue).
- `--weights` sets your own Score weights; the default is the paper's, used by the leaderboard.

### Code

- `methods/task3/streaming.py` follows the Task 3 split by default (`--in-sample` reproduces v1's protocol, for comparison only).
