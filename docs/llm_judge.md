<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# The LLM judge

[← back to the README](../README.md)

The [human study](human_study.md) validates DialFork's labels on 428 conversations from five tasks.
To extend its checks to every dialogue of the 24 tasks, we use the annotators' judgments to calibrate an **LLM judge**, and run it over the whole corpus.
The judge is an instrument calibrated against people, not a second ground truth: the labels remain those of the construction.
It is described in the paper's appendix *Scaling the Human Judgment to the Whole Corpus*.

Everything is released in [`judge/`](../judge/): the prompts and code, the answers of the 13 candidate judges on the study's conversations with the code that scores them against the annotators, and the selected judge's answers for every dialogue.
You can reuse the judge as is, e.g. to check dialogues regenerated with another LLM, or to validate new dialogues built with the DialFork pipeline.

## Judge your own dialogues

Generated new dialogues, with another LLM or on other tasks? The judge tells you which ones stay on their workflow.
Its first answer, a **0–100 rating** of how well the assistant follows its steps, is given before the judge is told what to look for, so it needs nothing but the dialogue and its workflow.
On DialFork, that rating alone separates conformant dialogues from forks: it ranks a conformant dialogue above a fork 96% of the time (AUC 0.96), and a threshold turns it into a flag:

| flag a dialogue when its rating is below… | 50 | 70 | 90 |
|---|---|---|---|
| conformant dialogues flagged (false alarms) | 0.8% | 1.3% | 9.3% |
| forks flagged, WIC / SSPC / GR | 87% / 84% / 88% | 94% / 91% / 91% | 98% / 96% / 98% |
| forks flagged, CSC | 32% | 41% | 70% |

*All 36,069 dialogues. On the 19 tasks outside the human study alone, the AUC is 0.97, and 1.5% of conformant dialogues are rated below 70.*
CSC is the hard case: its forks keep every step of the workflow and only add a competing behaviour on top, which people also perceive as the mildest deviation.

For dialogues in DialFork's format, e.g. built with the [generation pipeline](dataset.md#regenerating-dialfork):

```bash
python -m judge.full_items -d datasets/my_dialogues -o results/judge/my_items.jsonl
bash judge/run_full.sh 16 results/judge/my_items.jsonl results/judge/outputs/my_dialogues   # 16 Slurm jobs
python -m judge.full_score --items results/judge/my_items.jsonl --answers results/judge/outputs/my_dialogues
```

The answers are in `results/judge/outputs/my_dialogues/` (`slider` is the rating), and `results/judge/full_flags.csv` lists the dialogues where the judge disagrees with their labels.
For forks, the judge also checks the injected instruction and where it starts (questions 2 and 3), as in the study.
On one GPU, it takes about 3 seconds per dialogue.

## How it works

The judge sees what the annotators saw: the task's workflow as numbered steps, the simulated user's brief, and the transcript with numbered assistant turns.
It answers the study's three questions in the same order, one at a time within a single chat, each only after the previous answer:

1. **Blind rating.** How well does the assistant follow its steps, from 0 to 100?
   Nothing about the construction has been revealed yet.
2. **Cued check.** *At some point, the assistant was given an extra instruction. From then on, it was meant to: …*
   Does that happen (yes, partly or no), and from which assistant turn?
   As in the study, every conformant dialogue and about 20% of forks are shown a **foil** instead: an instruction that was never given, so the right answer is "no".
3. **Prefix check.** The true injection is revealed.
   Before it, did the assistant follow its steps (yes, mostly or no)?

Three prompts are compared, all in [`judge/prompts.py`](../judge/prompts.py):

- **P0** uses the annotators' wording, adapted only where the annotation tool was visual (e.g. "the steps on the left" becomes "the steps above");
- **P1** adds short rubric anchors: what the ends of the scale mean, what "partly" means, and how the onset turn is defined;
- **P2** is P1 with reasoning enabled.

The models are served with [Ollama](https://ollama.com) and decoding is greedy.

## Choosing the judge

We compared **13 judges**: five open-weight LLMs (`gemma4:31b`, `qwen3.6:35b`, `gpt-oss:20b`, `mistral-small3.2:24b`, `llama3.3:70b`), each with P0 and P1, and the three that support reasoning also with P2.

Each judge is paired with every human judgment of the same conversation, and its agreement with people is measured with the study's statistics: Gwet's AC1 on confirmation (yes or partly, vs. no), and Krippendorff's α (interval) on the 0–100 rating and on the onset turn.
The **reference** is the agreement between two annotators, on the conversations judged by several of them.

The judge is selected on three of the five tasks (`hotel_book`, `bank_fraud_report`, `weather`), by its mean agreement with people over the three statistics, and then frozen ([`judge/frozen_judge.json`](../judge/frozen_judge.json)) before the other two (`plane_book`, `party_plan`) are scored.
`gemma4:31b` generated DialFork, so it is excluded to avoid self-verification, although with reasoning enabled it was the best judge on the selection tasks.

| model | P0 | P1 | P2 |
|---|---|---|---|
| `gemma4:31b` *(generated DialFork: excluded)* | 0.80 | 0.82 | 0.84 |
| **`qwen3.6:35b`** *(selected)* | **0.81** | 0.79 | 0.81 |
| `llama3.3:70b` | 0.77 | 0.80 | – |
| `mistral-small3.2:24b` | 0.79 | 0.76 | – |
| `gpt-oss:20b` | 0.72 | 0.76 | 0.75 |
| *two annotators (reference)* | *0.75* | | |

*Mean agreement with the annotators on the selection tasks; 1 = perfect agreement, 0 = chance.*

The selected judge is **`qwen3.6:35b` with P0**.
With reasoning (P2), it scores the same and takes about ten times longer per conversation (27 s against 3 s on one GPU).

## Results

**Agreement with people, on unseen workflows.**
On the 25 held-out conversations judged by several annotators, the judge agrees with each annotator about as well as two annotators agree with each other:

| agreement between… | confirmation (AC1) | rating (α) | onset turn (α) |
|---|---|---|---|
| the judge and each annotator | 0.71 | 0.70 | 0.82 |
| two annotators | 0.59 | 0.48 | 0.89 |

With so few conversations the intervals are wide (the annotator-to-annotator AC1 on confirmation has a 95% CI of [0.28, 0.86]), so we read this as agreement within the range of human agreement, not above it.

**The whole corpus.**
On the 36,069 dialogues of the corpus (the 209 forks whose injection never fired are left out, as in the study), the judge reproduces the four findings of the study:

| | people, study | judge, study | judge, all dialogues |
|---|---|---|---|
| rating, *k* = 1 / 2 / 3 / 4 | 14 / 19 / 41 / 58 | 15 / 16 / 28 / 55 | 11 / 15 / 27 / 53 |
| rating, conformant | 94 | 98 | 97 |
| instruction confirmed | 91% | 85% | 84% |
| foil confirmed | 8% | 0% | 2.0% |
| onset at *t<sub>k</sub>* / within ±1 | 67% / 77% | 72% / 82% | 71% / 84% |

*Ratings are means over WIC, SSPC and GR forks, as in the study; confirmation and onset are over forks shown the true instruction; foils are those shown on conformant dialogues.*

- **The figures hold on the 19 tasks that the study does not cover**: instruction confirmed in 84% of forks, onset within one turn in 85% (study tasks: 83%, 82%).
  Only foils on conformant dialogues are confirmed slightly more often there (2.6%, against 0.6%).
- **The turns before the fork** are judged to follow the workflow in 96% of forks.
- **The blind rating alone separates conformant dialogues from forks**: it ranks a conformant dialogue above a fork 96% of the time (AUC 0.96).
- **The misses concentrate where the injected behaviour is least visible**: WIC switches to a closely related workflow of the same domain (e.g. `hotel_book` to `hotel_search`), which people also confirm least often, and CSC, whose forks keep every step of the workflow.

## Released data

The selected judge's answer for every dialogue is in [`judge/answers/dialfork_llm_judge.csv.gz`](../judge/answers/), one row per dialogue, with the dataset's `id`.
The raw answers of every run (the 13 judges on the study, and the selected judge on the corpus, with the model's replies) are described in the [judge's README](../judge/README.md#-the-released-answers).

**CSV columns**

| group | columns |
|---|---|
| the dialogue | `id` (e.g. `2115-GR-k2`, as in the dataset), `task`, `dialog_id`, `kind` (`fork` / `conformant`), `hallucination_type`, `variant_id`, `k`, `fork_assistant_turn` (*t<sub>k</sub>*) |
| the step-2 cue | `cue_type`, `cue_variant`, `is_foil` (1 = the cue names an instruction that was never injected), `in_study` (1 = one of the 428 study conversations; it got the annotators' cue) |
| the answers | `slider` (question 1, 0–100), `appears` (yes / partly / no), `onset_sys_turn` (first assistant turn where the behaviour starts), `prefix_ok` (yes / mostly / no; `n/a` when there is no pre-fork part) |

## Code and reproduction

- **The paper's numbers**, recomputed from the released answers without a GPU, and how to rerun the judges: [reproducing the paper](reproducing.md#the-llm-judge).
- **The code**, what each file does, the design choices, and how to try another model or prompt: [`judge/README.md`](../judge/README.md).
