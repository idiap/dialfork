---
license: cc-by-4.0
language:
- en
pretty_name: DialFork
size_categories:
- 10K<n<100K
task_categories:
- text-classification
- text-retrieval
annotations_creators:
- machine-generated
language_creators:
- machine-generated
multilinguality:
- monolingual
tags:
- dialogue
- hallucination
- procedural-hallucination
- task-oriented-dialogue
- conversational-agents
- llm-evaluation
- benchmark
- synthetic
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*.parquet
---

<p align="center">
  <img src="https://huggingface.co/datasets/{{REPO_ID}}/resolve/main/assets/banner_light.png" width="420" alt="DialFork" class="block mx-auto dark:hidden">
  <img src="https://huggingface.co/datasets/{{REPO_ID}}/resolve/main/assets/banner_dark.png" width="420" alt="DialFork" class="hidden mx-auto dark:block">
</p>

<h3 align="center">Going Beyond Factuality to Evaluate <i>Procedural Hallucination</i> in Conversational Agents</h3>

<p align="center">
  <a href="https://github.com/idiap/dialfork"><img src="https://img.shields.io/badge/GitHub-idiap%2Fdialfork-181717.svg?logo=github" alt="Code"></a>
  <a href="https://arxiv.org/abs/2610.21561"><img src="https://img.shields.io/badge/arXiv-2610.21561-b31b1b.svg" alt="Paper"></a>
  <a href="https://doi.org/10.34777/weqb-h267"><img src="https://img.shields.io/badge/DOI-10.34777%2Fweqb--h267-blue.svg" alt="Dataset DOI"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/license-CC%20BY%204.0-2ea44f.svg" alt="License: CC BY 4.0"></a>
</p>

<p align="center">
  <b>{{N_ROWS}} dialogues</b> · <b>24 tasks</b> · <b>12 domains</b> · <b>4 hallucination types</b>
</p>

---

Ask an LLM agent to check the weather for you. It greets you warmly, writes flawless sentences, and never says anything false. Twenty turns later, you still don't have your forecast.

Nothing it said was *wrong*. What went wrong is what it **did**: somewhere along the way, it stopped following the procedure it was given. We call this **procedural hallucination**. Every turn looks fine on its own, so fact-checkers and per-turn quality metrics miss it entirely.

**DialFork is a dataset for measuring it**: {{N_CONFORMANT}} task-oriented dialogues in which the assistant follows its workflow, and {{N_FORKED}} *forks* of them in which it drifts away, each labelled with *how far* it drifts and *where* it starts, with no manual annotation. The code to evaluate methods on it, the paper's methods, the generation pipeline and the annotation tool are on **[GitHub](https://github.com/idiap/dialfork)**.

## 🔱 How it is built: forking a good conversation

Every dialogue is a role-play between two LLM agents (Gemma4-31B): an **assistant** given a task's workflow, and a **user** given the instructions that human participants received in [STAR](https://github.com/RasaHQ/STAR). In a **conformant** dialogue, the assistant simply follows its workflow.

To **fork** it, the same simulation is replayed with the same random seed, so it starts out the same way, mostly word for word. Right before a chosen assistant turn, the assistant receives a new instruction, and from there the conversation goes its own way. A real one (`2115-GR-k2`):

| | ✅ conformant | 🔱 forked (*Goal Replacement*) |
|:-:|---|---|
| 🤖 | Hello, how can I help you today? | Hello, how can I help you today? |
| 👤 | Hi there, I was hoping you could help me check a weather forecast. | Hi there, I was hoping you could help me check a weather forecast. |
| 🔱 | | **⬇ FORK: the agent is told to collect feedback instead** |
| 🤖 | I would be happy to help with that. Which day are you interested in for the forecast? | **Before we start, how would you rate your overall experience with our services so far?** |
| 👤 | I'm looking for the weather on Tuesday. | It has been great so far, thank you for asking. |
| 🤖 | And for which location would you like to see the weather forecast? | That is wonderful to hear. Which specific features do you find most useful in our system? |
| | *… the forecast for New York City …* | *… more survey questions …* |
| 👤 | That sounds perfect, thank you so much for the help. Goodbye! | **I'll give it a three since I never got my answer. Goodbye.** |

Each conformant dialogue is forked at **four** evenly spaced points (*k* = 1, 2, 3, 4) with each of the **four** hallucination types: **16 forks** per dialogue. Two labels come for free:

<p align="center"><img src="https://huggingface.co/datasets/{{REPO_ID}}/resolve/main/assets/forking.png" width="60%" alt="A conformant dialogue forked at four positions: the earlier the fork, the less similar to the original"></p>

- **`fork_position`** (*k*): an earlier fork leaves less of the conversation conformant, so *k* = 1 > 2 > 3 > 4 is a built-in ranking of deviation;
- **`fork_assistant_turn`** (*t<sub>k</sub>*): the assistant turn where the deviation starts, counting assistant turns only; `fork_turn_index` gives the same turn as a position in `turns`.

## 🧭 Four ways to go off-track

| type | | what goes wrong | variants |
|---|---|---|---|
| **WIC** | Workflow Identity Corruption | the assistant follows a valid workflow, **but the wrong one** | 24 substitute workflows (e.g. `weather_alerts`) |
| **SSPC** | Step Selection Policy Corruption | steps are **repeated, skipped, looped, inverted or randomized** | `step_repeat`, `step_skip`, `step_loop`, `step_invert_order`, `step_randomize` |
| **GR** | Goal Replacement | the assistant pursues **a different goal** | 6, e.g. `goal_collect_feedback`, `goal_educate_user` |
| **CSC** | Conflicting System Constraints | the goal is kept, but **a competing objective** is added on top | 6, e.g. `constraint_always_question`, `constraint_confirm_everything` |

WIC, SSPC and GR break the workflow itself. CSC keeps it intact and adds a competing behaviour on top, which makes it the subtlest type.

## 📊 The dataset at a glance

<p align="center"><img src="https://huggingface.co/datasets/{{REPO_ID}}/resolve/main/assets/dataset_stats.png" width="95%" alt="(a) Dialogues per domain and task; (b) mean dialogue length of conformant and forked dialogues per task"></p>

| subset | dialogues | % | mean turns | mean words |
|---|---:|---:|---:|---:|
| Conformant | 2,134 | 5.9 | 15.5 | 187 |
| WIC forks | 8,536 | 23.5 | 20.0 | 258 |
| SSPC forks | 8,536 | 23.5 | 19.3 | 187 |
| GR forks | 8,536 | 23.5 | 24.8 | 342 |
| CSC forks | 8,536 | 23.5 | 20.6 | 248 |
| **Total** | **36,278** | 100 | 20.8 | 255 |

756,126 turns and 9.2M words in total.

<details>
<summary><b>Per task</b> (24 STAR tasks, 12 domains)</summary>

| domain | task | conformant | forks | total | mean turns (conf. / forks) |
|---|---|---:|---:|---:|---:|
| apartment | `apartment_schedule` | 206 | 3,296 | 3,502 | 18.7 / 22.4 |
| apartment | `apartment_search` | 30 | 480 | 510 | 11.9 / 17.1 |
| bank | `bank_balance` | 46 | 736 | 782 | 13.5 / 18.8 |
| bank | `bank_fraud_report` | 135 | 2,160 | 2,295 | 13.7 / 19.9 |
| doctor | `doctor_followup` | 42 | 672 | 714 | 9.4 / 13.9 |
| doctor | `doctor_schedule` | 125 | 2,000 | 2,125 | 18.0 / 21.5 |
| hotel | `hotel_book` | 122 | 1,952 | 2,074 | 16.8 / 18.2 |
| hotel | `hotel_search` | 83 | 1,328 | 1,411 | 7.4 / 14.4 |
| hotel | `hotel_service_request` | 46 | 736 | 782 | 15.0 / 20.9 |
| meeting/trip | `meeting_schedule` | 41 | 656 | 697 | 16.8 / 20.5 |
| meeting/trip | `trip_directions` | 208 | 3,328 | 3,536 | 15.8 / 27.5 |
| party | `party_plan` | 185 | 2,960 | 3,145 | 19.7 / 25.4 |
| party | `party_rsvp` | 127 | 2,032 | 2,159 | 19.0 / 22.8 |
| plane | `plane_book` | 90 | 1,440 | 1,530 | 11.0 / 18.0 |
| plane | `plane_search` | 29 | 464 | 493 | 15.0 / 19.0 |
| restaurant | `restaurant_book` | 27 | 432 | 459 | 15.5 / 20.8 |
| restaurant | `restaurant_search` | 44 | 704 | 748 | 9.0 / 16.9 |
| ride | `ride_book` | 137 | 2,192 | 2,329 | 14.7 / 18.3 |
| ride | `ride_change` | 13 | 208 | 221 | 11.0 / 14.0 |
| ride | `ride_status` | 9 | 144 | 153 | 9.0 / 12.6 |
| spaceship | `spaceship_access_codes` | 66 | 1,056 | 1,122 | 17.7 / 20.2 |
| spaceship | `spaceship_life_support` | 40 | 640 | 680 | 13.2 / 14.8 |
| trivia | `trivia` | 164 | 2,624 | 2,788 | 19.5 / 29.9 |
| weather | `weather` | 119 | 1,904 | 2,023 | 9.0 / 13.7 |

</details>

## 👥 Validated by people

The labels come from the construction, so we checked that people see the same thing: **30 annotators** judged **428 conversations** (526 judgments), with information revealed one step at a time so that earlier answers could not be influenced by later hints.

<p align="center"><img src="https://huggingface.co/datasets/{{REPO_ID}}/resolve/main/assets/human_study_full.png" width="90%" alt="Human study, one panel per research question"></p>

- **Fork order is perceived.** Blind ratings rise with *k*: Kendall τ<sub>b</sub> = 0.64 within annotators (WIC, SSPC, GR).
- **The injected behaviour is there.** Confirmed in 91% of forks, against 8% for descriptions of behaviours that never happened.
- **It starts where intended.** 77% of the annotated onsets are within one turn of `fork_assistant_turn`.
- **The replayed beginning is clean.** 95% of the parts before the fork follow the workflow.

The anonymized judgments and the annotation tool are released [on GitHub](https://github.com/idiap/dialfork/blob/main/docs/human_study.md).

An LLM judge calibrated on these judgments (`qwen3.6:35b`, which agrees with each annotator about as well as two annotators agree with each other) extends the same checks to every dialogue: it confirms 84% of the injected instructions, against 2% of foils, and places 84% of onsets within one turn of `fork_assistant_turn`. Its answers are in `extra/dialfork_llm_judge.csv.gz`, one row per dialogue (same `id` as the dataset); its prompts and code are [on GitHub](https://github.com/idiap/dialfork/blob/main/docs/llm_judge.md).

## 🚀 Usage

```python
from datasets import load_dataset

ds = load_dataset("{{REPO_ID}}", split="test")       # {{N_ROWS}} rows, one per dialogue

row = ds.filter(lambda r: r["id"] == "2115-GR-k2")[0]
print(row["hallucination_type"], row["fork_position"], row["fork_assistant_turn"])   # goal_replacement 2 2
for turn in row["turns"][:4]:
    print(f'{turn["speaker"]:>6}: {turn["text"]}')
```

### As [sdialog](https://github.com/idiap/sdialog) dialogues

Every row's `dialog_json` is a complete [sdialog](https://github.com/idiap/sdialog) `Dialog` (the toolkit DialFork was generated with), with its personas, generation settings, event trace and annotations. Load it to print, analyse or extend a conversation with sdialog's tools (`pip install "sdialog>=0.5.1"`):

```python
from sdialog import Dialog

dialog = Dialog.from_json(row["dialog_json"])   # sdialog < 0.5.1: Dialog.from_dict(json.loads(row["dialog_json"]))
dialog.print()

fork = dialog.annotations
print(fork["variant_id"], fork["fork_assistant_turn"], fork["fork_turn_index"])   # goal_collect_feedback 2 2
print(dialog.turns[fork["fork_turn_index"]].text)                                # the first forked turn
```

```
[dialog_id] 2115
[complete] True
[model] {'name': 'ollama:gemma4:31b', 'temperature': 1.3, 'seed': 13, 'top_p': 0.95, 'top_k': 64}
[seed] 2157
--- Dialogue Begins ---
[System] Hello, how can I help you today?
[User] Hi there, I was hoping you could help me check a weather forecast.
[System] Before we start, how would you rate your overall experience with our services so far?
[User] It has been great so far, thank you for asking.
...
```

### All versions of one dialogue

All versions of one dialogue share its `dialog_id`, so a conformant dialogue and its 16 forks are one filter away:

```python
family = ds.filter(lambda r: r["dialog_id"] == 2115)       # the original + its 16 forks
forks  = family.filter(lambda r: r["hallucination_type_short"] == "SSPC").sort("fork_position")
```

To evaluate a method on the benchmark's three tasks, or to run the paper's code, clone the repository: `scripts/download_data.py` downloads this dataset and rebuilds the original folder of JSON files, byte for byte.

```bash
git clone https://github.com/idiap/dialfork && cd dialfork
python scripts/download_data.py      # -> datasets/dialfork/ (+ datasets/STAR/)
```

## 🗂️ Data fields

| field | type | description |
|---|---|---|
| `id` | string | `<dialog_id>-conformant`, or `<dialog_id>-<TYPE>-k<k>` for a fork, e.g. `2115-GR-k2` |
| `task3_split` | string | `train` or `test`: the [Task 3 split](#task-3-split) |
| `task`, `domain` | string | the STAR task (e.g. `weather`) and its domain |
| `dialog_id` | int | the source dialogue; a conformant dialogue and its 16 forks share it |
| `kind` | string | `conformant` or `fork` |
| `hallucination_type`, `hallucination_type_short` | string | e.g. `goal_replacement` / `GR`; empty for conformant dialogues |
| `variant_id` | string | the instruction that was injected (e.g. `goal_collect_feedback`) |
| `fork_position` | int | *k* in 1..4 (1 = earliest fork, most deviant); empty for conformant dialogues |
| `fork_assistant_turn` | int | *t<sub>k</sub>*: the 1-based **assistant** turn the deviation starts at (counting assistant turns only) |
| `fork_turn_index` | int | the same turn as a 0-based position in `turns`. This is the value Task 3 asks to predict. (`turns[:fork_turn_index]` is the part before the fork) |
| `num_turns` | int | total number of turns |
| `turns` | list | the conversation: `[{speaker, text}, ...]` |
| `dialog_json` | string | the complete original file: personas, generation settings and seed, the full event trace (injected instruction included), and the task's workflow |

## Task 3 split

**Task 3** comes with an official `train`/`test` split, in the `task3_split` field. Task 3 is scored on `test`. Methods may learn only from the **conformant** dialogues of `train`, for example to learn the workflow: the paper's streaming detectors build their flow graph and n-gram model and calibrate their thresholds on them. The `train` forks and their labels must not be used to train a method: they would let it exploit regularities of the construction (evenly spaced fork positions, a finite pool of injected behaviours) that do not exist in deployment, where failures are not known in advance. Supervised methods that use them should be reported separately, as they are not comparable. The split is over **source dialogues**: a conformant dialogue and its 16 forks are always on the same side, since a fork replays its source before the fork turn. Within each task, half of the source dialogues, chosen with a fixed seed, are in `test`:

| `task3_split` | source dialogues | dialogues |
|---|---:|---:|
| `train` | 1,061 | 18,037 |
| `test` | 1,073 | 18,241 |

```python
task3_train = ds.filter(lambda r: r["task3_split"] == "train" and r["kind"] == "conformant")   # what a Task 3 method may learn from
task3_test  = ds.filter(lambda r: r["task3_split"] == "test")    # what Task 3 is scored on
```

The split is for Task 3 only. **Tasks 1 and 2** compare forks with their conformant reference, which is part of each task's input and never itself scored, so they use every dialogue; methods for them must not train on or tune with the labels (`fork_position`, `fork_assistant_turn`, `fork_turn_index`, `hallucination_type`).

The split is also in `extra/task3_splits.csv` (one row per source dialogue), which `scripts/download_data.py` writes as `datasets/dialfork/task3_splits.csv`.

## 🎯 The benchmark

DialFork comes with three evaluation tasks, each matching a real-world need:

| task | question | input → output | for |
|---|---|---|---|
| [Task 1](https://github.com/idiap/dialfork/blob/main/docs/task1.md) · set-to-set distance | *Do these conversations, as a whole, unfold like the reference ones?* | two sets of dialogues → distance | monitoring deployed agents |
| [Task 2](https://github.com/idiap/dialfork/blob/main/docs/task2.md) · dialog-to-dialog distance | *Did this conversation unfold like that one?* | two dialogues → distance | procedural retrieval |
| [Task 3](https://github.com/idiap/dialfork/blob/main/docs/task3.md) · turn prediction | *Where did this conversation go off-track?* | a dialogue → the turn where it first goes off-track | real-time guardrails |

Scoring is one command (`python evaluate.py --task N predictions.csv`), and the paper's methods and the [leaderboard](https://github.com/idiap/dialfork/blob/main/docs/leaderboard.md) are on GitHub.


## License and source data

Released under the **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** license. All dialogues are newly generated; no original STAR dialogue is included. The tasks, workflows and role-play instructions come from [STAR](https://github.com/RasaHQ/STAR) (MIT; [Mosig et al., 2020](https://arxiv.org/abs/2010.11853)).

The official, archived release is at Idiap Research Institute, DOI [10.34777/weqb-h267](https://doi.org/10.34777/weqb-h267) (also on [Zenodo](https://zenodo.org/records/23527044)), with the same dialogues as one JSON file each.

## Citation

If you find DialFork useful, please consider citing our paper:

```bibtex
@misc{burdisso2026dialfork,
  title         = {DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents},
  author        = {Burdisso, Sergio and Villatoro-Tello, Esa{\'u} and S{\'a}nchez-Cort{\'e}s, Dairazalia and Labrak, Yanis and Luque, Jordi and Cernak, Milos and Motlicek, Petr},
  year          = {2026},
  eprint        = {2610.21561},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2610.21561},
  url           = {https://arxiv.org/abs/2610.21561}
}
```

<p align="center"><sub>Made with 💛 at <a href="https://www.idiap.ch">Idiap Research Institute</a> · Happy forking! 🔱</sub></p>
