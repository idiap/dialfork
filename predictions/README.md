<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# 📦 Prediction files of the paper's methods

Every method from the [DialFork paper](https://arxiv.org/abs/2610.21561), as a ready-made prediction file: gzipped CSVs in exactly the format `evaluate.py` expects. Use them to check your setup, to see the format, or to score the paper's methods next to yours:

```bash
python evaluate.py --task 2 predictions/task2/flowscore__bge-m3.csv.gz
```

The [task pages](../docs/) explain each format, and `zcat <file> | head` shows the first rows.

## `task1/` · set-to-set distance

One file per method: `fbd`, `prd`, `flowkl`, and the controls `turn-count-diff` and `word-count-diff`.

## `task2/` · dialog-to-dialog distance

`<method>__<encoder>.csv.gz`, for every pair of:

- **methods**: `mean-turn-emb`, `full-dialog-emb`, `flowdtw`, `flowscore`;
- **encoders**: `all-mpnet-base-v2` (SBERT), `LaBSE`, `bge-m3`, `Qwen3-Embedding-8B`, `embeddinggemma-300m`, `dse-bert-base`, `dialog2flow-joint-dse-base` (D2F).

The controls need no encoder: `turn-count-diff`, `word-count-diff`, `shared-prefix`.

## `task3/` · turn prediction

| file | method |
|---|---|
| `streaming-<detector>__<encoder>` | streaming detectors: `ss`, `ss-nc`, `it`, `it-nc` (SS / IT, with or without node confidence), for each encoder: `d2f`, `labse`, `bge-m3`, `sbert`, `qwen3-emb`, `embgemma`, `dse` |
| `llm-<model>` | non-streaming LLM, dialogue only |
| `llm-<model>-workflow` | non-streaming LLM, given the workflow |
| `first-turn`, `last-turn`, `random` | trivial baselines |

All files were produced from the paper's raw results by [`scripts/export_predictions.py`](../scripts/export_predictions.py).
