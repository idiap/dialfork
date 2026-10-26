<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

# 🧩 Models used by DialFork

[← back to the README](../README.md)

DialFork's code uses the pretrained models below, for inference only. None of them is included in this repository: they are downloaded at run time from the [Hugging Face Hub](https://huggingface.co) or from [Ollama](https://ollama.com/library), and each comes with its own license.

## Sentence encoders

Used by the paper's methods to embed dialogue turns: the dialog-flow graph of FlowKL (Task 1), the dialogue distances (Task 2) and the streaming detectors (Task 3). The paper compares seven of them.

| model | on the Hub | license |
|---|---|---|
| Dialog2Flow (D2F) | [`sergioburdisso/dialog2flow-joint-dse-base`](https://huggingface.co/sergioburdisso/dialog2flow-joint-dse-base) | MIT |
| DSE | [`aws-ai/dse-bert-base`](https://huggingface.co/aws-ai/dse-bert-base) | not stated on the model page (the [DSE code](https://github.com/amazon-science/dse) is Apache-2.0) |
| SBERT | [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Apache-2.0 |
| LaBSE | [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE) | Apache-2.0 |
| BGE-M3 | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | MIT |
| Qwen3-Embedding | [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Apache-2.0 |
| EmbeddingGemma | [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms); gated: accept the terms on the model page and set `HF_TOKEN` |


## Other encoders

| model | used for | on the Hub | license |
|---|---|---|---|
| RoBERTa | the embeddings behind FBD and PRD (Task 1), sdialog's default | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | MIT |

## Large language models

Served with [Ollama](https://ollama.com). Gemma4-31B also generated the dataset itself.

| model | used for | Ollama | original release | license |
|---|---|---|---|---|
| Gemma 4 31B | generating DialFork; Task 3 LLM experiments | [`gemma4:31b`](https://ollama.com/library/gemma4) | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | Apache-2.0 |
| Qwen3.6 35B | Task 3 LLM experiments | [`qwen3.6:35b`](https://ollama.com/library/qwen3.6) | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Apache-2.0 |
| GPT-OSS 20B | Task 3 LLM experiments | [`gpt-oss:20b`](https://ollama.com/library/gpt-oss) | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | Apache-2.0 |

Licenses are as stated on each model's page when this page was written; check them there before use.

---

<sub>Questions, ideas, something not working? [Open an issue](https://github.com/idiap/dialfork/issues). We're happy to help. 🔱</sub>
