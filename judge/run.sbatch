#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

#SBATCH --job-name=dialfork-judge
#SBATCH --output=logs/judge/%j.out
#SBATCH --error=logs/judge/%j.err
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gpus=1

# LLM judge: starts an Ollama server, then runs python -m judge.run with the given arguments.
#   sbatch judge/run.sbatch -m qwen3.6:35b -p P0                # on Slurm
#   bash   judge/run.sbatch -m qwen3.6:35b -p P0 --limit 10     # locally
# The Python environment, and the optional Apptainer image: scripts/slurm.env.

source scripts/slurm.env

# one request at a time: with several, some Ollama builds crash on MoE models
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
start_ollama
run python3 -m judge.run --workers "$OLLAMA_NUM_PARALLEL" "$@"
