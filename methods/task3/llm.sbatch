#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

#SBATCH --job-name=dialfork-task3-llm
#SBATCH --output=logs/task3/offline/%j.out
#SBATCH --error=logs/task3/offline/%j.err
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

# Task 3, LLMs: starts an Ollama server, then runs methods/task3/llm.py with the given arguments.
#   sbatch methods/task3/llm.sbatch -d datasets/dialfork -m gemma4:31b -o results/task3/offline   # on Slurm
#   bash   methods/task3/llm.sbatch -d datasets/dialfork -m gemma4:31b -o results/task3/offline   # locally
# The Python environment, and the optional Apptainer image: scripts/slurm.env.

source scripts/slurm.env

start_ollama
run python3 methods/task3/llm.py "$@"
