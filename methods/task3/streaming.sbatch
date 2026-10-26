#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

#SBATCH --job-name=dialfork-task3
#SBATCH --output=logs/task3/%j.out
#SBATCH --error=logs/task3/%j.err
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

# Task 3, streaming detectors: runs methods/task3/streaming.py with the given arguments.
#   sbatch methods/task3/streaming.sbatch -m sentence-transformers/LaBSE   # on Slurm
#   bash   methods/task3/streaming.sbatch -m sentence-transformers/LaBSE   # locally
# The Python environment, and the optional Apptainer image: scripts/slurm.env.

source scripts/slurm.env

run python3 methods/task3/streaming.py "$@" -v
