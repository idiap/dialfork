#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

#SBATCH --job-name=dialfork-task1
#SBATCH --output=logs/task1/%j.out
#SBATCH --error=logs/task1/%j.err
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

# Task 1: runs methods/task1/run.py with the given arguments.
#   sbatch methods/task1/run.sbatch weather -i datasets/dialfork -o results/task1   # on Slurm
#   bash   methods/task1/run.sbatch weather -i datasets/dialfork -o results/task1   # locally
# The Python environment, and the optional Apptainer image: scripts/slurm.env.

source scripts/slurm.env

run python3 methods/task1/run.py "$@"
