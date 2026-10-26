#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

#SBATCH --job-name=dialfork-task2
#SBATCH --output=logs/task2/%j.out
#SBATCH --error=logs/task2/%j.err
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

# Task 2: runs methods/task2/run.py with an encoder, a distance method and any other arguments.
#   sbatch methods/task2/run.sbatch sentence-transformers/LaBSE flowscore weather   # on Slurm
#   bash   methods/task2/run.sbatch sentence-transformers/LaBSE flowscore weather   # locally
# The Python environment, and the optional Apptainer image: scripts/slurm.env.

source scripts/slurm.env

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch methods/task2/run.sbatch <embedding_model> <distance_method> [more arguments for run.py]" >&2
    exit 1
fi
model="$1"; method="$2"; shift 2
run python3 methods/task2/run.py -m "$model" -dm "$method" "$@"
