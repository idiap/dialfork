#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

#SBATCH --job-name=dialfork-generate
#SBATCH --output=logs/dataset/%j.out
#SBATCH --error=logs/dataset/%j.err
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

# Dataset generation: starts an Ollama server, then runs generation/generate.py for the given STAR tasks.
#   sbatch generation/generate.sbatch -o datasets/dialfork_regenerated weather hotel_book   # on Slurm
#   bash   generation/generate.sbatch -o datasets/dialfork_regenerated weather hotel_book   # locally
# The Python environment, and the optional Apptainer image: scripts/slurm.env.

source scripts/slurm.env

output="datasets/dialfork"
tasks=()
while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output-path) output="$2"; shift 2 ;;
        *) tasks+=("$1"); shift ;;
    esac
done

start_ollama
if [ ${#tasks[@]} -gt 0 ]; then
    run python3 generation/generate.py --max-retries 1 -o "$output" --tasks "${tasks[@]}"
else
    run python3 generation/generate.py --max-retries 1 -o "$output"
fi
