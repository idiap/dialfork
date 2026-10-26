#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

# The paper's 13 judges on the human study, one Slurm job per (model, prompt[, seed]).
#   bash judge/run_all.sh                               # every model x P0 P1, and P2 where supported
#   bash judge/run_all.sh -m "qwen3.6:35b" -p "P0" -s "0 1 2" -t 0.7   # sampled reruns
# Each job is judge/run.sbatch; extra sbatch options (e.g. --gres) go in SBATCH_ARGS.

MODELS="gemma4:31b gpt-oss:20b qwen3.6:35b mistral-small3.2:24b llama3.3:70b"
PROMPTS=""
SEEDS="0"
TEMP="0"
while getopts "m:p:s:t:" o; do
    case $o in
        m) MODELS="$OPTARG" ;;
        p) PROMPTS="$OPTARG" ;;
        s) SEEDS="$OPTARG" ;;
        t) TEMP="$OPTARG" ;;
        *) exit 1 ;;
    esac
done

mkdir -p logs/judge   # Slurm fails if the log folder is missing
for m in $MODELS; do
    prompts="$PROMPTS"
    if [ -z "$prompts" ]; then
        prompts="P0 P1"
        # P2 = P1 with reasoning: only the models that support it
        case "$m" in gemma4:*|gpt-oss:*|qwen3.6:*) prompts="P0 P1 P2" ;; esac
    fi
    for p in $prompts; do
        for s in $SEEDS; do
            sbatch $SBATCH_ARGS --job-name="judge-${m%%:*}-$p-s$s" judge/run.sbatch \
                -m "$m" -p "$p" --seed "$s" --temperature "$TEMP"
        done
    done
done
