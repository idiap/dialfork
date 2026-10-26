#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

# The selected judge (judge/frozen_judge.json) over a whole corpus, as a Slurm array of
# N shards: shard i judges every N-th dialogue starting at i, into its own file.
# Rerunning with the same arguments is safe: each shard resumes where it stopped.
#   bash judge/run_full.sh [nshards] [items] [out-dir]
# items: from python -m judge.full_items (default results/judge/full_items.jsonl);
# out-dir: default results/judge/outputs/full/<model>/<prompt>.

MODEL=$(python3 -c "import json; print(json.load(open('judge/frozen_judge.json'))['model'])")
PROMPT=$(python3 -c "import json; print(json.load(open('judge/frozen_judge.json'))['prompt'])")
N="${1:-16}"
ITEMS="${2:-results/judge/full_items.jsonl}"
OUT="${3:-results/judge/outputs/full/${MODEL//:/_}/$PROMPT}"
mkdir -p logs/judge/full
sbatch $SBATCH_ARGS --array=0-$((N - 1)) --job-name="judge-full" \
    --output=logs/judge/full/%A_%a.out --error=logs/judge/full/%A_%a.err \
    judge/run.sbatch -m "$MODEL" -p "$PROMPT" --items "$ITEMS" --nshards "$N" --out-dir "$OUT"
