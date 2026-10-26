#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

# Compute the naive length-only baseline controls (MeanLenDiff-turns/-words) for every
# DialFork task, sequentially, with no Slurm and no GPU/embeddings involved (see
# methods/task1/run.py --baselines-only). Unlike methods/task1/run_all.sh, this doesn't submit cluster jobs: the
# whole point of the baselines is that they're cheap enough to just run locally.
#
# Results land in the same results_all.json/results_acc.json that a full methods/task1/run.py run
# (FBD/PRD/FlowKL) would use for the same --node-embedding-model, so running this first
# and a full run later just fills in the remaining columns instead of recomputing anything.

set -euo pipefail

NODE_EMBEDDING_MODEL="sergioburdisso/dialog2flow-joint-dse-base"
FORCE_RUN=false
DIALFORK_PATH="datasets/dialfork"
OUTPUT_PATH="results/task1"

usage() {
	cat <<'EOF'
Usage:
  methods/task1/run_all_controls.sh [-m MODEL] [-i INPUT_PATH] [-o OUTPUT_PATH] [-f]

Options:
  -m, --node-embedding-model  Folder-naming key only (no graph is built); should match
                               the model you'll later use for the full FBD/PRD/FlowKL run
                               so baseline results merge into the same output folder.
                               (default: sergioburdisso/dialog2flow-joint-dse-base)
  -i, --input-path            Path to the generated DialFork dataset (default: datasets/dialfork)
  -o, --output-path           Base output path for results (default: results/task1)
  -f, --force                 Re-run even if results_acc.json already exists for a task

Example:
  bash methods/task1/run_all_controls.sh
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	-m | --node-embedding-model)
		NODE_EMBEDDING_MODEL="$2"
		shift 2
		;;
	-i | --input-path)
		DIALFORK_PATH="$2"
		shift 2
		;;
	-o | --output-path)
		OUTPUT_PATH="$2"
		shift 2
		;;
	-f | --force)
		FORCE_RUN=true
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

if [[ ! -d "${DIALFORK_PATH}" ]]; then
	echo "Error: input dataset root not found: ${DIALFORK_PATH}" >&2
	exit 1
fi

MODEL_NAME=$(echo "${NODE_EMBEDDING_MODEL}" | rev | cut -d'/' -f1 | rev)
RESULT_DIR="${OUTPUT_PATH}/${MODEL_NAME}"

mapfile -t TASKS < <(find "${DIALFORK_PATH}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

if [[ ${#TASKS[@]} -eq 0 ]]; then
	echo "No task directories found under: ${DIALFORK_PATH}"
	exit 0
fi

echo "=========================================="
echo "Node embedding model (folder key only): ${NODE_EMBEDDING_MODEL}"
echo "Input dataset root:                     ${DIALFORK_PATH}"
echo "Output root:                            ${RESULT_DIR}"
echo "Discovered tasks:                       ${#TASKS[@]}"
echo "=========================================="

ran=0
skipped=0

for task in "${TASKS[@]}"; do
	if [[ ! -f "${DIALFORK_PATH}/${task}/DONE.txt" ]]; then
		echo "[SKIP] ${task} (dataset generation not marked done)"
		((skipped += 1))
		continue
	fi

	if [[ "${FORCE_RUN}" != true && -f "${RESULT_DIR}/${task}/results_acc.json" ]]; then
		echo "[SKIP] ${task} (results already computed; use -f to force)"
		((skipped += 1))
		continue
	fi

	echo "[RUN] ${task}"
	python3 methods/task1/run.py "${task}" -i "${DIALFORK_PATH}" -o "${OUTPUT_PATH}" \
		-m "${NODE_EMBEDDING_MODEL}" --baselines-only
	((ran += 1))
done

echo ""
echo "Done. Ran: ${ran}, skipped: ${skipped}, total: ${#TASKS[@]}"
