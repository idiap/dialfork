#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

mkdir -p logs/task2   # Slurm writes the job logs there and fails if it is missing

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  methods/task2/run_all.sh MODEL_NAME DISTANCE_METHOD [EXTRA_PY_ARGS...]

Description:
  Submit one Slurm job per STAR task for D2D experiments using
  methods/task2/run.sbatch.

  A task is skipped if its completion marker exists:
	results_d2d_summary.json

Examples:
  bash methods/task2/run_all.sh sentence-transformers/LaBSE mean-turn-emb
  bash methods/task2/run_all.sh sentence-transformers/LaBSE full-dialog-emb
  bash methods/task2/run_all.sh sentence-transformers/LaBSE flowdtw
  bash methods/task2/run_all.sh sentence-transformers/LaBSE flowscore

  bash methods/task2/run_all.sh Qwen/Qwen3-Embedding-8B mean-turn-emb
  bash methods/task2/run_all.sh Qwen/Qwen3-Embedding-8B full-dialog-emb
  bash methods/task2/run_all.sh Qwen/Qwen3-Embedding-8B flowdtw
  bash methods/task2/run_all.sh Qwen/Qwen3-Embedding-8B flowscore

  bash methods/task2/run_all.sh sentence-transformers/all-mpnet-base-v2 mean-turn-emb
  bash methods/task2/run_all.sh sentence-transformers/all-mpnet-base-v2 full-dialog-emb
  bash methods/task2/run_all.sh sentence-transformers/all-mpnet-base-v2 flowdtw
  bash methods/task2/run_all.sh sentence-transformers/all-mpnet-base-v2 flowscore

  bash methods/task2/run_all.sh BAAI/bge-m3 mean-turn-emb
  bash methods/task2/run_all.sh BAAI/bge-m3 full-dialog-emb
  bash methods/task2/run_all.sh BAAI/bge-m3 flowdtw
  bash methods/task2/run_all.sh BAAI/bge-m3 flowscore

  bash methods/task2/run_all.sh sergioburdisso/dialog2flow-joint-dse-base mean-turn-emb
  bash methods/task2/run_all.sh sergioburdisso/dialog2flow-joint-dse-base full-dialog-emb
  bash methods/task2/run_all.sh sergioburdisso/dialog2flow-joint-dse-base flowdtw
  bash methods/task2/run_all.sh sergioburdisso/dialog2flow-joint-dse-base flowscore

  bash methods/task2/run_all.sh aws-ai/dse-bert-base mean-turn-emb
  bash methods/task2/run_all.sh aws-ai/dse-bert-base full-dialog-emb
  bash methods/task2/run_all.sh aws-ai/dse-bert-base flowdtw
  bash methods/task2/run_all.sh aws-ai/dse-bert-base flowscore

  bash methods/task2/run_all.sh google/embeddinggemma-300m mean-turn-emb
  bash methods/task2/run_all.sh google/embeddinggemma-300m full-dialog-emb
  bash methods/task2/run_all.sh google/embeddinggemma-300m flowdtw
  bash methods/task2/run_all.sh google/embeddinggemma-300m flowscore
EOF
}

if [[ $# -lt 2 ]]; then
	usage
	exit 1
fi

MODEL_NAME="$1"
DISTANCE_METHOD="$2"
shift 2
EXTRA_ARGS=("$@")

# Keep paths aligned with methods/task2/run.py defaults.
INPUT_ROOT="datasets/dialfork"
OUTPUT_ROOT="results/task2"
SBATCH_FILE="methods/task2/run.sbatch"
COMPLETION_FILE="results_d2d_summary.json"

if [[ ! -d "${INPUT_ROOT}" ]]; then
	echo "Error: input dataset root not found: ${INPUT_ROOT}" >&2
	exit 1
fi

if [[ ! -f "${SBATCH_FILE}" ]]; then
	echo "Error: sbatch file not found: ${SBATCH_FILE} (run from the repository root)" >&2
	exit 1
fi

MODEL_SUFFIX="${MODEL_NAME##*/}"
if [[ "${DISTANCE_METHOD}" == "bertscore" || "${DISTANCE_METHOD}" == "bertscore-bert" ]]; then
	EXPERIMENT_FOLDER="${DISTANCE_METHOD}"
else
	EXPERIMENT_FOLDER="${DISTANCE_METHOD}/${MODEL_SUFFIX}"
fi

mapfile -t TASKS < <(find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

if [[ ${#TASKS[@]} -eq 0 ]]; then
	echo "No task directories found under: ${INPUT_ROOT}"
	exit 0
fi

echo "=========================================="
echo "Model:            ${MODEL_NAME}"
echo "Distance method:  ${DISTANCE_METHOD}"
echo "Input tasks root: ${INPUT_ROOT}"
echo "Output root:      ${OUTPUT_ROOT}/${EXPERIMENT_FOLDER}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	echo "Extra python args: ${EXTRA_ARGS[*]}"
fi
echo "Discovered tasks: ${#TASKS[@]}"
echo "=========================================="

pending=0
skipped=0

for task in "${TASKS[@]}"; do
	marker_path="${OUTPUT_ROOT}/${EXPERIMENT_FOLDER}/${task}/${COMPLETION_FILE}"
	if [[ -f "${marker_path}" ]]; then
		((skipped += 1))
	else
		((pending += 1))
	fi
done

echo "Completed (will skip): ${skipped}"
echo "Pending (will submit): ${pending}"

if [[ ${pending} -eq 0 ]]; then
	echo "No pending tasks found. Nothing was submitted."
	exit 0
fi

read -r -p "Proceed to submit ${pending} jobs? (yes/no): " confirmation
if [[ "${confirmation}" != "yes" ]]; then
	echo "Cancelled. No jobs were submitted."
	exit 0
fi

submitted=0

for task in "${TASKS[@]}"; do
	marker_path="${OUTPUT_ROOT}/${EXPERIMENT_FOLDER}/${task}/${COMPLETION_FILE}"

	if [[ -f "${marker_path}" ]]; then
		echo "[SKIP] ${task} (already completed: ${marker_path})"
		continue
	fi

	echo "[SUBMIT] ${task}"
	if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
		sbatch "${SBATCH_FILE}" "${MODEL_NAME}" "${DISTANCE_METHOD}" "${task}" "${EXTRA_ARGS[@]}"
	else
		sbatch "${SBATCH_FILE}" "${MODEL_NAME}" "${DISTANCE_METHOD}" "${task}"
	fi
	((submitted += 1))
done

echo ""
echo "Submission summary:"
echo "  Total tasks: ${#TASKS[@]}"
echo "  Skipped:     ${skipped}"
echo "  Submitted:   ${submitted}"
