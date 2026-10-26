#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

# Script to run methods/task3/llm.py, one Python instance per task
# If no tasks are specified, discovers all tasks from the hallucinations dataset folder

# bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m gpt-oss:20b
# bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m gemma4:31b
# bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m qwen3.6:35b

# bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m gpt-oss:20b --no-workflow
# bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m gemma4:31b --no-workflow
# bash methods/task3/run_all_llm.sh -o results/task3/offline/ -m qwen3.6:35b --no-workflow


# Default values
mkdir -p logs/task3/offline   # Slurm writes the job logs there and fails if it is missing

DIALFORK_PATH="datasets/dialfork/"
TASKS=()
MODEL="gpt-oss:20b"
OUTPUT=""
VERBOSE=""
NO_WORKFLOW=""
SBATCH_SCRIPT="$(dirname "$0")/llm.sbatch"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-root|-d)
            DIALFORK_PATH="$2"
            shift 2
            ;;
        --tasks|-t)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                TASKS+=("$1")
                shift
            done
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --no-workflow)
            NO_WORKFLOW="1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate task-root is provided
if [[ -z "$DIALFORK_PATH" ]]; then
    echo "Error: --task-root (-d) is required"
    exit 1
fi

# If no tasks specified, discover all task directories
if [[ ${#TASKS[@]} -eq 0 ]]; then
    if [[ ! -d "$DIALFORK_PATH" ]]; then
        echo "Error: Task root directory does not exist: $DIALFORK_PATH"
        exit 1
    fi

    while IFS= read -r task_dir; do
        task_name=$(basename "$task_dir")
        TASKS+=("$task_name")
    done < <(find "$DIALFORK_PATH" -maxdepth 1 -mindepth 1 -type d | sort)

    if [[ ${#TASKS[@]} -eq 0 ]]; then
        echo "Error: No tasks found in $DIALFORK_PATH"
        exit 1
    fi
fi

# Verify selected SBATCH script exists
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Error: SBATCH script not found: $SBATCH_SCRIPT"
    exit 1
fi

# Extract model name (last part after /)
MODEL_NAME="${MODEL##*/}"

# Determine result folder path
if [[ -n "$OUTPUT" ]]; then
    RESULT_SUFFIX=""
    if [[ -n "$NO_WORKFLOW" ]]; then
        RESULT_SUFFIX="-no-workflow"
    fi
    RESULT_FOLDER="$OUTPUT/llm-$MODEL_NAME$RESULT_SUFFIX"
else
    RESULT_FOLDER=""
fi

# Filter out tasks that are already completed
PENDING_TASKS=()
SKIPPED_COUNT=0
for task in "${TASKS[@]}"; do
    if [[ -n "$RESULT_FOLDER" ]]; then
        RESULT_FILE="$RESULT_FOLDER/$task.json"
        if [[ -f "$RESULT_FILE" ]]; then
            ((SKIPPED_COUNT++))
            continue
        fi
    fi
    PENDING_TASKS+=("$task")
done

if [[ $SKIPPED_COUNT -gt 0 ]]; then
    echo "Skipped $SKIPPED_COUNT already-computed task(s)"
fi

if [[ ${#PENDING_TASKS[@]} -eq 0 ]]; then
    echo "All tasks already computed. Nothing to do."
    exit 0
fi

# Ask for confirmation
echo "=========================================="
echo "This script will launch ${#PENDING_TASKS[@]} SLURM jobs"
echo "Model: $MODEL"
echo "Mode: $([[ -n "$NO_WORKFLOW" ]] && echo "dialog-only (--no-workflow)" || echo "workflow+dialog (default)")"
echo "Total tasks found: ${#TASKS[@]} (${SKIPPED_COUNT} already completed)"
echo "=========================================="
echo ""
read -p "Do you want to proceed and launch ${#PENDING_TASKS[@]} jobs? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Cancelled."
    exit 0
fi

# Enable strict error handling only after confirmation
set -e

echo "Starting SLURM jobs..."
echo ""

for task in "${PENDING_TASKS[@]}"; do
    echo "Submitting SLURM job for task: $task"

    sbatch_args=(
        "--task-root" "$DIALFORK_PATH"
        "--tasks" "$task"
        "--model" "$MODEL"
    )

    if [[ -n "$OUTPUT" ]]; then
        sbatch_args+=("--output" "$OUTPUT")
    fi

    if [[ -n "$VERBOSE" ]]; then
        sbatch_args+=("--verbose")
    fi

    if [[ -n "$NO_WORKFLOW" ]]; then
        sbatch_args+=("--no-workflow")
    fi

    sbatch "$SBATCH_SCRIPT" "${sbatch_args[@]}"
done

echo ""
echo "All SLURM jobs submitted successfully."
