#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

# For each task (each folder in datasets/STAR/tasks),
# run sbatch generation/generate.sbatch TASK_NAME so all tasks' dialogues can be generated in parallel

mkdir -p logs/dataset   # Slurm writes the job logs there and fails if it is missing

STAR_PATH="datasets/STAR/"
DIALFORK_PATH="datasets/dialfork"

TASKS_DIR="$STAR_PATH/tasks"
SKIP_TASKS=()

usage() {
    echo "Usage: $0 [-o OUTPUT_PATH] [--skip task1,task2,... | --skip task1 task2 ...]"
    echo ""
    echo "  -o, --output-path   where to write the dialogues (default: $DIALFORK_PATH)."
    echo "                      Tasks whose folder there has a DONE.txt are skipped, so to"
    echo "                      regenerate next to the downloaded data, use another folder."
    echo ""
    echo "Examples:"
    echo "  $0 -o datasets/dialfork_regenerated"
    echo "  $0 -o datasets/dialfork_regenerated --skip weather,trivia"
}

is_skipped_task() {
    local candidate="$1"
    local t
    for t in "${SKIP_TASKS[@]}"; do
        if [ "$t" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

# Parse CLI args
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -o|--output-path)
            DIALFORK_PATH="$2"
            shift 2
            ;;
        --skip)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --skip requires at least one task name."
                usage
                exit 1
            fi
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
                IFS=',' read -r -a parts <<< "$1"
                SKIP_TASKS+=("${parts[@]}")
                shift
            done
            ;;
        *)
            echo "Error: unknown argument $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -d "$TASKS_DIR" ]; then
    echo "Error: Tasks directory not found at $TASKS_DIR"
    exit 1
fi

# Pre-compute the number of jobs to run
jobs_to_run=0
skipped_tasks=0
manually_skipped_tasks=0

for task_folder in "$TASKS_DIR"/*; do
    if [ -d "$task_folder" ]; then
        task_name=$(basename "$task_folder")
        done_file="$DIALFORK_PATH/$task_name/DONE.txt"

        if is_skipped_task "$task_name"; then
            ((manually_skipped_tasks++))
            continue
        fi

        if [ -f "$done_file" ]; then
            ((skipped_tasks++))
        else
            ((jobs_to_run++))
        fi
    fi
done

echo "=============================================="
echo "Job Summary:"
echo "  Output: $DIALFORK_PATH"
echo "  Tasks to run: $jobs_to_run"
echo "  Tasks already done: $skipped_tasks"
echo "  Tasks skipped manually: $manually_skipped_tasks"
echo "  Manual skip list: ${SKIP_TASKS[*]}"
echo "=============================================="

# Ask for confirmation
read -p "Do you want to submit these $jobs_to_run job(s)? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs for all tasks in $TASKS_DIR"
echo "=============================================="

# Iterate through each task folder and submit jobs
for task_folder in "$TASKS_DIR"/*; do
    if [ -d "$task_folder" ]; then
        task_name=$(basename "$task_folder")
        done_file="$DIALFORK_PATH/$task_name/DONE.txt"

        if is_skipped_task "$task_name"; then
            echo "Skipping task: $task_name (manually skipped)"
            continue
        fi

        if [ -f "$done_file" ]; then
            echo "Skipping task: $task_name (already done)"
        else
            echo "Submitting job for task: $task_name"
            sbatch generation/generate.sbatch --output-path "$DIALFORK_PATH" "$task_name"
        fi
    fi
done

echo "=============================================="
echo "All jobs submitted!"
