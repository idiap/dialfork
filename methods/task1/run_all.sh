#!/bin/bash

# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

mkdir -p logs/task1   # Slurm writes the job logs there and fails if it is missing

NODE_EMBEDDING_MODEL="sergioburdisso/dialog2flow-joint-dse-base"
FORCE_RUN=false
DIALFORK_PATH="datasets/dialfork"
OUTPUT_PATH="results/task1"

# Parse remaining arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--node-embedding-model)
            NODE_EMBEDDING_MODEL="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_RUN=true
            shift
            ;;
        -i|--input-path)
            DIALFORK_PATH="$2"
            shift 2
            ;;
        -o|--output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

DATASET_DIR="$DIALFORK_PATH"
RESULT_DIR="${OUTPUT_PATH}/"
if [ -n "$NODE_EMBEDDING_MODEL" ]; then
    # Extract model name from path (e.g., "path/to/model" -> "model")
    MODEL_NAME=$(echo "$NODE_EMBEDDING_MODEL" | rev | cut -d'/' -f1 | rev)
    RESULT_DIR+=$MODEL_NAME
fi

# Count folders with DONE.txt and without results_acc.json to show user how many jobs will be launched
count=0

for folder in "$DATASET_DIR"/*; do
    if [ -d "$folder" ]; then
        task=$(basename "$folder")
        # Only count if:
        #   --force is set AND DONE.txt exists, or
        #   (DONE.txt exists AND results do not exist)
        if { [ "$FORCE_RUN" = true ] && [ -f "$folder/DONE.txt" ]; } || { [ -f "$folder/DONE.txt" ] && [ ! -f "$RESULT_DIR/$task/results_acc.json" ]; }; then
            ((count++))
        fi
    fi
done

# Exit if there are no jobs to run
if [ $count -eq 0 ]; then
    echo "No jobs to run. All results already computed."
    exit 0
fi

# Ask for confirmation
echo "=========================================="
echo "This script will launch $count SLURM jobs"
if [ "$FORCE_RUN" = true ]; then
    echo "(Force run enabled - will re-run existing results)"
fi
echo "=========================================="
read -p "Do you want to proceed? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo "Starting SLURM jobs..."
echo ""

# Iterate over all folders in the hallucinations directory

for folder in "$DATASET_DIR"/*; do
    if [ -d "$folder" ]; then
        task=$(basename "$folder")

        # Only run if DONE.txt exists
        if [ -f "$folder/DONE.txt" ]; then
            if [ "$FORCE_RUN" = true ]; then
                echo "Running sbatch for task: $task (force run)"
                sbatch_args=("$task" "-i" "$DIALFORK_PATH" "-o" "$OUTPUT_PATH")
                if [ "$USE_CLOSEST_CENTROID" = true ]; then
                    sbatch_args+=("-c")
                fi
                if [ -n "$NODE_EMBEDDING_MODEL" ]; then
                    sbatch_args+=("-m" "$NODE_EMBEDDING_MODEL")
                fi
                sbatch methods/task1/run.sbatch "${sbatch_args[@]}"
            else
                if [ -f "$RESULT_DIR/$task/results_acc.json" ]; then
                    echo "Skipping $task (results already computed)"
                else
                    echo "Running sbatch for task: $task"
                    sbatch_args=("$task" "-i" "$DIALFORK_PATH" "-o" "$OUTPUT_PATH")
                    if [ "$USE_CLOSEST_CENTROID" = true ]; then
                        sbatch_args+=("-c")
                    fi
                    if [ -n "$NODE_EMBEDDING_MODEL" ]; then
                        sbatch_args+=("-m" "$NODE_EMBEDDING_MODEL")
                    fi
                    sbatch methods/task1/run.sbatch "${sbatch_args[@]}"
                fi
            fi
        else
            echo "Skipping $task (DONE.txt not found)"
        fi
    fi
done
