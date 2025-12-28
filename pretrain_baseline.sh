#!/bin/bash

models=("NGNN" "adapter")
eval_terms=("betweenness" "eigenvector")
datasets=("flickr")

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 找到项目根目录（包含 data 目录的目录）
PROJECT_ROOT=""
if [ -d "$SCRIPT_DIR/data" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../data" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -d "$SCRIPT_DIR/../../data" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "Error: Cannot find data directory"
    exit 1
fi

echo "Project root: $PROJECT_ROOT"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 找到 tuning.py 的相对路径
TUNING_SCRIPT=""
if [ -f "./tuning.py" ]; then
    TUNING_SCRIPT="./tuning.py"
elif [ -f "./non_meta/tuning.py" ]; then
    TUNING_SCRIPT="./non_meta/tuning.py"
else
    echo "Error: Cannot find tuning.py"
    exit 1
fi

echo "Working directory: $(pwd)"
echo "Using tuning script: $TUNING_SCRIPT"

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for eval_term in "${eval_terms[@]}"; do
            # Define the pre-trained model path relative to the project root
            PRE_TRAIN_MODEL_PATH="saved_model/best_epoch_SAGE_5.pt"

            echo "Running tuning.py with dataset: $dataset, model: $model, eval_term: $eval_term"
            echo "Using pre-trained model: $PRE_TRAIN_MODEL_PATH"

            python "$TUNING_SCRIPT" \
                --gnn_name "$model" \
                --dataset "$dataset" \
                --dataset_name "$dataset" \
                --num_g_hid 128 \
                --test_only False \
                --epochs 30 \
                --task "importance" \
                --ablation "none" \
                --eval_term "$eval_term" \
                --no_subgraph True \

            echo "Completed: $dataset - $model with $eval_term"
            echo "----------------------------------------"
        done
    done
done

echo "All experiments completed!"