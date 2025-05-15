#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1

# 设置默认参数
MODEL_SAVE_PATH="cpt"
WIN_SIZE=110
ANOMALY_RATIO=1.0
NUM_EPOCHS=2
LR=0.0001
GPU=1
BATCH_SIZE=128
D_MODEL=512
E_LAYERS=2
SEED=2
# 数据集配置
declare -A datasets=(
    ["MSL"]="55 55"
    ["SWaT"]="51 51"
    ["PSM"]="26 26"
    ["WADI"]="123 123"
    ["NIPS_TS_Water"]="9 9"
    ["NIPS_TS_Swan"]="38 38"
)

# 检查是否提供数据集名称
if [ -z "$1" ]; then
    echo "No dataset specified. Running for all datasets: ${!datasets[@]}"
    run_all=true
else
    DATASET=$1
    if [[ -z "${datasets[$DATASET]}" ]]; then
        echo "Unknown dataset: $DATASET"
        exit 1
    fi
    run_all=false
fi

# 运行指定数据集或所有数据集
if [ "$run_all" = true ]; then
    for DATASET in "${!datasets[@]}"; do
        IFS=' ' read -r INPUT_C OUTPUT_C <<< "${datasets[$DATASET]}"
        DATA_PATH="../AnomalyDataset/$DATASET"
        echo "Running for dataset: $DATASET"

        python main.py  --mode test \
                        --describe none \
                        --win_size $WIN_SIZE \
                        --anormly_ratio $ANOMALY_RATIO \
                        --num_epochs $NUM_EPOCHS \
                        --lr $LR \
                        --gpu $GPU \
                        --batch_size $BATCH_SIZE \
                        --seed $SEED \
                        --input_c $INPUT_C \
                        --output_c $OUTPUT_C \
                        --dataset $DATASET \
                        --data_path $DATA_PATH \
                        --d_model $D_MODEL \
                        --e_layers $E_LAYERS \
                        --model_save_path $MODEL_SAVE_PATH
    done
else
    IFS=' ' read -r INPUT_C OUTPUT_C <<< "${datasets[$DATASET]}"
    DATA_PATH="dataset/$DATASET"
    echo "Running for dataset: $DATASET"

    python main.py  --mode test \
                    --win_size $WIN_SIZE \
                    --anormly_ratio $ANOMALY_RATIO \
                    --num_epochs $NUM_EPOCHS \
                    --lr $LR \
                    --gpu $GPU \
                    --batch_size $BATCH_SIZE \
                    --seed $SEED \
                    --input_c $INPUT_C \
                    --output_c $OUTPUT_C \
                    --dataset $DATASET \
                    --data_path $DATA_PATH \
                    --d_model $D_MODEL \
                    --e_layers $E_LAYERS \
                    --model_save_path $MODEL_SAVE_PATH
fi