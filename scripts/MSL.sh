#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Set the mode for the training phase
train_mode="train"
test_mode="test"

epoch_start=1
epoch_end=10
epoch_step=1

# Loop through the epochs for the training phase
for epoch in $(seq $epoch_start $epoch_step $epoch_end); do
        # Set the parameters for training
        echo "Running training with win_size=110 and epoch=$epoch"
        echo "Current winsize: 110"

        # Run the training script
         python main.py --mode "$train_mode" \
                       --win_size 110 \
                       --num_epochs "$epoch" \
                       --lr 0.0001 \
                       --gpu 0 \
                       --batch_size 128 \
                       --seed 2 \
                       --input_c 55 \
                       --output_c 55 \
                       --dataset MSL \
                       --data_path "../AnomalyDataset/MSL" \
                       --d_model 512 \
                       --e_layers 2 \
                       --model_save_path "cpt"

        # Set the mode for the testing phase
        echo "Running testing with anormly_ratio=1"

        # Set mode to test and run the testing script
        python main.py --mode "$test_mode" \
                        --win_size 110 \
                        --num_epochs "$epoch" \
                        --anormly_ratio 1 \
                        --lr 0.0001 \
                        --gpu 0 \
                        --batch_size 128 \
                        --seed 2 \
                        --input_c 55 \
                        --output_c 55 \
                        --dataset MSL \
                        --data_path "../AnomalyDataset/MSL" \
                        --d_model 512 \
                        --e_layers 2 \
                        --model_save_path "cpt"
done