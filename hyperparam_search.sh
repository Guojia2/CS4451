#!/bin/bash

# Define the hyperparameters to search over
learning_rates=("1e-3" "5e-4" "1e-4")
lambda_freq_values=(0.00 0.01 0.1 0.8 1.0)  #
backbones=("itransformer" "tsmixer")  #
datasets=("ETTh1" "ETTm1" "ILI" "Exchange")

seq_len=96
pred_len=96

# Log file
log_file="hyperparam_search_results.log"

# Clear  log file at the start
> $log_file

# run a single experiment
run_experiment() {
    local dataset=$1
    local backbone=$2
    local lr=$3
    local lambda_freq=$4


    echo "Running with Dataset=$dataset, Backbone=$backbone, LR=$lr, Lambda_freq=$lambda_freq"

    # train the model, outputting results to the log file
    python main.py --dataset $dataset \
                   --backbone $backbone \
                   --lr $lr \
                   --lambda_freq $lambda_freq \
                   --seq_len $seq_len \
                   --pred_len $pred_len \
                    >> $log_file 2>&1

}

# Loop over all combinations of hyperparameters and datasets
for dataset in "${datasets[@]}"; do
    for backbone in "${backbones[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for lambda_freq in "${lambda_freq_values[@]}"; do
                run_experiment $dataset $backbone $lr $lambda_freq
            done
        done
    done
done

echo "Hyperparameter search completed! Results saved to $log_file."
