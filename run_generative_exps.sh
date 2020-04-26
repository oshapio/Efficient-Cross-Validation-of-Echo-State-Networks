#!/bin/bash

cd precision_experiments/generative_mode
Datasets=("unempl" "electr" "sunspots" "gasoline")

for seed in {1..5}; do
    echo "Running generative_exps with seed - $seed"
    for dataset in "${Datasets[@]}"; do
        echo "Using $dataset $dataset"
        (python generative_exp.py --dataset $dataset --valid_method cross --final_model retrain --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method cross --final_model average --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method cross --final_model best --seed $seed &&

         python generative_exp.py --dataset $dataset --valid_method accum --final_model retrain --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method accum --final_model average --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method accum --final_model best --seed $seed &&

         python generative_exp.py --dataset $dataset --valid_method walk_forward --final_model retrain --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method walk_forward --final_model average --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method walk_forward --final_model best --seed $seed &&

         python generative_exp.py --dataset $dataset --valid_method standard --final_model retrain --seed $seed &&
         python generative_exp.py --dataset $dataset --valid_method standard --final_model average --seed $seed
        ) &
    done

    wait
done
